#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from typing import Union, Optional
from dataclasses import dataclass, asdict
from contextlib import nullcontext

import torch
import torch.nn as nn
import numpy as np
import math
import einops
from timm.models.vision_transformer import Attention, Mlp

from icecream import ic
import torch as th
import torch.nn.functional as F

try:
    from diffusers.models.unet_1d import (
        UNet1DOutput
    )
except ImportError:
    from diffusers.models.unets.unet_1d import (
        UNet1DOutput
    )


from presto.network.layers import SinusoidalPositionalEncoding
from presto.network.pcd_encoder import PointCloudEncoderFPSMLPMLP


class WrappedAttention(Attention):
    def forward(self, x: th.Tensor):
        s = x.shape
        x = x.reshape(-1, *s[-2:])
        y = super().forward(x)
        y = y.reshape(*s[:-2], *y.shape[-2:])
        return y


class PatchEmbed(nn.Module):
    def __init__(self,
                 input_size: int,
                 in_channels: int,
                 patch_size: int,
                 hidden_size: int,
                 sin_emb: int = 0,
                 cat: bool = False,
                 split_c: bool = False
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.split_c = split_c

        emb_in: int = in_channels
        if split_c:
            emb_in = 1

        if sin_emb > 0:
            c_in = max(sin_emb, 2 * emb_in)
            self.emb_x = SinusoidalPositionalEncoding(emb_in,
                                                      c_in,
                                                      pad=True,
                                                      cat=cat)
        else:
            self.emb_x = nn.Identity()
            c_in = in_channels
        self.proj = nn.Linear(patch_size * c_in,
                              hidden_size)
        if split_c:
            self.num_patches = input_size // patch_size * in_channels
        else:
            self.num_patches = input_size // patch_size

    def forward(self, x):
        # x:  NXCxT -> NxTxD I guess
        if self.split_c:
            c: int = x.shape[-2]
            x = einops.rearrange(x, '... c s -> ... (s c)')[..., None]
            x = self.emb_x(x)
            x = einops.rearrange(x, '... (n g c) d -> ... (n c) (g d)',
                                 c=c,
                                 g=self.patch_size)
            x = self.proj(x)
        else:
            x = einops.rearrange(x, '... d s -> ... s d')
            x = self.emb_x(x)
            x = einops.rearrange(x, '... (n g) d -> ... n (g d)',
                                 g=self.patch_size)
            x = self.proj(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)


##########################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
##########################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(
            start=0,
            end=half,
            dtype=torch.float32,
            device=t.device) / half)
        # args = t[..., None].float() * freqs[None]
        args = th.einsum('..., d -> ... d', t, freqs)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, : 1])],
                dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class CondEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, cond_dim, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Linear(cond_dim, hidden_size)
        self.cond_dim = cond_dim
        self.dropout_prob = dropout_prob

    def forward(self, labels, train):
        use_dropout = self.dropout_prob > 0
        embeddings = self.embedding_table(labels)
        return embeddings


##########################################################################
#                                 Core DiT Model                                #
##########################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio=4.0,
                 use_cond: bool = True,
                 **block_kwargs):
        super().__init__()
        self.use_cond = use_cond
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=(not use_cond), eps=1e-6)
        self.attn = WrappedAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=(not use_cond), eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0)

        if use_cond:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

    def forward(self, x, c):
        if self.use_cond:
            # conditional
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=-1)
            x = x + gate_msa.unsqueeze(-2) * self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(-2) * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            # unconditional
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels,
                 use_cond: bool = True):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=(not use_cond),
            eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            patch_size * out_channels,
            bias=True)
        self.use_cond = use_cond
        if use_cond:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def forward(self, x, c):
        if self.use_cond:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            x = self.norm_final(x)
        x = self.linear(x)
        return x


class DiTImpl(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=1000,
        patch_size=20,
        in_channels=7,
        hidden_size=256,
        num_layer=4,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        cond_dim: int = 104,
        learn_sigma=True,
        use_cond: bool = True,
        use_pos_emb: bool = False,
        dim_pos_emb: int = 3 * 2 * 32,
        sin_emb_x: int = 0,
        cat_emb_x: bool = False,
        use_cond_token: bool = False,
        use_cloud: bool = False,

        cloud_size: int = 1024,
        num_cloud_token: int = 8,
        num_cloud_layer: int = 4,
        norm_cloud_token: bool = False,
        cloud_block2: bool = False,
        cond_type: str = 'mean',
        # split channels into
        # individual tokens.
        split_c: bool = False,
        cloud_embedder_type: str = 'patch'  # else 'point'
    ):
        super().__init__()
        # input_size=1
        # raise ValueError(F'Got input_size = {input_size}')
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        if split_c:
            self.out_channels = 2 if learn_sigma else 1
        else:
            self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cond = use_cond
        self.use_cond_token = use_cond_token
        self.use_cloud = use_cloud
        self.num_cloud_token = num_cloud_token
        self.norm_cloud_token = norm_cloud_token
        self.cloud_block2 = cloud_block2
        self.cond_type = cond_type
        self.split_c = split_c
        assert (self.use_cloud)
        if num_cloud_token > 0:
            assert (use_cond_token)

        self.x_embedder = PatchEmbed(input_size, in_channels,
                                     patch_size, hidden_size,
                                     sin_emb=sin_emb_x,
                                     cat=cat_emb_x,
                                     split_c=split_c)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if use_cloud:
            if cloud_embedder_type == 'patch':
                self.cloud_embedder = PointCloudEncoderFPSMLPMLP(
                    PointCloudEncoderFPSMLPMLP.Config(
                        num_layer=num_cloud_layer,
                        num_group=self.num_cloud_token,
                        patch_size=cloud_size // num_cloud_token,
                        embed_size=hidden_size,
                        point_size=3,
                        norm_out=self.norm_cloud_token,
                        use_block2=self.cloud_block2,
                    ))
        if use_pos_emb:
            self.y_embedder = nn.Sequential(
                SinusoidalPositionalEncoding(cond_dim, dim_pos_emb),
                CondEmbedder(dim_pos_emb, hidden_size, class_dropout_prob)
            )
        else:
            self.y_embedder = CondEmbedder(
                cond_dim, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches

        # Will use fixed sin-cos embedding:
        self.register_buffer('pos_embed',
                             # torch.zeros(1, num_patches, hidden_size),
                             torch.zeros(num_patches, hidden_size),
                             persistent=False)

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, num_heads,
                mlp_ratio=mlp_ratio,
                use_cond=True
            )
            for _ in range(num_layer)])
        self.final_layer = FinalLayer(
            hidden_size,
            patch_size,
            self.out_channels,
            use_cond=True)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches))
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float()
        )

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.use_cond:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Arg:
            x:
                if split_c:
                    (N, T * C //patch_size, patch_size)
                else:
                    (N, T//patch_size, patch_size * C)
        Return:
            y: (N, T, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size
        s = int(x.shape[1])

        if self.split_c:
            x = einops.rearrange(x, '... (s c) p -> ... c (s p)',
                                    c=self.in_channels
                                 )
        else:
            x = einops.rearrange(x, '... s (p c) -> ... c (s p)',
                                    c=self.out_channels)
        return x

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        if not th.is_tensor(t):
            t = th.tensor([t],
                          dtype=th.long,
                          device=x.device)
        elif th.is_tensor(t) and len(t.shape) == 0:
            # t = t[None]
            pass
        t = t.to(x.device)

        # (N, T, D), where T = H * W / patch_size ** 2
        x = self.x_embedder(x)
        x = x + self.pos_embed
        # raise ValueError('stop')

        t = self.t_embedder(t)                   # (N, D)

        y_in = y
        if self.use_cond:
            if self.use_cloud:
                y = self.y_embedder(y_in['task'], self.training)    # (N, D)
            else:
                y = self.y_embedder(y, self.training)    # (N, D)
            c = t + y                                # (N, D)
        else:
            c = t

        # Exactly _one_ cond token...!
        c = c[..., None, :]

        if self.use_cloud:
            if 'z_pcd' in y_in:
                z_pcd = y_in['z_pcd']
            else:
                z_pcd = self.cloud_embedder(y_in['cloud'])  # (N, #P, D)
            c = th.cat([z_pcd, c], dim=-2)  # (N, P+1, D)

        # Is averaging across tokens really a good idea
        # (for feeding into FiLM layers)?
        # Perhaps it's OK since we also use `cond` tokens...
        if self.cond_type == 'mean':
            c_flt = c.mean(dim=-2)
        else:
            c_flt = c[..., -1, :]

        x0 = x
        if self.use_cond_token:
            x = th.cat([x, c], dim=-2)

        for block in self.blocks:
            # (N, T, D)
            x = block(x, c_flt)
        # (N, T, patch_size ** 2 * out_channels)
        x = self.final_layer(x, c_flt)
        if self.use_cond_token:
            if self.use_cloud:
                x = x[..., :-(self.num_cloud_token + 1), :]
            else:
                x = x[..., :-1, :]
        # (N, out_channels, H, W)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """

        if not th.is_tensor(t):
            t = th.tensor([t],
                          dtype=th.long,
                          device=x.device)
        elif th.is_tensor(t) and len(t.shape) == 0:
            t = t[None]
        t = t.to(x.device)

        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiTCloud(DiTImpl):
    @dataclass
    class Config:
        input_size: int = 1000
        patch_size: int = 20
        in_channels: int = 7
        hidden_size: int = 256
        num_layer: int = 4
        num_heads: int = 16
        mlp_ratio: float = 4.0
        class_dropout_prob: float = 0.0
        cond_dim: int = 104
        learn_sigma: bool = True
        use_cond: bool = True
        use_pos_emb: bool = False
        dim_pos_emb: int = 3 * 2 * 32
        use_amp: Optional[bool] = None
        sin_emb_x: int = 0
        cat_emb_x: bool = False
        use_cond_token: bool = False
        use_cloud: bool = False
        cloud_size: int = 1024
        num_cloud_token: int = 8
        num_cloud_layer: int = 4
        norm_cloud_token: bool = False
        cloud_block2: bool = False
        cond_type: str = 'mean'
        split_c: bool = False

    def __init__(self, cfg: Config):
        kwds = asdict(cfg)
        use_amp = kwds.pop('use_amp', None)
        super().__init__(**kwds)
        self._use_amp = use_amp

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def forward(self,
                sample: th.FloatTensor,
                timestep: Union[th.Tensor, float, int],
                class_labels: Optional[th.Tensor] = None,
                return_dict: bool = True):

        amp_ctx = (nullcontext() if (self._use_amp is None)
                   else th.cuda.amp.autocast(enabled=self._use_amp))
        with amp_ctx:
            return UNet1DOutput(super().forward(sample, timestep,
                                                class_labels))


##########################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
##########################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_1d_sincos_pos_embed(
        embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=np.float32)
    grid = grid.reshape([1, 1, grid_size])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def test_attn():
    device: str = 'cuda:0'
    attn = WrappedAttention(128).to(device)
    x = th.zeros((1, 128), device=device)
    y = attn(x)


def test_dit():
    device: str = 'cuda:0'
    model = DiTCloud(DiTCloud.Config(128, 8, 4,
                                     hidden_size=256,
                                     num_layer=4,
                                     num_heads=4,
                                     cond_dim=10,
                                     sin_emb_x=16,
                                     cat_emb_x=True,
                                     use_cond_token=True,
                                     use_cloud=True,
                                     cloud_block2=True,
                                     split_c=True,
                                     learn_sigma=False
                                     )).to(device=device)
    ic(model)

    x = th.zeros((2, 5, 4, 128), device=device)
    # patch size = 8, 128/8*4=64 patches
    t = th.zeros((2, 5,), device=device)
    # c = th.zeros((2, 5, 10), device=device)
    c = dict(
        task=th.zeros((2, 5, 10), device=device),
        cloud=th.zeros((2, 5, 512, 3), device=device)
    )
    y = model(x, t, class_labels=c)  # -> 2,5,8,128
    print('y', y.sample.shape)


def test_xfm():
    model = PointTransformerWrapper(1024)


def main():
    test_xfm()


if __name__ == '__main__':
    main()
