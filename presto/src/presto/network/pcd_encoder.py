#!/usr/bin/env python3

from dataclasses import dataclass, replace
from typing import Optional, Dict, Tuple, Union

import torch as th
import torch.nn as nn
import einops

from flash_attn.modules.mha import MHA
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.sample_farthest_points import sample_farthest_points
from timm.models.vision_transformer import Attention, Mlp

from presto.network.common import MLP, get_activation_function

from icecream import ic


class WrappedAttention(Attention):
    def forward(self, x: th.Tensor):
        s = x.shape
        x = x.reshape(-1, *s[-2:])
        y = super().forward(x)
        y = y.reshape(*s[:-2], *y.shape[-2:])
        return y


class GroupFPS(nn.Module):
    """
    Group points via farthest-point sampling.
    Same as PointBERT / PointMAE variants.
    """

    @dataclass
    class Config:
        # number of groups
        num_group: int = 16
        # number of points per group
        num_point: int = 32

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self,
                x: th.Tensor,
                center: Optional[th.Tensor] = None,
                sort: bool = True,
                aux: Optional[Dict[str, th.Tensor]] = None
                ) -> Tuple[th.Tensor, th.Tensor]:
        cfg = self.cfg

        s = x.shape
        x = x.reshape(-1, *x.shape[-2:])

        if center is None:
            c, _ = sample_farthest_points(x, K=cfg.num_group)
        else:
            c = center

        _, nn_idx, p = knn_points(c, x,
                                  K=cfg.num_point,
                                  return_nn=True,
                                  return_sorted=sort)

        c = c.reshape(*s[:-2], *c.shape[1:])
        p = p.reshape(*s[:-2], *p.shape[1:])
        i = nn_idx
        return (p, c, i)

    def extra_repr(self):
        cfg = self.cfg
        return F'group={cfg.num_group}, point={cfg.num_point}'


class MLPPatchEncoder(nn.Module):
    """
    Encode patches via multi-layer perceptron (MLP).
    Only works if the patches are guaranteed to be sorted;
    @see `sort` argument in `GroupFPS`.
    """
    @dataclass
    class Config:
        patch_size: int = 32
        embed_size: int = 128
        point_size: int = 3
        hidden: Tuple[int, ...] = (256, 256)
        pre_ln_bias: bool = False
        act_cls: str = 'gelu'
        norm: str = 'layernorm'

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dims = (cfg.patch_size * cfg.point_size,
                *cfg.hidden,
                cfg.embed_size)
        self.mlp = MLP(
            dims,
            get_activation_function(cfg.act_cls),
            activate_output=False,
            norm=cfg.norm,
            bias=True,
            pre_ln_bias=cfg.pre_ln_bias
        )

    def forward(self, x: th.Tensor):
        cfg = self.cfg
        x = einops.rearrange(x,
                             '... g n p -> ... g (n p)',
                             p=cfg.point_size)
        out = self.mlp(x)
        return out

    def extra_repr(self):
        cfg = self.cfg
        return F'patch_size={cfg.patch_size}, embed_size={cfg.embed_size}'


class PosEncodingMLP(nn.Module):
    """
    Positional encoding with an MLP.
    Other popular choices are
    random fourier projections, NeRF/transformer-style
    analytical sinusoidal encodings, etc.
    """
    @dataclass
    class Config:
        dim_in: int = 3
        dim_out: int = 128
        dim_hidden: Tuple[int, ...] = (128,)
        act_cls: str = 'gelu'
        cat: bool = False
        norm: str = 'none'

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dims = (cfg.dim_in, *cfg.dim_hidden, cfg.dim_out)
        self.mlp = MLP(dims,
                       act_cls=get_activation_function(cfg.act_cls),
                       activate_output=False,
                       norm=cfg.norm,
                       bias=True)

    @property
    def out_dim(self):
        cfg = self.cfg
        return cfg.dim_in + cfg.dim_out

    def forward(self, x: th.Tensor) -> th.Tensor:
        cfg = self.cfg
        y = self.mlp(x)
        if cfg.cat:
            out = th.cat([x, y], dim=-1)
        else:
            out = y
        return out


class SABlock(nn.Module):
    """
    Self-attention block for a transformer.
    """
    @dataclass
    class Config:
        hidden_size: int = 128
        num_head: int = 4
        qkv_bias: bool = True
        p_dropout: float = 0.0

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        if cfg.hidden_size % cfg.num_head != 0:
            raise ValueError(
                F"Hidden size={cfg.hidden_size,}" +
                F"is not a multiple of the number of attention " +
                F"heads={cfg.num_head}.")
        self.attention = WrappedAttention(cfg.hidden_size,
                                          cfg.num_head,
                                          qkv_bias=cfg.qkv_bias)

    def forward(self, hidden_states: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        x = hidden_states.reshape(-1, *hidden_states.shape[-2:])
        x = self.attention(x)
        x = x.reshape(*hidden_states.shape[:-2], *x.shape[-2:])
        x = x.to(dtype=hidden_states.dtype)
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block for a ViT/Point-MAE style
    transformer.
    """
    @dataclass
    class Config:
        attention: SABlock.Config = SABlock.Config()
        hidden_size: int = 128
        layer_norm_eps: float = 1e-6

        def __post_init__(self):
            self.attention = replace(self.attention,
                                     hidden_size=self.hidden_size)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attention = SABlock(cfg.attention)
        self.ln1 = nn.LayerNorm(cfg.hidden_size,
                                eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size,
                                eps=cfg.layer_norm_eps)
        self.linear = nn.Linear(cfg.hidden_size,
                                cfg.hidden_size)

    def forward(self, x: th.Tensor):
        # dx, ap = self.attention(self.ln1(x))
        dx = self.attention(self.ln1(x))
        x = x + dx
        x = x + self.linear(self.ln2(x))
        return x

    def extra_repr(self):
        cfg = self.cfg
        return F'hidden_size={cfg.hidden_size}, eps={cfg.layer_norm_eps}'


class EncoderBlock2(nn.Module):
    """
    Encoder block for a ViT/Point-MAE style
    transformer.
    """
    @dataclass
    class Config:
        attention: SABlock.Config = SABlock.Config()
        hidden_size: int = 128
        layer_norm_eps: float = 1e-6

        def __post_init__(self):
            self.attention = replace(self.attention,
                                     hidden_size=self.hidden_size)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attention = SABlock(cfg.attention)
        self.ln1 = nn.LayerNorm(cfg.hidden_size,
                                eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size,
                                eps=cfg.layer_norm_eps)
        self.act = nn.GELU()
        self.linear1 = nn.Linear(cfg.hidden_size,
                                 cfg.hidden_size)
        self.linear2 = nn.Linear(cfg.hidden_size,
                                 cfg.hidden_size)

    def forward(self, x: th.Tensor):
        ln1 = self.ln1(x)
        dx = self.attention(ln1)
        x = x + dx
        dx = self.linear2(self.act(self.linear1(self.ln2(x))))
        x = x + dx
        return x

    def extra_repr(self):
        cfg = self.cfg
        return F'hidden_size={cfg.hidden_size}, eps={cfg.layer_norm_eps}'


class PointCloudEncoder(nn.Module):
    """
    Wrapper class around the general structure of
    PointBERT-style patch-based point cloud transformers.
    """

    def __init__(self,
                 group: nn.Module,
                 patch: nn.Module,
                 pos_enc: nn.Module,
                 mix: nn.Module):
        super().__init__()
        self.group = group
        self.patch = patch
        self.pos_enc = pos_enc
        self.mix = mix

    def forward(self,
                x: th.Tensor,
                z_ctx: Optional[th.Tensor] = None):
        p, c, i = self.group(x)
        z = self.patch(p - c[..., None, :])
        pe = self.pos_enc(c)
        z_pe = z + pe
        if z_ctx is not None:
            z_pe = th.cat([z_ctx, z_pe], dim=-2)
        z_mix = self.mix(z_pe)
        return z_mix


class PointCloudEncoderFPSMLPMLP(PointCloudEncoder):
    """
    Instance of `PointCloudEncoder` based on:
    * FPS grouping
    * MLP patch encoder
    * MLP positional encoder
    """
    @dataclass
    class Config:
        group: GroupFPS.Config = GroupFPS.Config()
        patch: MLPPatchEncoder.Config = MLPPatchEncoder.Config()
        pos_enc: PosEncodingMLP.Config = PosEncodingMLP.Config()
        block: EncoderBlock.Config = EncoderBlock.Config()
        block2: EncoderBlock2.Config = EncoderBlock2.Config()

        num_layer: int = 4
        num_group: int = 16
        patch_size: int = 32
        embed_size: int = 128
        point_size: int = 3
        norm_out: bool = False
        use_block2: bool = False

        def __post_init__(self):
            self.group = replace(self.group,
                                 num_group=self.num_group,
                                 num_point=self.patch_size)
            self.patch = replace(self.patch,
                                 point_size=self.point_size,
                                 patch_size=self.patch_size,
                                 embed_size=self.embed_size)
            self.pos_enc = replace(self.pos_enc,
                                   dim_in=self.point_size,
                                   dim_out=self.embed_size)
            self.block = replace(self.block,
                                 hidden_size=self.embed_size)
            self.block2 = replace(self.block2,
                                  hidden_size=self.embed_size)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        group = GroupFPS(cfg.group)
        patch = MLPPatchEncoder(cfg.patch)
        pos_enc = PosEncodingMLP(cfg.pos_enc)

        block_cls = EncoderBlock2 if cfg.use_block2 else EncoderBlock
        block_cfg = cfg.block2 if cfg.use_block2 else cfg.block

        mix = [
            block_cls(block_cfg)
            for _ in range(cfg.num_layer)
        ]
        if cfg.norm_out:
            mix.append(nn.LayerNorm(cfg.embed_size))
        mix = nn.Sequential(*mix)
        super().__init__(group, patch, pos_enc, mix)


def main_1():
    device: str = 'cuda:0'
    group = GroupFPS(GroupFPS.Config())
    patch = MLPPatchEncoder(MLPPatchEncoder.Config())
    pos_enc = PosEncodingMLP(PosEncodingMLP.Config())
    mix = nn.Sequential(*[
        EncoderBlock(EncoderBlock.Config())
        for _ in range(4)
    ])
    encoder = PointCloudEncoder(
        PointCloudEncoder.Config(),
        group, patch, pos_enc, mix)
    encoder.to(device)
    x = th.randn((2, 512, 3), device=device)
    y = encoder(x)
    print('y', y.shape)


def test_combinations():
    device: str = 'cuda:0'
    for batch_size in [1, 2]:
        for num_point in [32, 128, 512]:
            for num_layer in [1, 2]:
                for patch_size in [32, 64]:
                    for point_size in [3, 4]:
                        for embed_size in [64, 128]:
                            for extra_size in [0, 2]:
                                model = PointCloudEncoderFPSMLPMLP(
                                    PointCloudEncoderFPSMLPMLP.Config(
                                        num_layer=num_layer,
                                        patch_size=patch_size,
                                        embed_size=embed_size,
                                        point_size=point_size,
                                        # norm_out=1
                                        use_block2=True
                                    )).to(device)
                                ic(model)
                                # ic(model)
                                # ic(model.cfg)
                                x = th.randn(
                                    (batch_size, num_point, point_size), device=device)
                                if extra_size > 0:
                                    z = th.randn(
                                        (batch_size, extra_size, embed_size), device=device)
                                    y = model(x, z)
                                else:
                                    y = model(x)
                                # print('y', y.shape)
                                # ic(y.shape)
                                ic(y.shape, y.std())
                                return


def main():
    test_combinations()


if __name__ == '__main__':
    main()
