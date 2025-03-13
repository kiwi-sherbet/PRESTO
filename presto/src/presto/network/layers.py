#!/usr/bin/env python3

from typing import Optional, Tuple
from dataclasses import dataclass

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic

from presto.network.common import get_activation_function


class SinusoidalPositionalEncoding(nn.Module):
    """
    NeRF-style positional encoding.
    [Mildenhall et al. 2020].

    Computes the positional encoding for the
    normalized coordinate input in range (-1.0, +1.0).
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 flatten: bool = True,
                 pad: bool = False,
                 cat: bool = False):
        """
        Args:
            dim_in: dimensionality of coordinate input.
            num_frequencies: Number of higher-frequency elements.
            num_samples: Fallback computation of number of frequencies.
            flatten: If true, flatten positional encoding to one channel.
        """
        super().__init__()

        self.dim_in = dim_in
        self.cat = cat
        self.flatten = flatten

        c_out = dim_out
        if self.cat:
            assert (self.flatten)
            c_out = dim_out - dim_in

        self.pad = pad
        if not pad:
            assert ((c_out % (dim_in * 2)) == 0)
        self.num_frequencies = c_out // (dim_in * 2)
        self.dim_out = dim_out

        # Precompute the coefficient multipliers.
        self.register_buffer(
            'coefs', th.as_tensor(
                np.pi * (2 ** th.arange(self.num_frequencies)),
                dtype=th.float))

    def extra_repr(self):
        if self.cat:
            return F'{self.dim_in} -> {self.dim_out-self.dim_in}+{self.dim_in}'
        else:
            return F'{self.dim_in} -> {self.dim_out}'

    def forward(self, coords: th.Tensor) -> th.Tensor:
        """
        Args:
            coords: (..., D)
        Returns:
            pos_enc: (..., (2*F+1)*D) if flatten else (..., (2*F+1), D)
        """
        octaves = coords[..., None, :] * self.coefs[:, None]
        s = th.sin(octaves)
        c = th.cos(octaves)
        out = th.concat([s, c], dim=-2)
        # Optionally flatten the output.
        if self.flatten:
            out = out.view(coords.shape[:-1] + (-1,))

        if self.cat:
            out = th.cat([out, coords], dim=-1)

        if self.pad:
            out = F.pad(out, [0, self.dim_out - out.shape[-1]])
        return out


class Norm1d(nn.Module):
    """
    1D Normalization layer.

    Args:
        dim (int): feature dimension
        norm_type (str): normalization method
    """

    def __init__(self,
                 dim: int,
                 norm_type: str = 'batch_norm',
                 affine: bool = True):
        super().__init__()

        self.dim = dim
        self.norm_type = norm_type

        if norm_type == 'batch_norm':
            self.norm = nn.BatchNorm1d(dim, affine=affine)
        elif norm_type == 'instance_norm':
            self.norm = nn.InstanceNorm1d(dim, affine=affine)
        elif norm_type == 'group_norm':
            self.norm = nn.GroupNorm1d(dim, affine=affine)
        elif norm_type == 'layer_norm':
            self.norm = nn.LayerNorm(dim, elementwise_affine=affine)
        else:
            raise ValueError(F'Invalid normalization method = {norm_type}!')

    def forward(self, x: th.Tensor):
        """
        Args:
            x: (..., D_c); input.
            c: (..., D_c); conditioning variable.
        """
        s = x.shape
        return self.norm(x.reshape(-1, s[-1])).reshape(s)


class ConcatNorm1d(nn.Module):
    def __init__(self,
                 x_dim: int,
                 c_dim: int,
                 norm_type: str = 'batch_norm',
                 affine: bool = True,
                 project: bool = False):
        super().__init__()
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.norm_type = norm_type
        self.norm = Norm1d(x_dim + c_dim, norm_type, affine)
        if project:
            self.out = nn.Linear(x_dim + c_dim, x_dim)
        else:
            self.out = nn.Identity()

    def forward(self, x: th.Tensor, c: th.Tensor):
        c = c.expand(*x.shape[:-1], c.shape[-1])
        return self.out(self.norm(th.cat([x, c], dim=-1)))


class ConditionalBatchNorm1d(nn.Module):
    """
    Conditional batch normalization layer class.
    Adaptation from:
        https://github.com/autonomousvision/occupancy_networks

    Args:
        x_dim (int): feature dimension
        c_dim (int): condition dimension
        norm_type (str): normalization method
    """

    def __init__(self,
                 x_dim: int,
                 c_dim: int,
                 norm_type: str = 'batch_norm'
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.norm_type = norm_type

        # Submodules
        self.gamma = nn.Linear(c_dim, x_dim)
        self.beta = nn.Linear(c_dim, x_dim)

        if norm_type == 'batch_norm':
            self.norm = nn.BatchNorm1d(x_dim, affine=False)
        elif norm_type == 'instance_norm':
            self.norm = nn.InstanceNorm1d(x_dim, affine=False)
        elif norm_type == 'group_norm':
            self.norm = nn.GroupNorm1d(x_dim, affine=False)
        elif norm_type == 'layer_norm':
            self.norm = nn.LayerNorm(x_dim, elementwise_affine=False)
        else:
            raise ValueError(F'Invalid normalization method = {norm_type}!')

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: th.Tensor, c: th.Tensor):
        """
        Args:
            x: (..., D_c); input.
            c: (..., D_c); conditioning variable.
        """
        # Affine mapping
        gamma = self.gamma(c)
        beta = self.beta(c)

        # Allow broadcasting over arbitrary
        # number of batch dimensions.
        s = x.shape
        x = x.reshape(-1, x.shape[-1])
        net = self.norm(x)
        net = net.reshape(s)

        out = gamma * net + beta

        return out


class ConditionalResnetBlock(nn.Module):
    '''
    Conditional batch normalization-based Resnet block class.

    Adaptation from:
        https://github.com/autonomousvision/occupancy_networks

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_type (str): normalization method
    '''

    def __init__(self,
                 x_dim: int,
                 c_dim: int,
                 o_dim: int = None,
                 h_dim: int = None,
                 norm_type: str = 'batch_norm',
                 act_cls: str = 'relu'):
        super().__init__()

        # Attributes
        if h_dim is None:
            h_dim = x_dim
        if o_dim is None:
            o_dim = x_dim

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.o_dim = o_dim

        # Submodules
        self.norm_base = ConditionalBatchNorm1d(x_dim, c_dim,
                                                norm_type=norm_type)
        self.norm_residual = ConditionalBatchNorm1d(h_dim, c_dim,
                                                    norm_type=norm_type)

        self.base = nn.Linear(x_dim, h_dim)
        self.residual = nn.Linear(h_dim, o_dim)
        # self.actvn = nn.ReLU()
        self.actvn = get_activation_function(act_cls)()

        if x_dim == o_dim:
            self.project = nn.Identity()
        else:
            self.project = nn.Linear(x_dim,
                                     o_dim,
                                     bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x: th.Tensor, c: th.Tensor):
        net = self.base(self.actvn(self.norm_base(x, c)))
        dx = self.residual(self.actvn(self.norm_residual(net, c)))
        x_s = self.project(x)
        return x_s + dx


class ConcatResnetBlock(nn.Module):
    '''
    Conditional batch normalization-based Resnet block class.

    Adaptation from:
        https://github.com/autonomousvision/occupancy_networks

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_type (str): normalization method
    '''

    def __init__(self,
                 x_dim: int,
                 c_dim: int,
                 o_dim: int = None,
                 h_dim: int = None,
                 norm_type: str = 'batch_norm'):
        super().__init__()

        # Attributes
        if h_dim is None:
            h_dim = x_dim
        if o_dim is None:
            o_dim = x_dim

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.o_dim = o_dim

        # Submodules
        self.norm_base = ConcatNorm1d(x_dim, c_dim,
                                      norm_type=norm_type)
        self.norm_residual = ConcatNorm1d(h_dim, c_dim,
                                          norm_type=norm_type)

        self.base = nn.Linear(x_dim + c_dim, h_dim)
        self.residual = nn.Linear(h_dim + c_dim, o_dim)
        self.actvn = nn.ReLU()

        if x_dim == o_dim:
            self.project = nn.Identity()
        else:
            self.project = nn.Linear(x_dim,
                                     o_dim,
                                     bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x: th.Tensor, c: th.Tensor):
        net = self.base(self.actvn(self.norm_base(x, c)))
        dx = self.residual(self.actvn(self.norm_residual(net, c)))
        x_s = self.project(x)
        return x_s + dx


def test_pos_emb():
    pos_emb = SinusoidalPositionalEncoding(7, 14, flatten=True)
    print(pos_emb(th.zeros((3, 7))).shape)
    print(pos_emb(th.zeros((4, 3, 7))).shape)


def main():
    test_pos_emb()


if __name__ == '__main__':
    main()
