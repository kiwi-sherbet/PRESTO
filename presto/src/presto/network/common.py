#!/usr/bin/env python3

from typing import (Tuple, Optional, Union)


import torch as th
import torch.nn as nn


S = Union[int, Tuple[int, ...]]
T = th.Tensor

LN_NAMES = ['ln', 'layer_norm', 'layernorm']
BN_NAMES = ['bn', 'bayer_norm', 'batchnorm']


def get_activation_function(act_cls: str) -> nn.Module:
    if not isinstance(act_cls, str):
        return act_cls
    act_cls = act_cls.lower()
    if act_cls == 'tanh':
        out = nn.Tanh
    elif act_cls == 'relu':
        out = nn.ReLU
    elif act_cls == 'lrelu':
        out = nn.LeakyReLU
    elif act_cls == 'elu':
        out = nn.ELU
    elif act_cls == 'relu6':
        out = nn.ReLU6
    elif act_cls == 'gelu':
        out = nn.GELU
    elif act_cls == 'selu':
        out = nn.SELU
    elif act_cls == 'silu':
        out = nn.SiLU
    elif act_cls == 'none':
        out = nn.Identity
    else:
        raise KeyError(F'Unknown act_cls={act_cls}')
    return out


class LinearNorm(nn.Module):
    """ Linear layer with optimal batch normalization. """

    def __init__(self, dim_in: int, dim_out: int,
                 norm: Optional[str] = None, **kwds):
        super().__init__()
        affine = kwds.pop('affine', None)

        if norm in LN_NAMES:
            self.norm = nn.LayerNorm(dim_out, elementwise_affine=affine)
        elif norm in BN_NAMES:
            kwds['bias'] = False
            self.norm = nn.BatchNorm1d(dim_out, affine=affine)
        elif norm in ['none', None]:
            self.norm = nn.Identity()
        else:
            raise ValueError(F'Unknown norm={norm}')
        self.linear = nn.Linear(dim_in, dim_out, **kwds)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.linear(x)
        s = x.shape
        x = x.reshape(-1, s[-1])
        x = self.norm(x)
        x = x.reshape(s)
        return x


class MLP(nn.Module):
    """ Generic multilayer perceptron. """

    def __init__(self, dims: Tuple[int, ...],
                 act_cls: nn.Module = nn.LeakyReLU,
                 activate_output: bool = False,
                 norm: str = 'layernorm',
                 bias: bool = True,
                 pre_ln_bias: bool = True):
        super().__init__()
        assert (len(dims) >= 2)

        if isinstance(act_cls, str):
            act_cls = get_activation_function(act_cls)

        layers = []
        for d0, d1 in zip(dims[:-2], dims[1:-1]):
            # FIXME(ycho): incorrect `bias` logic
            if (norm not in LN_NAMES):
                layer_bias = bias
            else:
                layer_bias = pre_ln_bias
            layers.extend(
                (LinearNorm(
                    d0,
                    d1,
                    norm=norm,
                    bias=layer_bias),
                    act_cls(),
                 ))
        if activate_output:
            layers.extend((
                LinearNorm(
                    dims[-2],
                    dims[-1],
                    norm=norm,
                    bias=bias),
                act_cls()))
        else:
            # FIXME(ycho): not much I can do here except
            # hardcoding... for now
            layers.extend((
                nn.Linear(dims[-2], dims[-1], bias=bias),)
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor):
        return self.model(x)


def grad_step(
        loss: Optional[th.Tensor],
        optimizer: th.optim.Optimizer,
        scaler: Optional[th.cuda.amp.GradScaler] = None,
        parameters: Optional[nn.ParameterList] = None,
        max_grad_norm: Optional[float] = 1.0,
        step_grad: bool = True,
        skip_nan: bool = True,
        zero_grad: bool = True,
        ** bwd_args
):
    """
    Optimizer Step with optional AMP / grad clipping.

    Performs following operations:
        * loss.backward()
        * clip_grad_norm()
        * step() (optional)
        * zero_grad() (optional)

    Optionally applies scaler() related operations
    if AMP is enabled (triggered by scaler != None)
    """

    # Try to automatically fill out `parameters`
    if (max_grad_norm is not None) and (parameters is None):
        parameters = optimizer.param_groups[0]['params']

    if (scaler is not None):
        # With AMP + clipping
        if loss is not None:
            scaler.scale(loss).backward(**bwd_args)
        if step_grad:
            if (max_grad_norm is not None) and (parameters is not None):
                skip_step: bool = False
                try:
                    grad_norm = nn.utils.clip_grad_norm_(
                        parameters,
                        max_grad_norm,
                        error_if_nonfinite=skip_nan)
                except RuntimeError:
                    skip_step = True
                if not skip_step:
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                if zero_grad:
                    optimizer.zero_grad()
            else:
                scaler.step(optimizer)
                scaler.update()
                if zero_grad:
                    optimizer.zero_grad()
    else:
        # Without AMP
        if loss is not None:
            loss.backward(**bwd_args)
        if (max_grad_norm is not None) and (parameters is not None):
            skip_step: bool = False
            try:
                grad_norm = nn.utils.clip_grad_norm_(parameters,
                                                     max_grad_norm)
            except RuntimeError:
                skip_step = True
            if not skip_step:
                optimizer.step()
            if zero_grad:
                optimizer.zero_grad()  # set_to_none?
        if step_grad:
            optimizer.step()
            if zero_grad:
                optimizer.zero_grad()  # set_to_none?
