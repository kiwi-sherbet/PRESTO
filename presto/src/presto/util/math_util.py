#!/usr/bin/env python3

from typing import Optional
import torch as th
import numpy as np


def _quaternion_from_matrix(x: th.Tensor,
                            out: th.Tensor):
    # parse input
    m00, m01, m02 = [x[..., 0, i] for i in range(3)]
    m10, m11, m12 = [x[..., 1, i] for i in range(3)]
    m20, m21, m22 = [x[..., 2, i] for i in range(3)]
    th.stack([1. + m00 - m11 - m22,
              1. - m00 + m11 - m22,
              1. - m00 - m11 + m22,
              1. + m00 + m11 + m22], dim=-1,
             out=out)
    out.clamp_min_(0.0).sqrt_().mul_(0.5)
    out[..., 0].copysign_(m21 - m12)
    out[..., 1].copysign_(m02 - m20)
    out[..., 2].copysign_(m10 - m01)
    return out


def quaternion_from_matrix(
        x: th.Tensor, out: Optional[th.Tensor] = None) -> th.Tensor:
    if out is None:
        out = th.zeros(size=x.shape[:-2] + (4,),
                       dtype=x.dtype,
                       device=x.device)
    _quaternion_from_matrix(x, out)
    return out


def quat_xyzw2wxyz(q: th.Tensor):
    if isinstance(q, th.Tensor):
        return th.roll(q, +1, dims=-1)
    return np.roll(q, +1, axis=-1)


def quat_wxyz2xyzw(q: th.Tensor):
    if isinstance(q, th.Tensor):
        return th.roll(q, -1, dims=-1)
    return np.roll(q, -1, axis=-1)
