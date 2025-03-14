#!/usr/bin/env python3

from typing import Optional
from functools import reduce
from itertools import accumulate

import numpy as np
import torch as th
import einops


def matrix_from_dh(dh_base: th.Tensor, q: th.Tensor,
                   T_out: Optional[th.Tensor] = None):
    """
    Homogeneous transform matrix from DH parameters.
    """
    # Define Transformation matrix based on DH params
    alpha, a, d = th.unbind(dh_base, dim=-1)
    cq, sq = th.cos(q), th.sin(q)
    ca, sa = th.cos(alpha), th.sin(alpha)
    z, o = th.zeros_like(q), th.ones_like(q)
    T = th.stack([cq, -sq, z, a,
                  sq * ca, cq * ca, -sa, -sa * d,
                  sq * sa, cq * sa, ca, ca * d,
                  z, z, z, o], dim=-1)  # , out=out_buf)
    T = T.reshape(*dh_base.shape[:-1], 4, 4)
    return T


def franka_fk(q: th.Tensor,
              return_intermediate: bool = False,
              tool_frame: bool = True,
              z_offset: float = -0.05
              ):
    """
    Franka Panda forward kinematics in pytorch.
    (WARN(ycho): hardcoded link lengths & dh params)
    """

    # Partial DH paramters without joint values
    DH_BASE = th.as_tensor([[0, 0, 0.333],
                            [-th.pi / 2, 0, 0],
                            [th.pi / 2, 0, 0.316],
                            [th.pi / 2, 0.0825, 0],
                            [-th.pi / 2, -0.0825, 0.384],
                            [th.pi / 2, 0, 0],
                            # NOTE(ycho):
                            # `0.107` accounts for flange offset;
                            # `0.1034` accounts for gripper offset.
                            [th.pi / 2, 0.088,
                                (0.107 + 0.1034 + z_offset if tool_frame
                                 else 0.0)],
                            ],
                           dtype=q.dtype,
                           device=q.device)
    qs = q.reshape(-1, q.shape[-1])
    DHS = einops.repeat(DH_BASE, '... -> n ...', n=qs.shape[0])
    TS = matrix_from_dh(DHS, qs)

    if return_intermediate:
        # return intermediate transforms
        return list(accumulate(TS.unbind(dim=-3), th.bmm))
    else:
        # Only return final transform
        T = reduce(th.bmm, TS.unbind(dim=-3))
        # Account for rotational offset
        irt2 = float(1.0 / np.sqrt(2))
        T[..., :3, :3] = T[..., :3, :3] @ th.as_tensor([
            [irt2, irt2, 0],
            [-irt2, irt2, 0],
            [0, 0, 1.0000000]],
            dtype=T.dtype,
            device=T.device)
        return T.reshape(*q.shape[:-1], 4, 4)


def franka_link_transforms(q: th.Tensor) -> th.Tensor:
    # Obtain transforms from forward kinematics.
    # FIXME(ycho): because of this, THIS ROUTINE
    # ONLY WORKS FOR FRANKA PANDA !!
    transforms = franka_fk(q,
                           return_intermediate=True,
                           tool_frame=False,
                           z_offset=0.0
                           )  # ..., 4, 4
    # == Apply transform to convexes ==
    identity = transforms[0] * 0 + th.eye(
        4,
        dtype=transforms[0].dtype,
        device=transforms[0].device)
    transforms.insert(0, identity)
    transforms = th.stack(transforms, dim=-3)
    transforms = transforms.reshape(
        *q.shape[:-1],
        *transforms.shape[-3:])
    return transforms
