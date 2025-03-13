#!/usr/bin/env python3

from typing import Optional
from collections import namedtuple
from dataclasses import dataclass

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from mp_baselines.planners.costs.cost_functions import (
    CostGPTrajectory,
    CostGPTrajectoryPositionOnlyWrapper
)

# FIXME(ycho): hard-coded doF
Robot = namedtuple('Robot', [
    'q_dim',
])


class DistanceCost(nn.Module):
    """ Trajectory length cost. """

    def forward(self, q: th.Tensor):
        delta = q[..., 1:, :] - q[..., :-1, :]
        delta = (delta + th.pi) % (2 * th.pi) - th.pi
        dq = F.mse_loss(delta, th.zeros_like(delta))
        return dq  # or dq.sum()


class GPCost(nn.Module):
    """ Gaussian Prior Cost. """

    @dataclass
    class Config:
        q_dim: int = 7
        gp_dt: float = 5.0 / 1000
        gp_sigma: float = 1.0
        gp_n_support: int = 1000

    def __init__(self, cfg: Config,
                 device: Optional[str] = None):
        super().__init__()
        self.gp_cost = CostGPTrajectoryPositionOnlyWrapper(
            robot=Robot(cfg.q_dim),
            n_support_points=cfg.gp_n_support,
            dt=cfg.gp_dt,
            sigma_gp=cfg.gp_sigma,
            tensor_args=dict(device=device)
        )

    def forward(self, q: th.Tensor):
        c_gp = self.gp_cost(q)
        return c_gp
