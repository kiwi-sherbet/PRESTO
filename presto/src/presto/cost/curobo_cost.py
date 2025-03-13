#!/usr/bin/env python3

from dataclasses import replace, dataclass
from typing import Tuple, Optional, List, Dict
from collections import namedtuple
from pathlib import Path

import pickle
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from icecream import ic

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.rollout.cost.bound_cost import BoundCost, BoundCostConfig, BoundCostType
from curobo.util_file import get_robot_configs_path, get_task_configs_path, join_path, load_yaml
from curobo.rollout.arm_reacher import ArmReacher, ArmReacherConfig
from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutMetrics

from presto.util.torch_util import dcn
from presto.data.encode_primitives import decode_primitives
from presto.data.data_util import rename_obs


def make_default_env(n_env: int,
                     n_obs: int,
                     add_floor: bool = True,
                     add_wall: bool = False,
                     base_seed: int = 0):
    """ Dummy env to initialize CuroboCost """
    world_cfg = []
    world_datas = []

    for seed in range(n_env):
        # Prepare buffer...
        world_data = dict(cuboid={}, sphere={})
        # Prepare RNG...
        rng = np.random.default_rng(base_seed + seed)
        # First, optionally add walls/floors.
        if add_floor:
            world_data['cuboid'].update({
                'floor': {
                    'pose': [0.0, 0.0, -0.1,
                             0, 0, 0, 1],
                    'dims': [2.0, 2.0, 0.2]
                }
            })
        if add_wall:
            walls = {
                'wall_1': {
                    'pose': [-0.3, 0.0, 0.5,
                             0, 0, 0, 1],
                    'dims': [0.2, 1.0, 1.0]},
                'wall_2': {
                    'pose': [0.0, -0.3, 0.5,
                             0, 0, 0, 1],
                    'dims': [1.0, 0.2, 1.0]},
                'wall_3': {
                    'pose': [0.0, 0.0, 0.9,
                             0, 0, 0, 1],
                    'dims': [1.0, 1.0, 0.2]
                }}
            world_data['cuboid'].update(walls)

        # Then add obstacles.
        num_obstacles = rng.integers(1, n_obs + 1)
        radius_bound: Tuple[float, float] = (0.1, 0.1)
        lo = (0.143, 0.3, 0.45)
        hi = (0.5, 0.3, 0.45)
        for i in range(num_obstacles):
            shape_type = 0
            if shape_type == 0:  # sphere
                world_data['sphere'][F'obs_{seed:02d}_{i:02d}'] = {
                    'pose': [*rng.uniform(lo, hi).tolist(),
                             0, 0, 0, 1],
                    'radius': rng.uniform(*radius_bound),
                }
            else:
                world_data['cuboid'][F'obs_{seed:02d}_{i:02d}'] = {
                    'pose': [*rng.uniform(lo, hi).tolist(),
                             0, 0, 0, 1],
                    'dims': rng.uniform(*radius_bound, size=3).tolist(),
                }

        world_cfg.append(WorldConfig.create_collision_support_world(
            WorldConfig.from_dict(world_data)))
        world_datas.append(world_data)

    return world_cfg, world_datas


def world_from_code(cond: th.Tensor, aux=None):
    prims = decode_primitives(dcn(cond))
    out = []

    if aux is not None:
        aux['world_data'] = []

    for q_i, q in enumerate(prims):
        world_data = dict(cuboid={},
                          sphere={})
        for i, p in enumerate(q):
            if p['shape'] == 'sphere':
                world_data['sphere'][F'obs_{q_i:02d}_{i:02d}'] = {
                    'pose': [*p['base_pos'], 0, 0, 0, 1],
                    'radius': p['dims']
                }
            elif p['shape'] == 'box':
                world_data['cuboid'][F'obs__{q_i:02d}_{i:02d}'] = {
                    'pose': [*p['base_pos'], 0, 0, 0, 1],
                    # NOTE(ycho): it's always quite confusing,
                    # but the `dims` are encoded as
                    # halfExtents(a la pybullet) by default,
                    # so we have to multiply dims by 2.
                    'dims': [2.0 * x for x in p['dims']]
                }
        out.append(WorldConfig.create_collision_support_world(
            WorldConfig.from_dict(world_data)))
        if aux is not None:
            aux['world_data'].append(world_data)
    return out


class CuroboCost(nn.Module):

    @dataclass
    class Config:
        margin: float = 0.02
        robot_file: str = 'franka.yml'
        sweep: bool = True
        export_world: Optional[str] = None
        force_reset: bool = False
        relabel: bool = True
        esdf: bool = False
        classify: bool = False
        sum_dist: bool = True

    def __init__(self, cfg: Config,
                 batch_size: int,
                 device: str = None):
        super().__init__()
        self.cfg = cfg
        self.device = device

        tensor_args = TensorDeviceType(device)

        # == env ==
        # FIXME(ycho): fixed env...! for now
        world_cfg, world_datas = make_default_env(
            batch_size, 1, True, True)

        # == robot ==
        robot_file = cfg.robot_file

        # == motion generation ==
        config = RobotWorldConfig.load_from_config(
            robot_file, world_cfg, tensor_args,
            collision_activation_distance=cfg.margin,
        )
        robot_world = RobotWorld(config)
        self._robot_world = robot_world

        cost = self._robot_world.collision_cost
        cost.sum_distance = cfg.sum_dist

        # `classify`
        cost.classify = cfg.classify
        if cost.classify:
            cost.coll_check_fn = cost.world_coll_checker.get_sphere_collision
            cost.sweep_check_fn = cost.world_coll_checker.get_swept_sphere_collision
        else:
            cost.coll_check_fn = cost.world_coll_checker.get_sphere_distance
            cost.sweep_check_fn = cost.world_coll_checker.get_swept_sphere_distance

        # esdf
        if cfg.esdf:
            cost.coll_check_fn = partial(
                cost.coll_check_fn,
                compute_esdf=True)
            cost.sweep_check_fn = partial(
                cost.sweep_check_fn,
                compute_esdf=True)

        # sweep
        if cfg.sweep:
            cost.use_sweep = True
            cost.forward = cost.sweep_kernel_fn

        self._robot_cfg = robot_file

    def reset(self, c: th.Tensor):
        cfg = self.cfg
        tensor_args = TensorDeviceType(self.device)

        if isinstance(c, th.Tensor):
            cfgs = world_from_code(c)
        else:
            if isinstance(c, np.ndarray):
                if cfg.relabel:
                    c = rename_obs(c)
                cfgs = [
                    WorldConfig.create_collision_support_world(
                        WorldConfig.from_dict(d)
                    ) for d in c]
            elif isinstance(c, dict):
                if cfg.relabel:
                    c = rename_obs([c])
                cfgs = [
                    WorldConfig.create_collision_support_world(
                        WorldConfig.from_dict(d)
                    ) for d in c]
            else:
                raise ValueError(F'unsupported `c` type = {type(c)}')

        if cfg.force_reset:
            # Complete reset; not sure if this option is always better
            config = RobotWorldConfig.load_from_config(
                cfg.robot_file,
                cfgs,
                tensor_args,
                collision_activation_distance=cfg.margin,
                n_envs=len(cfgs),
                n_meshes=1,
                n_cuboids=5,
            )
            robot_world = RobotWorld(config)
            self._robot_world = robot_world
        else:
            # Reset cache only.
            self._robot_world.world_model.clear_cache()
            self._robot_world.world_model.load_batch_collision_model(cfgs)
            if cfg.sweep:
                self._robot_world.collision_cost.use_sweep = True
                self._robot_world.collision_cost.forward = (
                    self._robot_world.collision_cost.sweep_kernel_fn)

        if cfg.export_world is not None:
            Path(cfg.export_world).mkdir(
                parents=True,
                exist_ok=True
            )
            for i, c in enumerate(cfgs):
                c.save_world_as_mesh(F'{cfg.export_world}/{i}.obj')

    def forward_collision_cost(self,
                               q: th.Tensor,
                               c: Optional[th.Tensor] = None,
                               reduce: bool = False):
        cfg = self.cfg

        qs = q.shape

        # q <- (batch, seq_len, dim)
        q = q.reshape(-1, *q.shape[-2:])
        if (c is not None) and isinstance(c, th.Tensor):
            # c <- (batch, c_dim)
            c = c.reshape(-1, *c.shape[-1:])

        # FIXME(ycho): this way of updating the collision model is
        # _highly_ inefficient. (Prefer CachedCuroboCost instead)
        if c is not None:
            self.reset(c)

        env_query_idx = th.arange(q.shape[0],
                                  dtype=th.int32,
                                  device=q.device)
        if True:
            # Use Curobo's API for collision distance.
            sdf, _ = (self._robot_world
                      .get_world_self_collision_distance_from_joint_trajectory(
                          q, env_query_idx=env_query_idx)
                      )
        else:
            # Explicitly create spheres and query collision distance.
            # (NOTE(ycho): only for debugging)
            s = q.shape
            kin_state = self._robot_world.get_kinematics(
                q.reshape(-1, q.shape[-1]))
            spheres = kin_state.link_spheres_tensor
            spheres = spheres.reshape(*s[:-1], *spheres.shape[-2:])
            sdf = self._robot_world.get_collision_distance(
                spheres, env_query_idx=env_query_idx)
        sdf = sdf.reshape(*qs[:-1])

        if reduce:
            sdf = sdf.sum()

        return sdf

    def forward(self,
                q: th.Tensor,
                c: Optional[th.Tensor] = None,
                reduce: bool = False,

                q0: Optional[th.Tensor] = None,
                q1: Optional[th.Tensor] = None):
        return self.forward_collision_cost(q, c, reduce)


class CuroboCostWithNG(th.autograd.Function):
    """ CuroboCost with numerical gradients """
    @staticmethod
    def forward(ctx, q: th.Tensor, cost_fn):
        EPS: float = 1e-3
        with th.no_grad():
            q2 = q.detach().clone()
            dc_dq = th.zeros_like(q)
            for i in range(q.shape[-1]):
                q2[..., i] += EPS
                c_pos = cost_fn(q2)

                q2[..., i] -= 2 * EPS
                c_neg = cost_fn(q2)
                dc_dq[..., i] = (c_pos - c_neg) / (2 * EPS)
            ctx.save_for_backward(dc_dq)

            return cost_fn(q)

    @staticmethod
    def backward(ctx, grad_output):
        dc_dq, = ctx.saved_tensors
        # print(dc_dq.shape, grad_output.shape)
        return (dc_dq * grad_output[..., None], None)


curobo_cost_with_ng = CuroboCostWithNG.apply


def main():
    import open3d as o3d
    from yourdfpy import URDF
    device: str = 'cuda:1'
    batch_size: int = 5
    num_query: int = 10
    cost = CuroboCost(CuroboCost.Config(),
                      batch_size=batch_size,
                      device=device).to(device)
    q = th.randn((batch_size, num_query, 7),
                 dtype=th.float32,
                 device=device)
    q = (q + th.pi) % (2 * th.pi) - th.pi
    sph_list = cost._robot_world.kinematics.get_robot_as_spheres(q[:, 0])
    print(len(sph_list))
    for bi, ss in enumerate(sph_list):
        scene = []
        for i, s in enumerate(ss):
            sp = o3d.geometry.TriangleMesh.create_sphere(radius=s.radius)
            sp.translate(s.position)
            scene.append(sp)
        robot = URDF.load('franka_panda.urdf')
        robot.update_cfg(dcn(q[bi, 0]))
        robot = robot.scene.dump(concatenate=True).as_open3d
        robot = o3d.geometry.LineSet.create_from_triangle_mesh(robot)
        scene.append(robot)
        o3d.visualization.draw(scene)
    print(q)
    d = cost(q)

    print(d.min(), d.max())


def test_relabel():
    with open('/tmp/docker/dbgw.pkl', 'rb') as fp:
        cs = pickle.load(fp)
    print(cs[0])
    cs2 = rename_obs(cs)
    print(cs, cs2)


if __name__ == '__main__':
    test_relabel()
