#!/usr/bin/env python3

import os
from collections import defaultdict

from dataclasses import dataclass
from typing import Optional, Union, Dict, Tuple
from functools import partial
import numpy as np
import torch as th
import torch.nn as nn
import warp as wp

from curobo.types.base import TensorDeviceType
from curobo.geom.types import WorldConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.sdf.world import WorldCollision, WorldCollisionConfig
from curobo.types.robot import JointState, RobotConfig
from curobo.geom.sdf.world_mesh import WorldMeshCollision, WarpMeshData
from curobo.geom.transform import pose_inverse
from curobo.util_file import get_robot_configs_path, get_task_configs_path, join_path, load_yaml
from curobo.rollout.arm_reacher import ArmReacher, ArmReacherConfig
from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutMetrics

from presto.data.ext_util.spec import (
    Mesh,
    Cuboid,
    Sphere,
    Cylinder,
    Capsule,
    Scene,
    cls_from_str,
    str_from_cls
)
from presto.data.ext_util.curobo_spec import (
    to_curobo_dict,
    from_curobo
)

from presto.util.math_util import (
    quat_wxyz2xyzw,
    quat_xyzw2wxyz
)

from icecream import ic


def pad_mesh_verts(mesh_verts,
                   flat_index: Dict[Tuple[int, int], int],
                   shape: Tuple[int, int]):
    # pad list of arrays to same length.
    if len(mesh_verts) <= 0:
        num_vert = 0
    else:
        num_vert = max(len(v) for v in mesh_verts)
    out = wp.zeros((*shape, num_vert),
                   dtype=mesh_verts[0].dtype,
                   device=mesh_verts[0].device)
    for (i, j), k in flat_index.items():
        verts = mesh_verts[flat_index[(i, j)]]
        n = len(verts)
        out[i, j, :n].assign(verts)
    return out


@wp.kernel
def compute_batch_scale_point_(
    verts: wp.array(dtype=wp.vec3),
    scale: wp.array(dtype=wp.vec3),
    n_pts: wp.int32,
):  # given n,3 points and b poses, get b,n,3 transformed points
    # we tile as
    tid = wp.tid()
    b_idx = tid / (n_pts)
    p_idx = tid - (b_idx * n_pts)

    # read data:
    o_idx = b_idx * n_pts + p_idx

    tmp = wp.vec3(verts[o_idx][0] * scale[b_idx][0],
                  verts[o_idx][1] * scale[b_idx][1],
                  verts[o_idx][2] * scale[b_idx][2])
    verts[o_idx] = tmp


@wp.kernel
def compute_batch_scale_point(
    verts: wp.array(dtype=wp.vec3),
    scale: wp.array(dtype=wp.vec3),
    n_pts: wp.int32,
    verts_out: wp.array(dtype=wp.vec3),
):
    # out-of-place version of the above function
    tid = wp.tid()
    b_idx = tid / (n_pts)  # batch index
    p_idx = tid % n_pts
    # read data:
    o_idx = tid  # b_idx * n_pts + p_idx
    # output
    tmp = wp.vec3(verts[o_idx][0] * scale[b_idx][0],
                  verts[o_idx][1] * scale[b_idx][1],
                  verts[o_idx][2] * scale[b_idx][2])
    verts_out[o_idx] = tmp


def scale_point_(
        verts: wp.array(dtype=wp.vec3),
        scale: th.Tensor):
    # n=num points
    n: int = verts.shape[-1]
    b: int = 1

    wp.launch(
        kernel=compute_batch_scale_point_,
        dim=b * n,
        inputs=[
            verts,
            wp.from_torch(scale.detach().view(-1, 3).contiguous(),
                          dtype=wp.vec3),
            n,
        ],
        # outputs=[wp.from_torch(out_points.view(-1, 3).contiguous(), dtype=wp.vec3)],
        stream=wp.stream_from_torch(scale.device),
    )


def batch_scale_point(
        verts: wp.array(dtype=wp.vec3),
        scale: th.Tensor,
        verts_out: wp.array(dtype=wp.vec3)
):
    """
    verts: (b, m, p,) x3
    scale: (b, m,) x3
    """
    b: int = verts.shape[0]
    n: int = verts.shape[1]
    p: int = verts.shape[2]

    wp.launch(
        kernel=compute_batch_scale_point,
        dim=b * n * p,
        inputs=[
            verts.reshape((b * n * p,)),
            wp.from_torch(scale.detach().view(-1, 3).contiguous(),
                          dtype=wp.vec3),
            p,
        ],
        outputs=[verts_out.reshape((b * n * p,))],
        stream=wp.stream_from_torch(scale.device),
    )


def to_curobo_pose(pose: th.Tensor):
    """
    (1) xyzw -> wxyz
    (2) invert (world_from_self --> self_from_world)
    """
    t = pose[..., 0:3]
    q = quat_xyzw2wxyz(pose[..., 3:7])
    ti, qi = pose_inverse(t, q)
    return th.cat([ti, qi], axis=-1)


class ArrayHelper:
    def __getitem__(self, X):
        return np.asarray(X, dtype=np.float32)


def convert_n_prim(n_prim: Union[int, Dict[str, int]]):
    if isinstance(n_prim, int):
        n_prim = {str_from_cls(c): n_prim
                  for c in [Cuboid, Sphere,
                            # Capsule,
                            Cylinder]}
    return n_prim


def make_scene(batch_size: int,
               n_prim: Union[int, Dict[str, int]] = 6):
    n_prim = convert_n_prim(n_prim)
    A = ArrayHelper()
    # Create template scene with
    # all primitives pre-loaded.
    scenes = []
    for i in range(batch_size):
        scene = Scene({
            **{F'{str_from_cls(Cuboid)}_{i:02d}_{j:02d}': Cuboid(A[1, 1, 1], A[0, 0, 0, 0, 0, 0, 1])
               for j in range(n_prim.get(str_from_cls(Cuboid), 0))},
            **{F'{str_from_cls(Sphere)}_{i:02d}_{j:02d}': Sphere(1, A[0, 0, 0, 0, 0, 0, 1])
               for j in range(n_prim.get(str_from_cls(Sphere), 0))},
            **{F'{str_from_cls(Capsule)}_{i:02d}_{j:02d}': Capsule(1, 1, A[0, 0, 0, 0, 0, 0, 1])
               for j in range(n_prim.get(str_from_cls(Capsule), 0))},
            **{F'{str_from_cls(Cylinder)}_{i:02d}_{j:02d}': Cylinder(1, 1, A[0, 0, 0, 0, 0, 0, 1])
               for j in range(n_prim.get(str_from_cls(Cylinder), 0))},
        })
        scenes.append(scene)
    return scenes


def make_world(batch_size: int,
               n_prim: Union[int, Dict[str, int]] = 6,
               robot_file: str = 'franka.yml',
               device: str = 'cuda',
               margin: float = 0.0,
               aux=None):
    n_prim = convert_n_prim(n_prim)
    tensor_args = TensorDeviceType(device)
    scenes = make_scene(batch_size, n_prim)

    if aux is not None:
        aux['scene'] = scenes

    # Conversions
    cfgs = [to_curobo_dict(d) for d in scenes]
    scenes = [from_curobo(d) for d in cfgs]
    cfgs = [WorldConfig.create_collision_support_world(
            WorldConfig.from_dict(d)) for d in cfgs]
    # print('cfgs', cfgs)
    config = RobotWorldConfig.load_from_config(
        robot_file,
        cfgs,
        tensor_args,
        collision_activation_distance=margin,
        # sweep=cfg.sweep,
        n_envs=len(cfgs),
        # FIXME(ycho): hardcoded prim types (sphere/capsule/cylinder)
        n_meshes=sum(n_prim.get(str_from_cls(c), 0)
                     for c in [Sphere, Capsule, Cylinder]),
        n_cuboids=n_prim.get(str_from_cls(Cuboid), 0)
    )
    robot_world = RobotWorld(config)
    return robot_world


class CachedCuroboCost(nn.Module):

    @dataclass
    class Config:
        margin: float = 0.02
        robot_file: str = 'franka.yml'
        base_cfg_file: str = 'base_cfg.yml'
        gradient_file: str = 'gradient_trajopt.yml'
        sweep: bool = True
        sum_dist: bool = True
        export_world: Optional[str] = None
        force_reset: bool = False
        relabel: bool = True
        pad: bool = True
        esdf: bool = False
        classify: bool = False

    def __init__(self,
                 cfg: Config,
                 batch_size: int,
                 n_prim: Dict[str, int],
                 device: str = None,
                 ):
        super().__init__()
        self.cfg = cfg

        tensor_args = TensorDeviceType(device)
        self._robot_world = make_world(batch_size,
                                       n_prim,
                                       cfg.robot_file,
                                       device,
                                       cfg.margin)

        cost = self._robot_world.collision_cost

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

        self._robot_world.collision_cost.sum_distance = cfg.sum_dist
        self._robot_cfg = cfg.robot_file

        world_model = self._robot_world.world_model
        self.__mesh_verts = []
        self.__mesh_meshs = []
        self.__flat_index = {}

        count: int = 0
        for (k, v) in world_model._wp_mesh_cache.items():
            self.__mesh_verts.append(v.vertices)
            self.__mesh_meshs.append(v.mesh)  # for refit() maybe
            found = False
            for i in range(len(world_model._mesh_tensor_list[0])):
                if found:
                    break
                for j in range(len(world_model._mesh_tensor_list[0][0])):
                    if found:
                        break
                    if world_model._mesh_tensor_list[0][i, j] == v.m_id:
                        self.__flat_index[(i, j)] = count
                        found = True
            count += 1

            if not found:
                raise ValueError(F'Some mesh not found : {v.m_id}')
        self.__arry_index = {v: k for (k, v) in self.__flat_index.items()}

        # Replace `_wp_mesh_cache` with a
        # slightly more convenient version.
        if cfg.pad:
            if len(world_model._wp_mesh_cache) > 0:
                base_mesh_verts = pad_mesh_verts(
                    self.__mesh_verts,
                    self.__flat_index,
                    world_model._mesh_tensor_list[0].shape
                )
                scaled_mesh_verts = wp.clone(base_mesh_verts)
                my_wp_mesh_cache = {}
                self.__mesh_meshs = []
                for i_flt, (k, v) in enumerate(
                        world_model._wp_mesh_cache.items()):
                    (i, j) = self.__arry_index[i_flt]
                    vertices = scaled_mesh_verts[i, j]
                    # new mesh
                    m = wp.Mesh(points=vertices, indices=v.faces)
                    my_wp_mesh_cache[k] = WarpMeshData(v.name,
                                                       m.id,
                                                       vertices,
                                                       v.faces,
                                                       m)
                    world_model._mesh_tensor_list[0][i, j] = m.id
                    self.__mesh_meshs.append(m)

                world_model._wp_mesh_cache = my_wp_mesh_cache
                self.__base_mesh_verts = base_mesh_verts
                self.__scaled_mesh_verts = scaled_mesh_verts

        # == load `ArmReacher` ==
        base_config_data = load_yaml(
            join_path(
                get_task_configs_path(),
                cfg.base_cfg_file))
        grad_config_data = load_yaml(
            join_path(
                get_task_configs_path(),
                cfg.gradient_file))
        robot_config_data = load_yaml(
            join_path(
                get_robot_configs_path(),
                self._robot_cfg))['robot_cfg']
        self.reach_cfg = ArmReacherConfig.from_dict(
            robot_config_data,
            grad_config_data['model'],
            grad_config_data['cost'],
            base_config_data['constraint'],
            base_config_data['convergence'],
            world_coll_checker=self._robot_world.collision_cost.world_coll_checker,
            tensor_args=tensor_args,
        )
        self.reacher = ArmReacher(self.reach_cfg)

    def reset(self, c: th.Tensor):
        cfg = self.cfg

        world = self._robot_world.world_model
        (cube_dims,
         cube_pose,
         cube_mask) = world._cube_tensor_list
        (mesh_idxs,
         mesh_pose,
         mesh_mask) = world._mesh_tensor_list

        # NOTE(ycho): convert `half-dim` convention to full-dim.
        cube_dims[..., :3] = 2.0 * c['cube_dims']
        cube_pose[..., :7] = to_curobo_pose(c['cube_pose'])
        cube_mask[...] = c['cube_mask']

        # Apply mesh dim by scaling the verts and refitting cache.
        # FIXME(ycho): maybe there's a better way to do this?
        n_env: int = c['mesh_dims'].shape[0]
        n_mesh: int = c['mesh_dims'].shape[1]
        if n_mesh > 0:
            if not cfg.pad:
                for i in range(n_env):
                    for j in range(n_mesh):
                        # print('FI', self.__flat_index[(i, j)])
                        verts = self.__mesh_verts[self.__flat_index[(i, j)]]
                        scale_point_(verts, c['mesh_dims'][i, j])
                        th.cuda.synchronize()
            else:
                batch_scale_point(self.__base_mesh_verts,
                                  c['mesh_dims'],
                                  self.__scaled_mesh_verts)
            [m.refit() for m in self.__mesh_meshs]
            mesh_pose[..., :7] = to_curobo_pose(c['mesh_pose'])
            mesh_mask[...] = c['mesh_mask']

    def forward_old(self,
                    q: th.Tensor,
                    c: Optional[th.Tensor] = None,
                    reduce: bool = False):
        cfg = self.cfg
        qs = q.shape
        q = q.reshape(-1, *q.shape[-2:])
        if c is not None:
            self.reset(c)
        env_query_idx = th.arange(q.shape[0],
                                  dtype=th.int32,
                                  device=q.device)

        # NOTE(ycho): only returns `d_world` (self<->world collision)
        # optionally, consider summing with `d_self` (self<->self collision)
        sd_world, _ = (
            self._robot_world.get_world_self_collision_distance_from_joint_trajectory(
                q, env_query_idx=env_query_idx))
        sd_world = sd_world.reshape(*qs[:-1])
        return sd_world

    def forward(self,
                q: th.Tensor,
                c: Optional[th.Tensor] = None,
                reduce: bool = False,

                q0: Optional[th.Tensor] = None,
                q1: Optional[th.Tensor] = None):
        # NOTE(ycho):
        # In case q0/q1 is None, perform standard evaluation (collision only)
        # supplying q0&q1 _optionally_ toggles "vanilla curobo cost" mode,
        # but this code-path is quite fragile and untested.
        if q0 is None or q1 is None:
            return self.forward_old(q, c, reduce)

        # -- DANGER ZONE -- untested code
        cfg = self.cfg
        qs = q.shape
        q = q.reshape(-1, *q.shape[-2:])
        tensor_args = TensorDeviceType(device=q.device)

        if c is not None:
            # c is dict of (batch_size X num-prim...) things
            self.reset(c)

        # print(q.amin(), q.amax(), q.shape)
        env_query_idx = th.arange(q.shape[0],
                                  dtype=th.int32,
                                  device=q.device)

        init_state = JointState.from_position(q0)
        goal_state = JointState.from_position(q1)
        goal = Goal(goal_state=goal_state,
                    current_state=init_state)
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV,
            num_trajopt_seeds=1,
            batch_size=goal.batch,
            n_envs=goal.batch,
            n_goalset=1,
        )
        solve_state, goal_buffer, update_reference = solve_state.update_goal(
            goal,
            solve_state,
            None,
            tensor_args,
        )
        self.reacher.update_params(goal_buffer)
        return self.reacher.rollout_fn(q).costs


class CachedCuroboCostWithNG(th.autograd.Function):
    """ CachedCuroboCost with numerical gradients """
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
        return (dc_dq * grad_output[..., None], None)


cached_curobo_cost_with_ng = CachedCuroboCostWithNG.apply


def main():
    th.cuda.manual_seed_all(0)
    th.cuda.manual_seed(0)
    th.manual_seed(0)

    from presto.cost.curobo_cost import CuroboCost
    from presto.data.presto_shelf import (
        proc_scene,
        soa_from_aos
    )
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    batch_size: int = 4
    device: str = 'cuda'
    # n_prim: int = {'cuboid': 1,
    #                'sphere': 1}
    n_prim: int = 1
    cost = CachedCuroboCost(CachedCuroboCost.Config(margin=0.0,
                                                    sweep=False,
                                                    pad=True),
                            n_prim=n_prim,
                            batch_size=batch_size,
                            device=device)
    q = th.randn((batch_size, 1, 7),
                 dtype=th.float32,
                 device=device)

    # col_label = out['col-label']
    # col_label = [from_curobo(c) for c in col_label]
    aux = {}
    # robot_world = make_world(batch_size, n_prim, aux=aux)
    # col_label = aux['scene']
    # world_model = robot_world.world_model

    for _ in range(4):
        scenes = make_scene(batch_size, n_prim)

        # randomize...
        if True:
            for s in scenes:
                for k, v in s.geom.items():
                    if isinstance(v, (Sphere, Cylinder, Capsule)):
                        v.radius = np.random.uniform()
                    if isinstance(v, Cuboid):
                        v.radius = np.random.uniform(size=3)
                    if isinstance(v, (Cylinder, Capsule)):
                        v.half_length = np.random.uniform()

        col_label_2 = np.asarray([to_curobo_dict(s) for s in scenes])
        # print(col_label_2)

        col_label = [proc_scene(c, cost._robot_world.world_model)
                     for c in scenes]
        col_label = soa_from_aos(col_label)
        col_label = {k: np.stack(v) for (k, v) in col_label.items()}
        col_label = {
            k: th.as_tensor(
                v,
                dtype=th.float32,
                device=device) for (
                k,
                v) in col_label.items()}
        y = cost(q, c=col_label)
        print('y (cached curobo)', y)

        cost2 = CuroboCost(CuroboCost.Config(margin=0.0, sweep=False,
                                             relabel=False),
                           batch_size=batch_size,
                           device=device)
        y2 = cost2(q, c=col_label_2)
        # print('masks',
        #       cost2._robot_world.world_model._cube_tensor_list[2],
        #       cost2._robot_world.world_model._mesh_tensor_list[2],
        #       )
        # print('cube-tensor-list',cost2._robot_world.world_model._cube_tensor_list)
        print('y2 (original curobo)', y2)


if __name__ == '__main__':
    main()
