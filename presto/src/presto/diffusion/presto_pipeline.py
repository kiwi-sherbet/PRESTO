#!/usr/bin/env python3

from omegaconf import OmegaConf
from typing import Optional, Union, List, Dict
from dataclasses import dataclass, replace
from pathlib import Path
import copy
import time
from functools import partial, reduce
from itertools import accumulate, product
import pickle
import trimesh
import yaml
from yourdfpy import URDF
import scipy.signal

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops
import time
from tqdm.auto import tqdm
from matplotlib.pyplot import get_cmap
from matplotlib import pyplot as plt
from cho_util.math import transform as tx
import open3d as o3d
from icecream import ic

from diffusers import (
    DiffusionPipeline,
    DDIMScheduler
)
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig
from curobo.wrap.reacher.evaluator import TrajEvaluator, TrajEvaluatorConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.robot import JointState, RobotConfig
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutMetrics

from presto.data.encode_primitives import decode_primitives
from presto.data.franka_util import (franka_fk, franka_link_transforms)
from presto.data.sdf_util import build_geoms, build_rigid_body_chain
from presto.data.presto_shelf import PrestoDatasetShelf, retime_trajectory
from presto.util.ckpt import load_ckpt, last_ckpt
from presto.util.path import get_path
from presto.util.torch_util import dcn
from presto.cost.curobo_cost import CuroboCost, curobo_cost_with_ng
from presto.cost.cached_curobo_cost import CachedCuroboCost
from presto.network.factory import DDIMScheduler2
from presto.network.dit_cloud import DiTCloud
from presto.data.ext_util import (
    custom_spec,
    curobo_spec,
    open3d_spec
)
from presto.diffusion.util import (
    pred_x0, diffusion_loss,
)
try:
    from torch_robotics.robots.robot_panda import RobotPanda
    from mp_baselines.planners.costs.cost_functions import (
        CostComposite,
        CostGPTrajectory)
except ImportError:
    pass


MAX_VISUALIZATION_LENGTH: int = 1024


def xyzw2wxyz(x: np.ndarray):
    return np.roll(x, 1, axis=-1)


def wxyz2xyzw(x: np.ndarray):
    return np.roll(x, -1, axis=-1)


def transform_from_curobo_pose(pose):
    pose = np.asarray(pose, dtype=np.float32)
    t = pose[..., 0:3]
    q = wxyz2xyzw(pose[..., 3:7])
    R = tx.rotation.matrix.from_quaternion(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def scene_from_obs_info(obs_info,
                        lineset: bool = True,
                        skip: int = 0):
    """
    Open3D Geometry from `obs_info` dictionary.
    """
    meshes = []
    tool = None
    for i, o in enumerate(obs_info):
        if o is None:
            continue
        if i < skip:
            continue
        is_curobo: bool = False
        ms = []

        if 'attached_object' in o:
            v = o['attached_object']
            T = transform_from_curobo_pose(v['pose'])
            v['dims'] = np.asarray(v['dims'])
            m = (o3d.geometry.TriangleMesh.create_box(*v['dims'])
                    .translate(-0.5 * v['dims'])
                    .transform(T))
            tool = m

        if 'cuboid' in o:
            for k, v in o['cuboid'].items():
                T = transform_from_curobo_pose(v['pose'])
                v['dims'] = np.asarray(v['dims'])
                m = (o3d.geometry.TriangleMesh.create_box(*v['dims'])
                     .translate(-0.5 * v['dims'])
                     .transform(T))
                ms.append(m)
            is_curobo = True
        if 'sphere' in o:
            for k, v in o['sphere'].items():
                T = transform_from_curobo_pose(v['pose'])
                m = (o3d.geometry.TriangleMesh.create_sphere(v['radius'])
                     .transform(T))
                ms.append(m)
            is_curobo = True
        if 'mesh' in o:
            for k, v in o['mesh'].items():
                T = transform_from_curobo_pose(v['pose'])
                m = (o3d.io.read_triangle_mesh(v['file_path'])
                     .transform(T))
                ms.append(m)
            is_curobo = True
        if 'capsule' in o:
            for k, v in o['capsule'].items():
                T = transform_from_curobo_pose(v['pose'])
                height = v['tip'][2] - v['base'][2]
                m = (o3d.geometry.TriangleMesh.create_cylinder(
                    v['radius'], height)
                    .transform(T))
                ms.append(m)
            is_curobo = True
        if 'cylinder' in o:
            for k, v in o['cylinder'].items():
                T = transform_from_curobo_pose(v['pose'])
                height = v['height']
                m = (o3d.geometry.TriangleMesh.create_cylinder(
                    v['radius'], height)
                    .transform(T))
                ms.append(m)
            is_curobo = True

        if is_curobo:
            for m in ms:
                if lineset:
                    meshes.append(
                        o3d.geometry.LineSet.create_from_triangle_mesh(m))
                else:
                    meshes.append(m)
                    meshes.append(
                        o3d.geometry.LineSet.create_from_triangle_mesh(m))
            continue

        is_prim = False
        if 'mesh_mask' in o:
            is_prim = True

        for ds, ps, ms in zip(dcn(o['cube_dims']),
                              dcn(o['cube_pose']),
                              dcn(o['cube_mask'])):
            for d, p, m in zip(ds, ps, ms):
                if not m:
                    continue
                R = tx.rotation.matrix.from_quaternion(p[..., 3:7])
                T = np.zeros((*R.shape[:-2], 4, 4),
                             dtype=R.dtype)
                T[..., :3, :3] = R
                T[..., :3, 3] = p[..., :3]
                T[..., 3, 3] = 1

                m = (o3d.geometry.TriangleMesh.create_box(*(2 * d))
                        .translate(-d)
                        .transform(T))
                meshes.append(m)
                meshes.append(
                    o3d.geometry.LineSet.create_from_triangle_mesh(m))

        if is_prim:
            continue

        R = np.eye(3)
        if 'base_orn' in o:
            quat = np.asarray(o['base_orn'])
            R = o3d.geometry.Geometry3D.get_rotation_matrix_from_quaternion(
                quat[[3, 0, 1, 2]]
            )

        if o['shape'] == 'box':
            # halfextents -> extents
            if 'radius' in o:
                radius = o['radius']
            else:
                radius = o['dims']
            dims = np.asarray([2 * x for x in radius])
            m = (
                o3d.geometry.trianglemesh.create_box(*dims)
                .translate(-0.5 * dims)
                .rotate(R, center=(0, 0, 0))
                .translate(o['base_pos'])
            )
        elif o['shape'] == 'sphere':
            m = (
                o3d.geometry.trianglemesh.create_sphere(o['dims'])
                .rotate(R, center=(0, 0, 0))
                .translate(o['base_pos'])
            )
        elif o['shape'] in ['capsule', 'cylinder']:
            m = (
                o3d.geometry.trianglemesh.create_cylinder(
                    o['radius'],
                    2.0 *
                    o['half_length']) .rotate(
                    R,
                    center=(
                        0,
                        0,
                        0)) .translate(
                    o['base_pos']))
        elif o['shape'] == 'mesh':
            m = (o3d.io.read_triangle_mesh(o['file'])
                 .rotate(R, center=(0, 0, 0))
                 .translate(o['base_pos'])
                 )
        else:
            shape = o['shape']
            raise ValueError(f'unsupported shape={shape}')
        if lineset:
            meshes.append(
                o3d.geometry.lineset.create_from_triangle_mesh(m))
        else:
            meshes.append(m)
            meshes.append(
                o3d.geometry.lineset.create_from_triangle_mesh(m))
    return meshes, tool


def visualize_diffusion_chain(diffusion_chain,
                              cond,
                              vis_len: int = MAX_VISUALIZATION_LENGTH,
                              cloud: Optional[th.Tensor] = None,
                              cond_type: str = 'curobo'
                              ):
    # Truncate the visualization length...
    if True:
        T0 = len(diffusion_chain) - vis_len
        diffusion_chain = diffusion_chain[T0:]

    # == Get end-effector pose from forward kinematics ==
    ee_poses = (franka_fk(
        th.as_tensor(diffusion_chain[..., :7]))
        [..., :3, 3]).detach().cpu().numpy()

    # == Create Scene Geometry ==
    # FIXME(ycho): hack
    tool = None
    if cond_type == 'custom':
        geom_dict = open3d_spec.to_open3d(
            custom_spec.from_custom(cond)
        )
    elif cond_type == 'curobo':
        # geom_dict = open3d_spec.to_open3d(
        #    curobo_spec.from_curobo(cond)
        # )
        if isinstance(cond, dict):
            cond = [cond]
        obs_info = cond
        geoms, tool = scene_from_obs_info(obs_info)
        geom_dict = {}
        for i_g, g in enumerate(geoms):
            geom_dict[F'obs-{i_g:02d}'] = g
    elif cond_type == 'open3d':
        geom_dict = cond
    elif cond_type == 'cloud':
        cloud_0 = cloud.detach().cpu().numpy()[0]
        g = o3d.geometry.PointCloud()
        g.points = o3d.utility.Vector3dVector(cloud_0)
        geom_dict = {}
        geom_dict['cloud'] = g

    # == Create visualizer handle ==
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    cmap = get_cmap('turbo')

    # == Visualize each step of the diffusion chain ==
    start = True
    for T in range(0, len(diffusion_chain), 1):
        # == Create path geometry ==
        ee_paths = ee_poses[T]
        geom_paths = []
        for ee_path in ee_paths:
            n: int = len(ee_path)
            points = o3d.utility.Vector3dVector(ee_path)
            indices = o3d.utility.Vector2iVector(
                np.stack([np.arange(0, n - 1),
                          np.arange(1, n)],
                         axis=-1).astype(np.int32))
            path = o3d.geometry.LineSet(points=points, lines=indices)
            colors = cmap(np.linspace(0, 1, n - 1))
            path.colors = o3d.utility.Vector3dVector(colors[..., :3])
            geom_paths.append(path)

        for i_p, g in enumerate(geom_paths):
            geom_dict[F'path-{i_p:02d}'] = g

        # == Actually draw everything ==
        vis.clear_geometries()
        if True:
            init = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            init.translate(ee_paths[0][0])
            init.paint_uniform_color(cmap(0)[:3])

            goal = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            goal.translate(ee_paths[0][-1])
            goal.paint_uniform_color(cmap(1)[:3])

            geom_dict['init'] = init
            geom_dict['goal'] = goal

        for v in geom_dict.values():
            R = v.get_rotation_matrix_from_xyz((np.pi / 2 + np.deg2rad(15),
                                                -np.pi,
                                                np.pi / 4 - np.deg2rad(15)
                                                - np.deg2rad(90)
                                                ))
            v = copy.deepcopy(v).rotate(R, center=(0, 0, 0))
            vis.add_geometry(v, reset_bounding_box=start)

        for _ in range(2):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)
        vis.capture_screen_image(F'/tmp/docker/presto-{T:03d}.png')
        start = False

    # WAIT FOREVER...
    try:
        while True:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass

    vis.destroy_window()


def sphere_geoms(urdf: str = None):
    if urdf is None:
        urdf = get_path(
            'env/models/franka_description/robots/panda_arm_kinematics.urdf'
        )
    urdf = URDF.load(urdf,
                     build_scene_graph=True,
                     build_collision_scene_graph=True,
                     load_meshes=False,
                     load_collision_meshes=False,
                     force_mesh=False,
                     force_collision_mesh=False)
    chain, roots, rel_xfms = build_rigid_body_chain(urdf, col=True)

    d = OmegaConf.load(
        '/input/curobo/src/curobo/content/configs/robot/spheres/franka_mesh.yml'
    )
    mesh_list = [[] for _ in chain]
    for link_name, spheres in d['collision_spheres'].items():
        if link_name not in rel_xfms:
            continue
        index = chain.index(roots[link_name])
        T = np.asarray(rel_xfms[link_name], dtype=np.float32)
        ms = []
        for s in spheres:
            m = (trimesh.creation.uv_sphere(radius=s['radius'])
                 .apply_translation(s['center'])
                 .apply_transform(T))
            ms.append(m)
        mesh_list[index].extend(ms)
    mesh_list = [trimesh.util.concatenate(ms)
                 for ms in mesh_list]
    return mesh_list


def visualize_robot_motion(trajs, cond,
                           cloud: Optional[th.Tensor] = None,
                           CUTOFF: int = 192,
                           timeout: Optional[float] = None,
                           export_dir: Optional[str] = None,
                           collision_label=None,
                           true_traj=None,
                           draw_path_edge: bool = True,
                           kcfg=None,
                           draw_robot: bool = True
                           ):
    (urdf, chain, roots, mesh_list, rad_list) = build_geoms(
        get_path(
            'data/robots/franka_panda_simple/robot.urdf'
        ),
        merge=True, col=True, verbose=False, as_acd=False)

    # FIXME(ycho): hardcoded visualization limit
    trajs = trajs[..., :CUTOFF, :]
    CUTOFF = max(CUTOFF, trajs.shape[-2] + 1)
    B, S, _ = trajs.shape

    # Convert to open3d
    mesh_list = [m.as_open3d for m in mesh_list]
    edge_list = [
        o3d.geometry.LineSet.create_from_triangle_mesh(m) for m in mesh_list
    ]

    # Precompute forward kinematics
    Ts = dcn(franka_link_transforms(
        th.as_tensor(trajs[..., :7])
    ))

    Ts_true = None
    if true_traj is not None:
        Ts_true = dcn(franka_link_transforms(
            th.as_tensor(true_traj[..., :7])
        ))

    Ts_kq = None
    if kcfg is not None:
        kq, kl = kcfg
        Ts_kq = dcn(franka_link_transforms(
            th.as_tensor(kq[..., :7])
        ))

    # == Create Scene Geometry ==
    tool = None
    if cond is None:
        geom_dict = {}
    elif isinstance(cond, list):
        # already given as ~primitives
        obs_info = cond
        geoms, tool = scene_from_obs_info(obs_info)
        geom_dict = {}
        for i_g, g in enumerate(geoms):
            geom_dict[F'obs-{i_g:02d}'] = g
    elif isinstance(cond, np.ndarray):
        # already given as ~primitives
        obs_info = cond
        geoms, tool = scene_from_obs_info(obs_info)
        geom_dict = {}
        for i_g, g in enumerate(geoms):
            geom_dict[F'obs-{i_g:02d}'] = g
    elif isinstance(cond, dict):
        obs_info = [cond]
        geoms, tool = scene_from_obs_info(obs_info)
        geom_dict = {}
        for i_g, g in enumerate(geoms):
            geom_dict[F'obs-{i_g:02d}'] = g
    elif isinstance(cond, str):
        geom_dict = {}
        geom_dict['scene'] = o3d.io.read_triangle_mesh(cond)
    elif isinstance(cond, o3d.t.geometry.Geometry):
        geom_dict = {}
        geom_dict['scene'] = cond
    elif isinstance(cond, o3d.geometry.Geometry):
        geom_dict = {}
        geom_dict['scene'] = cond
    else:
        # FIXME(ycho): hardcoded cond-parsing logic;
        # should probably take `cond_type` as argument instead.
        if cond.shape[-1] in [90, 36, 45, 54]:
            # environment is encoded by `encode_primitives`
            obs_info = decode_primitives(cond.detach().cpu().numpy())
            if False:
                obs_info = [obs_info[0]]

            geom_dict = {}
            for i_o in range(len(obs_info)):
                geoms, tool = scene_from_obs_info(obs_info[i_o],
                                                  lineset=False)
                for i_g, g in enumerate(geoms):
                    geom_dict[F'obs-{i_o:02d}-{i_g:02d}'] = g
        elif (cloud is not None) and (cond.shape[-1] == 3):
            # show environment as point cloud
            cloud_0 = cloud.detach().cpu().numpy()[0]
            g = o3d.geometry.PointCloud()
            g.points = o3d.utility.Vector3dVector(cloud_0)
            geom_dict = {}
            geom_dict['cloud'] = g
        else:
            # In this case,
            # environment is defined as a position-varying
            # sphere of radius 0.1
            m = (
                o3d.geometry.TriangleMesh.create_sphere(0.1)
                .translate(dcn(cond[0]))
            )
            geom_dict = {}
            geom_dict['obs'] = m

    state = {
        'done': False,
        'play': False,
        'index': 0
    }

    def _on_quit(*args, **kwds):
        print('QUIT')
        state['done'] = True

    def _on_play(*args, **kwds):
        state['play'] = (not state['play'])

    def _on_next(*args, **kwds):
        state['index'] += 1
        state['index'] = min(state['index'], Ts.shape[-4])

    def _on_prev(*args, **kwds):
        state['index'] -= 1
        state['index'] = max(state['index'], 0)

    # == Create visualizer handle ==
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_callback(ord('Q'), _on_quit)
    vis.register_key_callback(ord('P'), _on_play)
    vis.register_key_callback(ord('J'), _on_prev)
    vis.register_key_callback(ord('K'), _on_next)
    cmap = get_cmap('turbo')

    # == Visualize robot motion ==
    # FIXME(ycho): draw the EE points at the tool-frame,
    # rather than the `panda_hand` frame.
    delta = np.eye(4)
    delta[..., 2, 3] = (0.107 + 0.1034)

    reset: bool = True
    for i_B in range(B):
        ee_path = (Ts @ delta)[..., i_B, :, -1, :3, 3]
        n: int = len(ee_path)
        points = o3d.utility.Vector3dVector(ee_path)
        indices = o3d.utility.Vector2iVector(
            np.stack([np.arange(0, n - 1),
                      np.arange(1, n)],
                     axis=-1).astype(np.int32))
        waypoints = o3d.geometry.PointCloud(points=points)

        # == color waypoints ==
        if collision_label is not None:
            colors = np.where(dcn(collision_label)[i_B][..., None],
                              [[1, 0, 0]],
                              [[0, 0, 1]])
            waypoints.colors = o3d.utility.Vector3dVector(colors.astype(
                np.float32))

        if draw_path_edge:
            path = o3d.geometry.LineSet(points=points,
                                        lines=indices)
            geom_dict[F'path-{i_B:02d}'] = path
        geom_dict[F'wpts-{i_B:02d}'] = waypoints

        if Ts_true is not None:
            true_ee_path = (Ts_true @ delta)[..., i_B, :, -1, :3, 3]
            n: int = len(true_ee_path)
            points = o3d.utility.Vector3dVector(true_ee_path)
            indices = o3d.utility.Vector2iVector(
                np.stack([np.arange(0, n - 1),
                          np.arange(1, n)],
                         axis=-1).astype(np.int32))

            if draw_path_edge:
                true_path = o3d.geometry.LineSet(points=points,
                                                 lines=indices)
                true_path.paint_uniform_color([0, 0, 0])
                geom_dict[F'true-path-{i_B:02d}'] = true_path

            waypoints = o3d.geometry.PointCloud(points=points)
            waypoints.paint_uniform_color([0, 0, 0])
            geom_dict[F'true-wpts-{i_B:02d}'] = waypoints

        if Ts_kq is not None:
            true_ee_path = dcn((Ts_kq @ delta)[..., -1, :3, 3])
            n: int = len(true_ee_path)
            points = o3d.utility.Vector3dVector(true_ee_path)
            indices = o3d.utility.Vector2iVector(
                np.stack([np.arange(0, n - 1),
                          np.arange(1, n)],
                         axis=-1).astype(np.int32))
            colors = np.where(dcn(kl[0, ..., None] > 0),
                              np.asarray([[1, 0, 0]]),
                              np.asarray([[0, 0, 1]]))

            waypoints = o3d.geometry.PointCloud(points=points)
            waypoints.colors = o3d.utility.Vector3dVector(
                dcn(colors.astype(np.float32))
            )
            geom_dict[F'true-wpts-{i_B:02d}'] = waypoints

    while True:
        i_S = state['index']
        # NOTE(ycho): we draw multiple robots simultaneously,
        # so we actually iterate over the outer (batch) axes
        # rather than the inner (sequence) axes.
        if draw_robot:
            for i_B in range(B):
                label = None
                try:
                    if collision_label is not None:
                        label = collision_label[i_B][i_S]
                except IndexError:
                    break
                # == Create path geometry ==
                link_xfms = Ts[..., i_B, i_S, :, :, :]

                ms = copy.deepcopy(mesh_list)
                es = copy.deepcopy(edge_list)
                if label is not None:
                    for e in es:
                        e.paint_uniform_color([1, 0, 0] if label
                                              else [0, 0, 1])

                [m.transform(T) for m, T in zip(ms, link_xfms)]
                [m.transform(T) for m, T in zip(es, link_xfms)]

                # NOTE(ycho): not adding the robot link `volumes` for now;
                # only the edges of the mesh, for clearer visualization.
                # for i_p, g in enumerate(ms):
                #     geom_dict[F'link-{i_B:02d}-{i_p:02d}'] = g
                for i_p, g in enumerate(es):
                    geom_dict[F'edge-{i_B:02d}-{i_p:02d}'] = g

                if tool is not None:
                    m_tool = copy.deepcopy(tool)
                    m_tool.transform(link_xfms[-1])
                    geom_dict['tool'] = m_tool
                    geom_dict['tool-edge'] = o3d.geometry.LineSet.create_from_triangle_mesh(
                        m_tool)

        # == Actually draw everything & animate ==
        vis.clear_geometries()
        for v in geom_dict.values():
            # NOTE(ycho): these values are chosen to show the scene
            # at the best default camera view.
            R = v.get_rotation_matrix_from_xyz((np.pi / 2 + np.deg2rad(15),
                                                -np.pi,
                                                -np.pi / 2
                                                ))
            v = copy.deepcopy(v).rotate(R, center=(0, 0, 0))
            vis.add_geometry(v, reset_bounding_box=reset)
        reset = False

        for _ in range(2):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)

        if export_dir is not None:
            Path(export_dir).mkdir(parents=True, exist_ok=True)
            vis.capture_screen_image(F'{export_dir}/{i_S:03d}.png')

        if state['play']:
            state['index'] += 1
            state['index'] = min(state['index'], Ts.shape[-4])

        if state['done'] or i_S >= CUTOFF - 1:
            break

    # WAIT FOREVER...
    if timeout is None:
        while True:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)
        vis.destroy_window()
    else:
        init = time.time()
        while True:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)
            if (time.time() - init) > timeout:
                break
        vis.destroy_window()


class GuideMPD(nn.Module):
    """
    ~Guidance pipeline implemented in MPD
    """

    def __init__(self,
                 device: str = 'cuda',
                 horizon: int = 256,
                 max_grad_norm: float = 1.0,
                 k_smooth: float = 1e-9,
                 k_coll: float = 1e-2
                 ):
        super().__init__()
        t_args = dict(device=device)
        self.robot = RobotPanda(tensor_args=t_args)
        # NOTE(ycho): "fixed" in MPD.
        dt: float = 5.0 / horizon
        self.max_grad_norm = max_grad_norm
        self.cost_gp = CostGPTrajectory(
            self.robot,
            n_support_points=horizon,
            dt=dt,
            sigma_gp=1.0,
            tensor_args=t_args
        )
        # NOTE(ycho): gaussian smoothing over trajectory parameters
        # for the purposes of propagating collision-cost gradients
        # across to neighboring waypoints. Parameters have been
        # selected somewhat arbitrarily.
        K = th.as_tensor([[scipy.signal.windows.gaussian(15, 4.)]],
                         dtype=th.float32,
                         device=device)
        K /= K.sum()
        self.kernel = K
        self.k_smooth = k_smooth
        self.k_coll = k_coll

    def cost(self,
             cost_coll,
             q: th.Tensor,
             smooth: bool = True):
        # (1) GP cost
        robot = self.robot
        p = robot.get_position(q)
        v = robot.get_velocity(q)
        pv = th.cat([p, v], dim=-1)
        c_smooth = self.cost_gp(pv)

        # (2) collision cost
        # NOTE(ycho):
        # This type of "trajectory smoothing" is not included
        # in the original MPD implementation,
        # but gradient-based guidance does _not_ work
        # unless collision-related corrections in
        # one timestep _also_ influences neighboring timesteps.
        # And NO, smoothing cost does not automatically take
        # care of this.
        if smooth:
            K = self.kernel
            r = K.shape[-1] // 2
            q_smooth = F.conv1d(q.swapaxes(-1, -2),
                                K.expand(q.shape[-1], -1, K.shape[-1]),
                                groups=q.shape[-1],
                                padding='valid').swapaxes(-1, -2)
        else:
            # This code-path optionally disables the smoothing.
            q_smooth = q
        c_coll = cost_coll(q_smooth.contiguous()).clamp_max_(0.1)
        return self.k_smooth * c_smooth.sum() + self.k_coll * c_coll.sum()

    def clip_gradient(self, grad, eps: float = 1e-6):
        grad_norm = th.linalg.norm(grad + eps, dim=-1, keepdims=True)
        scale_ratio = th.clip(grad_norm, 0., self.max_grad_norm) / grad_norm
        grad = scale_ratio * grad
        return grad

    def forward(self,
                cost_coll,
                normalizer,
                q: th.Tensor):
        q = q.detach().clone()
        with th.enable_grad():
            q.requires_grad_(True)
            cost = self.cost(cost_coll, normalizer.unnormalize(q))
            g, = th.autograd.grad([cost.sum()], [q], retain_graph=True)
            g = self.clip_gradient(g)

            # Clear grads @ T=0,-1
            # to prevent updates at endpoints.
            g[..., 0, :] = 0
            g[..., -1, :] = 0

        return q - g.detach()


class GuideSD(nn.Module):
    def __init__(self,
                 greedy_type: str = 'all_frame_exp',
                 clip_grad_by_value=dict(min=-0.1, max=+0.1),
                 scale: float = 1.0,
                 device: str = 'cuda'):
        super().__init__()
        self.scale = scale
        self.greedy_type = greedy_type
        self.clip_grad_by_value = clip_grad_by_value

    def cost(self, x: th.Tensor, q1: th.Tensor):
        """ Compute gradient for planner guidance

        Args:
            x: the denosied signal at current step, which is detached and is required grad
            data: data dict that provides original data

        Return:
            The optimizer objective value of current step
        """
        cost = 0.
        target = q1
        if self.greedy_type == 'last_frame':
            cost += F.l1_loss(x[:, -1, :], target, reduction='mean')
        elif self.greedy_type == 'all_frame':
            cost += F.l1_loss(x, target.unsqueeze(1), reduction='mean')
        elif self.greedy_type == 'all_frame_exp':
            traj_dist = th.norm(x - target.unsqueeze(1), dim=-1, p=1)
            cost += (-1.0) * th.exp(1 / traj_dist.clamp(min=0.01)).sum()
        else:
            raise Exception('Unsupported greedy type')
        return cost

    def forward(self, x: th.Tensor, normalizer, q1: th.Tensor) -> th.Tensor:
        """ Compute gradient for planner guidance
        Args:
            x: the denosied signal at current step
            data: data dict that provides original data

        Return:
            Commputed gradient
        """
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            cost = self.cost(normalizer.unnormalize(x_in), q1)
            grad, = th.autograd.grad(cost, x_in)
            grad = grad * self.scale
            grad = th.clip(grad, **self.clip_grad_by_value)

        return x.detach().clone() - grad.detach()


class PrestoPipeline(DiffusionPipeline):

    @dataclass
    class Config:
        # cost: CuroboCost.Config = CuroboCost.Config(margin=0.0)
        cost: CachedCuroboCost.Config = CachedCuroboCost.Config(margin=0.0)
        init_type: str = 'random'
        cond_type: str = 'data'

        n_denoise_step: int = 1

        n_guide_step: int = 0
        guide_start: int = 100
        guide_scale: float = 0.00001
        post_guide: int = 0

        # evaluate diffusion quality
        # with multiple seeds.
        expand: int = 1
        apply_constraint: bool = False
        index: Optional[int] = None

        optimize: bool = False
        opt_margin: float = 0.001
        use_cached: bool = False
        n_opt_step: int = 1

    def __init__(self,
                 cfg: Config,
                 unet,
                 scheduler,
                 batch_size: int = 4,
                 n_prim=None):
        super().__init__()
        self.cfg = cfg
        self.register_modules(unet=unet,
                              scheduler=scheduler)

        # guidance cost
        device = next(iter(unet.parameters())).device
        if cfg.use_cached:
            assert (n_prim is not None)
            # guidance cost
            self.cost_g = CachedCuroboCost(cfg.cost,
                                           batch_size * cfg.expand,
                                           n_prim=n_prim,
                                           device=device,
                                           )
            # evaluation cost
            self.cost_e = CachedCuroboCost(
                replace(cfg.cost, margin=0.0, sweep=0),
                batch_size * cfg.expand,
                n_prim=n_prim,
                device=device,
            ).to(device)

        else:
            # guidance cost
            self.cost_g = CuroboCost(
                cfg.cost,
                batch_size *
                cfg.expand,
                device=device).to(device)
            # evaluation cost
            self.cost_e = CuroboCost(
                replace(cfg.cost, margin=0.0, sweep=0),
                batch_size * cfg.expand, device=device
            ).to(device)

        if cfg.optimize:
            try:
                traj_eval_config = TrajEvaluatorConfig.from_basic(dof=7,
                                                                  max_dt=0.11)
            except Exception:
                traj_eval_config = TrajEvaluatorConfig(max_dt=0.11)
            tensor_args = TensorDeviceType(device)
            trajopt_config = TrajOptSolverConfig.load_from_robot_config(
                self.cost_g._robot_cfg,
                tensor_args=tensor_args,
                use_cuda_graph=False,
                world_coll_checker=self.cost_g._robot_world.world_model,
                # NOTE(ycho) +4 required due to numerical f.d.
                # for computation of higher-level derivatives in CuRobo.
                traj_tsteps=256 + 4,
                interpolation_steps=768,
                interpolation_dt=0.1,
                use_particle_opt=False,
                traj_evaluator_config=traj_eval_config,
                collision_activation_distance=cfg.opt_margin,
                collision_checker_type=CollisionCheckerType.MESH,
                minimize_jerk=False,
                grad_trajopt_iters=1,
                fixed_iters=True,
            )
            trajopt_solver = TrajOptSolver(trajopt_config)
            self.trajopt_solver = trajopt_solver

        if cfg.n_guide_step > 0:
            self.guide = GuideMPD(device=device)

    def _guide(self, dataset, traj, cond, t,
               constraint_fn, scale, col_label,
               q0=None,
               q1=None,
               i: int = None,
               guide_type: str = 'mpd'
               ):
        if guide_type == 'mpd':
            # Guide_type == `mpd`
            return self.guide(
                # NOTE(ycho): this assumes cost_g.reset() occurs
                # _outside_ of this function (which is true in our case).
                partial(self.cost_g, c=None),
                dataset.normalizer,
                traj.swapaxes(-1, -2)).swapaxes(-1, -2).detach().clone()
        elif guide_type == 'sd':
            # Guide_type == `sd`
            # Usually not a good idea
            # NOTE(ycho): presumably(??) does not need unnormalization
            return GuideSD(device=traj.device)(traj.swapaxes(-1, -2),
                                               # dataset.normalizer.unnormalize(traj.swapaxes(-1,-2)),
                                               dataset.normalizer,
                                               # .swapaxes(-1,-2),
                                               q1=q1).swapaxes(-1, -2).detach().clone()
        # Guide_type == `null`(default) -> just return the input.
        return traj

    @th.no_grad()
    def __call__(
            self,
            data_fn,
            generator: Optional[Union[th.Generator, List[th.Generator]]] = None,
            num_inference_steps: int = 32,
            return_dict: bool = True,
            shuffle_cond: bool = False):
        out = {'metric': {}}

        cfg = self.cfg

        dataset = None
        if hasattr(data_fn, 'dataset'):
            dataset = data_fn.dataset  # HACK

        init, cond, constraint_fn, extras = data_fn()

        # Make a copy-ish, just in case...!
        if isinstance(cond, dict):
            cond = dict(cond)

        traj = init.detach().clone()

        if cfg.apply_constraint:
            traj = constraint_fn(traj)
        else:
            constraint_fn = None

        if shuffle_cond:
            # NOTE(ycho): shuffle cond is _invalid_.
            # Use this option carefully,_only_ for testing
            # whether `cond` input meaningfully
            # influences the generated trajectory.
            index = th.randperm(traj.shape[0],
                                device=traj.device)
        else:
            index = th.arange(traj.shape[0],
                              device=traj.device)

        # Sync dev for perf. measures
        th.cuda.synchronize()
        t00 = time.time()
        if isinstance(cond, dict):
            cond_input = {k: v[index] for (k, v) in cond.items()}

            # Precompute cloud embeddings ... !
            if isinstance(self.unet, DiTCloud):
                cond_input['z_pcd'] = self.unet.cloud_embedder(
                    cond_input['cloud']
                )
        else:
            cond_input = cond[index]
        th.cuda.synchronize()
        t01 = time.time()
        out['metric']['pcd_dt'] = (t01 - t00)

        self.scheduler.set_timesteps(num_inference_steps,
                                     device=traj.device)
        print(F'Got num_inference_steps = {num_inference_steps}')
        trajs = []

        params = None
        if hasattr(self.unet, 'get_params'):
            params = self.unet.get_params(cond_input)

        t0 = time.time()

        traj_inner = traj

        if cfg.n_guide_step > 0 or cfg.optimize:
            # NOTE(ycho): `u_gt` (specifically, q_init/ q_goal)
            # is _necessary_ for optimization/guidance...!
            u_gt = dataset.normalizer.unnormalize(
                extras['true_traj']  # .swapaxes(-1, -2)
            )

        if cfg.n_guide_step > 0:
            if ('prim-label' in extras) and isinstance(self.cost_g,
                                                       CachedCuroboCost):
                guide_cond = extras['prim-label']
            else:
                guide_cond = extras['col-label']
            self.cost_g.reset(guide_cond)

        for t in self.progress_bar(self.scheduler.timesteps):
            trajs.append(traj_inner.detach().clone())

            R: int = (num_inference_steps - t) // 8 + 1

            for repeat in range(cfg.n_denoise_step):
                # 1. predict noise model_output
                # FIXME(ycho): single-dimensional batch assumption
                step_input = th.full((traj.shape[0],),
                                     fill_value=t,
                                     device=traj.device)

                if params is not None:
                    model_output = self.unet(traj_inner, step_input,
                                             class_labels=cond_input,
                                             params=params).sample
                else:
                    model_output = self.unet(traj_inner, step_input,
                                             class_labels=cond_input).sample

                # 2. compute previous traj: x_t -> x_t-1
                if isinstance(self.scheduler, DDIMScheduler):
                    traj_inner = self.scheduler.step(
                        model_output, t, traj_inner, generator=generator,
                        use_clipped_model_output=self.scheduler.config.clip_sample
                    ).prev_sample
                else:
                    traj_inner = self.scheduler.step(
                        model_output, t, traj_inner, generator=generator,
                    ).prev_sample

                # 3. apply constraint, if any
                if cfg.apply_constraint:
                    traj_inner = constraint_fn(traj_inner)

            # Guided sampling
            if t < cfg.guide_start:
                for i_g in range(cfg.n_guide_step):
                    traj_inner = self._guide(
                        dataset, traj_inner.detach_(), cond, t, constraint_fn,
                        cfg.guide_scale,
                        guide_cond,
                        q0=u_gt[..., 0, :],
                        q1=u_gt[..., -1, :],
                        i=i_g
                    )

        t1 = time.time()

        # Append final step
        trajs.append(traj_inner.detach().clone())

        if (cfg.post_guide > 0):
            trajs.append(traj.detach().clone())
            for i in range(cfg.post_guide):
                traj = self._guide(
                    dataset, traj, cond, t, constraint_fn, cfg.guide_scale,
                    i=i)
                trajs.append(traj.detach().clone())

        trajs = th.stack(trajs, dim=0)

        # Postprocess traj...
        if cfg.optimize:
            # NOTE(ycho): cost_g.reset() is needed at least once
            # for TrajOpt to function since they share the world model.
            # so we ensure that here in case n_guide_step<=0.
            if cfg.n_guide_step <= 0:
                if ('prim-label' in extras) and isinstance(self.cost_g,
                                                           CachedCuroboCost):
                    self.cost_g.reset(extras['prim-label'])
                else:
                    self.cost_g.reset(extras['col-label'])

            t00 = time.time()
            if cfg.n_denoise_step > 0:
                if dataset is not None:
                    assert (dataset.normalizer is not None)
                    u_traj = dataset.normalizer.unnormalize(
                        trajs[-1].swapaxes(-1, -2))
                else:
                    normalizer = data_fn.normalizer
                    u_traj = normalizer.unnormalize(
                        trajs[-1].swapaxes(-1, -2))
            else:
                u_traj = dataset.normalizer.unnormalize(
                    extras['true_traj']  # .swapaxes(-1, -2)
                )

            goal_state = JointState.from_position(
                u_gt[..., -1, :])
            init_state = JointState.from_position(
                u_gt[..., 0, :])
            goal = Goal(goal_state=goal_state,
                        current_state=init_state)
            with th.enable_grad():
                if cfg.n_denoise_step <= 0:
                    seed_traj = None
                else:
                    seed_traj = JointState.from_position(
                        u_traj
                    )
                soln = self.trajopt_solver.solve_batch_env(
                    goal,
                    seed_traj=seed_traj,
                    use_nn_seed=False,
                    return_all_solutions=True,
                    num_seeds=1,
                    newton_iters=cfg.n_opt_step
                )
            ic('trajopt succ ?', soln.success)
            if False:
                # NOTE(ycho): optionally interpolate solution to u_traj.shape
                u_traj = soln.interpolated_solution.position
                traj = dataset.normalizer.normalize(u_traj).swapaxes(-1, -2)
                traj = retime_trajectory(
                    traj.swapaxes(-1, -2), n=256).swapaxes(-1, -2)
            else:
                # 768 --> 256
                u_traj = soln.interpolated_solution.position[..., ::3, :]
                traj = dataset.normalizer.normalize(u_traj).swapaxes(-1, -2)

            trajs[-1] = traj
            t01 = time.time()
            ic('t00 ~ t01', t01 - t00)
            out['metric']['opt_dt'] = (t01 - t00)

        # Evaluate collision costs and print the result.
        cost = None
        if True:
            if dataset is not None:
                assert (dataset.normalizer is not None)
                u_traj = dataset.normalizer.unnormalize(
                    trajs[-1].swapaxes(-1, -2))
            else:
                normalizer = data_fn.normalizer
                u_traj = normalizer.unnormalize(
                    trajs[-1].swapaxes(-1, -2))

            # Potentially "densify" trajectory before evaluation.
            # This precludes trajectory evaluations from
            # "winging it" despite not actually being successful.
            # WARN(ycho): the consequence of retiming is that the
            # current visualization may _not_ actually match true collision
            # labels.
            if True:
                u_traj = retime_trajectory(u_traj, n=256)

            if ('prim-label' in extras) and isinstance(self.cost_e,
                                                       CachedCuroboCost):
                cost_label = extras['prim-label']
            else:
                cost_label = extras['col-label']
            self.cost_e.reset(cost_label)

            col_label = cost_label
            extras['col-label'] = col_label

            cost = self.cost_e(u_traj, None, reduce=False)
            cost_r = cost.reshape(cfg.expand, -1, *cost.shape[1:])

            # (r b) s ... -> r b s ...
            col_flag = (cost_r != 0)
            suc_rate = (~col_flag.any(dim=-1).all(dim=0)).float().mean()
            col_rate = col_flag.float().mean()
            col_cost = cost_r.mean()
            max_col = cost_r.max()

            best_seed = th.argmin(col_flag.sum(dim=-1), dim=0, keepdim=True)

            best_cost_traj = th.take_along_dim(
                cost_r,
                best_seed[..., None],
                dim=0).squeeze(dim=0)
            col_rate_in_best_traj = (best_cost_traj > 0).float().mean()
            col_rate = col_rate_in_best_traj
            worst_penetration_in_best_traj = best_cost_traj.amax(dim=-1)
            avg_pen_cost = worst_penetration_in_best_traj.mean()

            worst_cost = cost_r.amax(dim=-1).amin(dim=0)
            worst_idx = th.argmax(worst_cost)
            ic(suc_rate,
               col_rate,
               col_cost)
            ic('u_traj', u_traj.shape)
            normalizer = dataset.normalizer
            tracking_error = th.linalg.norm(
                (extras['true_traj']) - normalizer.normalize(u_traj),
                dim=-1).mean()
            ic('cost',
               cost.shape,
               # cost,
               (cost != 0).float().mean(),
               cost.mean(),
               '90%', th.quantile(cost, 0.99),
               '100%', cost.max(),
               'worst', worst_idx, worst_cost.shape,
               cost_r[:, worst_idx, :].amax(dim=-1),
               'avg_worst_penetration', avg_pen_cost,
               'time', (t1 - t0),
               'deviation', tracking_error
               )
            out['metric'].update(dict(
                suc_rate=suc_rate,
                model_dt=(t1 - t0),
                avg_pen_cost=avg_pen_cost,
                col_rate=(cost != 0).float().mean(),
                avg_col_cost=cost.mean(),
                # path_len=path_len.mean
            ))

        out.update({'traj': traj,
                    'trajs': trajs,
                    'cond': cond,
                    'cost': cost})
        if extras is not None:
            out.update(extras)
        return out


class PrestoGenerator:
    def __init__(self,
                 dataset,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 offset: int = 0,
                 init_type: str = 'random',
                 expand: int = 1,
                 index: Optional[int] = None,
                 cond_type: str = 'data',
                 ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.offset = offset
        self.init_type = init_type
        self.expand = expand
        self.index = index
        self.cond_type = cond_type

    def _apply_constraint(self, x, ref):
        x[..., 0] = ref[..., 0]
        x[..., -1] = ref[..., -1]
        return x

    def __call__(self):
        dataset = self.dataset
        batch_size = self.batch_size

        if isinstance(dataset, PrestoDatasetShelf):
            N = len(dataset)
        else:
            N = len(dataset['trajectory'])
            c = dataset['cond'][..., :-14]

        if self.shuffle:
            index = th.randint(0, N, size=(batch_size,))
        else:
            index = (self.offset + th.arange(batch_size)) % N

        if self.index is not None:
            index[...] = self.index

        # NOTE(ycho): Hack, to ensure every sample in the
        # batch reference the same trajectory:
        index = einops.repeat(index,
                              'b ... -> (r b) ...',
                              r=self.expand)

        if self.index is not None:
            print(dataset[index]['env-label'][..., -14:])

        if isinstance(dataset, PrestoDatasetShelf):
            data = dataset[index]
            traj0 = data['trajectory']
            traj0 = traj0.swapaxes(-1, -2)  # necessary??

            if 'cloud' in data:
                cond = dict(
                    cloud=data['cloud'],
                    task=data['env-label']
                )
            else:
                cond = data['env-label']

            if self.cond_type in ['swap', 'swap-coll', 'swap-goal']:
                alt_data = dataset[(index + 1) % len(dataset)]
                if dataset.cfg.embed_init_goal > 0:
                    d = dataset.cfg.embed_init_goal
                else:
                    d = 7

                # init/goal configs
                coll_cond = alt_data['env-label'][..., :-2 * d]
                task_cond = alt_data['env-label'][..., -2 * d:]

                # We replace the collision-conditioning variable
                # and observe whether there's any change in the
                # env outputs
                if self.cond_type in ['swap', 'swap-coll']:
                    data['env-label'][..., :-2 * d] = coll_cond
                elif self.cond_type in ['swap-goal']:
                    data['env-label'][..., -2 * d:] = task_cond
        else:
            # NOTE(ycho): old(deprecated) data-loading scheme.
            traj0 = dataset['trajectory'][index]
            traj0 = traj0.swapaxes(-1, -2)
            cond = dataset['cond'][index]

        if self.init_type == 'random':
            init = th.randn_like(traj0)  # => B, 7, 1000
        elif self.init_type == 'linear':
            init = th.lerp(
                traj0[..., :, :1],
                traj0[..., :, -1:],
                th.linspace(0.0, 1.0, traj0.shape[-1])[None, None, :].to(
                    device=traj0.device)
            )
        elif self.init_type in ['true', 'gt', 'data']:
            init = traj0.detach().clone()
        else:
            raise ValueError(F'Unknown init_type={self.init_type}')

        constraint_fn = partial(
            self._apply_constraint,
            ref=traj0)

        # Also output the input conditions for inspection.
        extras = {}
        if isinstance(dataset, PrestoDatasetShelf):
            extras['col-label'] = data['col-label']
            if 'prim-label' in data:
                extras['prim-label'] = data['prim-label']
        else:
            if 'cloud' in dataset:
                extras['cloud'] = dataset['cloud'][index]

        # Also output the true trajectory.
        if True:
            extras['true_traj'] = data['trajectory']

        return (init, cond, constraint_fn, extras)


def evaluate_presto(cfg,
                   dataset,
                   pipeline,
                   batch_size: int = 1,
                   seed: int = 0,
                   shuffle: bool = True,
                   offset: int = 0,
                   step: Optional[int] = None,
                   show: str = 'motion',
                   export_dir: Optional[str] = None):
    # == Infer Trajectory ==
    if step is None:
        step = cfg.diffusion.num_train_diffusion_iter

    output = pipeline(
        data_fn=PrestoGenerator(dataset,
                               batch_size,
                               shuffle=shuffle,
                               offset=offset,
                               init_type=cfg.pipeline.init_type,
                               expand=cfg.pipeline.expand),
        num_inference_steps=step
    )

    # Un-normalize the output trajectory.
    if cfg.data.normalize:
        lo, hi = dataset['min'], dataset['max']
        # output['traj'] = (0.5 * (output['traj'] + 1) * (hi - lo) + lo)
        hi = hi[None, None, :, None]
        lo = lo[None, None, :, None]
        output['trajs'] = (0.5 * (output['trajs'] + 1) * (hi - lo) + lo)

    # S B D T -> S B T D
    diffusion_chain = output['trajs'].swapaxes(-1, -2)
    if cfg.data.add_task_cond:
        cond = output['cond'][..., :-14]

    cloud = None
    if 'cloud' in output:
        cloud = output['cloud']

    if show == 'chain':
        visualize_diffusion_chain(diffusion_chain, cond=cond,
                                  cloud=cloud)
    elif show == 'motion':
        visualize_robot_motion(
            diffusion_chain[-1],
            cond=cond, cloud=cloud,
            export_dir=export_dir,
            CUTOFF=1000, collision_label=(output['cost'] > 0.0))
    else:
        pass
