#!/usr/bin/env python3

import h5py
import json
import numpy as np
import itertools
import pickle
from typing import Optional, List, Dict, Union
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from tqdm.auto import tqdm
from yourdfpy import URDF
from matplotlib import pyplot as plt

import torch as th
import torch.nn.functional as F
import einops
from curobo.geom.sdf.world_mesh import WorldMeshCollision

from presto.util.torch_util import dcn
from presto.util.math_util import quaternion_from_matrix
from presto.network.layers import SinusoidalPositionalEncoding
from presto.data.franka_util import franka_fk
from presto.data.normalize import Normalize
from presto.data.data_util import rename_obs
from presto.util.path import ensure_directory, get_path

from presto.data.ext_util.spec import (
    Mesh,
    Cuboid,
    Sphere,
    Cylinder,
    Capsule,
    Scene,
    cls_from_str,
    str_from_cls,
    idx_from_str
)
from presto.data.ext_util.curobo_spec import (
    from_curobo
)
from presto.cost.cached_curobo_cost import make_world
from icecream import ic

# NOTE(ycho): default path for datasets
root = get_path('../../../data/')


def retime_trajectory(q: th.Tensor,
                      n: Optional[int] = None,
                      const_vel: bool = True,
                      plot: bool = False):
    """
    interpolate trajectory with uniform spacing.
    """
    to_np = False
    if not isinstance(q, th.Tensor):
        to_np = True
        q = th.as_tensor(q, dtype=th.float32)
    inp = q
    s = q.shape
    q = q.reshape(-1, *q.shape[-2:])
    i = (q == q[..., -1:, :]).all(dim=-1).long().argmax(dim=-1)
    i = (i + 1).clamp_max_(q.shape[-2])
    if n is None:
        n = q.shape[-2]

    if not const_vel:
        # Uniform retiming
        grid = i[:, None] * th.linspace(0, 1, n,
                                        device=i.device)[None, :]
        # map grid from (0~n) to (-1,+1)
        grid = 2.0 * (grid / q.shape[-2]) - 1.0
        ic(grid.min(), grid.max())

        q = einops.repeat(q, 'b t c -> b c t one', one=1)
        grid = einops.repeat(grid, 'b t -> b t x y', x=1, y=1)
        grid = th.cat([grid * 0, grid], dim=-1)  # NHW2
        out = F.grid_sample(q, grid, 'bilinear', 'border')[..., 0]
        out = einops.rearrange(out, 'b c t -> b t c')
    else:
        # retime, proportional to distance
        # (=effectively makes the trajectory "constant velocity")
        grid = th.diff(q, dim=-2, prepend=q[..., :1, :])  # -> 256,768,6
        grid = th.linalg.norm(grid, dim=-1)
        grid = th.cumsum(grid, dim=-1)
        grid = grid / grid[..., -1:]  # 0 ~ 1; ex: (0.2, 0.3, 0.4, 0.8, 1.0)
        t = th.linspace(0, 1, n, device=grid.device).expand(
            *grid.shape[:-1], -1)
        i0 = (th.searchsorted(grid, t, right=True) - 1).clamp_min_(0)
        i1 = (i0 + 1).clamp_max_(n - 1)
        q0 = th.take_along_dim(q, i0[..., None], dim=-2)
        q1 = th.take_along_dim(q, i1[..., None], dim=-2)
        t0 = th.take_along_dim(grid, i0, dim=-1)
        t1 = th.take_along_dim(grid, i1, dim=-1)
        w = (t - t0) / (t1 - t0 + 1e-6)
        out = th.lerp(q0, q1, w[..., None])

    s = list(s)
    s[-2] = n
    out = out.reshape(s)

    if plot:
        x0 = inp[0]
        x1 = out[0]
        plt.plot(dcn(x0[..., 0]), label='pre')
        plt.plot(dcn(x1[..., 0]), label='post')
        plt.grid()
        plt.xlabel('time')
        plt.ylabel('pos[0]')
        plt.title('effect of retime')
        plt.legend()
        plt.show()

    if to_np:
        out = dcn(out)
    return out


def to_scale3(spec: Union[Mesh, Cuboid, Sphere, Cylinder, Capsule, None]):
    out = np.zeros(3)

    if isinstance(spec, Cuboid):
        out[...] = spec.radius
        return out

    if isinstance(spec, Sphere):
        out[...] = spec.radius
        return out

    if isinstance(spec, Cylinder):
        out[..., 0] = spec.radius
        out[..., 1] = spec.radius
        out[..., 2] = spec.half_length
        return out

    if isinstance(spec, Capsule):
        out[..., 0] = spec.radius
        out[..., 1] = spec.radius
        out[..., 2] = spec.half_length
        return out

    raise ValueError(F'Unknown Geometry = {spec}')


def soa_from_aos(x):
    """
    struct-of-arrays from array-of-structs
    """
    return {k: [xi[k] for xi in x]
            for k in x[0].keys()}


def proc_scene(scene: Scene,
               world: WorldMeshCollision):
    # Map configs to a form that _can_ actually
    # allow for in-place updates to CuRobo buffers.

    # TODO(ycho): consider assert for
    # `n` is **all same** across all enves.
    n_cube = world._env_n_obbs[0]
    n_mesh = world._env_n_mesh[0]

    counts = defaultdict(lambda: 0)

    cube_dims = np.zeros((n_cube, 3),
                         dtype=np.float32)
    cube_pose = np.zeros((n_cube, 7),
                         dtype=np.float32)
    cube_mask = np.zeros((n_cube, ), dtype=bool)

    mesh_dims = np.zeros((n_mesh, 3),
                         dtype=np.float32)
    mesh_pose = np.zeros((n_mesh, 7),
                         dtype=np.float32)
    mesh_mask = np.zeros((n_mesh, ), dtype=bool)
    mesh_type = np.zeros((n_mesh, ), dtype=np.int32)

    for (k, v) in scene.geom.items():
        c = str_from_cls(v)
        t = idx_from_str(c)
        i = counts[c]
        name = F'{c}_{0:02d}_{i:02d}'
        counts[c] += 1

        if isinstance(v, Cuboid):
            # cube
            cube_idx = world._env_obbs_names[0].index(name)
            cube_dims[cube_idx][...] = v.radius
            cube_pose[cube_idx][...] = v.pose
            cube_mask[cube_idx] = 1
            continue
        else:
            # mesh
            mesh_idx = world._env_mesh_names[0].index(name)
            mesh_dims[mesh_idx][...] = to_scale3(v)
            mesh_pose[mesh_idx][...] = v.pose
            mesh_mask[mesh_idx] = 1
            mesh_type[mesh_idx] = t
            continue
    return dict(
        cube_dims=cube_dims,
        cube_pose=cube_pose,
        cube_mask=cube_mask,
        mesh_dims=mesh_dims,
        mesh_pose=mesh_pose,
        mesh_mask=mesh_mask,
        mesh_type=mesh_type
    )


class PrestoDatasetShelf(th.utils.data.Dataset):
    """
    presto dataset with 1 sphere.
    """

    @dataclass
    class Config:
        dataset_dir: str = F'{root}/presto_shelf/rename'
        dataset_type: str = 'pkl'
        pattern: str = '*.pkl'

        normalize: bool = True
        device: str = 'cuda'
        load_count: Optional[int] = None
        use_kcfg: bool = True
        add_task_cond: bool = True
        binarize: bool = False
        one_hot: bool = False
        stride: int = 1
        embed_init_goal: int = 0
        retime: bool = False

        prim: bool = False
        index: Optional[int] = None
        prim_label: bool = False
        use_v2: bool = False
        add_cloud: bool = False

    def __init__(self,
                 cfg: Config,
                 split: str = 'train'):
        assert (cfg.dataset_type == 'pkl')
        assert (cfg.use_kcfg)

        if cfg.add_cloud:
            assert (cfg.use_v2)

        self.cfg = cfg
        self.split = split
        if cfg.dataset_type == 'pkl':
            (self.data, self.normalizer) = self.__load_pkl()
        elif cfg.dataset_type == 'h5':
            (self.data, self.normalizer) = self.__load()

    def renormalize_(self, normalizer):
        # unnormalize
        data = dict(self.data)
        for k in ['trajectory', 'start', 'goal']:
            data[k] = self.normalizer.unnormalize(data[k])

        # renormalize
        self.normalizer = normalizer
        for k in ['trajectory', 'start', 'goal']:
            data[k] = self.normalizer(data[k])
        self.data = data

    def __load_pkl(self):
        cfg = self.cfg

        data_root = Path(cfg.dataset_dir)
        device = cfg.device
        pkls = list(data_root.glob(cfg.pattern))
        print(F'loading from {pkls[0]} to {pkls[-1]}')

        data = {}

        for pkl in tqdm(pkls, desc='load-pkl'):
            with open(pkl, 'rb') as fp:
                datum = pickle.load(fp)
            for k, v in datum.items():
                if k not in data:
                    data[k] = []
                data[k].append(dcn(v))

        if len(data['qs']) == 1:
            data = {k: v[0] for (k, v) in data.items()}
        else:
            for k in tqdm(data.keys(), desc='cat'):
                if k in ['ws']:
                    continue
                data[k] = (
                    np.concatenate(data[k], axis=0)
                )
                if k not in ['ws']:
                    data[k] = data[k].astype(np.float32)
            data['ws'] = np.asarray(sum(data['ws'], []),
                                    dtype=object)

        out = {
            'trajectory': th.as_tensor(data['qs'],
                                       dtype=th.float32,
                                       device=device),
            'env-label': th.as_tensor(data['ys'],
                                      dtype=th.float32,
                                      device=device),
            # NOTE(ycho): `col-label` is annotated
            # as a dictionary of strings, so it's an
            # exception for sending to torch.
            # One alternative is to encode as zero-padded
            # string of bytes, which would allow the
            # below code block to be activated:
            # 'col-label': th.as_tensor(data['ws'],
            #                           dtype=th.uint8,
            #                           device=device),
            'col-label': data['ws'],

            'start': th.as_tensor(data['qs'][..., 0, :],
                                  dtype=th.float32,
                                  device=device),
            'goal': th.as_tensor(data['qs'][..., -1, :],
                                 dtype=th.float32,
                                 device=device),
        }

        if cfg.one_hot:
            assert (not cfg.binarize)
            lab = data['ss']
            _, idx = np.unique(lab, return_inverse=True)
            out['env-label'] = F.one_hot(th.as_tensor(idx, device=device))

        # Overwrite `col-label` with
        # primitive-centric affine transforms.
        if cfg.prim:
            if cfg.use_v2:
                if 'prim-data' in data:
                    out['prim-label'] = data['prim-data']
                # FIXME(ycho): hard-coded number of max prims per scene.
                # NOTE(ycho): maybe this _should_ be included in `data` as
                # metadata.
                self._num_prims = {'sphere': 12, 'cuboid': 19, 'cylinder': 14}
            else:
                col_label = out['col-label']
                if False:
                    num_prims = {k: max([(len(qq[k]) if k in qq else 0)
                                        for qq in col_label])
                                 for k in
                                 # col_label[0].keys()
                                 ['sphere', 'cuboid', 'cylinder']
                                 }
                else:
                    # FIXME(ycho): hardcoded; for debugging only
                    num_prims = {'sphere': 12, 'cuboid': 19, 'cylinder': 14}
                self._num_prims = num_prims
                # allow print here, to ensure users watch out for this
                print('num_prims', num_prims)
                col_label = [from_curobo(c) for c in col_label]

                robot_world = make_world(1, num_prims)
                world_model = robot_world.world_model
                col_label = [proc_scene(c, world_model) for c in col_label]
                col_label = soa_from_aos(col_label)
                col_label = {k: np.stack(v) for (k, v) in col_label.items()}
                col_label = {k: th.as_tensor(v,
                                             dtype=th.float32,
                                             device=device) for (k, v)
                             in col_label.items()}
                out['prim-label'] = col_label

        if cfg.prim_label:
            if cfg.use_v2:
                out['env-label'] = th.as_tensor(data['prim_code'],
                                                dtype=th.float32,
                                                device=device)
            else:
                assert (cfg.prim)
                if 'prim-label' in out:
                    env_ = out['prim-label']
                else:
                    env_ = out['col-label']

                # Sort cubes by volume, to break permutation invariance
                cube_vol = th.prod(
                    env_['cube_dims'] * env_['cube_mask'][..., None],
                    dim=-1)
                cube_idx = th.argsort(cube_vol, stable=True, dim=-1)
                cube_dims = th.take_along_dim(env_['cube_dims'],
                                              cube_idx[..., None], dim=-2)
                cube_pose = th.take_along_dim(env_['cube_pose'],
                                              cube_idx[..., None], dim=-2)
                cube_mask = th.take_along_dim(env_['cube_mask'],
                                              cube_idx, dim=-1)

                # Sort meshes by volume, to break permutation invariance
                mesh_vol = th.prod(
                    env_['mesh_dims'] * env_['mesh_mask'][..., None],
                    dim=-1)
                mesh_idx = th.argsort(mesh_vol, stable=True, dim=-1)
                mesh_dims = th.take_along_dim(env_['mesh_dims'],
                                              mesh_idx[..., None], dim=-2)
                mesh_pose = th.take_along_dim(env_['mesh_pose'],
                                              mesh_idx[..., None], dim=-2)
                mesh_mask = th.take_along_dim(env_['mesh_mask'],
                                              mesh_idx, dim=-1)

                prim_label = th.cat([
                    cube_dims.flatten(start_dim=-2),
                    cube_pose.flatten(start_dim=-2),
                    cube_mask,
                    mesh_dims.flatten(start_dim=-2),
                    mesh_pose.flatten(start_dim=-2),
                    mesh_mask], dim=-1)
                out['env-label'] = prim_label

        if cfg.add_cloud:
            # Somehow generate scene cloud...
            out['cloud'] = th.as_tensor(data['cloud'],
                                        dtype=th.float32,
                                        device=device)

        if cfg.retime:
            out['trajectory'] = retime_trajectory(out['trajectory'])

        if cfg.binarize:
            assert (not cfg.prim_label)
            out['env-label'] = (out['env-label'] > 0).float()

        if cfg.stride > 1:
            out['trajectory'] = out['trajectory'][..., ::cfg.stride, :]

        if cfg.add_cloud:
            assert (cfg.add_task_cond)

        if cfg.add_task_cond:
            q0 = out['start']
            q1 = out['goal']
            if cfg.embed_init_goal > 0:
                emb = SinusoidalPositionalEncoding(q0.shape[-1],
                                                   cfg.embed_init_goal,
                                                   flatten=True,
                                                   pad=True).to(q0.device)
                q0 = emb(q0 / (2 * th.pi))
                q1 = emb(q1 / (2 * th.pi))

            if cfg.add_cloud:
                # NOTE(ycho): omit preprocessed `env-label` from inputs
                out['env-label'] = th.cat([q0, q1], dim=-1)
            else:
                out['env-label'] = th.cat([out['env-label'], q0, q1], dim=-1)
        lower_limits = data['qs'].reshape(-1, 7).min(axis=0)
        upper_limits = data['qs'].reshape(-1, 7).max(axis=0)

        if cfg.normalize:
            normalizer = Normalize.from_minmax(
                th.as_tensor(lower_limits,
                             dtype=th.float32, device=device),
                th.as_tensor(upper_limits,
                             dtype=th.float32, device=device)
            )
        else:
            normalizer = Normalize.identity(device=device)

        out['trajectory'] = normalizer(out['trajectory'])
        out['start'] = normalizer(out['start'])
        out['goal'] = normalizer(out['goal'])

        # NOTE(ycho): for debugging, allow selecting
        # just a single index out of the list of items.
        # FIXME(ycho): Does _not_ work for nested structs!
        if cfg.index is not None:
            out = {k: v[cfg.index:cfg.index + 1] for (k, v) in out.items()}

        return (out, normalizer)

    @property
    def seq_len(self):
        return self.data['trajectory'].shape[-2]

    @property
    def obs_dim(self):
        return self.data['trajectory'].shape[-1]

    @property
    def cond_dim(self):
        return self.data['env-label'].shape[-1]

    def __len__(self):
        return len(self.data['trajectory'])

    def __getitem__(self, index):
        if self.cfg.prim:
            return {k: (v[index] if k not in ['prim-label']
                        else {kk: vv[index] for (kk, vv) in v.items()})
                    for k, v in self.data.items()}
        else:
            return {k: v[index] for k, v in self.data.items()}


def collate_fn(xlist):
    cols = [x.pop('col-label') for x in xlist]
    out = th.utils.data.default_collate(xlist)
    out['col-label'] = np.stack(cols, axis=0)
    return out
