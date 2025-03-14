#!/usr/bin/env python3

import numpy as np
import yaml

from collections import defaultdict
from presto.util.math_util import (
    quat_wxyz2xyzw,
    quat_xyzw2wxyz
)

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


def from_curobo_pose(pose: np.ndarray):
    pose = np.asarray(pose, dtype=np.float32)
    t = pose[..., 0:3]
    q = quat_wxyz2xyzw(pose[..., 3:7])
    return np.concatenate([t, q], axis=-1)


def to_curobo_pose(pose: np.ndarray):
    pose = np.asarray(pose, dtype=np.float32)
    t = pose[..., 0:3]
    q = quat_xyzw2wxyz(pose[..., 3:7])
    return np.concatenate([t, q], axis=-1)


def from_curobo(data: dict):
    out = {}

    # NOTE(ycho): legacy code
    if 'attached_object' in data:
        out['__tool__'] = from_curobo(
            data['attached_object']
        )

    if 'cuboid' in data:
        for k, v in data['cuboid'].items():
            g = Cuboid(
                radius=0.5 * np.asarray(v['dims']),
                pose=from_curobo_pose(v['pose'])
            )
            out[k] = g
    if 'sphere' in data:
        for k, v in data['sphere'].items():
            g = Sphere(
                radius=v['radius'],
                pose=from_curobo_pose(v['pose'])
            )
            out[k] = g
    if 'mesh' in data:
        for k, v in data['mesh'].items():
            g = Mesh(
                file=v['file_path'],
                pose=from_curobo_pose(v['pose'])
            )
            out[k] = g
    if 'capsule' in data:
        for k, v in data['capsule'].items():
            height = v['tip'][2] - v['base'][2]
            g = Capsule(
                radius=v['radius'],
                half_length=0.5 * height,
                pose=from_curobo_pose(v['pose'])
            )
            out[k] = g
    if 'cylinder' in data:
        for k, v in data['cylinder'].items():
            height = v['height']
            g = Cylinder(
                radius=v['radius'],
                half_length=0.5 * height,
                pose=from_curobo_pose(v['pose'])
            )
            out[k] = g
    return Scene(out)


class Convert:
    @classmethod
    def cuboid(cls, v: Cuboid):
        return {'dims': 2.0 * v.radius,
                'pose': to_curobo_pose(v.pose)}

    @classmethod
    def sphere(cls, v: Sphere):
        return {'radius': v.radius,
                'pose': to_curobo_pose(v.pose)}

    @classmethod
    def mesh(cls, v: Mesh):
        return {'file_path': v.file,
                'pose': to_curobo_pose(v.pose)}

    @classmethod
    def capsule(cls, v: Capsule):
        return {'radius': v.radius,
                'base': [0.0, 0.0, -v.half_length],
                'tip': [0.0, 0.0, v.half_length],
                'pose': to_curobo_pose(v.pose)}

    @classmethod
    def cylinder(cls, v: Cylinder):
        # return cls.capsule(v)
        return {'radius': v.radius,
                'height': 2.0 * v.half_length,
                'pose': to_curobo_pose(v.pose)}

    cls_from_str = cls_from_str
    str_from_cls = str_from_cls


def to_curobo_dict(data: Scene):
    out = defaultdict(dict)

    if data.base_pose is not None:
        # FIXME(ycho): only applies translation offset, for now
        base_from_world = -data.base_pose[..., :3]

    for k, v in data.geom.items():
        s = Convert.str_from_cls(type(v))
        y = getattr(Convert, s)(v)
        if data.base_pose is not None:
            y['pose'][..., :3] += base_from_world
        out[s][k] = y
    return dict(out)


def to_pod(x):
    if isinstance(x, dict):
        return {k: to_pod(v) for (k, v) in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_pod(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_pod(x.tolist())
    if isinstance(x, np.floating):
        return float(x)
    return x


def save_as_yaml(file: str, data: dict):
    data = {geom_key: {k: to_pod(v) for (k, v) in geom_val.items()}
            for (geom_key, geom_val) in data.items()}
    with open(file, 'w') as fp:
        yaml.dump(data, fp)
