#!/usr/bin/env python3

from typing import List, Dict, Any
import numpy as np
from presto.data.ext_util.spec import (
    Mesh,
    Cuboid,
    Sphere,
    Cylinder,
    Capsule,
    Scene
)


def from_custom(data: List[Dict[str, Any]]) -> Scene:
    out = {}
    for i, o in enumerate(data):
        # Parse pose
        quat = np.asarray([0, 0, 0, 1])
        if 'base_orn' in o:
            quat = np.asarray(o['base_orn'])
        pos = o.get('base_pos', (0, 0, 0))
        pose = np.cat([pos, quat], axis=-1)

        # Parse geometry
        if o['shape'] == 'box':
            if 'radius' in o:
                radius = o['radius']
            else:
                # NOTE(ycho): we assume `dims` annotates
                # halfExtents
                radius = o['dims']
            out[F'obs_{i:03d}'] = Cuboid(radius, pose)
        elif o['shape'] == 'sphere':
            out[F'obs_{i:03d}'] = Sphere(o['dims'], pose)
        elif o['shape'] == ['capsule']:
            out[F'obs_{i:03d}'] = Capsule(o['radius'], o['half_length'], pose)
        elif o['shape'] == ['cylinder']:
            out[F'obs_{i:03d}'] = Cylinder(o['radius'], o['half_length'], pose)
        elif o['shape'] == 'mesh':
            out[F'obs_{i:03d}'] = Mesh(file=o['file'], pose=pose)
        else:
            shape = o['shape']
            raise ValueError(F'Unsupported shape={shape}')
    return Scene(out)
