#!/usr/bin/env python3

import numpy as np
from itertools import product

SHAPE_TYPE = {
    'none': [1, 0, 0],
    'box': [0, 1, 0],
    'sphere': [0, 0, 1]
}


def encode_primitive(info):
    out = []
    # add shape type, 3
    out.extend(SHAPE_TYPE.get(info['shape']))
    # add position, 3
    out.extend(info['base_pos'])
    # add dimenison, 3
    if info['shape'] == 'box':
        out.extend(info['dims'])
    else:
        out.extend([info['dims']] * 3)
    return out


def encode_primitives(obs_info, max_len: int):
    out = []

    for o in obs_info:
        out.extend(encode_primitive(o))

    # zero-pad the remainder.
    assert (len(out) <= max_len)
    if len(out) < max_len:
        out.extend([0] * (max_len - len(out)))

    return out


def decode_primitives(cond):
    shape_type = {
        'none': [1, 0, 0],
        'box': [0, 1, 0],
        'sphere': [0, 0, 1]
    }
    # e.g. NxOx6
    cond = cond.reshape(*cond.shape[:-1], -1, 9)  # (1, 10, 9) in theory
    out = np.empty(cond.shape[:-2], dtype=object)
    for index in product(*[range(n) for n in cond.shape[:-2]]):
        c = cond[index]  # e.g. Mx6
        out[index] = []
        for q in c:
            shape_type = None
            if q[..., 1] == 1:
                shape_type = 'box'
            elif q[..., 2] == 1:
                shape_type = 'sphere'
            if shape_type is None:
                continue
            out[index].append({
                'shape': shape_type,
                'base_pos': q[..., 3:6],
                'dims': (q[..., 6:9] if shape_type == 'box' else q[..., 6]),
                # EXTRA FILEDS
                'mass': 0,
                'base_quat': [0, 0, 0, 1],
                # NOTE(ycho): `type` annotations unused, for now
                'type': ('wall' if shape_type == 'box' else 'obstacle')
            })
    return out
