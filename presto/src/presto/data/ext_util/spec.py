#!/usr/bin/env python3

from typing import Union, Optional, Dict
from dataclasses import dataclass
import numpy as np
import torch as th

AnyArray = Union[None, np.ndarray, th.Tensor]


@dataclass
class Mesh:
    vert: AnyArray = None
    face: AnyArray = None
    file: Optional[str] = None
    pose: AnyArray = None
    scale: AnyArray = None


@dataclass
class Cuboid:
    radius: AnyArray = None
    pose: AnyArray = None


@dataclass
class Sphere:
    radius: AnyArray = None
    pose: AnyArray = None


@dataclass
class Cylinder:
    radius: AnyArray = None
    half_length: AnyArray = None
    pose: AnyArray = None


@dataclass
class Capsule:
    radius: AnyArray = None
    half_length: AnyArray = None
    pose: AnyArray = None


@dataclass
class Scene:
    geom: Optional[
        Dict[str, Union[Mesh, Cuboid, Sphere, Cylinder, Capsule, None]]
    ] = None
    base_pose: AnyArray = None


cls_from_str = {
    'cuboid': Cuboid,
    'sphere': Sphere,
    'mesh': Mesh,
    'capsule': Capsule,
    'cylinder': Cylinder
}

_str_from_cls = {
    Cuboid: 'cuboid',
    Sphere: 'sphere',
    Mesh: 'mesh',
    Capsule: 'capsule',
    Cylinder: 'cylinder'
}

clss = sorted(cls_from_str)

# NOTE(ycho): offset by one to allow
# idx=0 to mean "none"
_idx_from_cls = {k: 1 + clss.index(v)
                 for (k,v) in _str_from_cls.items()}
_idx_from_str = {v: 1 + clss.index(v)
                 for (k,v) in _str_from_cls.items()}


def _is_cls(arg, cls):
    try:
        return isinstance(arg, cls) or issubclass(arg, cls)
    except TypeError as e:
        return False


def str_from_cls(cls):
    if _is_cls(cls, Cuboid):
        return 'cuboid'
    if _is_cls(cls, Sphere):
        return 'sphere'
    if _is_cls(cls, Mesh):
        return 'mesh'
    if _is_cls(cls, Capsule):
        return 'capsule'
    if _is_cls(cls, Cylinder):
        return 'cylinder'
    raise ValueError(F'unknown geom cls={cls}')

def idx_from_cls(cls):
    return _idx_from_cls.get(cls, 0)

def idx_from_str(cls):
    return _idx_from_str.get(cls, 0)
