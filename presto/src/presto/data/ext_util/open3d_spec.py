#!/usr/bin/env python3

import open3d as o3d
import numpy as np
from presto.data.ext_util.spec import (
    Mesh,
    Cuboid,
    Sphere,
    Cylinder,
    Capsule,
    Scene
)


def to_open3d(scene: Scene, wire: bool = False):
    """ This function is intended mostly for visualization. """
    out = {}

    for (k, v) in scene.geom.items():
        pos = v.pose[..., :3]
        quat = v.pose[..., 3:7]
        R = o3d.geometry.Geometry3D.get_rotation_matrix_from_quaternion(
            quat[[3, 0, 1, 2]]
        )

        if isinstance(v, Mesh):
            S = np.eye(4)
            if v.scale is not None:
                S[(0, 1, 2), (0, 1, 2)] = v.scale
            if v.file is not None:
                g = (o3d.io.read_triangle_mesh(v.file)
                     .transform(S)
                     .rotate(R, center=(0, 0, 0))
                     .translate(pos)
                     )
                out[k] = g
            elif (v.vert is not None) and (v.face is not None):
                g = o3d.geometry.TriangleMesh()
                g.vertices = o3d.utility.Vector3dVector(v.vert)
                g.triangles = o3d.utility.Vector3iVector(v.face)
                g = (g
                     .transform(S)
                     .rotate(R, center=(0, 0, 0))
                     .translate(pos)
                     )
                out[k] = g
        elif isinstance(v, Cuboid):
            dims = np.multiply(v.radius, 2.0)
            g = (
                o3d.geometry.TriangleMesh.create_box(*dims)
                .translate(-0.5 * dims)
                .rotate(R, center=(0, 0, 0))
                .translate(pos)
            )
            out[k] = g
        elif isinstance(v, Sphere):
            g = (o3d.geometry.TriangleMesh.create_sphere(v.radius)
                 .translate(pos))
            out[k] = g
        elif isinstance(v, Capsule):
            g_rod = (o3d.geometry.TriangleMesh.create_cylinder(
                v.radius, 2.0 * v.half_length)
            )
            g_pos_cap = (o3d.geometry.TriangleMesh.create_sphere(
                v.radius)
                .translate([0, 0, v.half_length])
            )
            g_neg_cap = (o3d.geometry.TriangleMesh.create_sphere(
                v.radius)
                .translate([0, 0, -v.half_length])
            )
            g = sum([g_pos_cap, g_neg_cap], g_rod)
            g = (g
                 .rotate(R, center=(0, 0, 0))
                 .translate(pos)
                 )
            out[k] = g
        elif isinstance(v, Cylinder):
            g = (o3d.geometry.TriangleMesh.create_cylinder(
                v.radius, 2.0 * v.half_length)
                .rotate(R, center=(0, 0, 0))
                .translate(pos))
            out[k] = g
    if wire:
        out = {k: o3d.geometry.LineSet.create_from_triangle_mesh(v)
               for (k, v) in out.items()}
    return out
