#!/usr/bin/env python3

from typing import Dict, Any
import yaml
import numpy as np
import trimesh

from cho_util.math import transform as tx
from presto.data.ext_util.spec import (
    Mesh,
    Cuboid,
    Sphere,
    Cylinder,
    Capsule,
    Scene
)


def _arr(x):
    if x is None:
        return None
    if isinstance(x, str):
        return np.fromstring(x, sep=' ')
    return x


def _parse_fromto(fromto, cancel: bool = True):
    p0, p1 = _arr(fromto).reshape(2, 3)
    dp = (p1 - p0)
    length = np.linalg.norm(dp, axis=-1)
    half_length = 0.5 * length

    u_z = np.zeros_like(dp)
    u_z[..., 2] = 1.0
    u_from_z = _align(dp / length[..., None],
                      u_z)
    q_uz = tx.rotation.quaternion.from_matrix(u_from_z)
    if cancel:
        # NOTE(ycho): it seems geom_xpos[...] _bakes in_
        # the offset rotations, assuming that
        # that tha capsules/cylinders are _always_
        # oriented along the +z axis anyway.
        q_uz[...] = 0
        q_uz[..., 3] = 1
    return half_length, q_uz


def _align(u_dst: np.ndarray, u_src: np.ndarray):
    """
    u_dst = R @ u_src
    """
    v = np.cross(u_src, u_dst)
    c = np.dot(u_src, u_dst)
    k = 1.0 / (1.0 + c)
    R = np.asarray([
        v[0] * v[0] * k + c,
        v[1] * v[0] * k - v[2],
        v[2] * v[0] * k + v[1],

        v[0] * v[1] * k + v[2],
        v[1] * v[1] * k + c,
        v[2] * v[1] * k - v[0],

        v[0] * v[2] * k - v[1],
        v[1] * v[2] * k + v[0],
        v[2] * v[2] * k + c
    ]).reshape(3, 3)
    return R


def convert_k(data: dict, k: str):
    geom_info = data['col_geom_infos'][k]
    pose_info = data['init_geom_xpose']
    mesh_info = data['col_mesh_infos']  # .get(k, None)
    geom_id = data['col_geom_ids']['object'][k]

    # Configure pose.
    pose = np.asarray(pose_info[geom_id])
    xyz = pose[..., 0:3]
    quat = pose[..., 3:7]
    pose = np.concatenate([xyz, quat], axis=-1)

    t = geom_info['type']
    s = _arr(geom_info.get('size', None))
    if t == 'mesh':
        m = mesh_info[geom_info['mesh']]
        mesh_file = m['file']

        # abs-coded filepaths are problematic,
        # But I guess we'll just tolerate this here.
        # FIXME(ycho): fragile path replacement logic,
        # may not always work, depending on the incoming data.
        mesh_file = mesh_file.replace(
            '/home/ming/Projects/etude/VIBRATO-dev/models/',
            '/input/PRESTO/external/legato/models/')
        mesh_file = mesh_file.replace(
            '/home/ming/Projects/etude/ETUDE-dev/vibrato-lite/models/',
            '/input/PRESTO/external/legato/models/')
        mesh_file = mesh_file.replace(
            '/home/ming/.pyenv/versions/etude/lib/python3.9/site-packages/robosuite/models/assets/arenas/../textures/',
            '/input/robosuite/robosuite/models/assets/textures/')

        if True:
            # Option 1: load as file
            scale = m.get('scale', '1 1 1')
            return Mesh(file=mesh_file,
                        pose=_arr(pose),
                        scale=_arr(scale))
        else:
            # Option 2: load as vertices-etc
            # In this scheme,
            # scale is pre-multiplied for the vertices.
            scale = _arr(m.get('scale', '1 1 1'))
            mesh = trimesh.load(mesh_file)
            mesh.apply_scale(scale)
            return Mesh(vert=np.array(mesh.vertices),
                        face=np.array(mesh.faces),
                        pose=pose)
    elif t == 'box':
        return Cuboid(radius=s, pose=_arr(pose))
    elif t == 'capsule':
        if 'fromto' in geom_info:
            radius = s[0]
            half_length, q_uz = (
                _parse_fromto(geom_info['fromto'])
            )
            pose[..., 3:7] = tx.rotation.quaternion.multiply(
                pose[..., 3:7],
                q_uz)
        else:
            radius, half_length = s[:2]
        return Capsule(radius=radius,
                       half_length=half_length,
                       pose=pose)
    elif t == 'cylinder':
        if 'fromto' in geom_info:
            radius = s[0]
            half_length, q_uz = (
                _parse_fromto(geom_info['fromto'])
            )
            pose[..., 3:7] = tx.rotation.quaternion.multiply(
                pose[..., 3:7],
                q_uz)
        else:
            radius, half_length = s[:2]
        return Cylinder(radius=radius,
                        half_length=half_length,
                        pose=pose)
    elif t == 'sphere':
        radius = s[0]
        return Sphere(radius=radius, pose=pose)
    raise ValueError(F'unknown t={t}')


def from_mingyo(data: dict, filter_fn=None):
    keys = list(data['col_geom_ids']['object'].keys())
    if filter_fn is not None:
        keys = filter_fn(keys)
    # Uncomment below line to visualize the robot
    # keys += list(data['col_geom_ids']['robot'].keys())
    out = {}
    for k in keys:
        out[k] = convert_k(data, k)

    link0_id = data['col_geom_ids']['robot']['robot_arm_link0_collision']
    link0_pose = np.asarray(data['init_geom_xpose'][link0_id])
    return Scene(out, base_pose=link0_pose)


def main():
    import open3d as o3d
    from presto.data.ext_util.open3d_spec import to_open3d
    from presto.data.ext_util.curobo_spec import to_curobo_dict, save_as_yaml
    with open('/tmp/docker/env_3_1_6_4757/info.yaml', 'r') as fp:
        data = yaml.safe_load(fp)
    scene = from_mingyo(data)
    scene_m = to_open3d(scene, wire=False)
    scene_w = to_open3d(scene, wire=True)
    if True:
        scene_c = to_curobo_dict(scene)
        save_as_yaml('/tmp/sample-2.yml', scene_c)
    smin = data['slot_range_min_global']
    smax = data['slot_range_max_global']
    bound = {k: np.stack([smin[k], smax[k]], axis=0)
             for k in smin.keys()}
    geoms = list(scene_m.values()) + list(scene_w.values())

    for k, v in bound.items():
        ext = np.abs(v[1] - v[0])
        ctr = 0.5 * (v[0] + v[1])
        print(ctr, ext)
        box = o3d.geometry.LineSet.create_from_triangle_mesh(
            o3d.geometry.TriangleMesh.create_box(*ext).translate(
                ctr - 0.5 * ext))
        box.paint_uniform_color([1, 0, 0])
        geoms.append(box)
    o3d.visualization.draw(geoms)


if __name__ == '__main__':
    main()
