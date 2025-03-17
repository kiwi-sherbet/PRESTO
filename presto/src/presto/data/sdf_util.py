#!/usr/bin/env python3

from typing import List
from pathlib import Path
from functools import partial
from yourdfpy import URDF, filename_handler_magic
import logging

import numpy as np
import torch
import torch as th

try:
    import coacd
except ImportError:
    logging.debug('CoACD not found; `assume_convex` needs to be True for `build_geoms`.')

import copy
import trimesh
from tqdm.auto import tqdm
from tempfile import TemporaryDirectory


try:
    import pymeshlab
except ImportError:
    print('pymeshlab import failed. Mesh simplification disabled...')
    pymeshlab = None
pymeshlab = None


def scene_to_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    if len(scene.graph.nodes_geometry) == 1:
        # Take cheaper option if possible.
        node_name = scene.graph.nodes_geometry[0]
        (transform, geometry_name) = scene.graph[node_name]
        mesh = scene.geometry[geometry_name]
        if not (transform == np.eye(4)).all():
            mesh.apply_transform(transform)
    else:
        # Default = dump
        mesh = scene.dump(concatenate=True)
    return mesh


def _simplify_mesh(input_parts: List[trimesh.Trimesh],
                   min_face_count: int = 4,
                   max_face_count: int = 128):
    if pymeshlab is None:
        return input_parts
    output_parts = []
    with TemporaryDirectory() as tmpdir:
        for i, p in enumerate(input_parts):
            # Export each part.
            dst_obj = F'{tmpdir}/p{i:03d}-pre.obj'
            p.export(dst_obj)

            # simplify part with meshlab
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(dst_obj)
            max_fc = int((max_face_count if isinstance(max_face_count, int)
                          else max_face_count[i]))
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=max(min_face_count, max_fc),
                preserveboundary=True,
                preservenormal=True,
                preservetopology=True)
            out_file = F'{tmpdir}/p{i:03d}-post.obj'
            ms.save_current_mesh(out_file)
            mesh = trimesh.load(out_file, file_type='obj')
            output_parts.append(mesh)
    return output_parts


def apply_coacd(mesh, simplify: bool = True, **kwds):
    max_concavity: float = kwds.pop('max_concavity', 0.01)
    max_convex_hull: int = kwds.pop('max_convex_hull', 4)
    preprocess: bool = kwds.pop('preprocess', True)
    resolution: int = kwds.pop('resolution', 2048)
    mcts_max_depth: int = kwds.pop('mcts_max_depth', 3)
    mcts_iterations: int = kwds.pop('mcts_iterations', 256)
    mcts_nodes: int = kwds.pop('mcts_nodes', 64)

    cmesh = coacd.Mesh()
    cmesh.vertices = mesh.vertices
    cmesh.indices = mesh.faces

    parts = coacd.run_coacd(
        cmesh,
        threshold=max_concavity,
        max_convex_hull=max_convex_hull,
        resolution=resolution,
        mcts_max_depth=mcts_max_depth,
        mcts_iterations=mcts_iterations,
        mcts_nodes=mcts_nodes,
        preprocess=preprocess
    )

    mesh_parts = [
        trimesh.Trimesh(
            np.asanyarray(p.vertices),
            np.asanyarray(p.indices).reshape((-1, 3))
        ) for p in parts
    ]
    if simplify:
        mesh_parts = _simplify_mesh(mesh_parts, **kwds)
    return mesh_parts


def load_acd_obj(filename: str, **kwds):
    return trimesh.load(filename,
                        split_object=True,
                        group_material=False,
                        skip_texture=True,
                        skip_materials=True,
                        **kwds)


def build_geoms(urdf_file: str,
                merge: bool,
                col: bool = True,
                verbose: bool = False,
                as_acd: bool = False,
                assume_convex: bool = True):
    if as_acd:
        assert (not merge)

    filename_handler = partial(
        filename_handler_magic,
        dir=Path(urdf_file).parent
    )

    urdf = URDF.load(urdf_file,
                     build_scene_graph=True,
                     build_collision_scene_graph=True,
                     load_meshes=False,
                     load_collision_meshes=False,
                     force_mesh=False,
                     force_collision_mesh=False)

    chain, roots, rel_xfms = build_rigid_body_chain(urdf, col=col)

    rad_list = None
    if (merge or as_acd):
        mesh_list = [[] for _ in chain]
        rad_list = [[] for _ in chain]
    else:
        mesh_list = [None for _ in chain]

    for link_name, link in tqdm(urdf.link_map.items(), disable=(not verbose)):
        link_meshes = []
        link_shapes = link.collisions if col else link.visuals
        for shape in link_shapes:
            g = shape.geometry

            # NOTE(ycho):
            # Currently cannot deal with URDFs with
            # primitive-type of geometries.
            assert (g.box is None)
            assert (g.cylinder is None)
            assert (g.sphere is None)

            if g.mesh is None:
                continue

            mesh_file: str = filename_handler(g.mesh.filename)

            if as_acd:
                cvx = load_acd_obj(mesh_file)
                if isinstance(cvx, trimesh.Scene):
                    scene = cvx
                    acd_parts = []
                    for node_name in scene.graph.nodes_geometry:
                        (transform, geometry_name) = scene.graph[node_name]
                        mesh = scene.geometry[geometry_name]
                        if (verbose) and (not trimesh.convex.is_convex(mesh)):
                            logging.warn(
                                F'part {node_name} for {link_name} was not convex'
                            )
                            mesh = mesh.convex_hull
                        part = copy.deepcopy(mesh).apply_transform(transform)
                        acd_parts.append(part)
                else:
                    if trimesh.convex.is_convex(cvx):
                        acd_parts = [cvx]
                    else:
                        if verbose:
                            logging.warn(
                                F'part {mesh_file} is not convex'
                            )
                        if assume_convex:
                            acd_parts = [cvx.convex_hull]
                        else:
                            acd_parts = apply_coacd(cvx)
                link_meshes.extend(acd_parts)
            else:
                mesh = trimesh.load(
                    mesh_file,
                    ignore_broken=False,
                    force="mesh",
                    skip_materials=True,
                )
                link_meshes.append(mesh)

        if len(link_meshes) <= 0:
            continue

        # NOTE(ycho): Consider optionlly simplifying meshes.
        # Low priority, since this is probably
        # less important when using `col`-type geometry.
        # meshes = _simplify_mesh(meshes)

        # Pre-apply transforms w.r.t. root link.
        index = chain.index(roots[link_name])
        T = np.asarray(rel_xfms[link_name],
                       dtype=np.float32)
        if g.mesh.scale is not None:
            S = np.eye(4, dtype=T.dtype)
            S[:3, :3] *= np.asarray(g.mesh.scale)
            T = T @ S
        if shape.origin is not None:
            T = T @ shape.origin

        if (merge or as_acd):
            link_meshes = [m.apply_transform(T) for m in link_meshes]
            mesh_list[index].extend(link_meshes)
            if as_acd:
                rad_list[index].extend([convex_radius(m) for m in link_meshes])
        else:
            # It seems a bit 'counterintuitive' but when `merge` is off,
            # since we don't do boolean union at the end,
            # we can just concatenate the triangles here online.
            if mesh_list[index] is not None:
                mesh_list[index] = trimesh.util.concatenate([
                    mesh_list[index], mesh.apply_transform(T)])
            else:
                mesh_list[index] = mesh.apply_transform(T)

    # NOTE(ycho): we cannot use concat(...) for computing
    # distance to robot, since it might return wrong distances
    # for interior points that could map to (nonexistent) interior faces.
    if merge:
        mesh_list = [
            (trimesh.boolean.union(ms)
             if len(ms) > 1
             else ms[0])
            for ms in mesh_list]

    return (urdf, chain, roots, mesh_list, rad_list)


def convex_radius(m):
    return np.linalg.norm(m.vertices, axis=-1).max()


def build_rigid_body_chain(urdf: URDF, col: bool = True):
    """Pre-compute the chain of links that share a common rigid body
    transform."""

    # Logic to compute that chain
    root_links = list(set([urdf.base_link] +
                          [j.child for j in urdf.actuated_joints]))
    ss = [len(urdf._successors(link)) for link in root_links]
    indices = np.argsort(ss)[::-1]
    chain = [root_links[i] for i in indices]

    # Logic to compute child links
    visited = set()
    sublinks = {}
    for c in chain[::-1]:
        s = list(urdf._successors(c))
        sublinks[c] = set(s).difference(visited)
        visited.update(s)

    # Compute (fixed) relative transforms w.r.t. root frames
    rel_xfms = {k: np.eye(4) for k in urdf.link_map.keys()}
    for root_link, child_links in sublinks.items():
        for child_link in child_links:
            T = urdf.get_transform(child_link, root_link,
                                   collision_geometry=col)
            rel_xfms[child_link] = T

    roots = {}
    for root, subs in sublinks.items():
        for sub in subs:
            roots[sub] = root
    return (chain, roots, rel_xfms)
