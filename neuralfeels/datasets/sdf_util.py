# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Utility functions for SDF computation and visualization

import colorsys
import hashlib
import os

import matplotlib as mpl
import numpy as np
import open3d as o3d
import trimesh
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy import ndimage
from scipy.spatial import cKDTree as KDTree

from neuralfeels.contrib.urdf import SceneGraph, URDFParser, URDFTree


def load_gt_mesh(file, color=True):
    """
    Load ground-truth mesh from URDF file
    """
    # Parse the URDF file
    parser = URDFParser(file)
    parser.parse()
    # Construct the URDF tree
    links = parser.links
    joints = parser.joints
    tree = URDFTree(links, joints)
    scene = SceneGraph(tree.root)
    mesh = scene.getMesh()[0]

    # SDF computation needs trimesh, but visualization needs open3d so we load both
    mesh_trimesh = trimesh.Trimesh(
        np.asarray(mesh.vertices),
        np.asarray(mesh.triangles),
        vertex_normals=np.asarray(mesh.vertex_normals),
    )
    mesh_path = tree.root.link.visuals[0].geometry_mesh["filename"]
    mesh_scale = tree.root.link.visuals[0].geometry_mesh["scale"][0]
    object_name = os.path.dirname(mesh_path).split("/")[-1]
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path, color)
    mesh_o3d = mesh_o3d.scale(mesh_scale, center=mesh_o3d.get_center())

    if not color:
        # assign random color, taken from viser: https://nerfstudio-project.github.io/viser/
        mesh_o3d.paint_uniform_color(
            colorsys.hls_to_rgb(
                np.random.default_rng(
                    np.frombuffer(
                        hashlib.md5(object_name.encode("utf-8")).digest(),
                        dtype="uint32",
                    )
                    + 5
                ).uniform(),
                0.6,
                0.9,
            )
        )

    return mesh_trimesh, mesh_o3d


def saturate_colors(rgb_array, factor):
    """Increase the saturation of an RGB array by a factor."""
    import colorsys

    # Convert the array to HSL color space
    hsl_array = np.zeros_like(rgb_array)
    for i in range(rgb_array.shape[0]):
        hsl_array[i] = colorsys.rgb_to_hls(*rgb_array[i])

    # Increase the saturation value
    hsl_array[:, 1] *= factor

    # Convert the array back to RGB color space
    rgb_array_out = np.zeros_like(rgb_array)
    for i in range(rgb_array.shape[0]):
        rgb_array_out[i] = colorsys.hls_to_rgb(*hsl_array[i])

    return rgb_array_out


def get_grid_pts(dims, transform):
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    z = np.arange(dims[2])
    x = x * transform[0, 0] + transform[0, 3]
    y = y * transform[1, 1] + transform[1, 3]
    z = z * transform[2, 2] + transform[2, 3]

    return x, y, z


def eval_sdf_interp(sdf_interp, pc, handle_oob="except", oob_val=0.0):
    """param:
    handle_oob: dictates what to do with out of bounds points. Must
    take either 'except', 'mask' or 'fill'.
    """

    reshaped = False
    if pc.ndim != 2:
        reshaped = True
        pc_shape = pc.shape[:-1]
        pc = pc.reshape(-1, 3)

    if handle_oob == "except":
        sdf_interp.bounds_error = True
    elif handle_oob == "mask":
        dummy_val = 1e99
        sdf_interp.bounds_error = False
        sdf_interp.fill_value = dummy_val
    elif handle_oob == "fill":
        sdf_interp.bounds_error = False
        sdf_interp.fill_value = oob_val
    else:
        assert True, "handle_oob must take a recognised value."

    sdf = sdf_interp(pc)

    if reshaped:
        sdf = sdf.reshape(pc_shape)

    if handle_oob == "mask":
        valid_mask = sdf != dummy_val
        return sdf, valid_mask

    return sdf


def get_colormap(sdf_range=[-2, 2], surface_cutoff=0.01):
    white = np.array([1.0, 1.0, 1.0, 1.0])
    sdf_range[1] += surface_cutoff - (sdf_range[1] % surface_cutoff)
    sdf_range[0] -= surface_cutoff - (-sdf_range[0] % surface_cutoff)

    positive_n_cols = int(sdf_range[1] / surface_cutoff)
    viridis = cm.get_cmap("viridis", positive_n_cols)
    positive_colors = viridis(np.linspace(0.2, 1, int(positive_n_cols)))
    positive_colors[0] = white

    negative_n_cols = int(np.abs(sdf_range[0]) / surface_cutoff)
    redpurple = cm.get_cmap("RdPu", negative_n_cols).reversed()
    negative_colors = redpurple(np.linspace(0.0, 0.7, negative_n_cols))
    negative_colors[-1] = white

    colors = np.concatenate((negative_colors, white[None, :], positive_colors), axis=0)
    sdf_cmap = ListedColormap(colors)

    norm = mpl.colors.Normalize(sdf_range[0], sdf_range[1])
    sdf_cmap_fn = cm.ScalarMappable(norm=norm, cmap=sdf_cmap)
    # plt.colorbar(sdf_cmap_fn)
    # plt.show()
    return sdf_cmap_fn


def voxelize_subdivide(
    mesh, pitch, origin_voxel=np.zeros(3), max_iter=10, edge_factor=2.0
):
    """
    Adapted from trimesh function allow for shifts in the origin
    of the SDF grid. i.e. there doesn't need to be a voxel with
    centere at [0, 0, 0].

    Voxelize a surface by subdividing a mesh until every edge is
    shorter than: (pitch / edge_factor)
    Parameters
    -----------
    mesh:        Trimesh object
    pitch:       float, side length of a single voxel cube
    max_iter:    int, cap maximum subdivisions or None for no limit.
    edge_factor: float,
    Returns
    -----------
    VoxelGrid instance representing the voxelized mesh.
    """
    max_edge = pitch / edge_factor

    if max_iter is None:
        longest_edge = np.linalg.norm(
            mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=1
        ).max()
        max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))), 0)

    # get the same mesh sudivided so every edge is shorter
    # than a factor of our pitch
    v, f = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=max_iter
    )

    # convert the vertices to their voxel grid position
    hit = (v - origin_voxel) / pitch

    # Provided edge_factor > 1 and max_iter is large enough, this is
    # sufficient to preserve 6-connectivity at the level of voxels.
    hit = np.round(hit).astype(int)

    # remove duplicates
    unique, inverse = trimesh.grouping.unique_rows(hit)

    # get the voxel centers in model space
    occupied_index = hit[unique]

    origin_index = occupied_index.min(axis=0)
    origin_position = origin_voxel + origin_index * pitch

    return trimesh.voxel.base.VoxelGrid(
        trimesh.voxel.encoding.SparseBinaryEncoding(occupied_index - origin_index),
        transform=trimesh.transformations.scale_and_translate(
            scale=pitch, translate=origin_position
        ),
    )


def sdf_from_occupancy(occ_map, voxel_size):
    # Convert occupancy field to sdf field
    inv_occ_map = 1 - occ_map

    # Get signed distance from occupancy map and inv map
    map_dist = ndimage.distance_transform_edt(inv_occ_map)
    inv_map_dist = ndimage.distance_transform_edt(occ_map)

    sdf = map_dist - inv_map_dist

    # metric units
    sdf = sdf.astype(float)
    sdf = sdf * voxel_size

    return sdf


def sdf_from_mesh(mesh, voxel_size, extend_factor=0.15, origin_voxel=np.zeros(3)):
    # Convert mesh to occupancy field
    voxels = voxelize_subdivide(mesh, voxel_size, origin_voxel=origin_voxel)
    voxels = voxels.fill()
    occ_map = voxels.matrix
    transform = voxels.transform

    # Extend voxel grid around object
    extend = np.array(occ_map.shape) * extend_factor
    extend = np.repeat(extend, 2).reshape(3, 2)
    extend = np.round(extend).astype(int)
    occ_map = np.pad(occ_map, extend)
    transform[:3, 3] -= extend[:, 0] * voxel_size

    sdf = sdf_from_occupancy(occ_map, voxel_size)

    return sdf, np.array(transform)


def colorize_mesh(color_pcd, mesh, sigma=0.01):
    """
    Colorize the mesh by interpolating the colors of the point cloud with Gaussian kernel
    """
    # downsample the point cloud
    color_pcd = color_pcd.voxel_down_sample(voxel_size=0.001)
    pc_positions = color_pcd.point.positions.numpy().astype(np.float64)
    pc_colors = color_pcd.point.colors.numpy()
    pc_tree = KDTree(pc_positions)
    # Compute the distances between each mesh vertex and all the points in the point cloud
    distances, indices = pc_tree.query(np.asarray(mesh.vertices), k=20)
    # Compute the weights for each neighboring point based on its distance to the vertex using a Gaussian kernel
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]
    mesh_colors = np.sum(weights[:, :, np.newaxis] * pc_colors[indices], axis=1)
    return o3d.utility.Vector3dVector(mesh_colors)
