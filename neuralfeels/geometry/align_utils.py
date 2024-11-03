# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/psodhi/tactile-in-hand

import copy

import numpy as np
import open3d as o3d


def visualize_registration(source, target, transformation, vis3d=None, colors=None):
    """Open3D visualizer for registration"""
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)

    source_copy.transform(transformation)

    clouds = [source, target_copy, source_copy]

    if colors is not None:
        clouds[0].paint_uniform_color(colors[0])  # black, source
        clouds[1].paint_uniform_color(colors[1])  # green, target
        clouds[2].paint_uniform_color(colors[2])  # red, transformed

    vis3d.add_geometry(clouds[0])
    vis3d.add_geometry(clouds[1])
    vis3d.add_geometry(clouds[2])

    vis3d.run()
    vis3d.destroy_window()


def icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    T_init=np.eye(4),
    mcd=0.01,
    max_iter=15,
):
    """Point to point ICP registration

    Args:
        source: source point cloud
        target: target point cloud
        T_init : Defaults to np.eye(4).
        mcd :  Defaults to 0.01.
        max_iter : Defaults to 15.
    """
    result = o3d.pipelines.registration.registration_icp(
        source=source,
        target=target,
        max_correspondence_distance=mcd,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter
        ),
    )
    transformation = result.transformation
    metrics = [result.fitness, result.inlier_rmse, result.correspondence_set]
    return transformation, metrics


def register(
    points3d_1,
    points3d_2,
    T_init=np.eye(4),
    debug_vis=False,
):
    """Register two point clouds using ICP and returns the 6DoF transformation"""

    cloud_1, cloud_2 = (points3d_1, points3d_2)

    T_reg, metrics_reg = icp(source=cloud_1, target=cloud_2, T_init=T_init)

    # print("T_reg: ", T_reg)
    if debug_vis:
        colors = [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
        ]  # black, green, red

        vis3d = o3d.visualization.Visualizer()
        vis3d.create_window()
        visualize_registration(
            source=cloud_1,
            target=cloud_2,
            transformation=T_reg,
            vis3d=vis3d,
            colors=colors,
        )

    return T_reg, metrics_reg
