# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/psodhi/tactile-in-hand

import copy

import numpy as np
import open3d as o3d

from thirdparty.fmr.model import PointNet, Decoder, SolveRegistration
from thirdparty.fmr.demo import Demo
import torch

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


def register_fmr(
    points3d_1, 
    points3d_2,
    voxed_size,
):
    """
    Register two point clouds using fmr
    
    Args:
        points3d_1: np.array()
        
    """
    # Load fmr model
    fmr = Demo()
    model = fmr.create_model()
    pretrained_path = "C:\Workspace\Codebase\\neuralfeels\\thirdparty\\fmr\\result\\fmr_model_7scene.pth"
    model.load_state_dict(torch.load(pretrained_path))
    device = "cpu"
    model.to(device)
    
    # Downsample points
    downpcd1 = points3d_1.voxel_down_sample(voxel_size=voxed_size)
    p1 = np.asarray(downpcd1.points)
    p1 = np.expand_dims(p1, 0)
    downpcd2 = points3d_2.voxel_down_sample(voxel_size=voxed_size)
    p2 = np.asarray(downpcd2.points)
    p2 = np.expand_dims(p2, 0)
    
    # Estimation point registration
    T_est = fmr.evaluate(model, p1, p2, device) # array(4, 4)
    
    # Convert data type to array
    T_est = np.array(T_est)
    
    return T_est

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source_pcd, target_pcd, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    return source_pcd, target_pcd, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(result_ransac, source, target, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def register_global(source_pcd, target_pcd, voxel_size):
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_pcd, target_pcd, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    result_icp = refine_registration(result_ransac, source, target, voxel_size)
    return result_icp.transformation
    
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

if __name__ == '__main__':
    source = o3d.io.read_point_cloud("C:\Workspace\Codebase\\neuralfeels\\thirdparty\\fmr\data\\neuralfeels\source_pcd.ply")
    target = o3d.io.read_point_cloud("C:\Workspace\Codebase\\neuralfeels\\thirdparty\\fmr\data\\neuralfeels\\target_pcd.ply")
    T_est = register_fmr(source, target)
    print(f"fmr: {T_est}")
    T_icp, _ = register(source, target)
    print(f"icp: {T_icp}")