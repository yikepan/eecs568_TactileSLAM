# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Compute metrics for neuralfeels evaluation

import time

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree

np.set_printoptions(precision=2, suppress=True)


def start_timing():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.perf_counter()
        end = None
    return start, end


def end_timing(start, end):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        end = time.perf_counter()
        elapsed_time = end - start
        # Convert to milliseconds to have the same units
        # as torch.cuda.Event.elapsed_time
        elapsed_time = elapsed_time * 1000
    return elapsed_time


def average_3d_error(point_cloud1, point_cloud2):
    # point_cloud1, point_cloud2: numpy arrays of shape (N, 3)
    # ADD-S: symmetric average 3D error pose metric (https://arxiv.org/pdf/1711.00199.pdf)
    # find nearest neighbors for each point in point_cloud1
    tree = KDTree(point_cloud2)
    distances, _ = tree.query(point_cloud1)  # returns euclidean distance
    return np.mean(distances)


def sample_trimesh_points(mesh, num_samples):
    """
    Sample points on trimesh surface
    """
    sampled_points = trimesh.sample.sample_surface(mesh, num_samples)[0]
    return sampled_points


def compute_f_score(
    gt_points_np, recon_mesh, num_mesh_samples=30000, T=[2e-2, 1e-2, 5e-3, 1e-3]
):
    """
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf
    Compute F-score between a ground truth point cloud and a reconstructed mesh.
    gt_points_np: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    recon_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = sample_trimesh_points(recon_mesh, num_mesh_samples)
    # print(f"ptp gen_points_sampled: {np.ptp(gen_points_sampled, axis=0)*1000}, gt_points_np: {np.ptp(gt_points_np, axis=0)*1000}")

    # one_distances is distance from each gen_points_sampled to its nearest neighbor in gt_points_np
    gt_points_kd_tree = KDTree(gt_points_np)
    one_distances, _ = gt_points_kd_tree.query(gen_points_sampled, p=2)

    # two_distances is distance from each gt point to its nearest neighbor in gen_points_sampled
    gen_points_kd_tree = KDTree(
        gen_points_sampled
    )  # build a KD tree for the generated points
    two_distances, _ = gen_points_kd_tree.query(
        gt_points_np, p=2
    )  # find nearest neighbors for all gt_points_np from gen_points_sampled

    f_scores, precisions, recalls = [], [], []
    for t in T:
        precision = (one_distances < t).sum() / len(
            gen_points_sampled
        )  # precision = percentage of gen_points_sampled that have a gt point within T mm
        recall = (two_distances < t).sum() / len(
            gt_points_np
        )  # recall = percentage of gt_points_np that have a gen_points_sampled within T mm
        # compupte F-score = 2 * (precision * recall) / (precision + recall) where
        # precision = percentage of gen_points_sampled that have a gt point within T mm
        f_score = 2 * (precision * recall) / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)

    _, vertex_ids = gen_points_kd_tree.query(np.array(recon_mesh.vertices))
    return (f_scores, precisions, recalls, one_distances, vertex_ids)
