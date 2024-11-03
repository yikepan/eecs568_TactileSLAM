# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Debug utilities for visualizing neuralfeels outputs

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def viz_dirs_C(dirs_C: torch.Tensor, poses: np.ndarray, skip: int = 100):
    """
    Visualize the vector field for a single image from camera origin
    """
    poses = poses.cpu().numpy()

    dirs_C = dirs_C.squeeze().cpu().numpy()
    dirs_C = dirs_C.reshape(-1, 3)

    dirs_C = dirs_C[::skip, :]
    mags = np.linalg.norm(dirs_C[:, :2], axis=1)

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    x, y, z = poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3]

    u = dirs_C[:, 0]
    v = dirs_C[:, 1]
    w = dirs_C[:, 2]

    ax.set_box_aspect((1, 1, 1))
    ax.quiver(x, y, z, u, v, w, length=0.01, colors=plt.cm.plasma(mags))
    ax.view_init(azim=-90, elev=90)  # x-y plane
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plotCameras(poses, ax)
    # ax.set_xlim(-1.5, 1.5)
    # ax.set_ylim(-1.5, 1.5)
    # ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect([1, 1, 1])
    plt.show()
    return


def viz_dirs_W(origins: torch.Tensor, dirs_W: torch.Tensor, skip: int = 10):
    """
    Visualize the vector field in world coordinates for a batch of images
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    origins = origins.cpu().numpy()
    dirs_W = dirs_W.squeeze().cpu().numpy()

    origins = origins[::skip, :]
    dirs_W = dirs_W[::skip, :]

    x, y, z = origins[:, 0], origins[:, 1], origins[:, 2]

    u, v, w = dirs_W[:, 0], dirs_W[:, 1], dirs_W[:, 2]

    ax.quiver(x, y, z, u, v, w, length=0.001, color="black")
    # ax.view_init(azim=-90, elev=90) # x-y plane
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))

    plt.show()
    return


def pose2axes(rotm: np.ndarray):
    """
    Convert rotation matrix to x, y, z axes
    """
    x, y, z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    r = R.from_matrix(rotm)  # (N, 3, 3) [qx, qy, qz, qw]
    quivers_u = r.apply(x)
    quivers_v = r.apply(y)
    quivers_w = r.apply(z)
    return quivers_u, quivers_v, quivers_w


def plotCameras(poses: np.ndarray, ax: None):
    """
    Plot camera matrices (XYZ -> RGB)
    """
    if type(poses) is not np.ndarray:
        poses = poses.cpu().numpy()

    axes_sz = 2e-2
    x, y, z = poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3]
    ax.scatter(x, y, z, color="k", s=1)
    u, v, w = pose2axes(poses[:, :3, :3])
    ax.quiver(
        x,
        y,
        z,
        u[:, 0],
        u[:, 1],
        u[:, 2],
        length=axes_sz,
        color="r",
        linewidths=0.5,
        alpha=0.5,
        normalize=True,
    )
    ax.quiver(
        x,
        y,
        z,
        v[:, 0],
        v[:, 1],
        v[:, 2],
        length=axes_sz,
        color="g",
        linewidths=0.5,
        alpha=0.5,
        normalize=True,
    )
    ax.quiver(
        x,
        y,
        z,
        w[:, 0],
        w[:, 1],
        w[:, 2],
        length=axes_sz,
        color="b",
        linewidths=0.5,
        alpha=0.5,
        normalize=True,
    )

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return


def viz_ray_samples(pc: torch.Tensor, poses: torch.Tensor, skip: int = 1):
    """
    Visualize the vector field in world coordinates for a batch of images
    """

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    pc = pc.cpu().numpy()
    pc = pc[::skip, :]

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

    ax.scatter(x, y, z)
    # ax.view_init(azim=-90, elev=90) # x-y plane
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))

    plotCameras(poses, ax)
    plt.show()
    return
