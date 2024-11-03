# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 2D and 3D visualization utilities for neuralfeels

import io

import cv2
import numpy as np
import skimage.measure
import torch
import trimesh
from PIL import Image

from neuralfeels import geometry


def draw_camera(camera, transform, color=(0.0, 1.0, 0.0, 0.8), marker_height=0.2):
    marker = trimesh.creation.camera_marker(camera, marker_height=marker_height)
    marker[0].apply_transform(transform)
    marker[1].apply_transform(transform)
    marker[1].colors = (color,) * len(marker[1].entities)

    return marker


def draw_cameras_from_eyes(eyes, ats, up, scene):
    for eye, at in zip(eyes, ats):
        R, t = geometry.transform.look_at(eye, at, up)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        transform = T @ geometry.transform.to_replica()
        camera = trimesh.scene.Camera(
            fov=scene.camera.fov, resolution=scene.camera.resolution
        )
        marker = draw_camera(camera, transform)
        scene.add_geometry(marker)


def draw_cams(
    batch_size, T_WC_batch_np, scene, color=None, latest_diff=True, cam_scale=1.0
):
    no_color = color is None
    if no_color:
        color = (0.0, 1.0, 0.0, 0.8)
    for batch_i in range(batch_size):
        # if batch_i == (batch_size - 1):
        #     color = (1., 0., 0., 0.8)
        T_WC = T_WC_batch_np[batch_i]

        camera = trimesh.scene.Camera(
            fov=scene.camera.fov, resolution=scene.camera.resolution
        )
        marker_height = 0.3 * cam_scale
        if batch_i == batch_size - 1 and latest_diff:
            if no_color:
                color = (1.0, 1.0, 1.0, 1.0)
                marker_height = 0.5 * cam_scale

        marker = draw_camera(camera, T_WC, color=color, marker_height=marker_height)
        scene.add_geometry(marker[1])


def draw_segment(t1, t2, color=(1.0, 1.0, 0.0)):
    line_segment = trimesh.load_path([t1, t2])
    line_segment.colors = (color,) * len(line_segment.entities)

    return line_segment


def draw_trajectory(trajectory, scene, color=(1.0, 1.0, 0.0)):
    for i in range(trajectory.shape[0] - 1):
        if (trajectory[i] != trajectory[i + 1]).any():
            segment = draw_segment(trajectory[i], trajectory[i + 1], color)
            scene.add_geometry(segment)


def draw_pc(batch_size, pcs_cam, T_WC_batch_np, im_batch=None, scene=None):
    pcs_w = []
    cols = []
    for batch_i in range(batch_size):
        T_WC = T_WC_batch_np[batch_i]
        pc_cam = pcs_cam[batch_i]

        col = None
        if im_batch is not None:
            img = im_batch[batch_i]
            col = img.reshape(-1, 3)
            cols.append(col)

        pc_tri = trimesh.PointCloud(vertices=pc_cam, colors=col)
        pc_tri.apply_transform(T_WC)
        pcs_w.append(pc_tri.vertices)

        if scene is not None:
            scene.add_geometry(pc_tri)

    pcs_w = np.concatenate(pcs_w, axis=0)
    if len(cols) != 0:
        cols = np.concatenate(cols)
    return pcs_w, cols


def marching_cubes_trimesh(numpy_3d_sdf_tensor, level=0.0):
    """
    Convert sdf samples to triangular mesh.
    """
    vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor,
        level=level,
        step_size=1,
    )

    dim = numpy_3d_sdf_tensor.shape[0]
    vertices = vertices / (dim - 1)
    mesh = trimesh.Trimesh(
        vertices=vertices, vertex_normals=vertex_normals, faces=faces
    )

    return mesh


def draw_mesh(sdf, color_by="normals", clean_mesh=True):
    """
    Run marching cubes on sdf tensor to return mesh.
    """
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()
    mesh = marching_cubes_trimesh(sdf)

    # Transform to [-1, 1] range
    mesh.apply_translation([-0.5, -0.5, -0.5])
    mesh.apply_scale(2)

    try:
        # from NICE-SLAM
        if clean_mesh:
            get_largest_components = False
            remove_small_geometry_threshold = 2
            # get connected components
            components = mesh.split(only_watertight=False)
            if get_largest_components:
                areas = np.array([c.area for c in components], dtype=np.float)
                print(areas)
                clean_mesh = components[areas.argmax()]
            else:
                new_components = []
                for comp in components:
                    if comp.area > remove_small_geometry_threshold:
                        new_components.append(comp)
                # print(f"Removed {len(components) - len(new_components)} blobs")
                clean_mesh = trimesh.util.concatenate(new_components)
            vertices = clean_mesh.vertices
            faces = clean_mesh.faces
            mesh = trimesh.Trimesh(vertices, faces)
    except:
        print("clean_mesh error: continuing")

    mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.3)
    if color_by == "normals":
        norm_cols = (-mesh.vertex_normals + 1) / 2
        norm_cols = np.clip(norm_cols, 0.0, 1.0)
        norm_cols = (norm_cols * 255).astype(np.uint8)
        alphas = np.full([norm_cols.shape[0], 1], 255, dtype=np.uint8)
        cols = np.concatenate((norm_cols, alphas), axis=1)
        mesh.visual.vertex_colors = cols
    elif color_by == "height":
        zs = mesh.vertices[:, 1]
        cols = trimesh.visual.interpolate(zs, color_map="viridis")
        mesh.visual.vertex_colors = cols
    else:
        mesh.visual.face_colors = [160, 160, 160, 255]
    return mesh


def capture_scene_im(scene, pose, tm_pose=False, resolution=(1280, 720)):
    if not tm_pose:
        pose = geometry.transform.to_trimesh(pose)
    scene.camera_transform = pose
    data = scene.save_image(resolution=resolution)
    image = np.array(Image.open(io.BytesIO(data)))

    return image


# adapted from https://github.com/NVlabs/BundleSDF/blob/878cee2f1cda23810ff861f6fef2922c96c7a67e/Utils.py#L309C1-L344C13
def draw_xyz_axis(
    color,
    obj_in_cam,
    fx,
    fy,
    cx,
    cy,
    h,
    w,
    scale=0.1,
    thickness=2,
    transparency=0.3,
    is_input_rgb=False,
):
    """
    @color: BGR
    """
    if is_input_rgb:
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    oo = np.array([0, 0, 0]).astype(float)
    xx = np.array([1, 0, 0]).astype(float) * scale
    yy = np.array([0, 1, 0]).astype(float) * scale
    zz = np.array([0, 0, 1]).astype(float) * scale
    pts_of = torch.tensor(np.vstack((oo, xx, yy, zz))).float()  # in object frame
    pts_of = pts_of.to(device=obj_in_cam.device, dtype=obj_in_cam.dtype)
    pts_cf = geometry.transform.transform_points(pts_of, obj_in_cam)  # in camera frame

    pts_2d = geometry.transform.point_cloud_to_image_plane(pts_cf, fx, fy, cx, cy, h, w)
    origin = tuple(pts_2d[0].cpu().numpy())
    xx = tuple(pts_2d[1].cpu().numpy())
    yy = tuple(pts_2d[2].cpu().numpy())
    zz = tuple(pts_2d[3].cpu().numpy())

    line_type = cv2.FILLED
    arrow_len = 0
    tmp = color.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        xx,
        color=(0, 0, 255),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        yy,
        color=(0, 255, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        zz,
        color=(255, 0, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp = tmp.astype(np.uint8)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    return tmp
