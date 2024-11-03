# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 3D transformations and geometry functions

import typing
import warnings

import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchlie.functional as lieF
import trimesh
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from skimage import morphology


def viz_boolean_img(x):
    x = x.to(torch.uint8)[..., None].repeat(1, 1, 3)
    return x.cpu().numpy() * 255


def euler2matrix(angles=[0, 0, 0], translation=[0, 0, 0], xyz="xyz", degrees=False):
    r = R.from_euler(xyz, angles, degrees=degrees)
    pose = np.eye(4)
    pose[:3, 3] = translation
    pose[:3, :3] = r.as_matrix()
    return pose


# TODO: fix this
@torch.jit.script
def ray_dirs_C(
    B: int,
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
    depth_type: str = "z",
):
    c, r = torch.meshgrid(
        torch.arange(W, device=device), torch.arange(H, device=device), indexing="ij"
    )
    c, r = c.t().float(), r.t().float()
    size = [B, H, W]

    C = torch.empty(size, device=device)
    R = torch.empty(size, device=device)
    C[:, :, :] = c[None, :, :]
    R[:, :, :] = r[None, :, :]

    z = torch.ones(size, device=device)
    x = -(C - cx) / fx
    y = (R - cy) / fy

    dirs = torch.stack((x, y, z), dim=3)
    if depth_type == "euclidean":
        norm = torch.norm(dirs, dim=3)
        dirs = dirs * (1.0 / norm)[:, :, :, None]

    return dirs


def transform_points(points, T):
    points_T = (
        T
        @ torch.hstack(
            (points, torch.ones((points.shape[0], 1), device=points.device))
        ).T
    ).T
    points_T = points_T[:, :3] / points_T[:, -1][:, None]
    return points_T


def transform_points_batch(points, T):
    points_one = torch.hstack(
        (points, torch.ones((points.shape[0], 1), device=points.device, dtype=T.dtype))
    )  # [B * S, 4]
    points_T = torch.bmm(T, points_one.unsqueeze(2)).squeeze()
    points_T = points_T[:, :3] / points_T[:, -1][:, None]
    return points_T


def transform_points_np(points, T):
    points_T = (T @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
    points_T = points_T[:, 0:3] / points_T[:, -1][:, None]
    points_T = points_T.reshape(-1, 3).astype(np.float32)
    return points_T


@torch.jit.script
def depth_image_to_point_cloud_GPU(
    depth,
    width: float,
    height: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
):
    """
    Based on https://github.com/PKU-MARL/DexterousHands/blob/main/docs/point-cloud.md
    """
    u = torch.arange(0, width, device=device)
    v = torch.arange(0, height, device=device)
    v2, u2 = torch.meshgrid(v, u, indexing="ij")

    Z = depth
    X = -(u2 - cx) / fx * Z
    Y = (v2 - cy) / fy * Z

    points = torch.dstack((X, Y, Z))
    return points


def point_cloud_to_image_plane(
    points,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
):
    """
    Project a set of 3D points to a camera image plane.
    """
    (X, Y, Z) = torch.unbind(points, dim=1)
    u = -(X * fx / Z) + cx
    v = (Y * fy / Z) + cy
    u = torch.clamp(u, 0, width - 1)
    v = torch.clamp(v, 0, height - 1)
    depth = torch.stack((u, v), dim=1).to(torch.int32)
    return depth


@torch.jit.script
def origin_dirs_W(T_WC: torch.Tensor, dirs_C: torch.Tensor):
    R_WC = T_WC[:, :3, :3]
    dirs_W = (R_WC * dirs_C[..., None, :]).sum(dim=-1)  # rotation
    origins = T_WC[:, :3, -1]
    return origins, dirs_W


def normalize(x):
    assert x.ndim == 1, "x must be a vector (ndim: 1)"
    return x / np.linalg.norm(x)


def look_at(
    eye,
    target: typing.Optional[typing.Any] = None,
    up: typing.Optional[typing.Any] = None,
) -> np.ndarray:
    """Returns transformation matrix with eye, at and up.
    Need to apply 180deg rotation about z axis to get the transform in the Open3D convention.

    Parameters
    ----------
    eye: (3,) float
        Camera position.
    target: (3,) float
        Camera look_at position.
    up: (3,) float
        Vector that defines y-axis of camera (z-axis is vector from eye to at).

    Returns
    -------
    T_cam2world: (4, 4) float (if return_homography is True)
        Homography transformation matrix from camera to world.
        Points are transformed like below:
            # x: camera coordinate, y: world coordinate
            y = trimesh.transforms.transform_points(x, T_cam2world)
            x = trimesh.transforms.transform_points(
                y, np.linalg.inv(T_cam2world)
            )
    """
    eye = np.asarray(eye, dtype=float)

    if target is None:
        target = np.array([0, 0, 0], dtype=float)
    else:
        target = np.asarray(target, dtype=float)

    if up is None:
        up = np.array([0, 0, -1], dtype=float)
    else:
        up = np.asarray(up, dtype=float)

    assert eye.shape == (3,), "eye must be (3,) float"
    assert target.shape == (3,), "target must be (3,) float"
    assert up.shape == (3,), "up must be (3,) float"

    # create new axes
    z_axis: np.ndarray = normalize(target - eye)
    x_axis: np.ndarray = normalize(np.cross(up, z_axis))
    y_axis: np.ndarray = normalize(np.cross(z_axis, x_axis))

    # create rotation matrix: [bs, 3, 3]
    R: np.ndarray = np.vstack((x_axis, y_axis, z_axis))
    t: np.ndarray = eye

    return R.T, t


def positionquat2tf(position_quat):
    try:
        position_quat = np.atleast_2d(position_quat)
        # position_quat : N x 7
        N = position_quat.shape[0]
        T = np.zeros((4, 4, N))
        T[0:3, 0:3, :] = np.moveaxis(
            R.from_quat(position_quat[:, 3:]).as_matrix(), 0, -1
        )
        T[0:3, 3, :] = position_quat[:, :3].T
        T[3, 3, :] = 1
    except ValueError:
        print("Zero quat error!")
    return T.squeeze() if N == 1 else T


def to_trimesh(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )


def to_replica(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(180), [0, 0, 1]
    )


def interpolation(keypoints, n_points):
    tick, _ = interpolate.splprep(keypoints.T, s=0)
    points = interpolate.splev(np.linspace(0, 1, n_points), tick)
    points = np.array(points, dtype=np.float64).T
    return points


def backproject_pointclouds(depths, fx, fy, cx, cy, device="cuda"):
    pcs = []
    batch_size = depths.shape[0]
    for batch_i in range(batch_size):
        batch_depth = depths[batch_i]
        h, w = batch_depth.shape
        batch_depth = torch.tensor(batch_depth).to(device)
        pcd = depth_image_to_point_cloud_GPU(batch_depth, w, h, fx, fy, cx, cy, device)
        if torch.is_tensor(pcd):
            pcd = pcd.cpu().numpy()
        pc_flat = pcd.reshape(-1, 3)
        pcs.append(pc_flat)

    pcs = np.stack(pcs, axis=0)
    return pcs


def project_pointclouds(pcs, fx, fy, cx, cy, w, h, device="cuda"):
    depths = []
    batch_size = pcs.shape[0]
    for batch_i in range(batch_size):
        batch_points = pcs[batch_i]
        batch_points = torch.tensor(batch_points).to(device)
        depth = point_cloud_to_image_plane(batch_points, fx, fy, cx, cy, w, h)
        if torch.is_tensor(depth):
            depth = depth.cpu().numpy()
        depths.append(depth)
    depths = np.stack(depths, axis=0)
    return depths


def pc_bounds(pc):
    min_x = np.min(pc[:, 0])
    max_x = np.max(pc[:, 0])
    min_y = np.min(pc[:, 1])
    max_y = np.max(pc[:, 1])
    min_z = np.min(pc[:, 2])
    max_z = np.max(pc[:, 2])
    extents = np.array([max_x - min_x, max_y - min_y, max_z - min_z])
    centroid = np.array(
        [(max_x + min_x) / 2.0, (max_y + min_y) / 2.0, (max_z + min_z) / 2]
    )

    return extents, centroid


def make_3D_grid(grid_range, dim, device, transform=None, scale=None):
    t = torch.linspace(grid_range[0], grid_range[1], steps=dim, device=device)
    grid = torch.meshgrid(t, t, t, indexing="ij")
    grid_3d = torch.cat(
        (grid[0][..., None], grid[1][..., None], grid[2][..., None]), dim=3
    )

    grid_3d = transform_3D_grid(grid_3d, transform=transform, scale=scale)

    return grid_3d


def transform_3D_grid(grid_3d, transform=None, scale=None):
    if scale is not None:
        grid_3d = grid_3d * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)

        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d


def equidist_points_sphere(n_samples, radius):
    offset = 2 / n_samples
    increment = np.pi * (3 - np.sqrt(5))

    points = np.zeros([n_samples, 3])
    for i in range(n_samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - np.power(y, 2))
        phi = (i + 1) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        points[i] = radius * np.array([x, y, z])

    return points


# Sample views on a sphere, looking at the center
def look_at_on_sphere(
    n_views, radius, sphere_center, look_at_noise=0.0, cam_axis=[0, 0, 1]
):
    view_centers = equidist_points_sphere(n_views, radius)
    view_centers += sphere_center

    poses = np.eye(4).reshape(1, 4, 4).repeat(n_views, axis=0)
    for i, center in enumerate(view_centers):
        look_at_pt = sphere_center + np.random.normal(0, look_at_noise, 3)
        R, t = look_at(center, look_at_pt, cam_axis)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        poses[i] = to_replica(T)

    return poses


# ray directions and camera centers in box/grid frame
def rays_box_frame(origins, dirs_W, box_transform):
    T_GW = torch.inverse(box_transform)
    origins_g = transform_points(origins, T_GW)
    dirs_g = torch.einsum("ij,mj-> mi", T_GW[:3, :3], dirs_W)

    return origins_g, dirs_g


# returns z vals at which the rays enter and leave the box
# returns torch.nan for rays that do not intersect the box
def ray_box_intersection(
    origins,
    dirs_W,
    box_extents,
    box_transform,
    visualize=False,
):
    # normal direction for each face of axis aligned box
    normals = torch.eye(3).repeat_interleave(2, 0).to(origins.device)
    normals[1::2] *= -1
    normals = normals[None, ...]

    # point at the centre of each face of the box
    plane_pts = torch.diag(box_extents / 2).repeat_interleave(2, 0)
    plane_pts[1::2] *= -1
    plane_pts = plane_pts[None, ...]

    # get origins and ray directions in the box frame
    origins_g, dirs_g = rays_box_frame(origins, dirs_W, box_transform)

    # intersection depth along ray with plane
    numerator = torch.hstack(
        [box_extents[None, :] / 2 - origins_g, box_extents[None, :] / 2 + origins_g]
    )
    denom = torch.hstack([dirs_g, -dirs_g])
    intsct_depth = numerator / denom

    # intersection points
    intsct_pts = origins_g[:, None, :] + intsct_depth[:, :, None] * dirs_g[:, None, :]

    # check intersection points within the face bounds
    check = torch.abs(intsct_pts) <= (box_extents / 2)
    check[:, torch.arange(6), [0, 1, 2, 0, 1, 2]] = True
    intersects_face = (check).all(dim=-1)
    intersects_box = intersects_face.sum(dim=-1) == 2
    # remove face intersections if there are not exactly 2 for the ray
    intersects_face = intersects_face * intersects_box[:, None]

    z_range = intsct_depth[intersects_face].reshape(-1, 2)
    z_range = z_range.sort(dim=-1)[0]

    n_rays = dirs_W.shape[0]
    z_near = torch.full([n_rays], torch.nan, device=origins.device)
    z_far = torch.full([n_rays], torch.nan, device=origins.device)
    z_near[intersects_box] = z_range[:, 0]
    z_far[intersects_box] = z_range[:, 1]

    if visualize:
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.primitives.Box(extents=box_extents.cpu()))
        scene.add_geometry(
            trimesh.PointCloud(plane_pts.view(-1, 3).cpu(), colors=[255, 0, 0])
        )
        scene.add_geometry(
            trimesh.PointCloud(origins_g.detach().cpu(), colors=[0, 0, 255])
        )
        line_starts_g = origins_g - dirs_g * 1e-6
        line_ends_g = origins_g + dirs_g * 0.2
        path_g = trimesh.load_path(
            torch.cat((line_starts_g[:, None], line_ends_g[:, None]), dim=1).cpu()
        )
        scene.add_geometry(path_g)
        scene.add_geometry(
            trimesh.PointCloud(
                intsct_pts[intersects_face].cpu().view(-1, 3), colors=[0, 255, 0]
            )
        )
        scene.show()

    return z_near, z_far


# to ensure the box is always fully in the frame
# use a zoomed out view and then reshape
def get_box_masks_zoomed(
    T_WC_batch,
    box_extents,
    box_transform,
    fx,
    fy,
    cx,
    cy,
    w,
    h,
    get_contours=False,
    reduce_factor=1,
    zoom_factor=2,
    do_viz_check=False,
):
    w_orig, h_orig = w, h

    # reduce resolution to reduce memory / computation
    h = h // reduce_factor
    w = w // reduce_factor
    fx = fx / reduce_factor
    fy = fy / reduce_factor
    cx = cx / reduce_factor
    cy = cy / reduce_factor

    # zoom out view
    fx_zoom = fx / zoom_factor
    fy_zoom = fy / zoom_factor

    mask_zoomed, contour = get_box_masks(
        T_WC_batch,
        box_extents,
        box_transform,
        fx_zoom,
        fy_zoom,
        cx,
        cy,
        w,
        h,
        get_contours,
        do_viz_check,
    )
    h_start = h // 2 - h // zoom_factor // 2
    h_end = h_start + h // zoom_factor
    w_start = w // 2 - w // zoom_factor // 2
    w_end = w_start + w // zoom_factor
    mask = mask_zoomed[:, h_start:h_end, w_start:w_end]
    mask = nn.functional.interpolate(
        mask[:, None, :, :].float(), size=(h_orig, w_orig), mode="nearest"
    )  # needs channel dim
    mask = mask[:, 0].bool()

    if contour is not None:
        contour = contour[:, h_start:h_end, w_start:w_end]
        contour = nn.functional.interpolate(
            contour[:, None, :, :].float(), size=(h_orig, w_orig), mode="nearest"
        )  # needs channel dim
        contour = contour[:, 0].bool()

    if do_viz_check:
        # mask_normal = get_box_masks(
        #     T_WC_batch, box_extents, box_transform, fx, fy, cx, cy, w, h, do_viz_check
        # )
        # if reduce_factor != 1:
        #     mask_normal = nn.functional.interpolate(
        #         mask_normal[:, None, :, :].float(), size=(h_orig, w_orig), mode="nearest"
        #     )[:, 0].bool()  # needs channel dim

        w_viz, h_viz = mask_zoomed.shape[2] // 4, mask_zoomed.shape[1] // 4
        viz = np.hstack(
            [
                # cv2.resize(viz_boolean_img(mask_normal[-1]), (w_viz, h_viz)),
                cv2.resize(viz_boolean_img(mask[-1]), (w_viz, h_viz)),
                # cv2.resize(viz_boolean_img(mask[-1] != mask_normal[0]), (w_viz, h_viz)),
                cv2.resize(viz_boolean_img(mask_zoomed[-1]), (w_viz, h_viz)),
            ]
        )
        cv2.imshow("zoom convex hull mask check", viz)
        cv2.waitKey(1)

    # erode mask 10% to remove rays that just penetrate edge of box
    eroded_mask = mask
    if torch.any(mask):
        kernel_size = int(0.1 * torch.sqrt(mask.sum()) / mask.shape[0])
        kernel_size += kernel_size % 2 - 1  # must be odd
        padding = kernel_size // 2
        maxpool = nn.MaxPool2d(kernel_size, stride=1, padding=padding)
        eroded_mask = ~(maxpool((~mask).float()).bool())

    return eroded_mask, contour


# compute projected masks for the box into the image planes
def get_box_masks(
    T_WC_batch,
    box_extents,
    box_transform,
    fx,
    fy,
    cx,
    cy,
    w,
    h,
    get_contours=False,
    do_viz_check=False,
):
    # box vertices in world frame
    mins = -box_extents / 2.0
    maxs = box_extents / 2.0
    min_max = torch.stack((mins, maxs), dim=0)
    xx, yy, zz = torch.meshgrid(min_max[:, 0], min_max[:, 1], min_max[:, 2])
    box_verts = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    box_verts_W = lieF.SE3.transform(box_transform[:3, :], box_verts)

    # box vertices in camera frame
    box_verts_C = lieF.SE3.untransform(T_WC_batch[:, :3, :], box_verts_W[:, None])
    box_verts_C = box_verts_C.transpose(0, 1)

    # project box vertices into image plane
    coords = point_cloud_to_image_plane(
        box_verts_C.reshape(-1, 3), fx, fy, cx, cy, w, h
    )
    coords = coords.reshape(-1, 8, 2)

    # check all coords are in view
    if (
        (coords == 0).any()
        or (coords[:, :, 0] == (w - 1)).any()
        or (coords[:, :, 1] == (h - 1)).any()
    ):
        warnings.warn("Box is not fully in view")

    # find the closes vertex to the camera and use that to infer which edges of
    # the box are on the boundary of the mask
    origins = T_WC_batch[:, :3, -1]
    vertex_dists = torch.norm(origins[:, None, :] - box_verts_W[None, :, :], dim=-1)
    closest_vertex = vertex_dists.argmin(dim=-1)
    boundaries = {
        0: {"start": torch.tensor([1, 3, 2, 6, 4, 5]).flip(0)},
        7: {"start": torch.tensor([1, 3, 2, 6, 4, 5])},
        1: {"start": torch.tensor([0, 2, 3, 7, 5, 4])},
        6: {"start": torch.tensor([0, 2, 3, 7, 5, 4]).flip(0)},
        2: {"start": torch.tensor([0, 1, 3, 7, 6, 4]).flip(0)},
        5: {"start": torch.tensor([0, 1, 3, 7, 6, 4])},
        3: {"start": torch.tensor([0, 1, 5, 7, 6, 2]).flip(0)},
        4: {"start": torch.tensor([0, 1, 5, 7, 6, 2])},
    }  # possible boundaries
    for k in boundaries.keys():
        boundaries[k]["end"] = torch.cat(
            (
                boundaries[k]["start"].roll(-1),
                torch.full([6], k),
                boundaries[k]["start"],
            )
        )
        boundaries[k]["start"] = torch.cat(
            (boundaries[k]["start"], boundaries[k]["start"], torch.full([6], k))
        )
    n_bnds = 18

    boundary_starts = torch.stack(
        [
            coords[i][boundaries[v]["start"]]
            for i, v in enumerate(closest_vertex.cpu().numpy())
        ]
    )  # [batch_size, n_bnds, 2]
    boundary_ends = torch.stack(
        [
            coords[i][boundaries[v]["end"]]
            for i, v in enumerate(closest_vertex.cpu().numpy())
        ]
    )  # [batch_size, n_bnds, 2]

    # compute the vector for each edge
    vecs = (boundary_ends - boundary_starts)[:, :, None, :]  # [nbatch, n_bnds, 1, 2]

    # check which edges are on the boundary of the mask
    # if on the boundary all other points should be to the rhs of the line
    nbatch = T_WC_batch.shape[0]
    end_pts = boundary_ends[:, None, :7].repeat(
        1, n_bnds, 1, 1
    )  # [nbatch, n_bnds, 7, 2]
    vecs_to_pt = end_pts - boundary_starts[:, :, None, :]  # [nbatch, n_bnds, 7, 2]
    cross_prod_z = vecs[..., 0] * vecs_to_pt[..., 1] - vecs[..., 1] * vecs_to_pt[..., 0]
    rhs = cross_prod_z >= 0
    valid_edge = rhs.all(dim=-1)

    # boundary coordinates should be in clockwise order
    device = T_WC_batch.device
    x = torch.arange(w, device=device)
    y = torch.arange(h, device=device)
    yy, xx = torch.meshgrid(y, x)
    grid = torch.stack((xx, yy), dim=-1)
    grid = grid[None, ...].repeat(nbatch, 1, 1, 1)
    grid = grid.reshape(nbatch, h * w, 2)[:, None, :, :].repeat(1, n_bnds, 1, 1)
    vecs_to_pt = grid - boundary_starts[:, :, None, :]  # [nbatch, n_bnds, h*w, 2]

    # component of 3D cross product along in z direction
    cross_prod_z = vecs[..., 0] * vecs_to_pt[..., 1] - vecs[..., 1] * vecs_to_pt[..., 0]
    rhs = cross_prod_z > 0
    rhs = rhs.reshape(nbatch, n_bnds, h, w)
    rhs[~valid_edge] = True  # don't use masks from non-boundary edges
    box_mask = torch.all(rhs, dim=1)

    # note this part is slow and not batched, only for viz
    contour_mask = None
    if get_contours:
        contour_masks = []
        for j in range(nbatch):
            contour_mask = np.full([h, w, 3], 0, dtype=np.uint8)
            for i in range(n_bnds):
                if valid_edge[j, i]:
                    start = boundary_starts[j, i].cpu().numpy()
                    end = boundary_ends[j, i].cpu().numpy()
                    cv2.line(contour_mask, start, end, (255, 0, 0), 2)
            contour_mask = torch.tensor(contour_mask == 255)[None, ..., 0]
            contour_mask = contour_mask.to(box_mask.device)
            contour_masks.append(contour_mask)
        contour_mask = torch.cat(contour_masks, dim=0)

    if do_viz_check:
        # compare with skimage convex hull
        sk_mask = np.full([nbatch, h, w], False)
        for i, coord in enumerate(coords.cpu().numpy()):
            sk_mask[i, coord[:, 1], coord[:, 0]] = True
            sk_mask[i] = morphology.convex_hull_image(sk_mask[i])
        sk_mask = torch.tensor(sk_mask)
        diff = box_mask.cpu() != sk_mask

        for j in range(nbatch):
            viz = []
            w_viz, h_viz = box_mask.shape[2] // 1, box_mask.shape[1] // 1

            # render image of box
            img = render_box_trimesh(
                box_extents.cpu(),
                box_transform.cpu(),
                T_WC_batch[j].cpu(),
                fx,
                fy,
                w,
                h,
            )
            viz.append(cv2.resize(img, (w_viz, h_viz)))

            img_mask = viz_boolean_img(box_mask[j])
            line_viz = []
            for i in range(n_bnds):
                if valid_edge[j, i]:
                    img_line = viz_boolean_img(rhs[j, i])
                    start = boundary_starts[j, i].cpu().numpy()
                    end = boundary_ends[j, i].cpu().numpy()
                    cv2.circle(img_mask, start, 8, [0, 255, 0], 2)
                    cv2.circle(img_line, start, 8, [0, 255, 0], 2)
                    cv2.circle(img_line, end, 8, [0, 255, 0], 2)
                    new_end = end + (end - start) * 5
                    new_start = start - (end - start) * 5
                    cv2.line(img_mask, new_start, new_end, (255, 0, 0), 2)
                    line_viz.append(cv2.resize(img_line, (w_viz, h_viz)))
            sk_img = viz_boolean_img(sk_mask[j])
            diff_img = viz_boolean_img(diff[j])
            viz.append(cv2.resize(img_mask, (w_viz, h_viz)))
            viz.append(cv2.resize(sk_img, (w_viz, h_viz)))
            viz.append(cv2.resize(diff_img, (w_viz, h_viz)))
            viz.extend([np.zeros([h_viz, w_viz, 3])] * (len(line_viz) - len(viz)))
            im = np.vstack((np.hstack(viz), np.hstack(line_viz)))
            cv2.imshow("convex hull", im)
            cv2.waitKey(0)

    return box_mask, contour_mask


def draw_camera(camera, transform, color=(0.0, 1.0, 0.0, 0.8), marker_height=0.2):
    marker = trimesh.creation.camera_marker(camera, marker_height=marker_height)
    marker[0].apply_transform(transform)
    marker[1].apply_transform(transform)
    marker[1].colors = (color,) * len(marker[1].entities)

    return marker


def render_box_trimesh(box_extents, box_transform, T_WC, fx, fy, w, h):
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.primitives.Box(extents=box_extents, transform=box_transform)
    )
    xfov = np.rad2deg(2 * np.arctan(0.5 * w / fx))
    yfov = np.rad2deg(2 * np.arctan(0.5 * h / fy))
    res = np.array([w, h])
    camera = trimesh.scene.Camera(fov=np.array([xfov, yfov]), resolution=res)
    scene.camera = camera
    marker = draw_camera(camera, to_trimesh(T_WC), color=[255, 0, 0], marker_height=0.3)
    scene.add_geometry(marker[1])
    scene.camera_transform = T_WC

    d = scene.save_image(resolution=res)
    with trimesh.util.wrap_as_stream(d) as f:
        i = PIL.Image.open(f)
        img = np.array(i)

    return img[..., :3]


def get_all_rotations():
    """
    Taken from https://stackoverflow.com/a/70413438
    """
    import itertools
    from itertools import permutations

    def rotations(array):
        for x, y, z in permutations([0, 1, 2]):
            for sx, sy, sz in itertools.product([-1, 1], repeat=3):
                rotation_matrix = torch.zeros((3, 3))
                rotation_matrix[0, x] = sx
                rotation_matrix[1, y] = sy
                rotation_matrix[2, z] = sz
                if torch.det(rotation_matrix) == 1:
                    yield torch.matmul(rotation_matrix, array)

    return list(rotations(torch.eye(3)))


class RotExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        ctx.save_for_backward(w)

        theta = torch.norm(w, dim=1)
        e_w_x = torch.eye(3, device=w.device).unsqueeze(0).repeat(w.shape[0], 1, 1)

        if torch.any(theta != 0):
            mask = theta != 0
            n_valid_thetas = mask.sum()
            w_x = torch.zeros((n_valid_thetas, 3, 3), device=w.device)
            valid_w = w[mask]
            w_x[:, 0, 1] = -valid_w[:, 2]
            w_x[:, 1, 0] = valid_w[:, 2]
            w_x[:, 0, 2] = valid_w[:, 1]
            w_x[:, 2, 0] = -valid_w[:, 1]
            w_x[:, 1, 2] = -valid_w[:, 0]
            w_x[:, 2, 1] = valid_w[:, 0]

            valid_theta = theta[mask]
            e_w_x[mask] = (
                e_w_x[mask]
                + (torch.sin(valid_theta) / valid_theta)[:, None, None] * w_x
                + ((1 - torch.cos(valid_theta)) / (valid_theta * valid_theta))[
                    :, None, None
                ]
                * w_x
                @ w_x
            )

        return e_w_x

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        G1 = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            device=grad_output.device,
        ).view([1, -1])

        G2 = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            device=grad_output.device,
        ).view([1, -1])

        G3 = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device=grad_output.device,
        ).view([1, -1])

        grad_input_flat = grad_input.view([grad_input.shape[0], -1])

        p1 = (grad_input_flat * G1).sum(1, keepdim=True)
        p2 = (grad_input_flat * G2).sum(1, keepdim=True)
        p3 = (grad_input_flat * G3).sum(1, keepdim=True)

        grad_input = torch.cat((p1, p2, p3), 1)

        return grad_input
