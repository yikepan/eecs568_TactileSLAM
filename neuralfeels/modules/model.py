# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Instant-NGP SDF model and custom interp/jacobian computation

from itertools import product
from typing import List, Optional

import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
import torchlie.functional as lieF
from omegaconf import DictConfig
from torch.autograd import grad

from neuralfeels.datasets import sdf_util
from neuralfeels.geometry.transform import transform_points, transform_points_batch


@torch.jit.script
def gradient(inputs: torch.Tensor, outputs: torch.Tensor):
    d_points: List[Optional[torch.Tensor]] = [
        torch.ones_like(outputs, device=outputs.device)
    ]
    points_grad = grad(
        outputs=[
            outputs,
        ],
        inputs=[
            inputs,
        ],
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
    )[0]
    return points_grad


class SDFModel(nn.Module):
    """
    Parent class to handle transformations and pose updates to object frames
    when usin SDFNetwork or SDFInterp.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.sdf_map_pose = torch.eye(4, device=device).unsqueeze(0)
        self.scale = 1.0
        self.obs_frames = None

    def update_pose_and_scale(self, pose, scale=None):
        """
        pose: [N, 4, 4] pose batch of object in world frame
        scale is dummy in this function
        """
        self.sdf_map_pose = pose
        if scale is not None:
            self.scale = scale

    def update_frames(self, obs_frames):
        """
        obs_frames: [n_rays] pose_ids of sensor observations, to know which pose to transform to
        """
        self.obs_frames = obs_frames

    def transform(self, x, return_jacobian=False):
        """
        Transforms sample points x to the object frames
        x: [n_rays, n_samples, 3] or [n_pts, 3]
        """
        # select corresponding pose for each ray
        T_obj = self.sdf_map_pose
        if self.obs_frames is not None:
            T_obj = self.sdf_map_pose[self.obs_frames]  # [n_rays, 4, 4]

        if x.ndim > 2:
            x = x.transpose(0, 1)  # [n_samples, n_rays, 3]
        if return_jacobian:
            jacs, x_obj = lieF.SE3.juntransform(T_obj[:, :3, :], x)
            jac_pose = jacs[0]
            if x.ndim > 2:
                x_obj = x_obj.transpose(0, 1).reshape(-1, 3)  # [n_rays*n_samples, 3]
                jac_pose = jac_pose.transpose(0, 1).reshape(
                    -1, 3, 6
                )  # [n_rays*n_samples, 3, 6]
            return x_obj, jac_pose
        else:
            # x_obj = lieF.SE3.untransform(T_obj[:, :3, :], x)
            # if x.ndim > 2:
            #     x_obj = x_obj.transpose(0, 1).reshape(-1, 3)  # [n_rays*n_samples, 3]

            if x.ndim > 2:
                x = x.transpose(0, 1)
            T_obj_inv = T_obj.inverse().to(x.device)
            if T_obj.shape[0] > 1:
                frames = torch.repeat_interleave(self.obs_frames, x.shape[1], dim=0)
                batch_poses = self.sdf_map_pose[frames]
                T_obj_inv = batch_poses.inverse().to(x.device)
                x_obj = transform_points_batch(x.reshape(-1, 3), T_obj_inv).squeeze()
            else:
                x_obj = transform_points(x.reshape(-1, 3), T_obj_inv[0]).squeeze()

            return x_obj


class SDFNetwork(SDFModel):
    def __init__(
        self,
        embed_config: DictConfig,
        num_layers=3,
        skips=[],
        hidden_dim=64,
        clip_sdf=None,
        scale_output=1.0,
        device=torch.device("cuda"),
    ):
        super().__init__(device)
        self.scale_output = scale_output
        self.device = device
        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf
        self.scale = 1.0
        assert (
            self.skips == []
        ), "TCNN does not support concatenating inside, please use skips=[]."

        """ hash encoding"""
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": embed_config.n_levels,  # 16
            "n_features_per_level": embed_config.n_features_per_level,  # 2
            "log2_hashmap_size": embed_config.log2_hashmap_size,  # 24
            "base_resolution": embed_config.base_resolution,  # 16
            "per_level_scale": embed_config.per_level_scale,  # 1.3819
        }

        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": hidden_dim,
            "n_hidden_layers": num_layers - 1,
        }

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1,
            encoding_config=encoding_config,
            network_config=network_config,
        )

    def forward(self, x, noise_std=None):
        # x: [B, 3]

        x = self.transform(x)
        x = x / self.scale
        x = torch.clamp(x, min=-1, max=1)  # TODO: investigate this

        x = (x + 1) / 2  # to [0, 1]
        h = self.model(x)

        if noise_std is not None:
            noise = torch.randn(h.shape, device=h.device) * noise_std
            h = h + noise

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        h = h * self.scale_output
        return h


class RegularGridInterpolator(nn.Module):
    """
    Interpolation grid for SDF pose tracking
    https://github.com/sbarratt/torch_interpolations/blob/master/torch_interpolations/multilinear.py
    """

    def __init__(self, points, values):
        super().__init__()
        self.points = points
        self.values = values

        assert isinstance(self.points, tuple) or isinstance(self.points, list)
        assert isinstance(self.values, torch.Tensor)

        self.ms = list(self.values.shape)
        self.n = len(self.points)

        assert len(self.ms) == self.n

        for i, p in enumerate(self.points):
            assert isinstance(p, torch.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def forward(self, points_to_interp, return_jacobian=False):
        assert self.points is not None
        assert self.values is not None

        assert len(points_to_interp) == len(self.points)
        K = points_to_interp[0].shape[0]
        for x in points_to_interp:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []

        # output_tensor = mask * custom_operation_if_mask_is_true(input_tensor) + (1 - mask) * input_tensor
        VMAP_COMPATIBLE = True
        for p, x in zip(self.points, points_to_interp):
            idx_right = torch.bucketize(x.contiguous(), p)

            if not VMAP_COMPATIBLE:
                idx_right[idx_right >= p.shape[0]] = (
                    p.shape[0] - 1
                )  # VMAP doesn't support batching operators with dynamic shape
            else:
                mask = (idx_right >= p.shape[0]).to(dtype=idx_right.dtype)
                idx_right = mask * (p.shape[0] - 1) + (1 - mask) * idx_right

            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x
            if not VMAP_COMPATIBLE:
                dist_left[dist_left < 0] = 0.0
                dist_right[dist_right < 0] = 0.0
            else:
                dist_left = dist_left * (dist_left >= 0).to(dtype=dist_left.dtype)
                dist_right = dist_right * (dist_right >= 0).to(dtype=dist_right.dtype)
            both_zero = (dist_left == 0) & (dist_right == 0)
            if not VMAP_COMPATIBLE:
                dist_left[both_zero] = dist_right[both_zero] = 1.0
            else:
                both_zero = both_zero.to(dtype=dist_left.dtype)
                dist_left = (
                    dist_left * (1 - both_zero).to(dtype=dist_left.dtype) + both_zero
                ).to(dtype=dist_left.dtype)
                dist_right = (
                    dist_right * (1 - both_zero).to(dtype=dist_right.dtype) + both_zero
                ).to(dtype=dist_right.dtype)

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        if return_jacobian:
            device = x.device
            jac = torch.zeros(K, self.n, device=device)
            # jacobians for left and right distances
            dist_jac = torch.tensor([1.0, -1.0])

        numerator = 0.0
        for indexer in product([0, 1], repeat=self.n):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[as_s] * torch.prod(torch.stack(bs_s), dim=0)

            if return_jacobian:
                # distance jacobian of bs_s through chain rule
                chain_term = torch.tensor([dist_jac[1 - onoff] for onoff in indexer])
                chain_term = chain_term.to(device=device)
                # jacobian of torch.prod operation
                prod = torch.prod(torch.stack(bs_s), dim=0, keepdim=True)
                prod = prod.repeat(self.n, 1, 1)
                prod_jac = prod / (torch.stack(bs_s) + 1e-6)  # eps for division error
                # jacobian of numerator += operation
                jac += (
                    self.values[as_s]
                    * prod_jac.transpose(0, 1).squeeze()
                    * chain_term[None]
                )
        denominator = torch.prod(torch.stack(overalls), dim=0)

        if return_jacobian:
            jac /= denominator
            return numerator / denominator, jac[:, None, :]

        return numerator / denominator

    def jacobian_numerical(self, points_to_interp):
        """
        Numerical jacobian with delta = grid size
        as in NeuroAngelo https://research.nvidia.com/labs/dir/neuralangelo/
        """
        # assert torch.any(
        #     torch.stack(
        #         [
        #             points_to_interp[:, 0].min() < self.points[0][0],
        #             points_to_interp[:, 0].max() > self.points[0][-1],
        #             points_to_interp[:, 1].min() < self.points[1][0],
        #             points_to_interp[:, 1].max() > self.points[1][-1],
        #             points_to_interp[:, 2].min() < self.points[2][0],
        #             points_to_interp[:, 2].max() > self.points[2][-1],
        #         ]
        #     )
        # ), "Point samples out of grid bounds"
        pts = points_to_interp[:, None, None, :].repeat(1, 3, 2, 1)
        offset = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            ],
            device=pts.device,
        )
        grid_size = torch.tensor(
            [
                self.points[0][1] - self.points[0][0],
                self.points[1][1] - self.points[1][0],
                self.points[2][1] - self.points[2][0],
            ],
            device=pts.device,
        )
        offset *= grid_size

        pts += offset[None, ...]
        shape_pts = pts.shape
        pts = pts.reshape(-1, 3)
        (xx, yy, zz) = torch.tensor_split(pts, 3, dim=1)
        sdf_offset = self.forward((xx, yy, zz))
        sdf_offset = sdf_offset.reshape(shape_pts[:-1])

        sdf_grad = (sdf_offset[..., 0] - sdf_offset[..., 1]) / (2 * grid_size[None, :])
        return sdf_grad[:, None, :]


class SDFInterp(SDFModel):
    def __init__(self, sdf_grid, sdf_transform, device):
        super().__init__(device)
        # Load ground-truth sdf grid and transform
        # create interpolation grid
        x_grid, y_grid, z_grid = sdf_util.get_grid_pts(sdf_grid.shape, sdf_transform)
        x_grid, y_grid, z_grid = (
            torch.tensor(x_grid).to(device).float(),
            torch.tensor(y_grid).to(device).float(),
            torch.tensor(z_grid).to(device).float(),
        )
        self.sdf_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), sdf_grid)
        self.x_grid, self.y_grid, self.z_grid = x_grid, y_grid, z_grid

    def updateSDFGrid(self, sdf_grid):
        """
        Update the trilinear grid with the latest SDF grid values
        """
        sdf_grid = sdf_grid.clone().detach()
        self.sdf_interp = RegularGridInterpolator(
            (self.x_grid, self.y_grid, self.z_grid), sdf_grid
        )

    def forward(self, x):
        """
        Takes in sample points x: [n_rays, n_samples, 3] and returns the interpolated SDF values
        """
        x = self.transform(x)  # transform points to obj frame [n_rays*n_samples, 3]
        (xx, yy, zz) = torch.tensor_split(x, 3, dim=1)
        # interpolate over latest SDF grid to get [n_rays*n_samples, 1]
        h = self.sdf_interp((xx, yy, zz))
        h = h.squeeze()
        return h  # [n_rays*n_samples]

    def jacobian(self, x, numerical_jacobian=False):
        x, jac_pose = self.transform(x, return_jacobian=True)  # jac_pose: [n_pts, 3, 6]
        (xx, yy, zz) = torch.tensor_split(x, 3, dim=1)

        if numerical_jacobian:
            jac_trilin = self.sdf_interp.jacobian_numerical(x)
            h = self.sdf_interp((xx, yy, zz))
            sdf = h.squeeze()
        else:  # analytic jacobian
            sdf, jac_trilin = self.sdf_interp(
                (xx, yy, zz), return_jacobian=True
            )  # jac_trilin: [n_pts, 1, 3]

        jac = torch.bmm(jac_trilin, jac_pose)  # [n_pts, 1, 6]
        return sdf, jac
