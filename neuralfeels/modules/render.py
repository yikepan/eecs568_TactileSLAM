# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# SDF depth rendering, based on iSDF: https://github.com/facebookresearch/iSDF

import torch

from neuralfeels.geometry import transform
from neuralfeels.modules.model import gradient


def sdf_render_depth(z_vals, sdf, t):
    """
    Basic method for rendering depth from SDF using samples along a ray.
    Assumes z_vals are ordered small -> large.
    Assumes sdf are ordered from expected small -> expected large
    """
    # assert (z_vals[0].sort()[1].cpu() == torch.arange(len(z_vals[0]))).all()

    # z_vals are sorted from gel to camera
    # sdfs sorted negative to positive (inside to outside)
    n = sdf.size(1)  # n_sample per ray

    inside = sdf < 0  # sdf indices outside object
    ixs = torch.arange(0, n, 1, device=sdf.device)  # ascending order [0, n]
    mul = inside * ixs  # keep only inside points
    max_ix = mul.argmax(dim=1)  # smallest -ve value before intersection

    arange = torch.arange(z_vals.size(0), device=sdf.device)  # [0 - n_pixels]
    depths = (
        z_vals[arange, max_ix] + sdf[arange, max_ix] * t
    )  # sdf will always be +ve, z_vals always -ve

    # if no zero crossing found
    depths[max_ix == 0] = torch.nan
    # print(torch.sum(~torch.isnan(depths)) / len(depths.view(-1)))
    return depths


# Compute surface normals in the camera frame
def render_normals(T_WC, render_depth, sdf_map, dirs_C):
    origins, dirs_W = transform.origin_dirs_W(T_WC, dirs_C)
    origins = origins.view(-1, 3)
    dirs_W = dirs_W.view(-1, 3)

    pc = origins + (dirs_W * (render_depth.flatten()[:, None]))
    pc.requires_grad_()
    sdf = sdf_map(pc.unsqueeze(0))
    sdf_grad = gradient(pc, sdf)

    surface_normals_W = -sdf_grad / (sdf_grad.norm(dim=1, keepdim=True) + 1e-6)
    R_CW = T_WC[:, :3, :3].inverse()
    surface_normals_C = (R_CW * surface_normals_W[..., None, :]).sum(dim=-1)

    surface_normals_C = surface_normals_C.view(
        render_depth.shape[0], render_depth.shape[1], 3
    )
    return surface_normals_C


def render_weighted(weights, vals, dim=-1, normalise=False):
    """
    General rendering function using weighted sum.
    """
    weighted_vals = weights * vals
    render = weighted_vals.sum(dim=dim)
    if normalise:
        n_samples = weights.size(dim)
        render = render / n_samples

    return render
