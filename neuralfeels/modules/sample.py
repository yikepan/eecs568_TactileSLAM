# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Camera ray sampling functions

import torch

from neuralfeels.geometry import transform


def sample_pixels(n_rays, n_frames, h, w, device, mask=None):
    # n_rays: rays per frame
    # mask: boolean mask of valid pixels

    if mask is not None:
        assert mask.shape == torch.Size([n_frames, h, w]), "Mask unexpected shape"

        # create meshgrid with all indices for each frame
        grid_h, grid_w = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )  # [h, w]
        grid_h = grid_h.unsqueeze(0).repeat(n_frames, 1, 1)  # [n_frames, h, w]
        grid_w = grid_w.unsqueeze(0).repeat(n_frames, 1, 1)  # [n_frames, h, w]
        grid_b = (
            torch.ones((n_frames, h, w), device=device, dtype=int)
            * torch.arange(n_frames, device=device, dtype=int)[:, None, None]
        )  # [n_frames, h, w] indices

        grid_h_along_batch = [g_h[m] for g_h, m in zip(grid_h, mask)]
        grid_w_along_batch = [g_w[m] for g_w, m in zip(grid_w, mask)]
        grid_b_along_batch = [g_b[m] for g_b, m in zip(grid_b, mask)]
        len_b_along_batch = torch.tensor([len(g_b) for g_b in grid_b_along_batch]).to(
            device=device
        )

        # Delete from list if no True values in mask
        # to avoid depth_batch with no depth values at all (happens often in tactile)
        # indices of frames that are all False in mask
        nan_depths = (torch.nansum(mask, dim=(1, 2)) == 0).nonzero().flatten()
        # indices of frames that have at least one True in mask
        valid_depths = (torch.nansum(mask, dim=(1, 2)) != 0).nonzero().flatten()
        for nan_idx in torch.flip(nan_depths, dims=[0]):
            del (
                grid_b_along_batch[nan_idx],
                grid_h_along_batch[nan_idx],
                grid_w_along_batch[nan_idx],
            )
        len_b_along_batch = len_b_along_batch[valid_depths]

        indices = torch.vstack(
            [
                torch.randint(0, l, (n_rays,), device=device)
                for l in len_b_along_batch[len_b_along_batch != 0]
            ]
        )
        assert indices.shape[1] == n_rays, "Wrong number of samples"

        grid_b = torch.hstack([g[idx] for g, idx in zip(grid_b_along_batch, indices)])
        grid_h = torch.hstack([g[idx] for g, idx in zip(grid_h_along_batch, indices)])
        grid_w = torch.hstack([g[idx] for g, idx in zip(grid_w_along_batch, indices)])

        assert torch.all(
            torch.unique(grid_b, return_counts=True)[1] == n_rays
        ), "Not all poses sampled equally"
        assert mask[grid_b, grid_h, grid_w].all(), "Sample outside mask"
        return grid_b, grid_h, grid_w

    else:
        return random_sample(n_rays, n_frames, h, w, device)


# randomly sample over all pixels
def random_sample(n_rays, n_frames, h, w, device):
    total_rays = n_rays * n_frames
    indices_h = torch.randint(0, h, (total_rays,), device=device)
    indices_w = torch.randint(0, w, (total_rays,), device=device)

    indices_b = torch.arange(n_frames, device=device)
    indices_b = indices_b.repeat_interleave(n_rays)
    return indices_b, indices_h, indices_w


def get_batch_data(
    T_WC_batch,
    dirs_C,
    indices_b,
    indices_h,
    indices_w,
    depth_batch=None,
    norm_batch=None,
):
    """
    Get depth, ray direction and pose for the sampled pixels.
    """
    T_WC_sample = T_WC_batch[indices_b]
    dirs_C_sample = dirs_C[0, indices_h, indices_w, :].view(
        -1, 3
    )  # sampled ray directions

    depth_sample = None
    if depth_batch is not None:  # retrieve sampled depths
        depth_sample = depth_batch[indices_b, indices_h, indices_w].view(
            -1
        )  # n_rays * n_frames elements

    norm_sample = None
    if norm_batch is not None:  # retrieve sampled normals
        norm_sample = norm_batch[indices_b, indices_h, indices_w, :].view(-1, 3)

    return (
        dirs_C_sample,
        T_WC_sample,
        depth_sample,
        norm_sample,
    )


def stratified_sample(
    min_depth,
    max_depth,
    n_rays,
    device,
    n_stratified_samples,
    bin_length=None,
):
    """
    Random samples between min and max depth
    One sample from within each bin.

    If n_stratified_samples is passed then use fixed number of bins,
    else if bin_length is passed use fixed bin size.
    """

    if n_stratified_samples is not None:  # fixed number of bins
        n_bins = n_stratified_samples
        if isinstance(max_depth, torch.Tensor):
            sample_range = (max_depth - min_depth)[:, None]
            bin_limits = torch.linspace(0, 1, n_bins + 1, device=device)[None, :]
            bin_limits = bin_limits.repeat(n_rays, 1) * sample_range
            if isinstance(min_depth, torch.Tensor):
                bin_limits = bin_limits + min_depth[:, None]
            else:
                bin_limits = bin_limits + min_depth
            bin_length = sample_range / (n_bins)
        else:
            bin_limits = torch.linspace(
                min_depth,
                max_depth,
                n_bins + 1,
                device=device,
            )[None, :]
            bin_length = (max_depth - min_depth) / (n_bins)

    elif bin_length is not None:  # fixed size of bins
        bin_limits = torch.arange(
            min_depth,
            max_depth,
            bin_length,
            device=device,
        )[None, :]
        n_bins = bin_limits.size(1) - 1

    increments = torch.rand(n_rays, n_bins, device=device) * bin_length
    # increments = 0.5 * torch.ones(n_rays, n_bins, device=device) * bin_length
    lower_limits = bin_limits[..., :-1]
    z_vals = lower_limits + increments

    z_vals, _ = torch.sort(z_vals, dim=1)

    return z_vals


def sample_along_rays(
    T_WC,
    dirs_C,
    n_stratified_samples,
    n_surf_samples=0,
    surf_samples_offset=None,
    min_depth=None,
    max_depth=None,
    box_extents=None,
    box_transform=None,
    gt_depth=None,
    grad=False,
    local=False,
):
    method1 = min_depth is not None and max_depth is not None
    method2 = box_transform is not None and box_extents is not None
    assert method1 or method2, "Need either min-max or box"

    n_rays = dirs_C.shape[0]
    device = T_WC.device
    if method1:
        if not isinstance(min_depth, torch.Tensor):
            min_depth = torch.full([n_rays], min_depth).to(device)
        if not isinstance(max_depth, torch.Tensor):
            max_depth = torch.full([n_rays], max_depth).to(device)
        # ensure min_depth < max_depth even if depths are negative
        indices = max_depth < min_depth
        min_depth[indices], max_depth[indices] = (
            max_depth[indices],
            min_depth[indices],
        )

    with torch.set_grad_enabled(grad):
        # rays in world coordinate
        # dirs_C are vectors in local coordinates (TODO: first visualize this)
        origins, dirs_W = transform.origin_dirs_W(T_WC, dirs_C)

        origins = origins.view(-1, 3)
        dirs_W = dirs_W.view(-1, 3)

        valid_ray = None
        if method2:
            min_depth_box, max_depth_box = transform.ray_box_intersection(
                origins,
                dirs_W,
                box_extents,
                box_transform,
            )
            # remove rays that don't intersect the box
            valid_ray = ~torch.isnan(min_depth_box)
            min_depth_box = min_depth_box[valid_ray]
            max_depth_box = max_depth_box[valid_ray]
            # if only single origin is passed rather than origins per ray
            if origins.shape[0] > 1:
                origins = origins[valid_ray]
            dirs_W = dirs_W[valid_ray]
            n_rays = dirs_W.shape[0]
            gt_depth = gt_depth[valid_ray] if gt_depth is not None else None

            # ensure min_depth < max_depth even if depths are negative
            indices = max_depth_box < min_depth_box
            min_depth_box[indices], max_depth_box[indices] = (
                max_depth_box[indices],
                min_depth_box[indices],
            )
            if method1:
                # use both methods to select tightest bounds
                min_depth = torch.max(min_depth, min_depth_box)
                max_depth = torch.min(max_depth, max_depth_box)
            else:
                min_depth = min_depth_box
                max_depth = max_depth_box

        # stratified sampling along rays # [total_n_rays, n_stratified_samples] between min_depth and max_depth
        z_vals = stratified_sample(
            min_depth,
            max_depth,
            n_rays,
            T_WC.device,
            n_stratified_samples,
            bin_length=None,
        )

        # if gt_depth is given, first sample at surface then around surface
        if gt_depth is not None and n_surf_samples > 0:
            assert surf_samples_offset is not None, "Need surf_samples_offset"
            surface_z_vals = gt_depth
            offsets = torch.normal(
                torch.zeros(gt_depth.shape[0], n_surf_samples - 1), surf_samples_offset
            ).to(z_vals.device)
            near_surf_z_vals = gt_depth[:, None] + offsets
            near_surf_z_vals = torch.clamp(
                near_surf_z_vals, min_depth[:, None], max_depth[:, None]
            )

            # 1 sample of surface, n_surf_samples around surface, n_stratified_samples along [min, max]
            z_vals = torch.cat(
                (
                    surface_z_vals[:, None],
                    near_surf_z_vals,
                    z_vals,
                ),
                dim=1,
            )

        # point cloud of 3d sample locations
        pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
    if local:
        return None, z_vals, valid_ray
    else:
        return pc, z_vals, valid_ray
