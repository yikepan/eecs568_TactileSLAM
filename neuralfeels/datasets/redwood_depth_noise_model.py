# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified from Habitat-Sim (https://aihabitat.org/docs/habitat-sim/habitat_sim.sensors.noise_models.RedwoodDepthNoiseModel.html) and based on the
# Redwood Depth Noise Model (http://redwood-data.org/indoor/data/simdepth.py) from
# choi2015robust (https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Choi_Robust_Reconstruction_of_2015_CVPR_paper.pdf)

import numba
import numpy as np

try:
    import torch
    from torch import Tensor
except ImportError:
    torch = None


# Read about the noise model here: http://www.alexteichman.com/octo/clams/
# Original source code: http://redwood-data.org/indoor/data/simdepth.py
@numba.jit(nopython=True, fastmath=True)
def _undistort(x, y, z, model):
    i2 = int((z + 1) / 2)
    i1 = int(i2 - 1)
    a = (z - (i1 * 2.0 + 1.0)) / 2.0
    x = x // 8
    y = y // 6
    f = (1.0 - a) * model[y, x, min(max(i1, 0), 4)] + a * model[y, x, min(i2, 4)]

    if f < 1e-5:
        return 0.0

    return z / f


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _simulate(gt_depth, model, noise_multiplier):
    noisy_depth = np.empty_like(gt_depth)

    H, W = gt_depth.shape
    ymax, xmax = H - 1.0, W - 1.0

    rand_nums = np.random.randn(H, W, 3).astype(np.float32)

    # Parallelize just the outer loop.  This doesn't change the speed
    # noticably but reduces CPU usage compared to two parallel loops
    for j in numba.prange(H):
        for i in range(W):
            y = int(
                min(max(j + rand_nums[j, i, 0] * 0.25 * noise_multiplier, 0.0), ymax)
                + 0.5
            )
            x = int(
                min(max(i + rand_nums[j, i, 1] * 0.25 * noise_multiplier, 0.0), xmax)
                + 0.5
            )

            # Downsample
            d = gt_depth[y - y % 2, x - x % 2]
            # If the depth is greater than 10, the sensor will just return 0
            if d >= 10.0:
                noisy_depth[j, i] = 0.0
            else:
                # Distort
                # The noise model was originally made for a 640x480 sensor,
                # so re-map our arbitrarily sized sensor to that size!
                undistorted_d = _undistort(
                    int(x / xmax * 639.0 + 0.5), int(y / ymax * 479.0 + 0.5), d, model
                )

                if undistorted_d == 0.0:
                    noisy_depth[j, i] = 0.0
                else:
                    denom = round(
                        (
                            35.130 / undistorted_d
                            + rand_nums[j, i, 2] * 0.027778 * noise_multiplier
                        )
                        * 8.0
                    )
                    if denom <= 1e-5:
                        noisy_depth[j, i] = 0.0
                    else:
                        noisy_depth[j, i] = 35.130 * 8.0 / denom

    return noisy_depth
