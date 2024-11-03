# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/antocad/FocusOnDepth and https://github.com/isl-org/DPT

# Wrapper for tactile depth estimation model

import collections
import os.path as osp

import cv2
import numpy as np
import tacto
import torch
from hydra import compose

from neuralfeels.contrib.tactile_transformer.touch_vit import TouchVIT

dname = osp.dirname(osp.abspath(__file__))


class TactileDepth:
    def __init__(self, depth_mode, real=False, device="cuda"):
        super(TactileDepth, self).__init__()

        cfg = compose(config_name=f"main/touch_depth/{depth_mode}").main.touch_depth

        cfg.weights = "dpt_real" if real else "dpt_sim"

        if depth_mode == "gt":
            self.model = None
            return
        if depth_mode == "vit":
            # print("Loading ViT depth model----")
            self.model = TouchVIT(cfg=cfg)
        else:
            raise NotImplementedError(f"Mode not implemented: {cfg.mode}")
        # print("done")
        self.device = device

        settings_config = cfg.settings.real if real else cfg.settings.sim
        self.b, self.r, self.clip = (
            settings_config.border,
            settings_config.ratio,
            settings_config.clip,
        )

        self.bg_id = settings_config.bg_id
        self.blend_sz = settings_config.blend_sz
        self.heightmap_window = collections.deque([])

        # background templates for heightmap2mask
        self.bg_template = {}

    def image2heightmap(self, image: np.ndarray, sensor_name: str = "digit_0"):
        if sensor_name not in self.bg_template:
            if self.bg_id is None:
                print(
                    f"{sensor_name} not in background images, generating new background template using first frame"
                )
                self.bg_template[sensor_name] = self.model.image2heightmap(image)
            else:
                print(
                    f"{sensor_name} not in background images, generating new background template from bg_id {self.bg_id}"
                )
                self.bg_template[sensor_name] = self.model.image2heightmap(
                    cv2.imread(tacto.get_background_image_path(self.bg_id))
                )
            self.bg_template[sensor_name] = self.bg_template[sensor_name].to(
                dtype=float, device=self.device
            )
        heightmap = self.model.image2heightmap(image)
        return self.blend_heightmaps(heightmap)

    def heightmap2mask(
        self, heightmap: torch.tensor, sensor_name: str = "digit_0"
    ) -> torch.Tensor:
        """Thresholds heightmap to return binary contact mask

        Args:
            heightmap: single tactile image

        Returns:
            padded_contact_mask: contact mask [True: is_contact, False: no_contact]

        """

        heightmap = heightmap.squeeze().to(self.device)
        bg_template = self.bg_template[sensor_name]
        # scale bg_template to match heightmap if different size
        if bg_template.shape != heightmap.shape:
            bg_template = torch.nn.functional.interpolate(
                bg_template[None, None, :, :], heightmap.shape[-2:], mode="bilinear"
            ).squeeze()

        init_height = bg_template
        if self.b:
            heightmap = heightmap[self.b : -self.b, self.b : -self.b]
            init_height = init_height[self.b : -self.b, self.b : -self.b]
        diff_heights = heightmap - init_height
        diff_heights[diff_heights < self.clip] = 0
        threshold = torch.quantile(diff_heights, 0.9) * self.r
        contact_mask = diff_heights > threshold
        padded_contact_mask = torch.zeros_like(bg_template, dtype=bool)

        if self.b:
            padded_contact_mask[self.b : -self.b, self.b : -self.b] = contact_mask
        else:
            padded_contact_mask = contact_mask
        return padded_contact_mask

    def blend_heightmaps(self, heightmap: torch.Tensor) -> torch.Tensor:
        """Exponentially weighted heightmap blending.

        Args:
            heightmap: input heightmap

        Returns:
            blended_heightmap: output heightmap blended over self.heightmap_window

        """

        if not self.blend_sz:
            return heightmap

        if len(self.heightmap_window) >= self.blend_sz:
            self.heightmap_window.popleft()

        self.heightmap_window.append(heightmap)
        n = len(self.heightmap_window)

        weights = torch.tensor(
            [x / n for x in range(1, n + 1)], device=heightmap.device
        )  # exponentially weighted time series costs

        weights = torch.exp(weights) / torch.sum(torch.exp(weights))

        all_heightmaps = torch.stack(list(self.heightmap_window))
        blended_heightmap = torch.sum(
            (all_heightmaps * weights[:, None, None]) / weights.sum(), dim=0
        )  # weighted average

        # view_subplots([heightmap, blended_heightmap], [["heightmap", "blended_heightmap"]])
        return blended_heightmap
