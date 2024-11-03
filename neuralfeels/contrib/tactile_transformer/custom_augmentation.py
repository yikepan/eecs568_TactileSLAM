# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/antocad/FocusOnDepth and https://github.com/isl-org/DPT

import numpy as np
import torch


class ToMask(object):
    """
    Convert a 3 channel RGB image into a 1 channel segmentation mask
    """

    def __init__(self, palette_dictionnary):
        self.nb_classes = len(palette_dictionnary)
        self.palette_dictionnary = palette_dictionnary

    def __call__(self, pil_image):
        # avoid taking the alpha channel
        image_array = np.array(pil_image)
        # get only one channel for the output
        output_array = np.zeros(image_array.shape, dtype="int")

        for label in self.palette_dictionnary.keys():
            rgb_color = self.palette_dictionnary[label]["color"]
            mask = image_array == rgb_color
            output_array[mask] = int(label)

        output_array = torch.from_numpy(output_array).long()
        return output_array
