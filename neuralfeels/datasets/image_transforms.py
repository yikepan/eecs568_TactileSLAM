# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np


class BGRtoRGB(object):
    """bgr format to rgb"""

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class DepthTransform(object):
    """
    Transforms tactile depth from the gel coordinate system to the camera coordinate system
    The camera is placed 0.022 m behind the gel surface
    """

    def __init__(self, cam_dist):
        self.cam_dist = cam_dist

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        depth += self.cam_dist
        depth[depth == self.cam_dist] = np.nan
        return depth.astype(np.float32)


class DepthScale(object):
    """scale depth to meters"""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        return depth * self.scale
