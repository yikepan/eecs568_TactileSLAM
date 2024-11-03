# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/Jianghanxiao/Helper3D

import numpy as np


class Link:
    def __init__(self, link_name):
        self.link_name = link_name
        self.color = [0.0, 0.0, 0.0]
        # Naming rule: concaten tag name as the variable name, and attribute name as the key
        self.visuals = []

    def hasVisual(self):
        if len(self.visuals) == 0:
            return False
        return True

    def addVisual(self, visual_name=None):
        self.visuals.append(Visual(visual_name))

    def setVisualMeshScale(self, scale):
        current_visual = len(self.visuals) - 1
        self.visuals[current_visual].geometry_mesh["scale"] = np.array(scale)

    def setVisualOriginXyz(self, xyz):
        current_visual = len(self.visuals) - 1
        self.visuals[current_visual].origin["xyz"] = np.array(xyz)

    def setVisualOriginRpy(self, rpy):
        current_visual = len(self.visuals) - 1
        self.visuals[current_visual].origin["rpy"] = np.array(rpy)

    def setVisualGeometryMeshFilename(self, filename):
        current_visual = len(self.visuals) - 1
        self.visuals[current_visual].geometry_mesh["filename"] = filename

    def __repr__(self):
        output = {}
        output["name"] = self.link_name
        output["visual"] = self.visuals
        return str(output)


class Visual:
    def __init__(self, visual_name=None):
        self.visual_name = visual_name
        self.origin = {"xyz": np.array([0, 0, 0]), "rpy": np.array([0, 0, 0])}
        self.geometry_mesh = {"filename": None, "scale": np.array([1.0, 1.0, 1.0])}

    def __repr__(self):
        output = {}
        output["origin"] = self.origin
        output["mesh"] = self.geometry_mesh["filename"]
        return str(output)
