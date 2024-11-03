# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/Jianghanxiao/Helper3D


import copy
import re

import open3d as o3d


class MeshNode:
    def __init__(self):
        self.mesh = None

    def addMesh(self, mesh):
        if self.mesh == None:
            self.mesh = mesh
        else:
            self.mesh += mesh

    def addMeshFile(self, mesh_file, color):
        # Read the mesh from obj file
        mesh_file = re.sub("allegro/allegro", "allegro", mesh_file)
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.paint_uniform_color(color)
        self.addMesh(mesh)

    def getMesh(self, worldMatrix):
        if self.mesh == None:
            return None
        new_mesh = copy.deepcopy(self.mesh)
        new_mesh.transform(worldMatrix)
        return new_mesh
