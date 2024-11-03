# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Test script to visualize the SDF of a mesh, SDFViewer is taken from iSDF

import os

import git
import numpy as np

from neuralfeels.datasets import sdf_util
from neuralfeels.datasets.sdf_util import load_gt_mesh
from neuralfeels.viz import sdf_viewer

root = git.Repo(".", search_parent_directories=True).working_tree_dir


def main():
    mesh_path = os.path.join(
        root, "data/assets/gt_models/ycb/contactdb_rubber_duck.urdf"
    )
    mesh, _ = load_gt_mesh(mesh_path)
    sdf, transform = sdf_util.sdf_from_mesh(
        mesh=mesh, voxel_size=5e-4, extend_factor=0.1, origin_voxel=np.zeros(3)
    )
    sdf_viewer.SDFViewer(
        mesh=mesh,
        sdf_grid=sdf,
        sdf_range=None,
        grid2world=transform,
        surface_cutoff=0.001,
        colormap=True,
    )


if __name__ == "__main__":
    main()
