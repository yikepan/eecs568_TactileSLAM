# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import git
import numpy as np
import open3d as o3d

from neuralfeels.contrib.urdf import SceneGraph, URDFParser, URDFTree

root = git.Repo(".", search_parent_directories=True).working_tree_dir


if __name__ == "__main__":
    URDF_file = os.path.join(root, "data/assets/allegro/allegro_digit_left_ball.urdf")
    # Parse the URDF file
    parser = URDFParser(URDF_file)
    parser.parse()
    # Construct the URDF tree
    links = parser.links
    joints = parser.joints
    tree = URDFTree(links, joints)
    # Construct the scene graph
    init_pose = np.array(
        [
            0.0627,
            1.2923,
            0.3383,
            0.1088,
            0.0724,
            1.1983,
            0.1551,
            0.1499,
            0.1343,
            1.1736,
            0.5355,
            0.2164,
            1.1202,
            1.1374,
            0.8535,
            -0.0852,
        ]
    )

    init_pose = np.zeros(16)
    init_pose[12] += 1.4
    scene = SceneGraph(tree.root, init_pose)
    mesh = scene.getMesh()

    o3d.visualization.draw_geometries(mesh)
