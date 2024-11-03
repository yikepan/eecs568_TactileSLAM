# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/Jianghanxiao/Helper3D

import re

import numpy as np
import open3d as o3d

from .SceneNode import SceneNode

# TODO : very slow, can we cache the SceneNodes?


class SceneGraph:
    def __init__(self, rootLink, joint_angles=None):
        self.root = SceneNode()
        self.joint_angles = joint_angles
        self.constructNode(self.root, rootLink)

    def update(self):
        self.root.update()

    def getMesh(self):
        self.update()
        meshes = self.root.getMesh()
        new_meshes = []
        for mesh in meshes:
            new_meshes.append(mesh)
        return new_meshes

    def updateJointAngles(self, joint_angles):
        self.joint_angles = joint_angles.cpu().numpy()
        return

    def rotateNode(self, node, joint_rpy):
        updates = np.nonzero(joint_rpy)[0]
        if len(updates) > 1:
            for i in [0, 1, 2]:
                _joint_rpy = np.zeros(3)
                _joint_rpy[i] += joint_rpy[i]
                node.rotateXYZ(_joint_rpy)
        else:
            node.rotateXYZ(joint_rpy)

    def getRPY(self, node):
        joint_axis = node.joint.axis
        axis_of_rotation = np.nonzero(joint_axis)[0].squeeze().item()
        rotate_rpy = np.zeros(3)
        joint_rpy = node.joint.origin["rpy"].astype(np.float64)

        if "tip" not in node.joint.joint_name:
            joint_id = re.findall("\d+\.\d+", node.joint.joint_name)[0]
            joint_id = int(float(joint_id))
            rotate_rpy[axis_of_rotation] += self.joint_angles[joint_id] * (
                -1.0 if joint_id == 13 else 1.0
            )
            # rotate_rpy[axis_of_rotation] += -3.14159 if joint_id == 12 else 0.0
            joint_rpy += rotate_rpy
        return joint_rpy

    def updateState(self, node=None):
        if node == None:
            node = self.root

        if "base_link" not in node.name:
            node.resetlocalTransform()

        if node.joint != None:
            # Construct the joint node firstly; Deal with xyz and rpy of the node
            joint_xyz = node.joint.origin["xyz"]
            joint_rpy = self.getRPY(node)

            # TODO: fix the
            # if node.name == "link_12.0":
            #     print("update state joint_rpy", joint_rpy)

            self.rotateNode(node, joint_rpy)
            node.translate(joint_xyz)

        for child_node in node.children:
            self.updateState(child_node)

    def constructNode(self, node, link):
        node.name = link.link.link_name

        node.joint = link.joint
        if node.joint != None:
            # Construct the joint node firstly; Deal with xyz and rpy of the node

            joint_xyz = node.joint.origin["xyz"]
            joint_rpy = self.getRPY(node)

            # if node.name == "link_12.0":
            #     print("construct state joint_rpy", joint_rpy)

            self.rotateNode(node, joint_rpy)
            node.translate(joint_xyz)

        # Construct the mesh nodes for multiple visuals in link
        visuals = link.link.visuals
        color = link.link.color
        for visual in visuals:
            visual_node = SceneNode(node)
            node.addChild(visual_node)
            visual_node.name = node.name + "_mesh:" + str(visual.visual_name)
            if visual.geometry_mesh["filename"] == None:
                raise RuntimeError("Invalid File path")
            visual_node.addMeshFile(visual.geometry_mesh["filename"], color)
            # Deal with xyz and rpy of the visual node
            visual_xyz = visual.origin["xyz"]
            visual_rpy = visual.origin["rpy"]
            visual_scale = visual.geometry_mesh["scale"]
            visual_node.rotateXYZ(visual_rpy)
            visual_node.translate(visual_xyz)
            visual_node.scale(visual_scale)

        # Construct node for the children
        for child in link.children:
            child_node = SceneNode(node)
            node.addChild(child_node)
            self.constructNode(child_node, child)
