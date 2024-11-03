# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/Jianghanxiao/Helper3D


class ExLink:
    def __init__(self, link):
        self.link = link
        self.parent = None
        self.children = []
        self.joint = None

    def setParent(self, parent):
        self.parent = parent

    def addChild(self, child):
        self.children.append(child)

    def setJoint(self, joint):
        self.joint = joint

    def __repr__(self):
        output = {}
        output["link"] = self.link
        if self.parent != None:
            output["parent"] = self.parent.link.link_name
        else:
            output["parent"] = None
        output["children"] = [child.link.link_name for child in self.children]
        output["joint"] = self.joint
        return str(output)
