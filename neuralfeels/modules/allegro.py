# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Class for Allegro hand joint state and forward kinematics

import os
from typing import Dict

import dill as pickle
import git
import numpy as np
import theseus as th
import torch
from torchkin import Robot, get_forward_kinematics_fns

from neuralfeels.modules.misc import pose_from_config

root = git.Repo(".", search_parent_directories=True).working_tree_dir


class Allegro:
    def __init__(
        self,
        dataset_path: str = None,
        base_pose: Dict = None,
        device: str = "cuda",
    ):
        """Allegro hand dataloader for neuralfeels data"""
        super(Allegro, self).__init__()
        assert (dataset_path is None) != (base_pose is None)
        self.device = device

        urdf_path = os.path.join(
            root, "data/assets/allegro/allegro_digit_left_ball.urdf"
        )  # Allegro hand URDF file
        self.robot, self.fkin, self.links, self.joint_map = load_robot(
            urdf_file=urdf_path, num_dofs=16, device=device
        )

        if dataset_path is not None:
            # Load base pose and jointstate vectors
            data_path = os.path.join(dataset_path, "data.pkl")
            with open(data_path, "rb") as p:
                self.data = pickle.load(p)
            self.allegro_pose = self.data["allegro"]["base_pose"]
            self.joint_states = torch.tensor(
                self.data["allegro"]["joint_state"], device=device, dtype=torch.float32
            )
        else:
            self.allegro_pose = pose_from_config(base_pose)

    def _hora_to_neural(self, finger_poses):
        """
        Convert the DIGIT urdf reference frame (bottom of the sensor) to neural SLAM frame
        """
        finger_poses = finger_poses @ np.linalg.inv(
            np.array(
                [
                    [0.000000, -1.000000, 0.000000, 0.000021],
                    [0.000000, 0.000000, 1.000000, -0.017545],
                    [-1.000000, 0.000000, 0.000000, -0.002132],
                    [0.000000, 0.000000, 0.000000, 1.000000],
                ]
            )
        )
        return finger_poses

    def get_fk(self, idx=None, joint_state=None):
        """Forward kinematics using theseus torchkin"""

        assert idx is None or joint_state is None
        if joint_state is not None:
            joint_states = torch.tensor(joint_state, device=self.device)
        else:
            if idx >= len(self.joint_states):
                return None
            joint_states = self.joint_states[idx].clone()

        # joint states is saved as [index, middle, ring, thumb]
        self.current_joint_state = joint_states  # for viz

        # Swap index and ring for left-hand, theseus FK requires this
        joint_states_theseus = joint_states.clone()
        joint_states_theseus[[0, 1, 2, 3]], joint_states_theseus[[8, 9, 10, 11]] = (
            joint_states_theseus[[8, 9, 10, 11]],
            joint_states_theseus[[0, 1, 2, 3]],
        )

        # Change to breadth-first order, theseus needs this too
        joint_states_theseus = joint_states_theseus[self.joint_map]
        j = th.Vector(
            tensor=joint_states_theseus.unsqueeze(0),
            name="joint_states",
        )
        link_poses = self.fkin(j.tensor)
        digit_poses = torch.vstack(link_poses).to(self.robot.device)
        digit_poses = th.SE3(tensor=digit_poses).to_matrix().cpu().numpy()

        base_tf = np.repeat(
            self.allegro_pose[np.newaxis, :, :], digit_poses.shape[0], axis=0
        )
        digit_poses = base_tf @ digit_poses
        digit_poses = self._hora_to_neural(digit_poses)
        return {k: v for k, v in zip(list(self.links.keys()), list(digit_poses))}

    def get_base_pose(self):
        return self.allegro_pose


def load_robot(urdf_file: str, num_dofs: int, device):
    """Load robot from URDF file and cache FK functions"""
    robot = Robot.from_urdf_file(urdf_file, device=device)
    links = {
        "digit_index": "link_3.0_tip",
        "digit_middle": "link_7.0_tip",
        "digit_ring": "link_11.0_tip",
        "digit_thumb": "link_15.0_tip",
    }

    # FK function is applied breadth-first, so swap the indices from the allegro convention
    joint_map = torch.tensor(
        [joint.id for joint in robot.joint_map.values() if joint.id < num_dofs],
        device=device,
    )
    # base, index, middle, ring, thumb
    fkin, *_ = get_forward_kinematics_fns(robot, list(links.values()))
    return (robot, fkin, links, joint_map)
