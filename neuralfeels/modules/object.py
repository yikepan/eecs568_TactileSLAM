# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Dataloader for object data in neuralfeels

import os

import dill as pickle
import numpy as np
import theseus as th
import torch


class Object:
    def __init__(
        self,
        map_mode: False,
        dataset_path: str = None,
        device: str = "cuda",
    ):
        """Dataloader for object data in neuralfeels"""
        super(Object, self).__init__()

        self.data_path = None
        if dataset_path is not None:
            self.data_path = os.path.join(dataset_path, "data.pkl")
            with open(self.data_path, "rb") as p:
                self.data = pickle.load(p)
            self.object_pose_gt = torch.tensor(
                np.array(self.data["object"]["pose"]),
                device=device,
                dtype=torch.float32,
            )
        else:
            self.object_pose_gt = torch.eye(4, device=device).unsqueeze(0)
            # tensor large enough for 180 seconds of data at 30 fps
            self.object_pose_gt = self.object_pose_gt.repeat(180 * 30, 1, 1)

        # current_pose_offset: computes the differences between the current and ground truth pose at every iteration. Needed to isolate
        # pose errors from map errors in F-score computation. (only applicable for mode=SLAM)
        self.current_pose_offset = np.eye(4)

        if map_mode:
            # if mapping, initialize the tracking problem with ground-truth
            self.object_pose_track = self.object_pose_gt.clone()
        else:
            # if slam/pure pose, initialize the tracking problem with identity
            self.object_pose_track = torch.zeros_like(self.object_pose_gt)
            self.object_pose_track[0] = torch.eye(4, device=device)

    def add_noise_to_poses(self, poses, noise_cfg):
        """
        Corrupt poses with noise
        """

        N = poses.shape[0]
        pose_noise = th.SE3.exp_map(
            torch.cat(
                [
                    noise_cfg.translation
                    * (
                        2.0 * torch.rand((N, 3), device=poses.device) - 1
                    ),  # scale translation noise n_t * [-1, 1]
                    noise_cfg.rotation
                    * (
                        2 * torch.rand((N, 3), device=poses.device) - 1
                    ),  # scale rotation noise n_r * [-1, 1]
                ],
                dim=1,
            )
        ).to_matrix()

        return poses @ pose_noise

    def save_baseline(self):
        # save pickle file with added baseline
        self.data["object"]["pose"] = list(self.object_pose_track.clone().cpu().numpy())
        with open(self.data_path, "wb") as p:
            pickle.dump(self.data, p)
        print("Saved baseline poses to: ", self.data_path)
