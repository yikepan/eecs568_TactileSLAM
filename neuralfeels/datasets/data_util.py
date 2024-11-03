# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Class to store image and depth data for each frame of the optimization

import copy

import numpy as np
import torch


class FrameData:
    def __init__(
        self,
        frame_id=None,
        im_batch=None,
        im_batch_np=None,
        depth_batch=None,
        depth_batch_np=None,
        T_WC_batch=None,
        T_WC_batch_np=None,
        normal_batch=None,
        seg_pixels=None,
        frame_avg_losses=None,
        format=None,
    ):
        super(FrameData, self).__init__()

        self.frame_id = frame_id
        self.im_batch = im_batch
        self.im_batch_np = im_batch_np
        self.depth_batch = depth_batch
        self.depth_batch_np = depth_batch_np
        self.T_WC_batch = T_WC_batch
        self.T_WC_batch_np = T_WC_batch_np

        self.normal_batch = normal_batch
        self.seg_pixels = seg_pixels
        self.frame_avg_losses = frame_avg_losses

        self.format = format

        self.frame_count = 0 if frame_id is None else len(frame_id)

    def add_frame_data(self, data, replace):
        """
        Add new FrameData to existing FrameData.
        """
        self.frame_count += len(data.frame_id)
        self.frame_id = expand_data(self.frame_id, data.frame_id, replace)

        self.im_batch = expand_data(self.im_batch, data.im_batch, replace)
        self.im_batch_np = expand_data(self.im_batch_np, data.im_batch_np, replace)

        self.depth_batch = expand_data(self.depth_batch, data.depth_batch, replace)
        self.depth_batch_np = expand_data(
            self.depth_batch_np, data.depth_batch_np, replace
        )

        self.T_WC_batch = expand_data(self.T_WC_batch, data.T_WC_batch, replace)
        self.T_WC_batch_np = expand_data(
            self.T_WC_batch_np, data.T_WC_batch_np, replace
        )

        self.normal_batch = expand_data(self.normal_batch, data.normal_batch, replace)

        self.seg_pixels = expand_data(self.seg_pixels, data.seg_pixels, replace)
        device = data.im_batch.device
        empty_dist = torch.zeros([data.im_batch.shape[0]], device=device)
        self.frame_avg_losses = expand_data(self.frame_avg_losses, empty_dist, replace)

        if type(data.format) is not list:
            data.format = [data.format]
        if self.format is None:
            self.format = data.format
        else:
            self.format += data.format

    def delete_frame_data(self, indices):
        """
        Delete FrameData at given indices.
        """
        self.frame_count -= len(indices)
        self.frame_id = np.delete(self.frame_id, indices)

        self.im_batch = torch.cat(
            [self.im_batch[: indices[0]], self.im_batch[indices[-1] + 1 :]]
        )
        self.im_batch_np = np.delete(self.im_batch_np, indices, axis=0)

        self.depth_batch = torch.cat(
            [self.depth_batch[: indices[0]], self.depth_batch[indices[-1] + 1 :]]
        )
        self.depth_batch_np = np.delete(self.depth_batch_np, indices, axis=0)

        self.T_WC_batch = torch.cat(
            [self.T_WC_batch[: indices[0]], self.T_WC_batch[indices[-1] + 1 :]]
        )
        self.T_WC_batch_np = np.delete(self.T_WC_batch_np, indices, axis=0)

        if self.normal_batch is not None:
            self.normal_batch = torch.cat(
                [self.normal_batch[: indices[0]], self.normal_batch[indices[-1] + 1 :]]
            )

        if self.seg_pixels is not None:
            self.seg_pixels = torch.cat(
                [self.seg_pixels[: indices[0]], self.seg_pixels[indices[-1] + 1 :]]
            )
        self.frame_avg_losses = torch.cat(
            [
                self.frame_avg_losses[: indices[0]],
                self.frame_avg_losses[indices[-1] + 1 :],
            ]
        )

    def __len__(self):
        if self.T_WC_batch is None:
            return 0
        else:
            return self.T_WC_batch.shape[0]

    def __getitem__(self, index):
        return FrameData(
            frame_id=np.array(self.frame_id[index]),
            im_batch=self.im_batch[index],
            im_batch_np=self.im_batch_np[index],
            depth_batch=self.depth_batch[index],
            depth_batch_np=self.depth_batch_np[index],
            T_WC_batch=self.T_WC_batch[index],
            T_WC_batch_np=self.T_WC_batch_np[index],
            normal_batch=(
                None if self.normal_batch is None else self.normal_batch[index]
            ),
            seg_pixels=None if self.seg_pixels is None else self.seg_pixels[index],
            frame_avg_losses=self.frame_avg_losses[index],
            format=self.format[index],
        )


def expand_data(batch, data, replace=False):
    """
    Add new FrameData attribute to exisiting FrameData attribute.
    Either concatenate or replace last row in batch.
    """
    cat_fn = np.concatenate
    if torch.is_tensor(data):
        cat_fn = torch.cat

    if batch is None:
        batch = copy.deepcopy(data)

    else:
        if replace is False:
            batch = cat_fn((batch, data))
        else:
            batch[-1] = data[0]

    return batch
