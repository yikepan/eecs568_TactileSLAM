# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings

import cv2
import dill as pickle
import git
import imgviz
import numpy as np
import torch
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R
from segment_anything import SamPredictor, sam_model_registry
from termcolor import cprint
from torch import nn
from torchvision import transforms

from neuralfeels import geometry
from neuralfeels.contrib.tactile_transformer import TactileDepth
from neuralfeels.datasets import dataset, image_transforms
from neuralfeels.datasets.data_util import FrameData
from neuralfeels.geometry.transform import transform_points_np
from neuralfeels.modules import sample
from neuralfeels.modules.misc import pose_from_config

# quicklink to the root and folder directories
root = git.Repo(".", search_parent_directories=True).working_tree_dir


class Sensor(nn.Module):
    def __init__(
        self,
        cfg_sensor: DictConfig,
        device: str = "cuda",
    ):
        super(Sensor, self).__init__()
        self.device = device
        self.sensor_name = cfg_sensor.name
        cprint(f"Adding Sensor: {self.sensor_name}", color="yellow")

        self.end = False

        self.kf_min_loss = cfg_sensor.kf_min_loss  # threshold for adding to kf set

        if "realsense" in self.sensor_name:
            self.optimal_mask_size = cfg_sensor.optimal_mask_size[self.sensor_name]
            self.sam_offset = cfg_sensor.sam_offset[self.sensor_name]

        # sampling parameters
        self.loss_ratio = cfg_sensor.sampling.loss_ratio
        self.free_space_ratio = cfg_sensor.sampling.free_space_ratio
        self.max_depth = -cfg_sensor.sampling.depth_range[1]  # -z towards the object
        self.min_depth = -cfg_sensor.sampling.depth_range[0]
        self.dist_behind_surf = cfg_sensor.sampling.dist_behind_surf
        self.n_rays = cfg_sensor.sampling.n_rays
        self.n_strat_samples = cfg_sensor.sampling.n_strat_samples
        self.n_surf_samples = cfg_sensor.sampling.n_surf_samples
        self.surface_samples_offset = cfg_sensor.sampling.surface_samples_offset

        # vizualisation/rendering parameters
        self.reduce_factor = cfg_sensor.viz.reduce_factor
        self.reduce_factor_up = cfg_sensor.viz.reduce_factor_up

    def set_intrinsics(self, intrinsics: dict):
        self.W = intrinsics["w"]
        self.H = intrinsics["h"]
        self.fx = intrinsics["fx"]
        self.fy = intrinsics["fy"]
        self.cx = intrinsics["cx"]
        self.cy = intrinsics["cy"]
        print(
            f"{self.sensor_name} intrinsics: {self.W}x{self.H}, fx: {self.fx}, fy: {self.fy}, cx: {self.cx}, cy: {self.cy}"
        )
        self.camera_matrix = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.distortion_coeffs = []
        if "k1" in intrinsics:
            self.distortion_coeffs.append(intrinsics["k1"])
        if "k2" in intrinsics:
            self.distortion_coeffs.append(intrinsics["k2"])
        if "p1" in intrinsics:
            self.distortion_coeffs.append(intrinsics["p1"])
        if "p2" in intrinsics:
            self.distortion_coeffs.append(intrinsics["p2"])
        if "k3" in intrinsics:
            self.distortion_coeffs.append(intrinsics["k3"])

        self.set_viz_cams()
        self.set_directions()
        self.set_active_sampling_params()

    def set_viz_cams(self):
        reduce_factor = self.reduce_factor
        self.H_vis = self.H // reduce_factor
        self.W_vis = self.W // reduce_factor
        self.fx_vis = self.fx / reduce_factor
        self.fy_vis = self.fy / reduce_factor
        self.cx_vis = self.cx / reduce_factor
        self.cy_vis = self.cy / reduce_factor

        reduce_factor_up = self.reduce_factor_up
        self.H_vis_up = self.H // reduce_factor_up
        self.W_vis_up = self.W // reduce_factor_up
        self.fx_vis_up = self.fx / reduce_factor_up
        self.fy_vis_up = self.fy / reduce_factor_up
        self.cx_vis_up = self.cx / reduce_factor_up
        self.cy_vis_up = self.cy / reduce_factor_up

    def set_directions(self):
        self.dirs_C = geometry.transform.ray_dirs_C(
            1,
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.device,
            depth_type="z",
        )

        self.dirs_C_vis = geometry.transform.ray_dirs_C(
            1,
            self.H_vis,
            self.W_vis,
            self.fx_vis,
            self.fy_vis,
            self.cx_vis,
            self.cy_vis,
            self.device,
            depth_type="z",
        ).view(1, -1, 3)

        self.dirs_C_vis_up = geometry.transform.ray_dirs_C(
            1,
            self.H_vis_up,
            self.W_vis_up,
            self.fx_vis_up,
            self.fy_vis_up,
            self.cx_vis_up,
            self.cy_vis_up,
            self.device,
            depth_type="z",
        ).view(1, -1, 3)

    def set_active_sampling_params(self):
        # for active_sampling
        self.loss_approx_factor = 8
        w_block = self.W // self.loss_approx_factor
        h_block = self.H // self.loss_approx_factor
        increments_w = (
            torch.arange(self.loss_approx_factor, device=self.device) * w_block
        )
        increments_h = (
            torch.arange(self.loss_approx_factor, device=self.device) * h_block
        )
        c, r = torch.meshgrid(increments_w, increments_h)
        c, r = c.t(), r.t()
        self.increments_single = torch.stack((r, c), dim=2).view(-1, 2)

    def sample_points(
        self,
        depth_batch,
        T_WC_batch,
        norm_batch=None,
        n_rays=None,
        dist_behind_surf=None,
        n_strat_samples=None,
        n_surf_samples=None,
        surface_samples_offset=None,
        box_extents=None,
        box_transform=None,
        free_space_ratio=None,
        grad=False,
        viz_masks=False,
    ):
        """
        Sample points by first sampling pixels, then sample depths along
        the backprojected rays.
        """
        if n_rays is None:
            n_rays = self.n_rays
        if dist_behind_surf is None:
            dist_behind_surf = self.dist_behind_surf
        if n_strat_samples is None:
            n_strat_samples = self.n_strat_samples
        if n_surf_samples is None:
            n_surf_samples = self.n_surf_samples
        if surface_samples_offset is None:
            surface_samples_offset = self.surface_samples_offset

        n_frames = depth_batch.shape[0]

        free_space_ratio = (
            self.free_space_ratio if free_space_ratio is None else free_space_ratio
        )
        free_space_rays = int(n_rays * free_space_ratio)
        object_rays = n_rays - free_space_rays

        # if box not provided use hardcoded min and max depths to sample
        if "realsense" in self.sensor_name and n_strat_samples > 0:
            assert box_extents is not None and box_transform is not None
        else:
            box_extents, box_transform = None, None

        # sample on the object surface (useful for tracking)
        obj_ray_sample = None
        obj_mask = ~torch.isnan(depth_batch)
        if object_rays:
            indices_b, indices_h, indices_w = sample.sample_pixels(
                object_rays,
                n_frames,
                self.H,
                self.W,
                device=self.device,
                mask=obj_mask,
            )
            (
                dirs_C_sample,
                T_WC_sample,
                depth_sample,
                norm_sample,
            ) = sample.get_batch_data(
                T_WC_batch,
                self.dirs_C,
                indices_b,
                indices_h,
                indices_w,
                depth_batch=depth_batch,
                norm_batch=norm_batch,
            )
            # use min and max depth for object rays, use box also if realsense
            max_depth = (
                depth_sample + torch.sign(depth_sample + 1e-8) * dist_behind_surf
            )
            pc, z_vals, valid_ray = sample.sample_along_rays(
                T_WC_sample,
                dirs_C_sample,
                n_strat_samples,
                n_surf_samples=n_surf_samples,
                surf_samples_offset=surface_samples_offset,
                min_depth=self.min_depth,
                max_depth=max_depth,
                box_extents=box_extents,
                box_transform=box_transform,
                gt_depth=depth_sample,
                grad=grad,
            )  # pc: (num_samples, N + M + 1, 3)

            if valid_ray is not None:
                if not valid_ray.all():
                    warnings.warn("Some object rays miss the box")
                    # filter out invalid rays to match the dimensions of the other tensors
                    indices_b, indices_h, indices_w, dirs_C_sample, depth_sample = (
                        indices_b[valid_ray],
                        indices_h[valid_ray],
                        indices_w[valid_ray],
                        dirs_C_sample[valid_ray],
                        depth_sample[valid_ray],
                    )

            # all rays should be valid
            obj_ray_sample = {
                "pc": pc,
                "z_vals": z_vals,
                "indices_b": indices_b,
                "indices_h": indices_h,
                "indices_w": indices_w,
                "dirs_C_sample": dirs_C_sample,
                "depth_sample": depth_sample,
                "T_WC_sample": T_WC_sample,
                "norm_sample": norm_sample,
            }

        # sample in the free-space, only realsense (for carving out objects in mapping)
        free_space_sample = None
        if free_space_rays:
            box_mask, _ = self.get_box_masks(T_WC_batch, box_extents, box_transform)
            free_space_mask = torch.logical_and(box_mask, ~obj_mask)
            if viz_masks:

                def vz(x):
                    x = x[0].to(torch.uint8)[..., None].repeat(1, 1, 3)
                    return x.cpu().numpy() * 255

                viz = np.hstack(
                    [
                        imgviz.depth2rgb(depth_batch[0].cpu().numpy()),
                        vz(obj_mask),
                        vz(box_mask),
                        vz(free_space_mask),
                    ]
                )
                cv2.imshow("masks", viz)
                cv2.waitKey(1)
            (
                indices_b,
                indices_h,
                indices_w,
            ) = sample.sample_pixels(
                free_space_rays,
                n_frames,
                self.H,
                self.W,
                device=self.device,
                mask=free_space_mask,
            )
            dirs_C_sample, T_WC_sample, _, _ = sample.get_batch_data(
                T_WC_batch,
                self.dirs_C,
                indices_b,
                indices_h,
                indices_w,
            )
            # use box extents and transform to sample
            pc, z_vals, valid_ray = sample.sample_along_rays(
                T_WC_sample,
                dirs_C_sample,
                n_strat_samples + n_surf_samples,  # must match tot for object rays
                box_extents=box_extents,
                box_transform=box_transform,
                grad=grad,
            )  # pc: (num_samples, N + M + 1, 3)

            if not valid_ray.all():
                warnings.warn("Some free spac rays miss the box")
                # filter out invalid rays to match the dimensions of the other tensors
                indices_b, indices_h, indices_w, dirs_C_sample = (
                    indices_b[valid_ray],
                    indices_h[valid_ray],
                    indices_w[valid_ray],
                    dirs_C_sample[valid_ray],
                )

            depth_sample = torch.full(
                indices_b.shape, fill_value=torch.nan, device=self.device
            )
            free_space_sample = {
                "pc": pc,
                "z_vals": z_vals,
                "indices_b": indices_b,
                "indices_h": indices_h,
                "indices_w": indices_w,
                "dirs_C_sample": dirs_C_sample,
                "depth_sample": depth_sample,
                "T_WC_sample": T_WC_sample,
                "norm_sample": norm_sample,
            }

        # merge samples from object and free space rays
        if obj_ray_sample is None:
            samples = free_space_sample
        elif free_space_sample is None:
            samples = obj_ray_sample
        else:
            samples = {
                k: (
                    torch.cat([obj_ray_sample[k], free_space_sample[k]], dim=0)
                    if obj_ray_sample[k] is not None
                    else None
                )  # as norm_sample can be None
                for k in obj_ray_sample.keys()
            }

        free_space_mask = torch.isnan(depth_batch)
        binary_masks = torch.zeros(depth_batch.shape, device=depth_batch.device)
        binary_masks[indices_b, indices_h, indices_w] = 1

        sample_pts = {
            "depth_batch": depth_batch,
            "free_space_mask": free_space_mask,
            "binary_masks": binary_masks,
            "format": self.sensor_name,
            "object_rays": object_rays * n_frames,
        }
        return {**samples, **sample_pts}

    def batch_indices(self):
        indices = np.arange(len(self.scene_dataset), dtype=int)  # use all frames
        return indices

    def project(self, pc):
        return geometry.transform.project_pointclouds(
            pc,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.W,
            self.H,
            device=self.device,
        )

    def backproject(self, depth):
        pc = geometry.transform.backproject_pointclouds(
            depth,
            self.fx_vis,
            self.fy_vis,
            self.cx_vis,
            self.cy_vis,
            device=self.device,
        ).squeeze()
        return pc, ~np.isnan(pc).any(axis=1)


class DigitSensor(Sensor):
    tac_depth = None

    def __init__(
        self,
        cfg_sensor: DictConfig,
        dataset_path: str = None,
        calibration: dict = None,
        device: str = "cuda",
    ):
        super(DigitSensor, self).__init__(cfg_sensor, device)
        assert (dataset_path is None) != (
            calibration is None
        ), "Pass only one of dataset path or frame from ROS"

        self.gt_depth = cfg_sensor.tactile_depth.mode == "gt"
        if dataset_path is not None:
            sensor_location = cfg_sensor.name.replace("digit_", "")
            seq_dir = os.path.join(root, dataset_path, "allegro", sensor_location)

            pkl_path = os.path.join(root, dataset_path, "data.pkl")
            with open(pkl_path, "rb") as p:
                digit_info = pickle.load(p)["digit_info"]

            self.scene_dataset = dataset.TactileDataset(
                root_dir=seq_dir, gt_depth=self.gt_depth
            )

            cfg_sensor.tactile_depth.use_real_data = "real" in dataset_path
            # More conservative loss ratio for real data
            if cfg_sensor.tactile_depth.use_real_data:
                self.loss_ratio *= 0.1
        else:
            digit_info = calibration

        self.set_intrinsics(digit_info["intrinsics"])

        # Load one model for all sensor classes
        if not self.gt_depth and DigitSensor.tac_depth is None:
            DigitSensor.tac_depth = TactileDepth(
                cfg_sensor.tactile_depth.mode,
                real=cfg_sensor.tactile_depth.use_real_data,
                device=device,
            )

        # image transforms
        self.cam_dist = digit_info["cam_dist"]
        self.inv_depth_scale = 1.0 / digit_info["depth_scale"]
        self.rgb_transform = transforms.Compose([image_transforms.BGRtoRGB()])
        self.depth_transform = transforms.Compose(
            [
                image_transforms.DepthScale(self.inv_depth_scale),
                image_transforms.DepthTransform(self.cam_dist),
            ]
        )

        self.outlier_thresh = 5.0

        # gel depth for thresholding renderer
        self.gel_depth = self.get_gel_depth(cfg_sensor)

    def get_frame_data(self, idx, poses, msg_data=None):
        if msg_data is not None:
            image = msg_data["color"]
        else:
            image, depth = self.scene_dataset[idx]  # extract rgb, d, transform

        if not self.gt_depth:
            depth = DigitSensor.tac_depth.image2heightmap(
                image[:, :, ::-1], sensor_name=self.sensor_name
            )  # RGB -> BGR

            mask = DigitSensor.tac_depth.heightmap2mask(
                depth, sensor_name=self.sensor_name
            )
            depth = (
                depth.cpu().numpy().astype(np.int64)
                if torch.is_tensor(depth)
                else depth.astype(np.int64)
            )

            mask = (
                mask.cpu().numpy().astype(np.int64)
                if torch.is_tensor(mask)
                else mask.astype(np.int64)
            )

            depth = depth * mask  # apply contact mask

        image = self.rgb_transform(image)
        # scale from px to m and transform to gel frame
        depth = self.depth_transform(depth)

        # gt_depth = self.depth_transform(gt_depth.cpu().numpy())
        im_np = image[None, ...]  # (1, H, W, C)
        depth_np = depth[None, ...]  # (1, H, W)

        T_np = poses[None, ...]  # (1, 4, 4)

        im = torch.from_numpy(im_np).float().to(self.device) / 255.0
        depth = torch.from_numpy(depth_np).float().to(self.device)
        T = torch.from_numpy(T_np).float().to(self.device)

        data = FrameData(
            frame_id=np.array([idx]),
            im_batch=im,
            im_batch_np=im_np,
            depth_batch=depth,
            depth_batch_np=depth_np,
            T_WC_batch=T,
            T_WC_batch_np=T_np,
            format=[self.sensor_name],
            frame_avg_losses=torch.zeros([1], device=self.device),
        )
        return data

    def outlier_rejection_depth(self, depth):
        # outlier thresholding
        abs_depth = np.abs(depth[depth != 0.0])
        if len(abs_depth) > 0:
            outlier_thresh_max = np.percentile(abs_depth, 100 - self.outlier_thresh)
            outlier_thresh_min = np.percentile(abs_depth, 0.1)
            outlier_mask = (outlier_thresh_min < np.abs(np.nan_to_num(depth))) & (
                np.abs(np.nan_to_num(depth)) < outlier_thresh_max
            )
            reject_fraction = 1 - np.sum(depth * outlier_mask) / np.sum(depth)
            # Flat surfaces can cause outlier_thresh_min and outlier_thresh_max to be almost equal
            if reject_fraction > 0.2:
                outlier_mask = np.ones_like(depth, dtype=bool)
            depth = depth * outlier_mask

        return depth

    def get_gel_depth(self, cfg: DictConfig):
        g = cfg.gel
        origin = g.origin

        X0, Y0, Z0 = origin[0], origin[1], origin[2]
        # Curved gel surface
        N = g.countW
        W, H = g.width, g.height
        M = int(N * H / W)
        R = g.R
        zrange = g.curvatureMax

        y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
        z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
        yy, zz = np.meshgrid(y, z)
        h = R - np.maximum(0, R**2 - (yy - Y0) ** 2 - (zz - Z0) ** 2) ** 0.5
        xx = X0 - zrange * h / h.max()

        gel_depth = torch.tensor(-xx).to(
            self.device
        )  # negative for our sign convention
        return gel_depth


class RealsenseSensor(Sensor):
    segment_pred = None

    def __init__(
        self,
        cfg_sensor: DictConfig,
        dataset_path: str = None,
        calibration: dict = None,
        device: str = "cuda",
    ):
        super(RealsenseSensor, self).__init__(cfg_sensor, device)
        assert (dataset_path is None) != (
            calibration is None
        ), "Pass only one of dataset path or frame from ROS"

        self.gt_seg = cfg_sensor.masks == "read"
        if dataset_path is not None:
            # create dataset to load rgb and depth images
            sensor_location = cfg_sensor.name.replace("realsense_", "")
            sensor_location = sensor_location.replace("_", "-")
            seq_dir = os.path.join(root, dataset_path, "realsense", sensor_location)

            if "feelsight_real" in seq_dir:
                self.sim_noise_iters = 0
            else:
                self.sim_noise_iters = cfg_sensor.sim_noise_iters
            self.scene_dataset = dataset.VisionDataset(
                seq_dir, gt_seg=self.gt_seg, sim_noise_iters=self.sim_noise_iters
            )

            self.is_real = "real" in dataset_path

            # load pose and intrinsics which are fixed during sequence
            pkl_path = os.path.join(root, dataset_path, "data.pkl")
            with open(pkl_path, "rb") as p:
                realsense_data = pickle.load(p)["realsense"][sensor_location]
            self.set_intrinsics(realsense_data["intrinsics"])
            self.set_realsense_pose(realsense_data["pose"])

        else:
            self.set_intrinsics(calibration["intrinsics"])
            self.set_realsense_pose(pose_from_config(calibration["pose"]))
            self.depth_scale = calibration["depth_scale"]

        self.masks = cfg_sensor.masks

        # for filtering and segmentation
        self.outlier_thresh = 7.0
        self.cutoff_depth = 0.7

        self.mask_pixels = None
        self.sensor_pixels = None
        self.visible_sensors = None
        self.logits = None
        self.seg_area = None

        # Load one model for all sensor classes
        if not self.gt_seg and RealsenseSensor.segment_pred is None:
            # Load ViT-H SAM model and set masks and logits to None
            if "vit_h" in self.masks:
                sam_checkpoint = os.path.join(
                    root, "data", "segment-anything", "sam_vit_h_4b8939.pth"
                )
                model_name = "vit_h"
            elif "vit_l" in self.masks:
                sam_checkpoint = os.path.join(
                    root, "data", "segment-anything", "sam_vit_l_0b3195.pth"
                )
                model_name = "vit_l"
            elif "vit_b" in self.masks:
                sam_checkpoint = os.path.join(
                    root,
                    "data",
                    "segment-anything",
                    "sam_vit_b_01ec64.pth",
                )
                model_name = "vit_b"
            else:
                raise NotImplementedError
            sam = sam_model_registry[model_name](checkpoint=sam_checkpoint)
            sam.to(device=device)
            RealsenseSensor.segment_pred = SamPredictor(sam)

        if self.gt_seg:
            self.seg_area = self.scene_dataset.get_avg_seg_area()
            print(f"Average segmentation area: {self.seg_area}")

        self.minor_tf = np.eye(4)

    def set_realsense_pose(self, pose):
        # because sim and real datasets are saved inconsistently
        if pose.ndim == 2:
            self.pose = pose
        else:
            self.pose = pose[0]

    def get_realsense_pose(self):
        return self.pose

    def get_frame_data(self, idx, digit_poses, latest_render_depth, msg_data=None):
        if msg_data is not None:
            image = msg_data["color"]
            depth = msg_data["depth"]
            depth = depth * self.depth_scale
        else:
            image, depth = self.scene_dataset[idx]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = undistort_depth(depth, self.camera_matrix, self.distortion_coeffs)

        seg_pixels = None
        if not self.gt_seg:
            (
                sensor_pixels,
                visible_sensors,
                focus_pixels,
            ) = self.kinematics_pixel_prompt(
                digit_poses, self.pose, latest_render_depth
            )
            self.set_mask(focus_pixels, sensor_pixels, visible_sensors)
            mask, seg_pixels = self.segment_object(image)
            depth = depth * mask

            seg_pixels = seg_pixels[None, ...]

        depth = filter_depth(
            depth,
            self.outlier_thresh,
            1.0,
            self.cutoff_depth,
            do_reject_frac=False,
        )

        # depth = filter_depth(
        #     depth, self.outlier_thresh, self.outlier_thresh, self.cutoff_depth
        # )

        im_np = image[None, ...]  # (1, H, W, C)
        depth_np = depth[None, ...]  # (1, H, W)

        im = torch.from_numpy(im_np).float().to(self.device) / 255.0
        depth = torch.from_numpy(depth_np).float().to(self.device)

        T_np = self.pose[None, ...]  # (1, 4, 4)
        T = torch.from_numpy(T_np).float().to(self.device)

        data = FrameData(
            frame_id=np.array([idx]),
            im_batch=im,
            im_batch_np=im_np,
            depth_batch=depth,
            depth_batch_np=depth_np,
            T_WC_batch=T,
            T_WC_batch_np=T_np,
            seg_pixels=seg_pixels,
            format=[self.sensor_name],
            frame_avg_losses=torch.zeros([1], device=self.device),
        )
        return data

    def define_sample_range(self, grasp_center):
        # compute distance between base_pose and self.pose
        dist = np.linalg.norm(grasp_center - self.pose[:3, -1])
        self.max_depth = -(dist + 0.1)
        self.min_depth = -(dist - 0.1)

    def set_mask(self, mask_pixels, sensor_pixels, visible_sensors):
        self.mask_pixels = mask_pixels
        self.sensor_pixels = sensor_pixels
        self.visible_sensors = visible_sensors

    def get_optimal_mask(self, mask, logits, all_sensor_pixels):
        """
        Compute area of each mask and choose the one closest to the optimal area
        Diregard masks that are too small or too large
        """

        min_mask_size = self.optimal_mask_size // 5
        max_mask_size = self.optimal_mask_size * 1.5
        valid_masks_min = [i for i, m in enumerate(mask) if np.sum(m) > min_mask_size]
        valid_masks_max = [i for i, m in enumerate(mask) if np.sum(m) < max_mask_size]
        valid_masks = list(set(valid_masks_min) & set(valid_masks_max))
        if len(valid_masks) > 0:
            mask, logits = mask[valid_masks], logits[valid_masks]
            mask = [
                mask[i]
                for i in np.argsort(
                    np.abs(np.sum(mask, axis=(1, 2)) - self.optimal_mask_size)
                )
            ]
            logits = [
                logits[i]
                for i in np.argsort(
                    np.abs(np.sum(mask, axis=(1, 2)) - self.optimal_mask_size)
                )
            ]
        mask, logits = mask[0], logits[0]
        return mask, logits

    def segment_object(self, image):
        image_shape = image.shape[:-1]
        target_shape = (480, 640)
        resized_image = cv2.resize(
            image,
            dsize=(target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        if not self.is_real:
            # Gaussian blur to handle un-photrealistic scenes in IsaacGym
            resized_image = cv2.GaussianBlur(resized_image, (15, 15), 0)
        RealsenseSensor.segment_pred.set_image(resized_image)

        # object points: positive labels for SAM
        object_points = (self.mask_pixels / image_shape[::-1]) * target_shape[::-1]
        object_points = object_points.astype(np.int32)
        object_labels = np.ones(object_points.shape[0])

        # DIGIT points: negative labels for SAM
        all_sensor_pixels = self.sensor_pixels.copy()
        visible_sensor_pixels = all_sensor_pixels.copy()
        visible_sensor_pixels[~self.visible_sensors] = 0

        sensor_points = (visible_sensor_pixels / image_shape[::-1]) * target_shape[::-1]
        sensor_points = sensor_points.astype(np.int32)
        sensor_labels = np.zeros(sensor_points.shape[0])

        # without negative prompts
        # input_points = object_points
        # input_labels = object_labels

        # with negative prompts
        input_points = np.concatenate((object_points, sensor_points), axis=0)
        input_labels = np.concatenate((object_labels, sensor_labels), axis=0)

        # to visualize the prompts
        # im = resized_image.copy()
        # for point, label in zip(input_points, input_labels):
        #     color = (0, 255, 0) if label == 1 else (255, 0, 0)
        #     cv2.circle(im, point, 5, color, -1)
        # cv2.imshow('prompts', im[..., ::-1]); cv2.waitKey(1)

        # multi-mask prediction usually segments out: (object part, object, foreground) len(mask) = 3
        mask, scores, logits = RealsenseSensor.segment_pred.predict(
            point_coords=input_points,
            point_labels=input_labels,
            mask_input=self.logits,
            multimask_output=True,
        )
        mask, logits = self.get_optimal_mask(mask, logits, all_sensor_pixels)

        self.logits = logits[None, :]
        mask = mask.squeeze().astype(np.uint8)
        mask = cv2.resize(
            mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC
        )

        # scale input_points back to original image dimensions
        input_points = (input_points / target_shape[::-1]) * image_shape[::-1]
        return mask.astype(bool), input_points

    def get_box_masks(
        self,
        T_WC_batch,
        box_extents,
        box_transform,
        vis=False,
        get_contours=False,
        reduce_factor=1,
        zoom_factor=2,
    ):
        if vis:
            return geometry.transform.get_box_masks_zoomed(
                T_WC_batch,
                box_extents,
                box_transform,
                self.fx_vis,
                self.fy_vis,
                self.cx_vis,
                self.cy_vis,
                self.W_vis,
                self.H_vis,
                get_contours=get_contours,
                reduce_factor=reduce_factor,
                zoom_factor=zoom_factor,
            )
        else:
            return geometry.transform.get_box_masks_zoomed(
                T_WC_batch,
                box_extents,
                box_transform,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                self.W,
                self.H,
                get_contours=get_contours,
                reduce_factor=reduce_factor,
                zoom_factor=zoom_factor,
            )

    def kinematics_pixel_prompt(self, digit_poses, realsense_pose, latest_render_depth):
        """
        Returns the pixel coordinates of the tactile sensors, and which ones are visible from the camera
        Also returns the pixel coordinate of the center of the grasp
        """
        # offset the kinematic tracking point to the tip of the DIGIT
        digit_adjust = np.eye(4)
        digit_adjust[1, 3] = 8e-3
        digit_poses = {k: v @ digit_adjust for k, v in digit_poses.items()}

        sensor_points, focus_point = get_center_of_grasp(
            digit_poses, offset=self.sam_offset
        )

        sensor_pixels = self.point_to_pixel(sensor_points, realsense_pose)
        focus_pixels = self.point_to_pixel(focus_point, realsense_pose)

        # Use the latest rendering to figure out if the sensor is in view, or occluded by the
        sensor_depth = None
        if latest_render_depth is not None:
            sensor_pixels_torch = torch.from_numpy(sensor_pixels).long().to(self.device)
            sensor_depth = latest_render_depth[
                0, sensor_pixels_torch[:, 1], sensor_pixels_torch[:, 0]
            ]
            sensor_depth = torch.abs(sensor_depth).cpu().numpy()
        visible_sensors = get_visible_sensors(digit_poses, realsense_pose, sensor_depth)

        return sensor_pixels, visible_sensors, focus_pixels

    def point_to_pixel(self, point, T_WC):
        if point.ndim == 1:
            point = point[None, :]
        focus_point = transform_points_np(point, np.linalg.inv(T_WC))
        mask_pixels = self.project(focus_point[None, :, :])
        mask_pixels = mask_pixels.reshape(-1, 2)
        return mask_pixels

    def minor_adjustment(self, key):
        """
        Adjust the minor errors in the realsense calibration with the GUI
        (1, 2, 3, 4, 5, 6) for controls the translation
        (8, i, 9, o, 0, p) for controls the rotation
        7 for saving the adjustment
        """
        if key == 49:  # 1
            self.minor_tf[:3, 3] += np.array([0.0025, 0.0, 0.0])
        elif key == 50:  # 2
            self.minor_tf[:3, 3] -= np.array([0.0025, 0.0, 0.0])
        elif key == 51:  # 3
            self.minor_tf[:3, 3] += np.array([0.0, 0.0025, 0.0])
        elif key == 52:  # 4
            self.minor_tf[:3, 3] -= np.array([0.0, 0.0025, 0.0])
        elif key == 53:  # 5
            self.minor_tf[:3, 3] += np.array([0.0, 0.0, 0.0025])
        elif key == 54:  # 6
            self.minor_tf[:3, 3] -= np.array([0.0, 0.0, 0.0025])
        elif key == 56:  # 8
            self.minor_tf[:3, :3] = (
                self.minor_tf[:3, :3]
                @ R.from_euler("zxy", [1, 0, 0], degrees=True).as_matrix()
            )
        elif key == 57:  # 9
            self.minor_tf[:3, :3] = (
                self.minor_tf[:3, :3]
                @ R.from_euler("zxy", [0, 1, 0], degrees=True).as_matrix()
            )
        elif key == 48:  # 0
            self.minor_tf[:3, :3] = (
                self.minor_tf[:3, :3]
                @ R.from_euler("zxy", [0, 0, 1], degrees=True).as_matrix()
            )
        elif key == 105:  # i
            self.minor_tf[:3, :3] = (
                self.minor_tf[:3, :3]
                @ R.from_euler("zxy", [-1, 0, 0], degrees=True).as_matrix()
            )
        elif key == 111:  # o
            self.minor_tf[:3, :3] = (
                self.minor_tf[:3, :3]
                @ R.from_euler("zxy", [0, -1, 0], degrees=True).as_matrix()
            )
        elif key == 112:  # p
            self.minor_tf[:3, :3] = (
                self.minor_tf[:3, :3]
                @ R.from_euler("zxy", [0, 0, -1], degrees=True).as_matrix()
            )
        print(f"Minor adjustment: {self.minor_tf}")


def get_center_of_grasp(digit_poses, offset=0.0):
    sensor_points = np.dstack(
        [digit_pose[:3, -1] for digit_pose in digit_poses.values()]
    )
    grasp_center = sensor_points.mean(axis=2)
    # add small offset in the Z axis
    grasp_center[:, 2] += offset

    sensor_points = sensor_points.squeeze().T
    return sensor_points, grasp_center


def get_visible_sensors(digit_poses, realsense_pose, sensor_depth, far_thresh=4e-2):
    """
    Returns which DIGIT sensors are in front of the object w.r.t. the realsense.
    Uses either the rendered depth or some handcrafted 3D checks
    """
    digit_positions = np.dstack(list(digit_poses.values()))[:3, -1, :].T
    mean_digit_position = digit_positions.mean(axis=0)
    realsense_position = realsense_pose[:3, -1]

    if sensor_depth is not None:
        # Mode 1: using only rendered depth to reason about sensor visibility
        realsense_to_digit_dist = np.linalg.norm(
            realsense_position - digit_positions, axis=1
        )
        # check if sensor depth is greater than euclidean distance
        is_in_front = (realsense_to_digit_dist < sensor_depth) | (sensor_depth == 0)
        return is_in_front
    else:
        # Mode 2: use 3D distance to reason about sensor visibility (less preferred)
        # check 1: is the sensor in front of the object?
        realsense_to_mean_dist = np.linalg.norm(
            realsense_position - mean_digit_position
        )
        # print(f"Realsense to mean digit dist: {realsense_to_mean_dist}")
        realsense_to_digit_dist = np.linalg.norm(
            realsense_position - digit_positions, axis=1
        )
        is_in_front = realsense_to_digit_dist < realsense_to_mean_dist
        # check 2: is the sensor far enough from the object?
        digit_to_mean_dist = np.linalg.norm(
            digit_positions - mean_digit_position, axis=1
        )
        is_far_enough = digit_to_mean_dist > far_thresh
        return is_in_front | is_far_enough


def undistort_depth(depth, camera_matrix, distortion_coeffs):
    img_size = (depth.shape[1], depth.shape[0])
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        np.array(distortion_coeffs),
        np.eye(3),
        camera_matrix,
        img_size,
        cv2.CV_32FC1,
    )
    depth = cv2.remap(depth, map1, map2, cv2.INTER_NEAREST)
    return depth


def filter_depth(
    depth,
    outlier_thresh_max_perc,
    outlier_thresh_min_perc,
    cutoff_depth,
    do_reject_frac=True,
):
    abs_depth = np.abs(depth[depth != 0.0])
    if len(abs_depth) > 0:
        outlier_thresh_max = np.percentile(abs_depth, 100 - outlier_thresh_max_perc)
        outlier_thresh_min = np.percentile(abs_depth, outlier_thresh_min_perc)
        outlier_mask = (
            (outlier_thresh_min < np.abs(np.nan_to_num(depth)))
            & (np.abs(np.nan_to_num(depth)) < outlier_thresh_max)
            & (np.abs(np.nan_to_num(depth)) < cutoff_depth)
        )

        if do_reject_frac:
            # Flat surfaces can cause outlier_thresh_min and outlier_thresh_max to be almost equal
            reject_fraction = 1 - np.sum(depth * outlier_mask) / np.sum(depth)
            if reject_fraction > 0.1:
                outlier_mask = np.abs(np.nan_to_num(depth)) < cutoff_depth
                # print(
                #     f"% reject_fraction: {reject_fraction},"
                #     f"outlier_min: {outlier_thresh_min:.2f},"
                #     f"outlier_max: {outlier_thresh_max:.2f}"
                # )

        depth = depth * outlier_mask

    depth[depth == 0.0] = torch.nan
    return depth
