# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Trainer for neural field

import collections
import copy
import os
import pickle
import shutil
import time

import cv2
import git
import imgviz
import matplotlib.pylab as plt
import numpy as np
import open3d as o3d
import open3d.core as o3c
import theseus as th
import torch
import trimesh
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict
from scipy import ndimage
from scipy.spatial import cKDTree as KDTree
from termcolor import cprint
from tqdm import tqdm

from neuralfeels import geometry
from neuralfeels.datasets import data_util, sdf_util
from neuralfeels.datasets.data_util import FrameData
from neuralfeels.eval.feelsight_init import feelsight_init_poses
from neuralfeels.eval.metrics import (
    average_3d_error,
    compute_f_score,
    end_timing,
    start_timing,
)
from neuralfeels.geometry import transform
from neuralfeels.geometry.transform import transform_points_np
from neuralfeels.modules import loss, model, pose_optimizer, render, sample, sensor
from neuralfeels.modules.allegro import Allegro
from neuralfeels.modules.model import gradient
from neuralfeels.modules.object import Object
from neuralfeels.modules.sensor import get_center_of_grasp
from neuralfeels.viz import draw

from neuralfeels.viz import plot_pose

# quicklink to the root and folder directories
root = git.Repo(".", search_parent_directories=True).working_tree_dir


class Trainer:
    def __init__(self, cfg: DictConfig, gpu_id: int = 0, ros_node=None):
        super(Trainer, self).__init__()

        self.cfg_main = cfg.main
        self.cfg_data = cfg.main.data
        self.cfg_train = cfg.main.train
        self.cfg_eval = cfg.main.eval
        self.cfg_scene = cfg.main.scene
        self.cfg_viz = cfg.main.viz
        self.cfg_pose = cfg.main.pose

        self.device = f"cuda:{gpu_id}"
        self.train_mode = self.cfg_train.train_mode

        # loop over sensors, change the name when multiple, assign seq_dir based on object name (name and object must not be part of sensor definition)
        self.cfg_sensors = [
            value for key, value in cfg.main.items() if "sensor" in key.lower()
        ]
        self.sensor = {}
        self.frames = {}
        self.steps_since_frame = {}

        self.init_map_loss, self.init_pose_loss = None, None
        self.n_keyframes, self.frame_id = {}, {}
        self.gt_depth_vis = {}
        self.gt_im_vis = {}
        self.prev_kf_time = {}
        self.sample_pts = {}
        self.render_samples = {}
        self.render_frames = {}

        self.set_params()
        self.latest_render_depth = {}

        dataset_path = os.path.join(root, cfg.main.data.dataset_path)

        if self.cfg_main.occlusion:
            # replace feelsight in dataset path with occlusion
            dataset_path = dataset_path.replace("feelsight", "feelsight_occlusion")

        self.object_name = cfg.main.data.object
        self.log_id = dataset_path.split("/")[-1]

        for sensor_cfg in self.cfg_sensors:
            sensor_class = (
                sensor.RealsenseSensor
                if "realsense" in sensor_cfg.name
                else sensor.DigitSensor
            )
            self.sensor[sensor_cfg.name] = sensor_class(
                sensor_cfg,
                dataset_path,
                calibration=None,
                device=self.device,
            )
            self.frames[sensor_cfg.name] = FrameData()  # keyframes
            self.steps_since_frame[sensor_cfg.name] = 0
            self.gt_depth_vis[sensor_cfg.name] = None
            self.gt_im_vis[sensor_cfg.name] = None
            self.sample_pts[sensor_cfg.name] = None
            self.n_keyframes[sensor_cfg.name] = 0
            self.frame_id[sensor_cfg.name] = []
            self.render_frames[sensor_cfg.name] = None
            self.latest_render_depth[sensor_cfg.name] = None

        self.sensor_list = list(self.sensor.keys())

        self.render_samples = {"pc": None, "sdf": None}
        base_pose = None
        self.allegro = Allegro(
            dataset_path=dataset_path,
            base_pose=base_pose,
            device=self.device,
        )
        self.object = Object(
            map_mode="map" in self.train_mode,
            dataset_path=dataset_path,
            device=self.device,
        )

        self.is_baseline = "baseline" in self.cfg_main.mode
        if self.is_baseline:
            cprint("Running baseline, will save the poses on completion", color="green")

        self.train_fps = 30.0
        with open(os.path.join(dataset_path, "data.pkl"), "rb") as p:
            data = pickle.load(p)
        self.dataset_time = data["time"]
        # actual fps of the dataset (not the one used for training)
        self.real_fps = len(self.dataset_time) / self.dataset_time[-1]
        if self.real_fps == 0:
            self.real_fps = 30.0
        self.train_fps = self.cfg_data.train_fps  # fps used for training

        real_time = self.dataset_time[-1]
        train_time = self.dataset_time[-1] * self.real_fps / self.train_fps

        # store 75% and 80% of the real_time as the schedule times
        self.map_schedule_times = [real_time * 0.9, real_time * 0.95, real_time]
        self.map_schedule_idx = 0
        cprint(
            f"real_fps: {self.real_fps} ({real_time} secs),"
            f"train_fps: {self.train_fps} ({train_time:.1f} secs or {train_time/60:.1f} mins)",
            color="yellow",
        )

        self.incremental = self.cfg_train.optimizer.incremental
        self.map_iters = 0
        self.tot_step_time = 0.0

        # keyframe params
        self.time_since_kf = np.inf
        self.prev_kf_time = -np.inf
        self.last_is_keyframe = False
        self.current_time = 0.0
        self.t0 = time.time()

        self.touch_tree = None

        self.save_stats = {
            "pose": {
                "opt_pose": [],
                "gt_pose": [],
                "errors": [], "timing": []},
            "map": {"errors": [], "timing": []},
            "optimizer": self.cfg_pose.second_order.tsdf_method,
            "cameras": [s for s in self.sensor_list if "realsense" in s],
        }

        # Add occlusion stats to save
        if self.cfg_main.occlusion:
            # save average segmentation mask area
            self.save_stats["seg_area"] = self.sensor[
                self.save_stats["cameras"][0]
            ].seg_area
            # save pose of the camera
            self.save_stats["camera_pose"] = self.sensor[
                self.save_stats["cameras"][0]
            ].pose

        self.grasp_threshold = self.cfg_pose.grasp_threshold
        self.gt_voxel_size = self.cfg_train.gt_voxel_size
        self.load_checkpoint_model = self.cfg_train.load_checkpoint_model
        self.train_time_min = self.cfg_train.batch.train_time_min

        self.step_window = collections.deque([])

        self.full_pc = {}
        for sensor_name in self.sensor_list:
            self.full_pc[sensor_name] = {
                "points": [],
                "colors": [],
            }

        # eval params
        self.gt_sdf_interp = None
        self.stage_sdf_interp = None
        self.sdf_dims = None
        self.sdf_transform = None

        self.mesh_interval = self.cfg_scene.mesh_interval
        self.grid_dim = self.cfg_scene.grid_dim
        self.crop_dist = self.cfg_scene.crop_dist
        self.prev_frame = -1

        self.new_grid_dim = None

        # scene params for visualisation
        self.scene_center = None
        self.active_idxs = None
        self.active_pixels = None
        self.last_track = 0  # last optimized frame_id pose
        self.tracked_frames = np.array([self.last_track])

        self.is_palm_init = False

        self.define_scene()
        if self.train_mode in ["pose", "pose_grasp"]:
            self.load_gt_sdf(path=dataset_path)
        self.load_networks()
        self.update_scene_properties(idx=0)
        if (
            self.cfg_train.optimizer.checkpoint is not None
            and self.load_checkpoint_model
        ):
            self.load_checkpoint(self.cfg_train.optimizer.checkpoint)

        self.sdf_map.train()
        self.cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.map_ratio, self.pose_ratio = 1.0, 1.0

        if not self.incremental:
            print("Offline mode - adding all frames")
            self.add_all_frames()
            
        # check memory usage
        available, total = torch.cuda.mem_get_info("cuda:0")
        print(
            f"Trainer Initialization Memory usage: {available / (1024**3)}GB available out of {total / (1024**3)}GB"
        )

    # Init functions ---------------------------------------
    
    def visualize_pose(self):
        plot_pose.visualize_pose(self.save_stats, save_path='./plot')
        

    def get_latest_frame_id(self):
        """
        data was collect at self.real_fps rate, but we train at self.train_fps rate
        """
        return int(self.tot_step_time * self.train_fps)

    def define_scene(self):
        # Scale and extents of object
        self.obj_extents_np = np.array(self.cfg_scene.extents)
        self.obj_extents = torch.tensor(self.obj_extents_np).float().to(self.device)
        self.grid_range = [-1.0, 1.0]
        self.obj_scale_np = self.obj_extents_np / (
            self.grid_range[1] - self.grid_range[0]
        )
        self.obj_scale = torch.from_numpy(self.obj_scale_np).float().to(self.device)
        self.up = np.array([0.0, 0.0, 1.0])

    def init_first_pose(self, digit_poses):
        """
        Initialize the first pose of the object with the center of grasp.
        """
        if self.train_mode in ["map"]:
            return
        # intialize object either between the phalanges (is_palm_init) or at the fingertips
        init_offset = -0.02 if self.is_palm_init else 0.01
        _, grasp_center = get_center_of_grasp(digit_poses, offset=init_offset)
        self.object.object_pose_track[0, :3, 3] = torch.tensor(
            grasp_center, device=self.device
        )

        # define sampling range for vision sensors
        vision_sensor_list = [x for x in self.sensor_list if "digit" not in x.lower()]
        for viz_sensor in vision_sensor_list:
            self.sensor[viz_sensor].define_sample_range(grasp_center)

        # Get a rough estimate of rotation for the initial pose
        if self.is_baseline:
            # if baseline method, we load from hand-computed initial poses, which will break the symmetry of the objects
            # First get the default camera view for the front-left camera
            standard_rot = (
                np.array(feelsight_init_poses["default"].split(" "))
                .astype(np.float32)
                .reshape(4, 4)[:3, :3]
            )
            # Next get the rotated camera view to align with the first frame of the log
            init_pose = feelsight_init_poses[self.object_name][self.log_id]
            init_pose = np.array(init_pose.split(" ")).astype(np.float32).reshape(4, 4)
            init_rot, init_trans = init_pose[:3, :3], init_pose[:3, 3]
            init_rot = np.dot(np.linalg.inv(standard_rot), init_rot)  # apply diff
            self.object.object_pose_track[0, :3, :3] = torch.tensor(
                init_rot, device=self.device
            )  # assign rotation to first frame
            self.object.object_pose_track[0, 2, 3] += init_trans[
                2
            ]  # assign translation offset
        elif self.train_mode in ["pose", "slam"]:
            # initialize with ground-truth for pose tracking evaluations
            self.object.object_pose_track[0] = self.object.object_pose_gt[0]
        else:
            # Initialize rotations with closest cardinal rotation to gt. This is to avoid ambiguious RMSE stats for symmetric objects
            all_cardinal_rotations = transform.get_all_rotations()
            cardinal_errors = []
            for i, R in enumerate(all_cardinal_rotations):
                cardinal_errors.append(
                    pose_optimizer.rot_rmse(
                        self.object.object_pose_gt[:1, :3, :3],
                        R.unsqueeze(0).to(self.device),
                    )
                )
            cardinal_errors = torch.hstack(cardinal_errors)
            best_rotation = all_cardinal_rotations[torch.argmin(cardinal_errors)]
            self.object.object_pose_track[0, :3, :3] = best_rotation

    def update_scene_properties(self, idx):
        # monogram notation: https://manipulation.mit.edu/pick.html#monogram
        # transform from (W)orld to (O)rigin in (W)orld frame
        if self.train_mode in ["map"]:
            self.last_track = idx
        if idx == self.last_track:
            # update current pose only if last tracked
            self.p_WO_W_np = (
                self.object.object_pose_track[idx].detach().cpu().numpy().copy()
            )
            self.p_WO_W = torch.from_numpy(self.p_WO_W_np).float().to(self.device)
        self.p_WO_W_gt_np = (
            self.object.object_pose_gt[idx].detach().cpu().numpy().copy()
        )

        # Center of the neural field
        self.scene_center = self.p_WO_W_np[:3, 3]

        # Neural field query grid
        self.grid_pc = geometry.transform.make_3D_grid(
            self.grid_range,  # [-1, 1]
            self.grid_dim,  # 200
            self.device,
            transform=self.p_WO_W,
            scale=self.obj_scale,
        )
        self.grid_pc = self.grid_pc.view(-1, 3).to(self.device)

        self.up_ix = np.argmax(np.abs(np.matmul(self.up, self.p_WO_W_np[:3, :3])))
        self.grid_up = self.p_WO_W_np[:3, self.up_ix]
        self.up_aligned = np.dot(self.grid_up, self.up) > 0

        # Update neural field parameters
        self.sdf_map.update_pose_and_scale(self.p_WO_W.unsqueeze(0), self.obj_scale)

        self.update_current_time()

    def update_current_time(self):
        self.current_time = self.tot_step_time * self.train_fps / self.real_fps

    def set_params(self):
        if "gt_sdf_dir" in self.cfg_data:
            gt_sdf_dir = self.cfg_data.gt_sdf_dir
            # check if object belongs to ycb or feelsight
            if "gt_models" in gt_sdf_dir:
                object_class = "ycb"
                if self.cfg_data.object in [
                    "bell_pepper",
                    "large_dice",
                    "peach",
                    "pear",
                    "pepper_grinder",
                    "rubiks_cube_small",
                ]:
                    object_class = "feelsight"
                gt_sdf_dir = os.path.join(gt_sdf_dir, object_class)
            self.gt_obj_file = os.path.join(
                root, gt_sdf_dir, f"{self.cfg_data.object}.urdf"
            )

            # load both trimesh and open3d mesh from file
            is_color_mesh = True  # self.train_mode in ["map", "slam"]
            mesh_trimesh, mesh_o3d = sdf_util.load_gt_mesh(
                self.gt_obj_file, color=is_color_mesh
            )
            self.gt_obj_mesh = mesh_trimesh
            self.gt_obj_mesh_o3d = mesh_o3d

            # Check if we need to intialize in the palm or between the fingers
            bounds = self.gt_obj_mesh.bounds
            max_obj_bounds = np.max(np.abs(bounds[0, :] - bounds[1, :]))
            self.is_palm_init = (
                max_obj_bounds > 0.1
            )  # if obj is > 10cm, initialize in palm
            # print(f"Palm init: {self.is_palm_init}")

            # ground-truth pointcloud from mesh in canonical object frame
            self.num_points_f_score = self.cfg_eval.num_points_f_score
            self.which_f_score = self.cfg_eval.which_f_score
            self.f_score_T = self.cfg_eval.f_score_T
            self.gt_mesh_sampled = np.empty((0, 3))
            while True:
                sP, f = trimesh.sample.sample_surface_even(
                    self.gt_obj_mesh, count=self.num_points_f_score
                )
                self.gt_mesh_sampled = np.vstack([self.gt_mesh_sampled, sP])
                if len(self.gt_mesh_sampled) > self.num_points_f_score:
                    self.gt_mesh_sampled = self.gt_mesh_sampled[
                        : self.num_points_f_score, :
                    ]
                    break

            self.latest_f_score = None
            self.digit_file = os.path.join(root, "data", "assets", "digit/digit.STL")

            self.realsense_file = os.path.join(
                root, "data", "assets", "realsense", "realsense.stl"
            )

            self.gt_scene = True

        # Model
        self.do_active = bool(self.cfg_train.model.do_active)
        # scaling applied to network output before interpreting value as sdf
        self.scale_output = self.cfg_train.model.scale_output
        self.noise_std = self.cfg_train.model.noise_std[
            self.cfg_data.dataset
        ]  # different noises depending on real or synthetic dataset

        self.kf_time = self.cfg_train.model.kf_time
        # max time beyond which keyframe is forced
        self.max_kf_time = self.kf_time * 10

        # sliding window size for optimising latest frames
        self.window_size = self.cfg_train.model.window_size
        self.pose_window_size = self.cfg_pose.window_size

        self.num_layers = self.cfg_train.model.num_layers
        self.hidden_feature_size = self.cfg_train.model.hidden_feature_size

        # optimizer
        self.learning_rate = self.cfg_train.optimizer.lr
        self.weight_decay = self.cfg_train.optimizer.weight_decay

        # Evaluation
        self.do_eval = self.cfg_eval.do_eval
        self.eval_freq_s = self.cfg_eval.eval_freq_s
        self.sdf_eval = bool(self.cfg_eval.sdf_eval)
        self.mesh_eval = bool(self.cfg_eval.mesh_eval)
        self.eval_times = []

        # save
        self.save_period = self.cfg_eval.save_period
        self.save_times = np.arange(self.save_period, 2000, self.save_period).tolist()
        self.save_slices = bool(self.cfg_eval.save_slices)
        self.save_meshes = bool(self.cfg_eval.save_meshes)

        # Loss
        self.loss_params = self.cfg_train.loss

    def load_networks(self):
        # Load neural field and optimiser
        embed_config = self.cfg_train.pos_encoding

        self.map_optimizer, self.map_scheduler = None, None
        self.pose_optimizer, self.pose_scheduler = None, None
        if self.train_mode in ["pose", "pose_grasp"]:
            # Load ground truth sdf and optimize pose
            self.sdf_map = model.SDFInterp(
                self.sdf_grid, self.sdf_transform, device=self.device
            )
        if self.train_mode in ["pose", "pose_grasp", "slam"]:
            with open_dict(self.cfg_pose):
                # assign same truncation distance as mapping
                self.cfg_pose.trunc_distance = self.cfg_train.loss.trunc_distance
                self.cfg_pose.trunc_weight = self.cfg_train.loss.trunc_weight

            self.pose_optimizer = pose_optimizer.PoseOptimizer(
                sensors=[self.sensor[sensor_name] for sensor_name in self.sensor_list],
                cfg=self.cfg_pose,
                train_mode=self.train_mode,
                device=self.device,
            )
        if self.train_mode in ["map", "slam"]:
            # Initialise neural field and optimize sdf
            self.sdf_map = model.SDFNetwork(
                embed_config=embed_config,
                num_layers=self.num_layers,
                skips=[],
                hidden_dim=self.hidden_feature_size,
                clip_sdf=None,
                scale_output=self.scale_output,
                device=self.device,
            )
            milestones, gamma = (
                self.cfg_train.model.milestones,
                self.cfg_train.model.gamma,
            )
            self.map_optimizer = torch.optim.AdamW(
                params=self.sdf_map.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.99),
                weight_decay=self.weight_decay,
                eps=1e-15,
            )
            self.map_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.map_optimizer, milestones=milestones, gamma=gamma
            )
            self.current_lr = self.map_scheduler.get_last_lr()
            self.frozen_transform = np.eye(4)
            self.frozen_transform[:3, 3] -= self.obj_scale_np
            self.frozen_transform[:3, :3] = np.diag(self.obj_extents_np / self.grid_dim)

            random_sdf_grid = torch.zeros(
                (self.grid_dim, self.grid_dim, self.grid_dim), device=self.device
            )
            self.frozen_sdf_map = model.SDFInterp(
                random_sdf_grid, self.frozen_transform, device=self.device
            )

        self.sdf_map.to(device=self.device)
        self.sdf_map.share_memory()

    def load_checkpoint(self, checkpoint_load_file):
        checkpoint = torch.load(os.path.join(root, checkpoint_load_file))
        self.sdf_map.load_state_dict(checkpoint["model_state_dict"])
        self.sdf_map.update_pose_and_scale(
            checkpoint["sdf_map_pose"], checkpoint["sdf_map_scale"]
        )

    def load_gt_sdf(self, path):
        gt_sdf_npy = os.path.join(
            path,
            f"gt_sdf_voxel={self.gt_voxel_size}.npz",
        )

        if os.path.exists(gt_sdf_npy):
            # if gt_sdf_npy exists, load it
            print(f"Loading gt sdf from {gt_sdf_npy}")
            npzfile = np.load(gt_sdf_npy)
            self.sdf_grid = npzfile["sdf"]
            self.sdf_transform = npzfile["tf"]
        else:
            # else, compute it and save it
            print(f"Computing ground-truth sdf (voxel size: {self.gt_voxel_size})")
            self.sdf_grid, self.sdf_transform = sdf_util.sdf_from_mesh(
                mesh=self.gt_obj_mesh,
                voxel_size=self.gt_voxel_size,
                extend_factor=0.15,
                origin_voxel=np.zeros(3),
            )
            print(f"Saving gt sdf to {gt_sdf_npy}")
            np.savez(gt_sdf_npy, sdf=self.sdf_grid, tf=self.sdf_transform)

        self.sdf_grid = torch.tensor(
            self.sdf_grid, device=self.device, dtype=torch.float32
        )
        self.sdf_dims = torch.tensor(self.sdf_grid.shape)

    def backproject_depth(self, sensor_name, depth, colors=None, T_WC=None):
        pcs_cam, valid = self.sensor[sensor_name].backproject(depth)
        pcs_cam = pcs_cam[valid]
        if colors is not None:
            colors = colors[valid, :]
        # sensor to world transform
        if T_WC is not None:
            pcs_world = transform_points_np(pcs_cam, T_WC)
        else:
            pcs_world = pcs_cam
        return pcs_world, colors

    def add_pc_to_memory(self, sensor_name, keyframe_n):
        """
        Backproject depth to world frame given the sensor poses and downsample.
        These are stored whenever there is a new keyframe. get_pcd() reads these from memory and transforms
        to the current object pose to give a "dynamic" point cloud.
        """
        frames = self.frames[sensor_name]

        self.update_vis_vars(sensor_name)
        depth = self.gt_depth_vis[sensor_name][[keyframe_n]]
        any_depth = np.sum(~np.isnan(depth))

        if any_depth:
            colors = self.gt_im_vis[sensor_name][keyframe_n].reshape(-1, 3)
            colors = colors.astype(np.float32) / 255
            T_WC = frames.T_WC_batch_np[keyframe_n].squeeze()
            pcs_world, colors = self.backproject_depth(sensor_name, depth, colors, T_WC)
            temp_pc = pcs_world[None, :]
            temp_colors = colors[None, :]
            temp_pc, temp_colors = self.voxel_downsample(
                temp_pc.squeeze(axis=0), temp_colors.squeeze(axis=0), 1e-4
            )
        else:
            temp_pc = np.zeros((1, 3))
            temp_colors = np.zeros((1, 3))

        if len(self.full_pc[sensor_name]["points"]) <= keyframe_n:
            # if new keyframe, append to list
            self.full_pc[sensor_name]["points"].append(np.vstack(temp_pc))
            self.full_pc[sensor_name]["colors"].append(np.vstack(temp_colors))
        else:
            # if existing keyframe, overwrite pointcloud
            self.full_pc[sensor_name]["points"][keyframe_n] = np.vstack(temp_pc)
            self.full_pc[sensor_name]["colors"][keyframe_n] = np.vstack(temp_colors)

    def check_end(self, idx):
        return idx >= len(self.dataset_time)

    def add_data(self, data, replace=False):
        """
        Adds new keyframe data and pointcloud for given sensor format.
        If last frame isn't a keyframe then the new frame replaces last frame in batch.
        """

        format = data.format[-1]
        frame_id = data.frame_id[-1]

        if frame_id in self.frame_id[format]:
            return False
        replace = self.last_is_keyframe is False

        self.frames[format].add_frame_data(data, replace)
        # self.frame_id is for bookkeeping the pose optimizer, refer get_pcd()
        if not replace or self.n_keyframes[format] == 0:
            # print("appending to keyframes")
            self.n_keyframes[format] += 1
            self.frame_id[format].append(frame_id)
        else:
            # print("replacing last keyframe")
            self.frame_id[format][-1] = frame_id
        # print(f"Added frame {frame_id} to {format} format")
        self.add_pc_to_memory(format, self.n_keyframes[format] - 1)
        return True

    def transform_to_object(self, frames):
        T = frames.T_WC_batch.clone()
        frame_obj_state = self.object.object_pose_track[frames.frame_id]
        tf_pose = frame_obj_state.inverse() @ T
        return self.p_WO_W @ tf_pose

    def add_frame(self, frame_data):
        format = frame_data.format[-1]

        added_frame = self.add_data(frame_data)

        self.steps_since_frame[format] = 0
        return added_frame

    def add_all_frames(self):
        sensor0_name = list(self.sensor.keys())[0]
        indices = self.sensor[sensor0_name].batch_indices()
        # check all sensors have same number of frames
        for sensor_name in self.sensor.keys():
            assert len(indices) == len(self.sensor[sensor_name].batch_indices())

        max_frames = self.cfg_train.batch.max_frames
        indices = np.random.choice(indices, max_frames, replace=False)
        print("Frame indices added in offline mode:", indices)

        for idx in indices:
            digit_poses = self.allegro.get_fk(idx=idx)

            for sensor_name in self.sensor.keys():
                if "digit" in sensor_name:
                    frame_data = self.sensor[sensor_name].get_frame_data(
                        idx, digit_poses[sensor_name]
                    )
                else:
                    frame_data = self.sensor[sensor_name].get_frame_data(
                        idx, digit_poses
                    )
                    self.last_is_keyframe = True
                self.add_frame(frame_data)

            self.update_scene_properties(idx)

    # Keyframe methods ----------------------------------

    def get_latest_depth_renders(self):
        for sensor_name in self.sensor_list:
            frames = self.frames[sensor_name]
            if frames.T_WC_batch is None:
                continue
            T_WC = frames.T_WC_batch[-1].unsqueeze(0)
            depth_gt = torch.nan_to_num(frames.depth_batch[-1].unsqueeze(0))

            sensor = self.sensor[sensor_name]
            with torch.set_grad_enabled(False):
                # efficient depth and normals render
                self.latest_render_depth[sensor_name] = self.render_depth(
                    T_WC, sensor, depth_gt
                )

    def is_keyframe(self):
        if self.time_since_kf > self.max_kf_time:
            # if time since last keyframe is too long, force a new keyframe
            return True
        for sensor_name in self.sensor_list:
            sensor = self.sensor[sensor_name]
            frames = self.frames[sensor_name]
            depth_gt = torch.nan_to_num(frames.depth_batch[-1].unsqueeze(0))
            loss = torch.abs(
                self.latest_render_depth[sensor_name].view(-1) - depth_gt.view(-1)
            )
            # ignore zero depth pixels of depth_gt
            loss = loss[depth_gt.view(-1) != 0]
            avg_loss = loss.mean()
            is_keyframe = avg_loss > sensor.kf_min_loss
            if is_keyframe:
                print(
                    f"[{sensor_name}, t = {self.current_time:.2f}s] Current keyframe loss: {avg_loss:.3f} (thresh: {sensor.kf_min_loss}) ---> is keyframe: {is_keyframe}",
                    end="\r",
                )
                return True
        # print(
        #     f"view_depth.mean(): {view_depth.mean():.3f}, depth_sample.mean(): {depth_sample.mean():.3f}"
        # )

        # view_subplots(
        #     [render_depth.squeeze().cpu().numpy(), depth_gt.squeeze().cpu().numpy()],
        #     [["render", "gt"]],
        # )
        return False

    def update_keyframe_time(self):
        self.time_since_kf = self.current_time - self.prev_kf_time

    def check_keyframe_latest(self):
        """
        returns whether or not to add a new frame.
        """
        add_new_frame = False

        # number of kfs for each sensor
        num_kfs = list(self.n_keyframes.values())[0]

        self.update_keyframe_time()
        if self.last_is_keyframe:
            # Latest frame is already a keyframe. We have now
            # finished the extra steps and want to add a new frame
            add_new_frame = True
        else:
            if self.time_since_kf > self.kf_time or num_kfs <= 1:
                is_keyframe = self.is_keyframe()
                if is_keyframe:
                    self.last_is_keyframe = True
                    self.prev_kf_time = self.current_time

                # self.enforce_watertight()
            if not self.last_is_keyframe:
                add_new_frame = True

        return add_new_frame

    def select_keyframes(self, frames, pose=False):
        """
        Use most recent two keyframes then fill rest of window
        based on loss distribution across the remaining keyframes.
        """
        n_frames = len(frames)
        limit = n_frames - 2

        select_size = self.window_size - 2 if not pose else self.pose_window_size - 2
        last = n_frames - 1

        if select_size == -2:
            # window_size = 0
            idxs = [last]
        elif select_size == -1:
            # window_size = 1
            idxs = [last - 1, last]
        else:
            denom = frames.frame_avg_losses[:-2].sum()
            loss_dist = frames.frame_avg_losses[:-2] / denom
            loss_dist = torch.nan_to_num(loss_dist, 1 / len(loss_dist))  # remove NaNs
            loss_dist_np = loss_dist.cpu().numpy()

            # remove outliers from keyframes if they are more than 2*window_size in the past
            outlier_val = np.percentile(loss_dist_np, 80)
            frame_outliers = np.where(loss_dist_np > outlier_val)[0]
            ignore_flag = frame_outliers < len(loss_dist_np) - 2 * self.window_size
            frame_outliers = frame_outliers[ignore_flag]
            loss_dist_np[frame_outliers] = 0
            loss_dist_np = loss_dist_np / loss_dist_np.sum()

            num_nonzero = len(np.nonzero(loss_dist_np)[0])
            replace = True if num_nonzero < select_size else False

            rand_ints = np.random.choice(
                np.arange(0, limit),
                size=select_size,
                replace=replace,
                p=loss_dist_np if not np.isnan(loss_dist_np).any() else None,
            )
            rand_ints = np.sort(rand_ints)
            idxs = [*rand_ints, last - 1, last]

        return idxs

    def clear_keyframes(self):
        self.frames = FrameData()  # keyframes
        for key, value in self.frames.items():
            self.gt_depth_vis[key] = None
            self.gt_im_vis[key] = None

    # Main training methods ----------------------------------

    def sdf_eval_and_loss(
        self,
        all_samples,
        sensor_name,
        vision_weights=None,
        do_avg_loss=True,
    ):
        sample = all_samples[sensor_name]
        pc = sample["pc"]
        z_vals = sample["z_vals"]
        indices_b = sample["indices_b"]
        indices_h = sample["indices_h"]
        indices_w = sample["indices_w"]
        dirs_C_sample = sample["dirs_C_sample"]
        depth_sample = sample["depth_sample"]
        T_WC_sample = sample["T_WC_sample"]
        norm_sample = sample["norm_sample"]
        binary_masks = sample["binary_masks"]
        depth_batch = sample["depth_batch"]
        free_space_mask = sample["free_space_mask"]
        object_rays = sample["object_rays"]
        format = sample["format"]

        loss_params = self.loss_params

        do_sdf_grad = loss_params.eik_weight != 0
        if do_sdf_grad:
            pc.requires_grad_()

        # add noise to prevent overfitting to high-freq data from a noisy sensor  # [vision, touch]
        if "digit" in sensor_name:
            noise = torch.randn(pc.shape, device=pc.device) * self.noise_std[1]
        else:
            noise = torch.randn(pc.shape, device=pc.device) * self.noise_std[0]

        pc = pc + noise

        # sdf and sdf gradient
        sdf = self.sdf_map(pc, noise_std=None)  # SDF train

        sdf = sdf.reshape(pc.shape[:-1])

        sdf_grad = None
        if do_sdf_grad:
            sdf_grad = gradient(pc, sdf)

        # compute bounds
        # start, end = start_timing()
        bounds, grad_vec = loss.bounds(
            loss_params.bounds_method,
            dirs_C_sample,
            depth_sample,
            T_WC_sample,
            z_vals,
            pc,
            object_rays,
            loss_params.trunc_distance,
            norm_sample,
            do_grad=False,
        )
        # bounds_time = end_timing(start, end)
        # print(f"\n{loss_params.bounds_method} bounds time: {bounds_time:.3f}", end="\n")
        # compute loss

        # equation (8)
        sdf_loss_mat, free_space_ixs = loss.sdf_loss(
            sdf, bounds, loss_params.trunc_distance, loss_type=loss_params.loss_type
        )

        #### added, test
        eik_loss_mat = None
        if loss_params.eik_weight != 0:
            eik_loss_mat = torch.abs(sdf_grad.norm(2, dim=-1) - 1)

        if vision_weights is not None:
            vision_weights = vision_weights.reshape(sdf.shape)
        total_loss, total_loss_mat, losses = loss.tot_loss(
            sdf_loss_mat,
            eik_loss_mat,
            free_space_ixs,
            bounds,
            loss_params.trunc_weight,
            loss_params.eik_weight,
            vision_weights=vision_weights,
        )

        loss_approx, frame_avg_loss = None, None

        W, H, loss_approx_factor = (
            self.sensor[format].W,
            self.sensor[format].H,
            self.sensor[format].loss_approx_factor,
        )

        if do_avg_loss:
            # remove free space from depth batch
            loss_approx, frame_avg_loss = loss.frame_avg(
                total_loss_mat,
                depth_batch,
                indices_b,
                indices_h,
                indices_w,
                W,
                H,
                loss_approx_factor,
                binary_masks,
                free_space_mask,
            )

        return (
            total_loss,
            losses,
            loss_approx,
            frame_avg_loss,
        )

    def check_gt_sdf(
        self,
        depth_sample,
        z_vals,
        dirs_C_sample,
        pc,
        target_ray,
        target_pc,
        target_normal,
    ):
        # origins, dirs_W):
        # reorder in increasing z vals
        z_vals, indices = z_vals.sort(dim=-1)
        row_ixs = torch.arange(pc.shape[0])[:, None].repeat(1, pc.shape[1])

        pc = pc[row_ixs, indices]
        target_ray = target_ray[row_ixs, indices]

        z2euc_sample = torch.norm(dirs_C_sample, dim=-1)
        z_vals = z_vals * z2euc_sample[:, None]

        scene = trimesh.Scene(trimesh.load(self.gt_obj_file))

        with torch.set_grad_enabled(False):
            j = 0
            fig, ax = plt.subplots(3, 1, figsize=(11, 10))

            for i in [9, 19, 23]:  # range(0, 100):
                gt_sdf = sdf_util.eval_sdf_interp(
                    self.gt_sdf_interp,
                    pc[i].reshape(-1, 3).detach().cpu().numpy(),
                    handle_oob="fill",
                    oob_val=np.nan,
                )

                x = z_vals[i].cpu()
                lw = 2.5
                ax[j].hlines(0, x[0], x[-1], color="gray", linestyle="--")
                ax[j].plot(
                    x, gt_sdf, label="True signed distance", color="C1", linewidth=lw
                )
                ax[j].plot(
                    x, target_ray[i].cpu(), label="Ray", color="C3", linewidth=lw
                )
                if target_normal is not None:
                    target_normal = target_normal[row_ixs, indices]
                    ax[j].plot(
                        x,
                        target_normal[i].cpu(),
                        label="Normal",
                        color="C2",
                        linewidth=lw,
                    )
                if target_pc is not None:
                    target_pc = target_pc[row_ixs, indices]
                    ax[j].plot(
                        x,
                        target_pc[i].cpu(),
                        label="Batch distance",
                        color="C0",
                        linewidth=lw,
                    )

                # print("diffs", target_sdf[i].cpu() - gt_sdf)
                if j == 2:
                    ax[j].set_xlabel("Distance along ray, d [m]", fontsize=24)
                    ax[j].set_yticks([0, 4, 8])
                # ax[j].set_ylabel("Signed distance (m)", fontsize=21)
                ax[j].tick_params(axis="both", which="major", labelsize=24)
                # ax[j].set_xticks(fontsize=20)
                # ax[j].set_yticks(fontsize=20)
                # if j == 0:
                #     ax[j].legend(fontsize=20)
                j += 1

            fig.text(
                0.05,
                0.5,
                "Signed distance [m]",
                va="center",
                rotation="vertical",
                fontsize=24,
            )
            # plt.tight_layout()
            plt.show()

        scene.show()

    def end_all(self, sensor_list):
        ended = [self.sensor[sensor_name].end == True for sensor_name in sensor_list]
        return all(ended)

    def depth_time(self, sensor_list, sensor="digit"):
        depth_time = [
            self.sensor[sensor_name].fetch_time()
            for sensor_name in sensor_list
            if sensor in sensor_name
        ]
        if depth_time:
            return np.average(np.vstack(depth_time))
        else:
            return 0.0

    def render_pc_from_sdf(
        self, n_points: int = 2000, visualize: bool = False
    ) -> torch.Tensor:
        # by rendering along rays from virtual cameras around the object
        # generate a point cloud around the object, return pc in object frame

        # generate n views around the object on a sphere that is just
        # outside of the object instant NGP grid
        radius = np.linalg.norm(self.obj_extents_np / 2)
        poses = transform.look_at_on_sphere(
            n_points,
            radius,
            self.scene_center,
            look_at_noise=0.05,  # add 5cm noise to look at point
        )
        poses_on_sphere = torch.tensor(poses, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            # single ray out through cam center
            dirs_C = torch.tensor([[0, 0, 1]], device=self.device)
            pt_samples, z_vals, valid_ray = sample.sample_along_rays(
                poses_on_sphere,
                dirs_C,
                n_stratified_samples=100,
                box_extents=self.obj_extents,
                box_transform=self.p_WO_W,
                grad=False,
                gt_depth=None,
            )

            pt_samples = pt_samples.float()
            sdf = self.sdf_map(pt_samples)  # latest_frame_vis
            sdf = sdf.view(pt_samples.shape[:-1], -1)
            depth_vals = render.sdf_render_depth(
                z_vals, sdf, self.loss_params.trunc_distance
            )[:, None]

            # remove rays that don't intersect instant NGP grid
            poses_on_sphere = poses_on_sphere[valid_ray]

            # backproject depth to 3D points in object frame
            origins, dirs_W = transform.origin_dirs_W(poses_on_sphere, dirs_C)
            origins_g, dirs_g = transform.rays_box_frame(origins, dirs_W, self.p_WO_W)
            pc = origins_g[:, None, :] + (dirs_g[:, None, :] * depth_vals[:, :, None])
            pc = pc[~torch.isnan(depth_vals)]

        if visualize:
            scene = trimesh.Scene()
            # scene.add_geometry(trimesh.PointCloud(poses[:, :3, 3], colors=[0, 255, 0]))
            scene.add_geometry(
                trimesh.PointCloud(pc.detach().cpu().reshape(-1, 3), colors=[255, 0, 0])
            )
            rec_mesh, full_rec_mesh, _, _ = self.mesh_rec(crop_mesh_with_pc=False)
            plot_mesh = full_rec_mesh.apply_scale(self.obj_scale_np[0])
            scene.add_geometry(plot_mesh)
            # bb = trimesh.primitives.Box(extents=self.obj_extents_np, transform=np.eye(4))
            # scene.add_geometry(bb)
            scene.show()

        print(f"Sampled {len(pc)} points on surface of object neural SDF")
        return pc

    def render_depth(self, pose, sensor, depth):
        # render image/depth from current pose + neural field
        # efficient depth and normals render
        depth_shape = depth.shape

        if len(depth_shape) < 3:
            depth_shape = (1, depth_shape[0], depth_shape[1])

        # only render in rays through the box
        box_mask = None
        dirs_C = sensor.dirs_C_vis
        box_extents, box_transform = None, None
        min_depth, max_depth = None, None
        if "realsense" in sensor.sensor_name:
            box_extents, box_transform = self.obj_extents, self.p_WO_W
            box_mask, _ = sensor.get_box_masks(
                pose, self.obj_extents, self.p_WO_W, vis=True
            )
            dirs_C = dirs_C.view(-1, sensor.H_vis, sensor.W_vis, 3)[box_mask][None, ...]
        else:
            min_depth, max_depth = sensor.min_depth, sensor.max_depth

        pc, z_vals, valid_rays = sample.sample_along_rays(
            pose,
            dirs_C,
            n_stratified_samples=100,
            box_extents=box_extents,
            box_transform=box_transform,
            min_depth=min_depth,
            max_depth=max_depth,
            grad=False,
        )

        # [TODO: fix] Edge case where the ray passes through 3 faces and is marked as invalid, assign as nan
        depth_vals_vis = torch.full(
            (dirs_C.shape[1], 1),
            torch.nan,
            device=self.device,
        ).squeeze()

        try:
            # visualize the points sampled
            sdf = self.sdf_map(pc)  # latest_frame_vis
            sdf = sdf.view(pc.shape[:-1], -1)
            depth_vals_vis[valid_rays] = render.sdf_render_depth(
                z_vals, sdf, self.loss_params.trunc_distance
            )
        except RuntimeError:
            # TODO: fix this edge case
            print("RuntimeError, skipping depth_vals_vis")

        if box_mask is not None:
            depth_vis = torch.full(
                [sensor.H_vis, sensor.W_vis], torch.nan, device=self.device
            )
            depth_vis[box_mask[0]] = depth_vals_vis
        else:
            depth_vis = depth_vals_vis.view(sensor.H_vis, sensor.W_vis)

        # upsample rendered depth to image_up size (160, 120)
        render_depth = torch.nn.functional.interpolate(
            depth_vis.view(1, 1, sensor.H_vis, sensor.W_vis),
            size=depth_shape[1:],
            mode="bilinear",
            align_corners=True,
        ).squeeze()

        if "digit" in sensor.sensor_name:
            # remove elements away from gel (DIGIT isn't really a camera)
            gel_depth = torch.nn.functional.interpolate(
                sensor.gel_depth.view(
                    1, 1, sensor.gel_depth.shape[0], sensor.gel_depth.shape[1]
                ),
                size=depth_shape[1:],
                mode="bilinear",
                align_corners=True,
            ).squeeze()
            render_depth[render_depth < gel_depth] = torch.nan  # gel thresholding
            # render_depth[render_depth < sensor.cam_dist] = torch.nan  # standard way
        mask = ~torch.isnan(render_depth)
        render_depth = render_depth * mask

        return torch.nan_to_num(render_depth).squeeze().unsqueeze(0)

    def save_current_pc(self, points, frame_id):
        save_dir = f"./pointsclouds/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        o3d.t.io.write_point_cloud(save_dir + f"{frame_id}.ply", points)
        print(f"Saved pointcloud to {save_dir + f'{frame_id}.ply'}")

    def smoothed(self, time, window_sz=10):
        if not window_sz:
            return time

        if len(self.step_window) >= window_sz:
            self.step_window.popleft()
        self.step_window.append(time)
        return np.mean(np.stack(list(self.step_window)))

    # Visualisation methods -----------------------------------
    def update_vis_vars(self, sensor_name):
        frames = self.frames[sensor_name]
        depth_batch_np = frames.depth_batch_np
        im_batch_np = frames.im_batch_np

        # from neuralfeels.datasets.dataset import view_subplots

        # view_subplots(
        #     [depth_batch_np[0].squeeze(), im_batch_np[0].squeeze()], [["depth", "rgb"]]
        # )

        if self.gt_depth_vis[sensor_name] is None:
            updates = depth_batch_np.shape[0]
        else:
            diff_size = (
                depth_batch_np.shape[0] - self.gt_depth_vis[sensor_name].shape[0]
            )
            updates = diff_size + 1

        for i in range(updates, 0, -1):
            prev_depth_gt = depth_batch_np[-i]
            prev_im_gt = im_batch_np[-i]

            prev_depth_gt_resize = imgviz.resize(
                prev_depth_gt,
                width=self.sensor[sensor_name].W_vis,
                height=self.sensor[sensor_name].H_vis,
                interpolation="nearest",
            )[None, ...]
            prev_im_gt_resize = imgviz.resize(
                prev_im_gt,
                width=self.sensor[sensor_name].W_vis,
                height=self.sensor[sensor_name].H_vis,
            )[None, ...]

            replace = False
            if i == updates:
                replace = True

            self.gt_depth_vis[sensor_name] = data_util.expand_data(
                self.gt_depth_vis[sensor_name], prev_depth_gt_resize, replace=replace
            )
            self.gt_im_vis[sensor_name] = data_util.expand_data(
                self.gt_im_vis[sensor_name], prev_im_gt_resize, replace=replace
            )

    def latest_pose(self, sensor):
        frames = self.frames[sensor]
        return frames.T_WC_batch_np[-1]

    def latest_rgb_vis(self, sensor_name):
        frames = self.frames[sensor_name]
        image = frames.im_batch_np[-1]
        sel_sensor = self.sensor[sensor_name]
        image = cv2.resize(image, (sel_sensor.W_vis_up, sel_sensor.H_vis_up))
        return image

    def frame_vis(self, N, sensor, do_render=True, debug=False):
        # get latest frame from camera
        frames = copy.deepcopy(self.frames[sensor])
        image = frames.im_batch_np[N]
        depth = frames.depth_batch_np[N]

        # from neuralfeels.datasets.dataset import view_subplots
        # view_subplots([image, depth], [["image", "depth"]])

        T_WC = frames.T_WC_batch_np[N]
        T_WC = torch.FloatTensor(T_WC).to(self.device)[None, ...]
        sel_sensor = self.sensor[sensor]

        image = cv2.resize(image, (sel_sensor.W_vis_up, sel_sensor.H_vis_up))
        depth = cv2.resize(depth, (sel_sensor.W_vis_up, sel_sensor.H_vis_up))
        min_depth, max_depth = -sel_sensor.min_depth, -sel_sensor.max_depth

        if np.sum(depth < 0):
            # get min and max filtering out 0 and nan
            max_depth, min_depth = np.min(depth[depth < 0]), np.max(depth[depth < 0])
            # add some leeway to the min and max if realsense
            if "realsense" in sensor:
                max_depth, min_depth = max_depth - 0.01, min_depth + 0.01
        # Visualize the segmented depth map and draw a small contour around it
        depth_viz = imgviz.depth2rgb(
            depth,
            min_value=min_depth,
            max_value=max_depth,
        )

        # if "realsense" not in sensor:
        #     print(f"Sensor: {sensor}, min_depth: {(0.022+min_depth)*1000}mm")

        if "realsense" in sensor:
            seg_depth_mask = (~np.isnan(depth)).astype(np.uint8) * 255
            seg_contour, _ = cv2.findContours(
                seg_depth_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            depth_viz = cv2.addWeighted(depth_viz, 1.0, image, 0.3, 0.0)
            cv2.drawContours(depth_viz, seg_contour, -1, (0, 0, 0), 1)

        if "realsense" in sensor and debug:
            # draw segmentation pixels on image feed
            if frames.seg_pixels is not None and debug:
                seg_pixels = frames.seg_pixels[N]
                seg_pixels[:, 0] *= sel_sensor.W_vis_up / sel_sensor.W
                seg_pixels[:, 1] *= sel_sensor.H_vis_up / sel_sensor.H
                seg_pixels = seg_pixels.astype(np.int32)

                # Crosshair for the object center
                cv2.line(
                    image,
                    (seg_pixels[0, 0], seg_pixels[0, 1] - 10),
                    (seg_pixels[0, 0], seg_pixels[0, 1] + 10),
                    (255, 255, 0),
                    4,
                )
                cv2.line(
                    image,
                    (seg_pixels[0, 0] - 10, seg_pixels[0, 1]),
                    (seg_pixels[0, 0] + 10, seg_pixels[0, 1]),
                    (255, 255, 0),
                    4,
                )
                # circles for each of the DIGIT pixels
                for seg_pixel in seg_pixels[1:]:
                    cv2.circle(image, tuple(seg_pixel), 5, (0, 255, 255), 4)

                # Draw contour around neural field box
                _, contour = self.sensor[sensor].get_box_masks(
                    T_WC,
                    self.obj_extents,
                    self.p_WO_W,
                    vis=True,
                    get_contours=True,
                )

                contour = contour[0].cpu().numpy()
                if sel_sensor.H_vis_up != sel_sensor.H_vis:
                    # resize contour to the render_depth_viz shape
                    contour = imgviz.resize(
                        contour.astype(dtype=np.uint8),
                        width=sel_sensor.W_vis_up,
                        height=sel_sensor.H_vis_up,
                        interpolation="nearest",
                    )
                    contour = contour > 0
                image[contour] = 100

        rgbd_vis = np.hstack((image, depth_viz))

        latest_render_depth = self.latest_render_depth[sel_sensor.sensor_name]
        if not do_render or latest_render_depth is None:
            return rgbd_vis, None, None
        else:
            # render image/depth from current pose + neural field
            with torch.set_grad_enabled(False):
                # efficient depth and normals render
                # resize depth to depth_up
                depth_up = torch.nn.functional.interpolate(
                    latest_render_depth.view(1, 1, sel_sensor.H_vis, sel_sensor.W_vis),
                    size=(sel_sensor.H_vis_up, sel_sensor.W_vis_up),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze()
                depth_vals = depth_up.view(-1)

            surf_normals_C = render.render_normals(
                T_WC, depth_vals[None, ...], self.sdf_map, sel_sensor.dirs_C_vis_up
            )

            render_depth = (
                depth_vals.view(sel_sensor.H_vis_up, sel_sensor.W_vis_up).cpu().numpy()
            )

            render_depth_viz = imgviz.depth2rgb(
                render_depth, min_value=min_depth, max_value=max_depth
            )

            surf_normals_C = (-surf_normals_C + 1.0) / 2.0
            surf_normals_C = torch.clip(surf_normals_C, 0.0, 1.0)
            normals_viz = (
                surf_normals_C.view(sel_sensor.H_vis_up, sel_sensor.W_vis_up, 3)
                .detach()
                .cpu()
            )
            normals_viz = (normals_viz.numpy() * 255).astype(np.uint8)
            normals_viz = cv2.GaussianBlur(normals_viz, (3, 3), 0)  # smooth normals_viz
            # increase color saturation for visualization
            # normals_viz = cv2.cvtColor(normals_viz, cv2.COLOR_BGR2HSV)
            # normals_viz[:, :, 1] = 150
            # normals_viz = cv2.cvtColor(normals_viz, cv2.COLOR_HSV2BGR)

            # Get rendered mask and inverse of that
            render_mask = (render_depth != 0.0)[:, :, None]  # where render depth exists
            inverse_render_mask = (~render_mask.squeeze()).astype(
                np.uint8
            )  # where render depth doesn't exist
            inverse_render_mask = np.repeat(
                inverse_render_mask[:, :, np.newaxis], 3, axis=2
            )

            # Mask rendered normals and depth
            normals_viz = normals_viz * render_mask
            render_depth_viz = render_depth_viz * render_mask
            render_vis = np.hstack((normals_viz, render_depth_viz))

            if "realsense" in sensor:
                # draw green contour around object border
                depth_mask = (render_depth != 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    depth_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )

                # draw coordinate axis of object frame
                T_CW = torch.tensor(np.linalg.inv(self.latest_pose(sensor))).to(
                    device=self.device
                )
                obj_pose = self.object.object_pose_track[self.last_track].to(
                    dtype=T_CW.dtype
                )
                obj_in_cam = T_CW @ obj_pose
                image_normals = copy.deepcopy(image)
                image_normals = draw.draw_xyz_axis(
                    image_normals,
                    obj_in_cam,
                    self.sensor[sensor].fx_vis_up,
                    self.sensor[sensor].fy_vis_up,
                    self.sensor[sensor].cx_vis_up,
                    self.sensor[sensor].cy_vis_up,
                    self.sensor[sensor].W_vis_up,
                    self.sensor[sensor].H_vis_up,
                    scale=0.15,
                    thickness=3,
                    transparency=0.0,
                    is_input_rgb=True,
                )
                image_normals = image_normals * inverse_render_mask
                image_normals = cv2.addWeighted(
                    normals_viz, 1.0, image_normals, 0.8, 0.0
                )
                cv2.drawContours(image_normals, contours, -1, (0, 255, 0), 1)

                # overlay image on render_depth_viz, but remove the regions where the render_depth_viz exists in the image
                inverse_image = copy.deepcopy(image) * inverse_render_mask
                render_depth_viz = cv2.addWeighted(
                    render_depth_viz, 1.0, inverse_image, 0.8, 0.0
                )
                render_vis = np.hstack((image_normals, render_depth_viz))

            # view_subplots([rgbd_vis, render_vis], [["rgbd_vis", "render_vis"]])
            w_up = int(render_vis.shape[1])
            h_up = int(render_vis.shape[0])
            render_vis = cv2.resize(render_vis, (w_up, h_up))

            return rgbd_vis, render_vis, render_depth

    def latest_frame_vis(self, sensor, do_render=True, debug=False):
        return self.frame_vis(-1, sensor, do_render, debug=debug)

    def keyframe_vis(self, reduce_factor=2):
        start, end = start_timing()

        h, w = self.frames.im_batch_np.shape[1:3]
        h = int(h / reduce_factor)
        w = int(w / reduce_factor)

        kf_vis = []
        for i, kf in enumerate(self.frames.im_batch_np):
            kf = cv2.resize(kf, (w, h))
            kf = cv2.cvtColor(kf, cv2.COLOR_BGR2RGB)

            pad_color = [255, 255, 255]
            if self.active_idxs is not None and self.active_pixels is not None:
                if i in self.active_idxs:
                    pad_color = [0, 0, 139]

                    # show sampled pixels
                    act_inds_mask = self.active_pixels["indices_b"] == i
                    h_inds = self.active_pixels["indices_h"][act_inds_mask]
                    w_inds = self.active_pixels["indices_w"][act_inds_mask]
                    mask = np.zeros([self.H, self.W])
                    mask[h_inds.cpu().numpy(), w_inds.cpu().numpy()] = 1
                    mask = ndimage.binary_dilation(mask, iterations=6)
                    mask = (mask * 255).astype(np.uint8)
                    mask = cv2.resize(mask, (w, h)).astype(np.bool)
                    kf[mask, :] = [0, 0, 139]

            kf = cv2.copyMakeBorder(
                kf, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=pad_color
            )
            kf = cv2.copyMakeBorder(
                kf, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            kf_vis.append(kf)

        kf_vis = np.hstack(kf_vis)
        elapsed = end_timing(start, end)
        print("Time for kf vis", elapsed)
        return kf_vis

    def get_weights(self):
        return self.sdf_map.state_dict()

    def load_weights(self, state_dict):
        self.sdf_map.load_state_dict(state_dict)

    def get_sdf_grid_pc(self, include_gt=False, mask_near_pc=False):
        # grid_pc = geometry.transform.make_3D_grid(
        #     self.grid_range, self.grid_dim, self.device  # [-1, 1]  # 200
        # )

        # Get the SDF values and the grid (x, y, z) positions and concatenate into a (dim, dim, dim, 4)
        grid_pc = self.grid_pc.reshape(self.grid_dim, self.grid_dim, self.grid_dim, 3)
        # 1. Query the SDF from instant-NGP model
        with torch.set_grad_enabled(False):
            sdf_grid = self.sdf_map(self.grid_pc)  # SDF query
        sdf_grid = sdf_grid.squeeze(dim=-1)
        dim = self.grid_dim
        sdf_grid = sdf_grid.view(dim, dim, dim)

        # Scale to [-1, 1]

        sdf_grid_pc = torch.cat((grid_pc, sdf_grid[..., None]), dim=-1)
        sdf_grid_pc = sdf_grid_pc.detach().cpu().numpy()
        return sdf_grid_pc

    def get_proximity(self, pcd, mesh, prox_dist):
        """
        Get mesh vertices close to pointcloud, useful for coloring the mesh based on proximity
        """
        start, end = start_timing()
        tree = KDTree(pcd)
        dists, _ = tree.query(mesh.vertices, k=1)
        proximity_ixs = dists < prox_dist
        elapsed = end_timing(start, end)
        # print("Time KDTree", elapsed)
        return proximity_ixs, tree

    def voxel_downsample(self, points, color=None, voxel_size=1e-3):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downsample_color = None
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(color)
        downsample_pcd = o3d.geometry.PointCloud(pcd.voxel_down_sample(voxel_size))
        downsample_points = np.array(downsample_pcd.points)
        if color is not None:
            downsample_color = np.array(downsample_pcd.colors)
        return downsample_points, downsample_color

    def mesh_rec(self, crop_mesh_with_pc=True):
        """
        Generate mesh reconstruction, returns in the object centric frame (cropped and uncropped versions)
        """

        # 1. Query the SDF from instant-NGP model
        with torch.set_grad_enabled(False):
            sdf = self.sdf_map(self.grid_pc.unsqueeze(0))  # SDF query

        sdf = sdf.squeeze(dim=-1)

        dim = self.grid_dim
        sdf = sdf.view(dim, dim, dim)

        # 2. Run marching cubes and clean-up mesh
        sdf_mesh = draw.draw_mesh(
            sdf,
            color_by="none",
            clean_mesh=True,
        )

        sdf_mesh_copy = sdf_mesh.copy()  # [-0.5, 0.5] scale
        self.latest_f_score = self.get_f_score(sdf_mesh_copy)

        if self.new_grid_dim is not None:
            self.grid_dim = self.new_grid_dim
            self.grid_pc = self.new_grid_pc
            self.new_grid_dim = None
            self.new_grid_pc = None

        # print(
        #     f"full_pc size: {self.full_pc['pc'].shape[0]}, touch_pc size: {self.full_pc['touch_pc'].shape[0]}"
        # )

        touch_points = np.full(sdf_mesh.vertices.shape[0], False)
        point_count = 0
        for sensor_name in self.sensor_list:
            point_count += len(self.full_pc[sensor_name]["points"])
        if point_count == 0:
            return (
                sdf_mesh,
                sdf_mesh_copy,
                touch_points,
                np.full(sdf_mesh.vertices.shape[0], False),
            )
        # Get full and tactile pointclouds in world frame
        pc_world = self.get_pcd(source="both")
        touch_pc_world = self.get_pcd(source="touch")
        pc_world = pc_world.point.positions.numpy()
        touch_pc_world = touch_pc_world.point.positions.numpy()

        sdf_mesh_world = sdf_mesh.copy().apply_scale(self.obj_scale_np)
        sdf_mesh_world.apply_transform(self.p_WO_W_np)

        # print("proximity for all points")
        # 4. Find all mesh vertices close to full pointcloud and crop
        dont_crop_points, _ = self.get_proximity(
            pc_world, sdf_mesh_world, self.crop_dist
        )
        face_mask = dont_crop_points[sdf_mesh.faces].any(axis=1)

        if crop_mesh_with_pc:
            sdf_mesh.update_faces(face_mask)
            sdf_mesh.remove_unreferenced_vertices()
            sdf_mesh_world.update_faces(face_mask)
            sdf_mesh_world.remove_unreferenced_vertices()

        # 5. Find all mesh vertices close to tactile pointcloud

        if len(touch_pc_world):
            # print("proximity for touch points")
            touch_points, self.touch_tree = self.get_proximity(
                touch_pc_world, sdf_mesh_world, 5e-3
            )

        # Return mesh with crop idxs
        if crop_mesh_with_pc:
            # sdf_mesh.visual.vertex_colors[~keep_ixs, 3] = 10
            return (
                sdf_mesh,
                sdf_mesh_copy,
                touch_points,
                np.full(sdf_mesh.vertices.shape[0], False),
            )
        else:
            return sdf_mesh, sdf_mesh_copy, touch_points, ~dont_crop_points

    def get_f_score(self, rec_mesh):
        """
        Compute F-score distance b/w estimated mesh and ground-truth mesh
        """
        if self.train_mode in ["slam"]:
            # Compute the offset between the tracked object pose and gt object pose
            self.object.current_pose_offset = (
                torch.inverse(self.object.object_pose_gt[self.last_track])
                @ self.object.object_pose_track[self.last_track]
            )
            self.object.current_pose_offset = (
                self.object.current_pose_offset.cpu().numpy()
            )

        # transform rec_mesh to gt frame
        rec_mesh_object = rec_mesh.copy().apply_scale(
            self.obj_scale_np
        )  # metric, in tracked pose frame
        rec_mesh_object.apply_transform(
            self.object.current_pose_offset
        )  # metric, in object centric gt frame

        (
            f_scores,
            precisions,
            recalls,
            mesh_error,
            _,
        ) = compute_f_score(
            self.gt_mesh_sampled,  # metric, in object centric gt frame
            rec_mesh_object,  # metric, in object centric gt frame
            num_mesh_samples=self.num_points_f_score,
            T=self.f_score_T,
        )

        avg_mesh_error = np.mean(mesh_error)
        return {
            "f_score": f_scores,
            "precision": precisions,
            "recall": recalls,
            "f_score_T": self.f_score_T,
            "mesh_error": avg_mesh_error,
            "time": self.current_time,
        }

    def get_pose_error(self, pose_opt_batch, pose_gt_batch, timestamp):
        """
        Compute the symmetric average 3D error between the ground-truth and estimated poses
        """

        pose_gt_batch = pose_gt_batch[-1].cpu().numpy().squeeze()
        points_gt = transform_points_np(
            copy.deepcopy(self.gt_mesh_sampled), pose_gt_batch
        )  # in world frame

        pose_opt_batch = pose_opt_batch[-1].cpu().numpy().squeeze()
        points_est = transform_points_np(
            copy.deepcopy(self.gt_mesh_sampled),
            pose_opt_batch,
        )  # in world frame

        # Convert numpy arrays to Open3D point cloud format
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(points_gt)

        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(points_est)
        # o3d.visualization.draw_geometries([pcd1, pcd2])

        avg_3d_error = average_3d_error(points_est, points_gt)
        return {"avg_3d_error": avg_3d_error, "time": timestamp}

    def get_pcd(self, source: str = None, frame_ids=None, max_points: int = 100000):
        """
        Get concatenated pointcloud in the "dynamic" object frame.
        """

        # Get the list of sensors to visualize
        viz_sensor_list = self.sensor_list
        if source is not None:
            if "vision" in source.lower():
                viz_sensor_list = [
                    x for x in viz_sensor_list if "digit" not in x.lower()
                ]
            elif "touch" in source.lower():
                viz_sensor_list = [x for x in viz_sensor_list if "digit" in x.lower()]

        pcd = o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))
        if len(viz_sensor_list):
            T_WC = self.p_WO_W_np  # current estimated pose of the object
            pcd_list, pcd_color_list = [], []
            # stack points from all sensors
            for sensor_name in viz_sensor_list:
                # loop over all keyframes (TODO find a better way to do this)

                # select elements of self.frame_id[sensor_name] less than self.last_track
                all_frame_ids = np.array(self.frame_id[sensor_name])
                if len(all_frame_ids) == 0:
                    continue
                if frame_ids is None:
                    frame_ids = all_frame_ids

                if self.incremental:
                    indices = frame_ids <= self.last_track
                    frame_ids = frame_ids[indices]
                for f in frame_ids:
                    idx = np.where(all_frame_ids == f.item())[0][0]
                    if not len(self.full_pc[sensor_name]["points"]):
                        continue
                    # get the pointcloud in the world frame from add_pc_to_memory()
                    pcs_world = self.full_pc[sensor_name]["points"][idx]
                    color = self.full_pc[sensor_name]["colors"][idx]

                    # get the tracked object pose at the keyframe
                    T_obj_i = (
                        self.object.object_pose_track[f].detach().cpu().numpy().copy()
                    )
                    T_obj_i_inv = np.linalg.inv(T_obj_i)
                    T_adjusted = T_WC @ T_obj_i_inv  # adjust w.r.t. current pose
                    pcd = transform_points_np(pcs_world, T_adjusted)

                    pcd_list.append(pcd)
                    pcd_color_list.append(color)
                # print(f"sensor {sensor_name} has {len(pcd_list)} points")
            if len(pcd_list):
                pcd = np.vstack(pcd_list)
                pcd_color = np.vstack(pcd_color_list)

                # Randomly downsample if too many points
                if len(pcd) > max_points:
                    idxs = np.random.choice(
                        np.arange(len(pcd)), max_points, replace=False
                    )
                    pcd = pcd[idxs, :]
                    pcd_color = pcd_color[idxs, :]

                pcd = o3d.t.geometry.PointCloud(o3c.Tensor(pcd.astype(np.float32)))
                pcd.point["colors"] = o3c.Tensor(pcd_color.astype(np.float32))
        return pcd

    def balance_compute_buget(self, map_loss, pose_loss):
        # compute ratio of map loss to total loss
        # if both losses are zero, then use 0.5
        if self.init_map_loss is None:
            self.init_map_loss = map_loss
        if self.init_pose_loss is None:
            self.init_pose_loss = pose_loss

        # check if both losses are zero/nan
        if map_loss in [0, np.nan] and pose_loss in [0, np.nan]:
            return
        map_loss = map_loss / self.init_map_loss
        pose_loss = pose_loss / self.init_pose_loss
        self.map_ratio = map_loss / (map_loss + pose_loss)
        self.pose_ratio = 1.0 - self.map_ratio
        # print(f"map_ratio: {self.map_ratio}, pose_ratio: {self.pose_ratio}")

    def step_map(self):
        start, end = start_timing()

        if self.map_optimizer is None:
            return 0

        sensor_losses = None

        num_iters = int(self.map_ratio * self.cfg_train.optimizer.num_iters)
        if num_iters == 0:
            num_iters += 1
        if self.map_iters == 0:
            num_iters = int(self.cfg_train.optimizer.map_init_iters)
        # print(f"num map iters: {num_iters}")

        for _ in (
            tqdm(range(num_iters), desc="Init. map")
            if num_iters > 100
            else range(num_iters)
        ):
            idxs = {}
            sensor_losses = None
            for sensor_name in self.sensor_list:
                frames = self.frames[sensor_name]
                if len(frames) == 0:
                    continue

                # Don't optimize tactile for first 10 seconds
                # if "digit" in sensor_name and self.current_time < 5:
                #     continue

                # remove un-optimized poses
                unoptimized_poses = None
                if self.incremental:
                    unoptimized_poses = np.sum(
                        np.array(frames.frame_id) > self.last_track
                    )
                # print(f"unoptimized poses: {unoptimized_poses}")
                if unoptimized_poses:
                    frames = frames[:-unoptimized_poses]

                sensor = self.sensor[sensor_name]
                loss_ratio = sensor.loss_ratio

                T_WC_batch = self.transform_to_object(frames)

                depth_batch = frames.depth_batch  # (1, H, W)
                norm_batch = (
                    None  # self.frames.normal_batch if self.do_normal else None
                )

                if len(frames) > self.window_size and self.incremental:
                    idxs[sensor_name] = self.select_keyframes(frames)
                else:
                    idxs[sensor_name] = np.arange(T_WC_batch.shape[0])
                depth_batch = depth_batch[idxs[sensor_name]]
                T_WC_select = T_WC_batch[idxs[sensor_name]]
                # print(f"batch size: {T_WC_select.shape[0]}, len(frames): {len(frames)}")

                # print(f"{sensor_name} :{idxs}")
                if not torch.nansum(depth_batch):
                    continue
                try:
                    self.sample_pts[sensor_name] = sensor.sample_points(
                        depth_batch,
                        T_WC_select,
                        norm_batch=norm_batch,
                        box_extents=(
                            self.obj_extents if "realsense" in sensor_name else None
                        ),
                        box_transform=(
                            self.p_WO_W if "realsense" in sensor_name else None
                        ),
                    )
                except RuntimeError:
                    continue

                # vision-weights (removed the downweighting part)
                dist = None
                # if "realsense" in sensor_name and self.touch_tree:
                #     vision_pc = self.sample_pts[sensor_name]["pc"].reshape((-1, 3))
                #     dist, _ = self.touch_tree.query(vision_pc.cpu().numpy(), k=1)
                #     dist = torch.from_numpy(dist).to(self.device).float()
                #     # dist[dist < 0.01] = 0.0
                #     dist = (dist - dist.min()) / (dist.max() - dist.min())

                (
                    total_loss,
                    losses,
                    active_loss_approx,
                    frame_avg_loss,
                ) = self.sdf_eval_and_loss(
                    self.sample_pts,
                    sensor_name,
                    vision_weights=dist,
                    do_avg_loss=True,
                )

                if sensor_losses is None:
                    sensor_losses = loss_ratio * total_loss
                else:
                    sensor_losses += loss_ratio * total_loss
                frames.frame_avg_losses[idxs[sensor_name]] = frame_avg_loss
                self.steps_since_frame[sensor_name] += 1
                # print(f"{sensor_name} : {self.steps_since_frame[sensor_name]}")

            if sensor_losses is not None and self.map_optimizer is not None:
                if torch.isnan(sensor_losses):
                    print("Error: SDF NaN loss")
                sensor_losses.backward()
                self.map_optimizer.step()
                self.map_optimizer.zero_grad()
                for param_group in self.map_optimizer.param_groups:
                    params = param_group["params"]
                    for param in params:
                        param.grad = None
                sensor_losses = sensor_losses.item()

        # Save the f_score distance stats
        if self.latest_f_score is not None:
            self.save_stats["map"]["errors"].append(self.latest_f_score)
            self.save_stats["map"]["timing"].append(end_timing(start, end) / 1000.0)

        self.map_iters += 1
        return sensor_losses

    def step_pose(self):
        start, end = start_timing()

        if self.map_optimizer is None:
            self.frozen_sdf_map = copy.deepcopy(self.sdf_map)

        if self.pose_optimizer is None:
            return 0

        num_iters = int(self.pose_ratio * self.pose_optimizer.num_iters)
        if num_iters == 0:
            num_iters += 1
        # print(f"num pose iters: {num_iters}")

        if self.train_mode in ["map", "slam"]:
            with torch.set_grad_enabled(False):
                sdf = self.sdf_map(self.grid_pc.unsqueeze(0))  # SDF query
            sdf = sdf.squeeze(dim=-1)
            dim = self.grid_dim
            sdf = sdf.view(dim, dim, dim)
            self.frozen_sdf_map.updateSDFGrid(sdf)

        for _ in range(num_iters):
            self.pose_optimizer.addSDF(self.frozen_sdf_map)
            all_frames = []
            opt_sensors = []
            for sensor_name in self.sensor_list:
                opt_sensors.append(sensor_name)
                frames = self.frames[sensor_name]
                T_WC_batch = frames.T_WC_batch.clone()  # (N, 4, 4)

                if len(frames) > self.pose_window_size and self.incremental:
                    idxs = np.arange(
                        len(frames) - self.pose_window_size, len(frames)
                    )  # sequential keyframes
                else:
                    idxs = np.arange(T_WC_batch.shape[0])
                # idxs = [idxs[-1]]  # Only latest frame

                # Get ground-truth and tracked poses
                depth_batch = frames.depth_batch[idxs]  # (N, H, W)
                frame_id_batch = frames.frame_id  # (N,)

                frame_id_batch = list(frame_id_batch[idxs])
                # print(f"{sensor_name}: {idxs}, {frame_id_batch}")

                all_frames += frame_id_batch

                # Get sensor poses in global frame
                pose_batch = T_WC_batch[idxs]  # [len(idxs), 4, 4]
                pose_batch = th.SE3(tensor=pose_batch.clone()[:, :3, :])

                self.pose_optimizer.addVariables(
                    depth_batch, pose_batch, frame_id_batch, sensor_name
                )

            # initialize latest pose with previous newest pose
            prev_pose = self.object.object_pose_track[self.last_track]
            self.last_track = frame_id_batch[-1]
            self.object.object_pose_track[self.last_track] = prev_pose

            # if "digit" in sensor_name:
            # for now, skip ICP loss in offline mode
            if (
                self.cfg_pose.second_order.icp
                and len(frame_id_batch) > 1
                and self.incremental
            ):
                frame_t_1 = self.get_pcd(
                    source=None, frame_ids=torch.tensor(frame_id_batch[-2])
                )  # t - 1 frame visuo-tactile pointcloud
                frame_t = self.get_pcd(
                    source=None, frame_ids=torch.tensor([frame_id_batch[-1]])
                )  # t frame visuo-tactile pointcloud

                self.pose_optimizer.addPointCloud(
                    frame_t_1.point.positions.numpy(), frame_t.point.positions.numpy()
                )

            all_frames = torch.unique(torch.tensor(all_frames, device=self.device))
            if len(all_frames) > self.pose_window_size and self.incremental:
                return 0.0

            # find if they are in the tracked_frame set, else init with the closest frame
            init_frames = all_frames.clone()
            f = init_frames.cpu().numpy()
            t = self.tracked_frames
            match_idx = np.argmin(
                np.abs(f - t[:, None]), axis=0
            )  # which element in tracked_frames is closest to init_frames

            # print(match_idx)
            init_frames = t[match_idx]

            object_pose_batch = th.SE3(
                tensor=self.object.object_pose_track[init_frames][:, :3, :].clone()
            )
            self.pose_optimizer.addPoses(object_pose_batch)
            

            try:
                pc, pose_loss = self.pose_optimizer.optimize_pose_theseus(
                    opt_sensors, self.pose_optimizer.lm_iters
                )
            except Exception as e:
                print(f"Pose optimization failed for {self.current_time}: {e}")
                return 0.0
            if pc is not None:
                self.viz_pc(pc, None, sensor_name)

            updated_pose_batch, pose_list = self.pose_optimizer.getOptimizedPoses(
                matrix=True
            )

            if self.incremental:
                object_pose_gt = self.object.object_pose_gt[[self.last_track]]
            else:
                object_pose_gt = self.object.object_pose_gt[pose_list]

            object_pose_gt.to(dtype=updated_pose_batch.dtype)

            self.object.object_pose_track[pose_list] = updated_pose_batch.float()

            if self.incremental:
                self.tracked_frames = np.append(self.tracked_frames, self.last_track)
                self.tracked_frames = np.unique(self.tracked_frames)
            else:
                self.tracked_frames = pose_list.cpu().numpy()
            self.update_scene_properties(self.last_track)

        self.save_stats["pose"]["timing"].append(end_timing(start, end) / 1000.0)

        # avg. 3d error b/w optimized and ground-truth poses
        pose_error = self.get_pose_error(
            updated_pose_batch, object_pose_gt, self.current_time
        )
        self.save_stats["pose"]["errors"].append(pose_error)
        
        # save optimized pose and ground truth pose
        self.save_stats["pose"]["gt_pose"].append(
            object_pose_gt[-1].cpu().numpy().squeeze()
        )
        self.save_stats["pose"]["opt_pose"].append(
            updated_pose_batch[-1].cpu().numpy().squeeze()
        )

        return pose_loss

    def viz_pc(self, pc, sdf, sensor):
        # Plot and debug the pointcloud
        self.render_samples["pc"] = pc.view(-1, 3).detach().cpu().numpy()

        if len(self.render_samples["pc"]) > 1000:
            idxs = np.random.choice(
                np.arange(len(self.render_samples["pc"])),
                1000,
                replace=False,
            )
            self.render_samples["pc"] = self.render_samples["pc"][idxs, :]

        if sdf is not None:
            self.render_samples["sdf"] = sdf[idxs].detach().cpu().numpy()
        else:
            self.render_samples["sdf"] = sdf
