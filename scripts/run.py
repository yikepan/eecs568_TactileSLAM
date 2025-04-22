# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Entrypoint python script for neuralfeels

import gc
import os
import sys
import traceback
from typing import TYPE_CHECKING

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pyvirtualdisplay import Display
from termcolor import cprint

if TYPE_CHECKING:
    from neuralfeels.modules.trainer import Trainer


class OptionalDisplay:
    def __init__(self, size=(1900, 1084), use_xauth=True, active=False):
        self.display = None
        if active:
            self.display = Display(size=size, use_xauth=use_xauth)

    def __enter__(self):
        if self.display is not None:
            self.display.__enter__()
            print(f"Display created at :{self.display.display}.")

    def __exit__(self, *args, **kwargs):
        if self.display is not None:
            self.display.__exit__()


def _load_frames_incremental(trainer: "Trainer", t):
    # lazy imports for tinycudann compatibility issues in cluster
    from neuralfeels.modules.misc import print_once

    kf_set = {sensor: None for sensor in trainer.sensor_list}

    trainer.update_current_time()
    add_new_frame = True if t == 0 else trainer.check_keyframe_latest()

    end_all = False
    if add_new_frame:
        new_frame_id = trainer.get_latest_frame_id()

        digit_poses = trainer.allegro.get_fk(idx=new_frame_id)
        end_all = trainer.check_end(new_frame_id)

        if end_all:
            if not os.path.exists(f"./visualizer/{trainer.cfg_data.object}.mp4"):
                print_once("******End of sensor stream******")
            return kf_set, end_all

        trainer.update_scene_properties(new_frame_id)

        if t == 0:
            trainer.init_first_pose(digit_poses)

        added_frame = False
        for sensor_name in trainer.sensor_list:
            n_keyframes_start = trainer.n_keyframes[sensor_name]

            if "digit" in sensor_name:
                frame_data = trainer.sensor[sensor_name].get_frame_data(
                    new_frame_id,
                    digit_poses[sensor_name],
                    msg_data=None,
                )
            else:
                frame_data = trainer.sensor[sensor_name].get_frame_data(
                    new_frame_id,
                    digit_poses,
                    trainer.latest_render_depth[sensor_name],
                    msg_data=None,
                )

            added_frame = trainer.add_frame(frame_data)
            if t == 0:
                trainer.prev_kf_time = trainer.tot_step_time

            # kf_set thumbnails for visualizer
            if trainer.n_keyframes[sensor_name] - n_keyframes_start:
                new_kf = trainer.frames[sensor_name].im_batch_np[-1]
                h = int(new_kf.shape[0] / 6)
                w = int(new_kf.shape[1] / 6)
                try:
                    kf_set[sensor_name] = cv2.resize(new_kf, (w, h))
                except:
                    # print("Error in resizing keyframe image")
                    kf_set[sensor_name] = new_kf

    if add_new_frame and added_frame:
        trainer.last_is_keyframe = False

    return kf_set, end_all


def optim_iter(trainer: "Trainer", t, start_optimize=True):
    # lazy imports for tinycudann compatibility issues in cluster
    from neuralfeels.modules.misc import gpu_usage_check

    if trainer.incremental:
        kf_set, end_all = _load_frames_incremental(trainer, t)
    else:
        kf_set = {sensor: None for sensor in trainer.sensor_list}
        end_all = False

    status = ""
    # optimization step---------------------------------------------
    if start_optimize:
        # Run map and pose optimization sequentially
        pose_loss = trainer.step_pose()
        map_loss = trainer.step_map()

        # Store losses
        map_loss, pose_loss = float(map_loss or 0.0), float(pose_loss or 0.0)
        pose_stats, map_stats = trainer.save_stats["pose"], trainer.save_stats["map"]
        pose_error_dict, map_error_dict = pose_stats["errors"], map_stats["errors"]
        pose_time_dict, map_time_dict = pose_stats["timing"], map_stats["timing"]
        pose_time, pose_errors = 0.0, 0.0
        map_time, map_errors, f_score_T = 0.0, 0.0, 0
        if len(map_error_dict) > 0:
            map_time, map_errors, f_score_T = (
                map_time_dict[-1],
                map_error_dict[-1]["f_score"][trainer.which_f_score],
                map_error_dict[-1]["f_score_T"][trainer.which_f_score],
            )
        if len(pose_error_dict) > 0:
            pose_time, pose_errors = (
                pose_time_dict[-1],
                pose_error_dict[-1]["avg_3d_error"],
            )

        # retrieve the next frame based on optimization time
        trainer.tot_step_time += (map_time + pose_time) * (t > 0)

        # Print useful information
        status = f"Map time: {map_time:.2f} s, Pose time: {pose_time:.2f} s, Total: {trainer.tot_step_time:.2f} s, Dataset: {trainer.current_time:.2f} s\n"
        status = (
            "".join(status)
            + f"Pose err [{pose_errors*1000:.2f} mm] Map err (< {f_score_T*1000:.2f} mm): [{map_errors:.2f}]"
        )
    else:
        print("Waiting for visualizer..")

    trainer.get_latest_depth_renders()
    gpu_usage_check()
    
    # Check memory usage
    available, total = torch.cuda.mem_get_info("cuda:0")
    print(f"Optim_iter GPU memory usage: {available / (1024**3):.2f} GB available out of {total / (1024**3):.2f} GB total")
    
    # Visualize current SE(3) pose
    trainer.visualize_pose()
    
    return status, kf_set, end_all


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main function to run neuralfeels

    Args:
        cfg (DictConfig): Hydra configuration
    """
    gpu_id = cfg.gpu_id
    torch.set_default_device(f"cuda:{gpu_id}")
    cprint(f"Using GPU: {gpu_id}", color="yellow")
    try:
        import open3d.visualization.gui as gui

        # lazy imports to avoid tinycudann errors when launching locally for a
        # different architecture
        from neuralfeels.modules.trainer import Trainer
        from neuralfeels.viz import neuralfeels_gui

        seed = cfg.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        with OptionalDisplay(
            size=(3840, 1644), use_xauth=True, active=cfg.create_display
        ):
            tac_slam_trainer = Trainer(cfg=cfg, gpu_id=gpu_id, ros_node=None)
            # open3d vis window
            app = gui.Application.instance
            app.initialize()
            mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
            size_ratio = 0.4  # scaling ratio w.r.t. tkinter resolution
            w = neuralfeels_gui.GUI(
                tac_slam_trainer, optim_iter, mono, size_ratio, cfg.profile
            )
            app.run()
        w.save_data()  # save all the images, meshes, plots, etc.
        w.visualize_optimized_pose()
        # clear memory
        gc.collect()
        torch.cuda.empty_cache()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
