# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Plot group statistics for a sweep of neuralfeels experiments
Usage: python neuralfeels/eval/group_plot.py log_path=<LOG_PATH> # e.g. multirun/2023-07-31/14-27-43 
"""

import os

import git
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from neuralfeels.viz.plot_metrics import (
    avg_map_error_over_time,
    avg_map_error_per_experiment,
    avg_map_error_per_modality,
    avg_map_error_per_object,
    avg_pose_error_over_time,
    avg_pose_error_per_camera_placement,
    avg_pose_error_per_experiment,
    avg_pose_error_per_modality,
    avg_pose_error_per_object,
    avg_pose_error_per_optimizer,
    avg_pose_error_per_shape_res,
    avg_precision_over_time,
    avg_recall_over_time,
    avg_timing_per_modality,
    avg_timing_per_optimizer,
    draw_map_error,
    draw_pose_error,
    get_dataframe,
    map_error_vs_thresh,
    success_failure_stats,
)

root = git.Repo(".", search_parent_directories=True).working_tree_dir

from pathlib import Path


@hydra.main(version_base=None, config_path="config/", config_name="group_error")
def main(cfg: DictConfig) -> None:
    log_path = os.path.join(root, cfg.log_path)
    if log_path[-1] == "/":
        log_path = log_path[:-1]
    all_expts = []
    for path in Path(log_path).rglob("stats.pkl"):
        expt_path = str(path.parent).replace(log_path + "/", "")
        all_expts.append(expt_path)
    which_f_score = cfg.which_f_score
    print(f"Found {len(all_expts)} experiments in {log_path}: {all_expts}")
    df_combined = get_dataframe(all_expts, log_path, which_f_score)

    # assert len(df_combined["slam_mode"].unique()) == 1  # only one slam_mode per plot
    slam_mode = df_combined["slam_mode"].unique()[0]

    avg_timing_per_optimizer(df_combined, log_path)
    avg_timing_per_modality(df_combined, log_path)

    if slam_mode in ["pose", "slam"]:
        avg_pose_error_over_time(df_combined, log_path)
        avg_pose_error_per_modality(df_combined, log_path)
        avg_pose_error_per_optimizer(df_combined, log_path)
        avg_pose_error_per_object(df_combined, log_path)
        avg_pose_error_per_camera_placement(df_combined, log_path)
        success_failure_stats(df_combined)
    if slam_mode in ["map", "slam"]:
        avg_map_error_over_time(df_combined, log_path)
        avg_precision_over_time(df_combined, log_path)
        avg_recall_over_time(df_combined, log_path)
        avg_map_error_per_modality(df_combined, log_path)
        avg_map_error_per_object(df_combined, log_path)
        map_error_vs_thresh(all_expts, log_path)

    if slam_mode in ["pose", "slam"]:
        avg_pose_error_per_experiment(df_combined, log_path)
        avg_pose_error_per_shape_res(df_combined, log_path)
    if slam_mode in ["map", "slam"]:
        avg_map_error_per_experiment(df_combined, log_path)
    if cfg.individual:
        print("Drawing individual plots")
        for expt_path in tqdm(all_expts):
            if "map" in expt_path or "slam" in expt_path:
                draw_map_error(expt_path=expt_path)
            if "pose" in expt_path or "slam" in expt_path:
                draw_pose_error(expt_path=expt_path, slam_mode=slam_mode)
    print(f"All outputs saved at {log_path}")


if __name__ == "__main__":
    main()
