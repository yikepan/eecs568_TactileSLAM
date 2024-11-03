# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Plot graph of pose error v.s. noise for a sweep of neuralfeels experiments
Usage: python neuralfeels/eval/noise_plot.py log_path=<LOG_PATH> # e.g. multirun/2023-07-31/14-27-43 
"""

import os

import git
import hydra
from omegaconf import DictConfig

from neuralfeels.viz.plot_metrics import (
    pose_error_vs_occlusion,
    pose_errors_vs_camera_frustums,
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

    pose_errors_vs_camera_frustums(all_expts, log_path)
    pose_error_vs_occlusion(all_expts, log_path)
    print(f"All outputs saved at {log_path}")


if __name__ == "__main__":
    main()
