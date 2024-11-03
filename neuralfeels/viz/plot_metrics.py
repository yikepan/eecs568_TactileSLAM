# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Plotting utilities for neuralfeels evaluation

import matplotlib

matplotlib.use("Agg")
import functools
import os
import re

import dill as pickle
import git
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import ttest_ind
from termcolor import cprint

from neuralfeels.datasets import sdf_util
from neuralfeels.geometry import transform

root = git.Repo(".", search_parent_directories=True).working_tree_dir

from neuralfeels.viz.plot_utils import smooth_data

# https://seaborn.pydata.org/tutorial/color_palettes.html
colorblind_pal = sns.color_palette("colorblind")
my_pal = {
    "vision": colorblind_pal[0],
    "touch": colorblind_pal[-1],
    "vision + touch": colorblind_pal[-2],
    "precision": colorblind_pal[1],
    "recall": colorblind_pal[2],
    "map_timings": colorblind_pal[6],
    "pose_timings": colorblind_pal[7],
}


def load_pose_stats(stats, i=0):
    """
    Separates the pose error stats into a dataframe and smooths the data.
    """
    pose_error_stats = stats["pose"]["errors"]
    pose_timing_stats = stats["pose"]["timing"]

    if len(pose_error_stats) == 0:
        return pd.DataFrame({"avg_3d_error": [], "pose_timestamps": []})
    avg_3d_error = np.array(
        [
            pose_error_stats[i]["avg_3d_error"] * 1000.0
            for i in range(len(pose_error_stats))
        ]
    )

    pose_timestamps = np.array(
        [pose_error_stats[i]["time"] for i in range(len(pose_error_stats))]
    )
    pose_timings = np.array(
        [pose_timing_stats[i] for i in range(len(pose_timing_stats))]
    )
    # Ignore the first 5 seconds of each experiment, because we initialize at ground-truth for consistency
    avg_3d_error[pose_timestamps < 5] = np.nan
    # remove outliers due to ground-truth failures
    avg_3d_error[avg_3d_error > 50] = np.nan

    return pd.DataFrame(
        {
            "avg_3d_error": [np.array(avg_3d_error)],
            "pose_timestamps": [np.array(pose_timestamps)],
            "pose_timings": [np.array(pose_timings)],
        },
        index=[i],
    )


def load_map_stats(stats, i=0, which_f_score=10):
    """
    Separates the map error stats into a dataframe and smooths the data.
    """
    map_error_stats = stats["map"]["errors"]
    map_timing_stats = stats["map"]["timing"]
    if len(map_error_stats) == 0:
        return pd.DataFrame(
            {
                "f_score": [],
                "precision": [],
                "recall": [],
                "mesh_error": [],
                "f_score_T": [],
                "map_timestamps": [],
            }
        )

    f_score = np.array(
        [
            map_error_stats[i]["f_score"][which_f_score]
            for i in range(len(map_error_stats))
        ]
    )
    precision = np.array(
        [
            map_error_stats[i]["precision"][which_f_score]
            for i in range(len(map_error_stats))
        ]
    )
    recall = np.array(
        [
            map_error_stats[i]["recall"][which_f_score]
            for i in range(len(map_error_stats))
        ]
    )

    # check if "mesh_error" is in map_error_stats
    if "mesh_error" not in map_error_stats[0]:
        mesh_error = np.array([0.0 for i in range(len(map_error_stats))])
    else:
        mesh_error = np.array(
            [map_error_stats[i]["mesh_error"] for i in range(len(map_error_stats))]
        )

    map_timestamps = np.array(
        [map_error_stats[i]["time"] for i in range(len(map_error_stats))]
    )
    map_timings = np.array([map_timing_stats[i] for i in range(len(map_timing_stats))])
    f_score_T = [1000 * map_error_stats[0]["f_score_T"][which_f_score]]

    return pd.DataFrame(
        {
            "f_score": [np.array(f_score)],
            "precision": [np.array(precision)],
            "recall": [np.array(recall)],
            "mesh_error": [np.array(mesh_error)],
            "map_timestamps": [np.array(map_timestamps)],
            "map_timings": [np.array(map_timings)],
            "f_score_T": f_score_T,
        },
        index=[i],
    )


def load_metadata(expt_path, stats, i=0):
    """
    Breaks down the expt_path into object, expt_number, modality, slam_mode and returns a dataframe.
    Note: expt_path should be arranged as: object/expt_number/modality/slam_mode
    """
    parts = expt_path.split("/")
    # get elemnts before underscore
    parts[3] = parts[3].split("_")[0]

    optimizer = None
    if "optimizer" in stats:
        optimizer = stats["optimizer"]
    if "cameras" in stats:
        cameras = stats["cameras"]
    if not cameras:
        cameras = "None"

    if parts[2] == "vitac":
        parts[2] = "vision + touch"
    elif parts[2] == "tactile":
        parts[2] = "touch"

    return pd.DataFrame(
        {
            "object": parts[0],
            "expt": parts[1],
            "modality": parts[2],
            "slam_mode": parts[3],
            "optimizer": optimizer,
            "cameras": cameras,
        },
        index=[i],
    )


def get_dataframe(expt_paths, root_path, which_f_score=10):
    """
    Loads the pickled stats.pkl file for each expt_path and returns a dataframe with the following columns:
    object, expt_number, modality, slam_mode, optimizer, cameras, avg_3d_error, pose_timestamps, f_score,
    precision, recall, map_timestamps, f_score_T
    """
    df_combined = pd.DataFrame()
    # split each expt_path into object, expt_number, modality, slam_mode
    for i, expt_path in enumerate(expt_paths):
        pkl_path = os.path.join(root_path, expt_path, "stats.pkl")
        with open(pkl_path, "rb") as p:
            stats = pickle.load(p)

        metadata_df = load_metadata(expt_path, stats, i)
        pose_df = load_pose_stats(stats, i)
        map_df = load_map_stats(stats, i, which_f_score=which_f_score)
        df_combined = pd.concat(
            [df_combined, pd.concat([metadata_df, pose_df, map_df], axis=1)], axis=0
        )
    return df_combined


def avg_pose_error_per_modality(df, log_path, width=0.8):
    """
    Groups all experiments by modality (vision/vitac/tac) and plots the average avg_3d_error for each modality.
    """
    save_path = os.path.join(log_path, f"avg_pose_error_per_modality.pdf")

    plot_df = df.copy()

    # remove modality = touch
    # plot_df = plot_df[plot_df["modality"] != "touch"]

    # count number of modalities in plot_df
    mod_count = plot_df["modality"].unique()

    # Sort the dataframe by modality
    plot_df = plot_df.sort_values("modality")

    # set color palette based on the modalities in plot_df
    colors = {}
    for modality in mod_count:
        colors[modality] = my_pal[modality]

    # only save avg_3d_error, pose_timestamps and modality
    plot_df = plot_df[["modality", "avg_3d_error", "pose_timestamps"]]

    final_df = plot_df.copy()
    # extract the final avg_3d_error for each modality
    final_df["avg_3d_error"] = final_df["avg_3d_error"].apply(
        lambda x: x[-1] if len(x) > 0 else np.nan
    )
    # print the final avg_3d_error for the vision + touch modality
    print(
        f"Final avg_3d_error for vision + touch: {final_df[final_df['modality'] == 'vision + touch']['avg_3d_error'].mean()}"
    )
    # check if vision and vision + touch are in the dataframe
    if "vision" in mod_count and "vision + touch" in mod_count:
        # run p_test between vision and vision + touch (paired sample t-test)
        t_test_df = plot_df.copy()
        # take mean avg 3d error for each row
        t_test_df["avg_3d_error"] = t_test_df["avg_3d_error"].apply(
            lambda x: np.nanmedian(x)
        )
        vision_data = t_test_df[t_test_df["modality"] == "vision"]["avg_3d_error"]
        vision_touch_data = t_test_df[t_test_df["modality"] == "vision + touch"][
            "avg_3d_error"
        ]
        t_stat, p_value = ttest_ind(vision_data, vision_touch_data)
        print(f"T-statistic: {t_stat}, p-value: {p_value}")

    # breakpoint()

    # explode pairwise the avg_3d_error and pose_timestamps
    plot_df = plot_df.explode(["avg_3d_error", "pose_timestamps"]).reset_index(
        drop=True
    )

    # keep only modality and avg_3d_error
    plot_df = plot_df[["modality", "avg_3d_error"]]

    boxprops, whiskerprops, capprops, medianprops = {}, {}, {}, {}
    if "sim" in log_path:
        boxprops = whiskerprops = capprops = medianprops = {
            "linestyle": "--",
            "linewidth": 2,
        }

    # Create a boxplot of the avg_3d_error for each modality
    fig, ax = plt.subplots()
    sns.boxplot(
        x="modality",
        y="avg_3d_error",
        palette=colors,
        whis=1.0,
        data=plot_df,
        width=width,
        linewidth=2,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        showfliers=False,
        dodge=True,
        ax=ax,
    )

    means = plot_df.groupby("modality")["avg_3d_error"].mean()
    print(f"Average pose error per modality:\n {means}")

    if len(mod_count) > 1:
        # percentage improvement with vision + touch
        print(
            f"Percentage improvement with vision + touch: {100*(means['vision'] - means['vision + touch'])/means['vision']}"
        )

    # ax.plot(range(len(means)), means, color="red", marker="s", linewidth=0)

    ax.set_ylabel("Avg. pose error (mm)", fontsize=25)
    ax.tick_params(axis="y")
    plt.xlabel("", fontsize=25)

    # y-ticks should be integers
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    y_lim = (0, 8)
    y_plot_ticks = np.arange(y_lim[0], y_lim[1] + 1, 2)

    # pose error plots
    # y_lim = (0, 4.25)
    # y_plot_ticks = np.arange(y_lim[0], y_lim[1] + 1, 2)
    plt.yticks(y_plot_ticks)

    ax.tick_params(axis="y", labelsize=25)
    ax.tick_params(axis="x", labelsize=25)

    max_x = len(mod_count) - 1 + 0.6 * width
    min_x = -0.6 * width
    plt.axvspan(min_x, max_x, facecolor="gray", alpha=0.05)
    ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
    ax.set_xlim(min_x, max_x)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0.2)
    print(f"Saved {save_path}")


def avg_pose_error_per_modality_all_three(df, log_path, width=0.8):
    """
    Special version of avg_pose_error_per_modality for the paper to plot (a) vision, (b) vision + touch, (c) touch
    """
    save_path = os.path.join(log_path, f"avg_pose_error_per_modality.pdf")

    plot_df = df.copy()

    # remove modality = touch
    # plot_df = plot_df[plot_df["modality"] != "touch"]

    # count number of modalities in plot_df
    mod_count = plot_df["modality"].unique()

    # Sort the dataframe by modality
    plot_df = plot_df.sort_values("modality")

    # set color palette based on the modalities in plot_df
    colors = {}
    for modality in mod_count:
        colors[modality] = my_pal[modality]

    # only save avg_3d_error, pose_timestamps and modality
    plot_df = plot_df[["modality", "avg_3d_error", "pose_timestamps"]]
    # explode pairwise the avg_3d_error and pose_timestamps
    plot_df = plot_df.explode(["avg_3d_error", "pose_timestamps"]).reset_index(
        drop=True
    )

    # keep only modality and avg_3d_error
    plot_df = plot_df[["modality", "avg_3d_error"]]

    boxprops, whiskerprops, capprops, medianprops = {}, {}, {}, {}
    if "sim" in log_path:
        boxprops = whiskerprops = capprops = medianprops = {
            "linestyle": "--",
            "linewidth": 2,
        }

    # Create a boxplot of the avg_3d_error for each modality
    fig, ax = plt.subplots()
    sns.boxplot(
        x="modality",
        y="avg_3d_error",
        palette=colors,
        whis=0.25,
        data=plot_df,
        width=width,
        linewidth=2,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        showfliers=False,
        dodge=True,
        ax=ax,
    )

    # Set the y-axis to log scale
    ax.set_yscale("log")

    means = plot_df.groupby("modality")["avg_3d_error"].mean()
    print(f"Average pose error per modality:\n {means}")

    if len(mod_count) > 1:
        # percentage improvement with vision + touch
        print(
            f"Percentage improvement with vision + touch: {100*(means['vision'] - means['vision + touch'])/means['vision']}"
        )

    # ax.plot(range(len(means)), means, color="red", marker="s", linewidth=0)

    ax.set_ylabel("Avg. pose error (mm)", fontsize=25)
    plt.xlabel("", fontsize=25)

    # y ticks
    y_plot_ticks = [1, 5, 10, 20]
    plt.yticks(y_plot_ticks)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    y_lim = [0, 25]

    ax.set_ylim(bottom=0.7, top=y_lim[1])
    # ax.set_yscale("log")

    ax.tick_params(axis="y", labelsize=25)
    ax.tick_params(axis="x", labelsize=25)

    max_x = len(mod_count) - 1 + 0.6 * width
    min_x = -0.6 * width
    plt.axvspan(min_x, max_x, facecolor="gray", alpha=0.05)
    ax.set_xlim(min_x, max_x)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0.2)
    print(f"Saved {save_path}")


def avg_map_error_per_modality(df, log_path, width=0.8):
    """
    Groups all experiments by modality (vision/vitac/tac) and plots the average f_score distance for each modality.
    """

    save_path = os.path.join(log_path, f"avg_map_error_per_modality.pdf")

    plot_df = df.copy()

    # replace all nans in df with 0.0
    f_score_T = int(plot_df["f_score_T"].unique()[0])

    # count number of modalities in plot_df
    mod_count = plot_df["modality"].unique()

    # Sort the dataframe by modality
    plot_df = plot_df.sort_values("modality")

    # set color palette based on the modalities in plot_df
    colors = {}
    for modality in mod_count:
        colors[modality] = my_pal[modality]

    # only plot the final f-score
    plot_df["mesh_error"] = plot_df["mesh_error"].apply(lambda x: np.nanmean(x))
    plot_df["mesh_error"] = plot_df["mesh_error"].astype("float")
    # get avg mesh_error for each modality and print
    means = plot_df.groupby("modality")["mesh_error"].median()
    print(f"Average mesh error per modality:\n {means * 1000.0} mm")

    # only plot the final f-score
    plot_df["f_score"] = plot_df["f_score"].apply(lambda x: x[-1])
    plot_df = plot_df.explode("f_score")
    plot_df["f_score"] = plot_df["f_score"].astype("float")

    # check if vision and vision + touch are in the dataframe
    if "vision" in mod_count and "vision + touch" in mod_count:
        # run p_test between vision and vision + touch (paired sample t-test)
        t_test_df = plot_df.copy()
        # take mean avg 3d error for each row
        t_test_df["f_score"] = t_test_df["f_score"].apply(lambda x: np.nanmedian(x))
        vision_data = t_test_df[t_test_df["modality"] == "vision"]["f_score"]
        vision_touch_data = t_test_df[t_test_df["modality"] == "vision + touch"][
            "f_score"
        ]
        t_stat, p_value = ttest_ind(vision_data, vision_touch_data)
        print(f"f_score T-statistic: {t_stat}, p-value: {p_value}")

    means = plot_df.groupby("modality")["f_score"].mean()
    print(f"Average map error per modality:\n {means}")

    # percentage improvement with vision + touch
    # first check if vision and vision + touch are in the dataframe
    if "vision" in mod_count and "vision + touch" in mod_count:
        print(
            f"Percentage improvement with vision + touch: {100*(means['vision + touch'] - means['vision'])/means['vision']}"
        )

    # Create a boxplot of the f_score for each modality
    fig, ax = plt.subplots()

    boxprops, whiskerprops, capprops, medianprops = {}, {}, {}, {}
    if "sim" in log_path:
        boxprops = whiskerprops = capprops = medianprops = {
            "linestyle": "--",
            "linewidth": 2,
        }

    sns.boxplot(
        x="modality",
        y="f_score",
        palette=colors,
        whis=1.0,
        data=plot_df,
        width=width,
        linewidth=2,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        showfliers=False,
        # gap=0.1,
        dodge=True,
        ax=ax,
    )

    ax.set_ylabel(f"Avg. map F-score (< {f_score_T} mm)", fontsize=25)

    plt.xlabel("", fontsize=25)

    ax.tick_params(axis="y", labelsize=25)
    ax.tick_params(axis="x", labelsize=25)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0, top=1.02)
    y_plot_ticks = np.arange(0, 1.1, 0.2)
    plt.yticks(y_plot_ticks)

    max_x = len(mod_count) - 1 + 0.6 * width
    min_x = -0.6 * width
    plt.axvspan(min_x, max_x, facecolor="gray", alpha=0.05)

    ax.set_xlim(min_x, max_x)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0.2)
    print(f"Saved {save_path}")


def avg_map_error_over_time(df, log_path):
    """
    Groups all experiments by modality (vision/vitac/tac) and plots the average f_score distance for each modality.
    """
    save_path = os.path.join(log_path, f"avg_map_error_over_time.pdf")

    plot_df = df.copy()

    f_score_T = int(plot_df["f_score_T"].unique()[0])

    # count number of modalities in plot_df
    mod_count = plot_df["modality"].unique()

    # Sort the dataframe by modality
    plot_df = plot_df.sort_values("modality")

    # set color palette based on the modalities in plot_df
    colors = {}
    for modality in mod_count:
        colors[modality] = my_pal[modality]

    # only store the f_score and map_timestamps of each modality
    plot_df = plot_df[["modality", "f_score", "map_timestamps"]]
    # Create a boxplot of the f_score for each modality

    # interpolate each f_score to the same length
    plot_df["f_score"] = plot_df["f_score"].apply(
        lambda x: np.interp(np.arange(0, 1, 0.005), np.linspace(0, 1, len(x)), x)
    )
    plot_df["map_timestamps"] = plot_df["map_timestamps"].apply(
        lambda x: np.interp(np.arange(0, 1, 0.005), np.linspace(0, 1, len(x)), x)
    )

    fig, ax = plt.subplots()

    # for each experiment in vision + touch, compute the difference between plot_df.iloc[i]['f_score'][-1] and plot_df.iloc[0]['f_score'][0]
    shape_est_diff = {"positive": 0, "negative": 0}
    for i in range(len(plot_df)):
        # if modality is vision + touch
        if plot_df.iloc[i]["modality"] == "vision + touch":
            avg_f_score_in_end = np.mean(
                plot_df.iloc[i]["f_score"][int(0.5 * len(plot_df.iloc[i]["f_score"])) :]
            )
            avg_f_score_in_start = np.mean(
                plot_df.iloc[i]["f_score"][: int(0.5 * len(plot_df.iloc[i]["f_score"]))]
            )
            f_score_diff = avg_f_score_in_end - avg_f_score_in_start
            if f_score_diff > 0:
                shape_est_diff["positive"] += 1
            else:
                shape_est_diff["negative"] += 1
                # print the object name and expt name of the i-th experiment from df
                print(
                    f"DIVERGING Object: {df.iloc[i]['object']}, Expt: {df.iloc[i]['expt']}"
                )

    print(
        f"Number of positive shape estimation differences: {shape_est_diff['positive']} fraction: {shape_est_diff['positive']/(shape_est_diff['positive'] + shape_est_diff['negative'])}",
        f"\n Number of negative shape estimation differences: {shape_est_diff['negative']}, fraction: {shape_est_diff['negative']/(shape_est_diff['positive'] + shape_est_diff['negative'])}",
    )

    # for each modality, append the row of f_score and map_timestamps to a numpy array
    fscores_mean, fscores_std = {}, {}
    max_timestamp = 0
    for modality in mod_count:
        mod_df = plot_df[plot_df["modality"] == modality]
        fscores_mean[modality] = np.array(mod_df["f_score"].tolist())
        fscores_mean[modality] = np.mean(fscores_mean[modality], axis=0)
        fscores_std[modality] = np.array(mod_df["f_score"].tolist())
        fscores_std[modality] = np.std(fscores_std[modality], axis=0)
        # timestamps.append(np.array(mod_df["map_timestamps"].tolist()))

        # find max_timestamp for modality
        all_expt_timestamps = plot_df[plot_df["modality"] == modality]["map_timestamps"]
        max_t = np.max([np.max(timestamps) for timestamps in all_expt_timestamps])
        if max_t > max_timestamp:
            max_timestamp = max_t

    timestamps = np.linspace(0, max_timestamp, len(fscores_mean[modality]))

    # plot mean with standard deviation bars, colored by modality with matplotlib
    for modality in mod_count:
        x, y_mean, y_std = timestamps, fscores_mean[modality], fscores_std[modality]
        ax.plot(x, y_mean, color=colors[modality], label=modality)
        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std, alpha=0.1, color=colors[modality]
        )

    plt.xlabel("Time (secs)", fontsize=15)
    ax.set_ylabel(f"Map F-score (< {f_score_T} mm)", fontsize=25)

    ax.tick_params(axis="y", labelsize=25)
    ax.tick_params(axis="x", labelsize=25)

    max_time, min_time = np.max(timestamps), np.min(timestamps)
    plt.axvspan(min_time, max_time, facecolor="gray", alpha=0.05)
    ax.set_ylim(bottom=0.4)

    ax.set_xlim(min_time, max_time)

    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")
    plt.close("all")


def avg_recall_over_time(df, log_path):
    """
    Groups all experiments by modality (vision/vitac/tac) and plots the average f_score distance for each modality.
    """
    save_path = os.path.join(log_path, f"avg_recall_over_time.pdf")

    plot_df = df.copy()

    f_score_T = int(plot_df["f_score_T"].unique()[0])

    # count number of modalities in plot_df
    mod_count = plot_df["modality"].unique()

    if len(mod_count) < 2:
        return

    # Sort the dataframe by modality
    plot_df = plot_df.sort_values("modality")

    # only store the f_score and map_timestamps of each modality
    plot_df = plot_df[["modality", "recall", "map_timestamps"]]
    # Create a boxplot of the f_score for each modality

    # interpolate each f_score to the same length
    plot_df["recall"] = plot_df["recall"].apply(
        lambda x: np.interp(np.arange(0, 1, 0.005), np.linspace(0, 1, len(x)), x)
    )
    plot_df["map_timestamps"] = plot_df["map_timestamps"].apply(
        lambda x: np.interp(np.arange(0, 1, 0.005), np.linspace(0, 1, len(x)), x)
    )

    fig, ax = plt.subplots()

    # for each modality, append the row of recall and map_timestamps to a numpy array
    fscores_mean, fscores_std = {}, {}
    timestamps = []
    for modality in mod_count:
        mod_df = plot_df[plot_df["modality"] == modality]
        fscores_mean[modality] = np.array(mod_df["recall"].tolist())
        fscores_mean[modality] = np.mean(fscores_mean[modality], axis=0)
        fscores_std[modality] = np.array(mod_df["recall"].tolist())
        fscores_std[modality] = np.std(fscores_std[modality], axis=0)
        timestamps.append(np.array(mod_df["map_timestamps"].tolist()))

    timestamps = np.vstack(timestamps)
    timestamps = np.mean(timestamps, axis=0)

    mod_count = ["vision", "vision + touch"]

    # set color palette based on the modalities in plot_df
    colors = {}
    for modality in mod_count:
        colors[modality] = my_pal[modality]
    # plot mean with standard deviation bars, colored by modality with matplotlib
    for modality in mod_count:
        x, y_mean, y_std = (
            timestamps,
            fscores_mean[modality] * 100,
            fscores_std[modality] * 100,
        )
        # smooth the data
        y_mean = smooth_data(y_mean, 10)
        y_std = smooth_data(y_std, 10)
        ax.plot(x, y_mean, color=colors[modality], label=modality)
        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std, alpha=0.1, color=colors[modality]
        )
        print(f"plotting {modality}")

    plt.xlabel("Time (secs)", fontsize=15)
    ax.set_ylabel(f"Recall % (< {f_score_T} mm)", fontsize=25)

    ax.tick_params(axis="y", labelsize=25)
    ax.tick_params(axis="x", labelsize=25)

    max_time, min_time = np.max(timestamps), np.min(timestamps)
    plt.axvspan(min_time, max_time, facecolor="gray", alpha=0.05)
    ax.set_ylim(bottom=40)

    ax.set_xlim(min_time, max_time)

    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")
    plt.close("all")


def avg_precision_over_time(df, log_path):
    """
    Groups all experiments by modality (vision/vitac/tac) and plots the average f_score distance for each modality.
    """
    save_path = os.path.join(log_path, f"avg_precision_over_time.pdf")

    plot_df = df.copy()

    f_score_T = int(plot_df["f_score_T"].unique()[0])

    # count number of modalities in plot_df
    mod_count = plot_df["modality"].unique()

    if len(mod_count) < 2:
        return

    # Sort the dataframe by modality
    plot_df = plot_df.sort_values("modality")

    # only store the f_score and map_timestamps of each modality
    plot_df = plot_df[["modality", "precision", "map_timestamps"]]
    # Create a boxplot of the f_score for each modality

    # interpolate each f_score to the same length
    plot_df["precision"] = plot_df["precision"].apply(
        lambda x: np.interp(np.arange(0, 1, 0.005), np.linspace(0, 1, len(x)), x)
    )
    plot_df["map_timestamps"] = plot_df["map_timestamps"].apply(
        lambda x: np.interp(np.arange(0, 1, 0.005), np.linspace(0, 1, len(x)), x)
    )

    fig, ax = plt.subplots()

    # for each modality, append the row of precision and map_timestamps to a numpy array
    fscores_mean, fscores_std = {}, {}
    timestamps = []
    for modality in mod_count:
        mod_df = plot_df[plot_df["modality"] == modality]
        fscores_mean[modality] = np.array(mod_df["precision"].tolist())
        fscores_mean[modality] = np.mean(fscores_mean[modality], axis=0)
        fscores_std[modality] = np.array(mod_df["precision"].tolist())
        fscores_std[modality] = np.std(fscores_std[modality], axis=0)
        timestamps.append(np.array(mod_df["map_timestamps"].tolist()))

    timestamps = np.vstack(timestamps)
    timestamps = np.mean(timestamps, axis=0)

    mod_count = ["vision", "vision + touch"]

    # set color palette based on the modalities in plot_df
    colors = {}
    for modality in mod_count:
        colors[modality] = my_pal[modality]
    # plot mean with standard deviation bars, colored by modality with matplotlib
    for modality in mod_count:
        x, y_mean, y_std = (
            timestamps,
            fscores_mean[modality] * 100,
            fscores_std[modality] * 100,
        )
        # smooth the data
        y_mean = smooth_data(y_mean, 10)
        y_std = smooth_data(y_std, 10)
        ax.plot(x, y_mean, color=colors[modality], label=modality)
        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std, alpha=0.1, color=colors[modality]
        )
        print(f"plotting {modality}")

    plt.xlabel("Time (secs)", fontsize=15)
    ax.set_ylabel(f"Precision % (< {f_score_T} mm)", fontsize=25)

    ax.tick_params(axis="y", labelsize=25)
    ax.tick_params(axis="x", labelsize=25)

    max_time, min_time = np.max(timestamps), np.min(timestamps)
    plt.axvspan(min_time, max_time, facecolor="gray", alpha=0.05)
    ax.set_ylim(bottom=40)

    ax.set_xlim(min_time, max_time)

    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")
    plt.close("all")


def avg_pose_error_over_time(df, log_path):
    """
    Groups all experiments by modality (vision/vitac/tac) and plots the average f_score distance for each modality.
    """
    save_path = os.path.join(log_path, f"avg_pose_error_over_time.pdf")

    plot_df = df.copy()

    # remove touch modality
    plot_df = plot_df[plot_df["modality"] != "touch"]

    # count number of modalities in plot_df
    mod_count = plot_df["modality"].unique()

    # Sort the dataframe by modality
    plot_df = plot_df.sort_values("modality")

    # set color palette based on the modalities in plot_df
    colors = {}
    for modality in mod_count:
        colors[modality] = my_pal[modality]

    # only store the f_score and map_timestamps of each modality
    plot_df = plot_df[["modality", "avg_3d_error", "pose_timestamps"]]
    # Create a boxplot of the f_score for each modality

    # interpolate each f_score to the same length
    plot_df["avg_3d_error"] = plot_df["avg_3d_error"].apply(
        lambda x: np.interp(np.arange(0, 1, 0.0005), np.linspace(0, 1, len(x)), x)
    )
    plot_df["pose_timestamps"] = plot_df["pose_timestamps"].apply(
        lambda x: np.interp(np.arange(0, 1, 0.0005), np.linspace(0, 1, len(x)), x)
    )

    fig, ax = plt.subplots()

    # for each modality, append the row of f_score and map_timestamps to a numpy array
    avg_3d_error_mean, avg_3d_error_std = {}, {}
    for modality in mod_count:
        mod_df = plot_df[plot_df["modality"] == modality]
        pose_timestamps_numpy = np.array(mod_df["pose_timestamps"].tolist())
        avg_3d_error_numpy = np.array(mod_df["avg_3d_error"].tolist())
        avg_3d_error_mean[modality] = np.nanmean(avg_3d_error_numpy, axis=0)
        avg_3d_error_std[modality] = np.nanstd(avg_3d_error_numpy, axis=0)

    # plot mean with standard deviation bars, colored by modality with matplotlib
    for modality in mod_count:
        # find max_timestamp for modality
        all_expt_timestamps = plot_df[plot_df["modality"] == modality][
            "pose_timestamps"
        ]
        max_timestamp = np.max(
            [np.max(timestamps) for timestamps in all_expt_timestamps]
        )
        timestamps = np.linspace(0, max_timestamp, len(avg_3d_error_mean[modality]))
        x, y_mean, y_std = (
            timestamps,
            avg_3d_error_mean[modality],
            avg_3d_error_std[modality],
        )
        ax.plot(x, y_mean, color=colors[modality], label=modality)
        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std, alpha=0.1, color=colors[modality]
        )
        print(
            f"Min timestamp: {np.min(timestamps)}, Max timestamp: {np.max(timestamps)}"
        )

    plt.xlabel("Time (secs)", fontsize=15)
    ax.set_ylabel("Pose drift from ground-truth\n initialization (mm)", fontsize=18)
    ax.tick_params(axis="y", labelsize=25)
    ax.tick_params(axis="x", labelsize=25)

    max_time, min_time = np.max(timestamps), np.min(timestamps)

    plt.axvspan(min_time, max_time, facecolor="gray", alpha=0.05)

    if "slam" in log_path:
        ax.set_ylim(bottom=0.0, top=10.0)
    else:
        ax.set_ylim(bottom=0.0, top=6.0)

    ax.set_xlim(min_time, max_time)

    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {save_path}")
    plt.close("all")


def avg_pose_error_per_optimizer(df, log_path):
    """
    Groups all experiments by tsdf optimizer (analytic/numerical/autodiff) and plots the average avg_3d_error for each optimizer
    """
    save_path = os.path.join(log_path, f"avg_pose_error_per_optimizer.pdf")

    plot_df = df.copy()

    # count number of modalities in df
    opt_count = plot_df["optimizer"].unique()

    # Sort the dataframe by the mean avg_3d_error
    plot_df = pd.melt(
        plot_df,
        id_vars=["optimizer"],
        value_vars=["avg_3d_error"],
        var_name="metric",
    )

    plot_df = plot_df.explode("value")  # turns list in column into separate cells
    # Create a boxplot of the avg_3d_error for each optimizer
    fig, ax = plt.subplots()
    sns.boxplot(
        x="optimizer",
        y="value",
        whis=1.0,
        data=plot_df,
        width=0.5,
        linewidth=1.5,
        showfliers=False,
        dodge=True,
        ax=ax,
    )
    ax.tick_params(axis="y")
    plt.xlabel("", fontsize=25)

    ax.set_ylabel("Avg. pose error (mm)", fontsize=25)

    ax.tick_params(axis="y", labelsize=25)

    plt.axvspan(-0.5, len(opt_count) - 0.5, facecolor="gray", alpha=0.05)
    ax.set_ylim(bottom=0.0)

    ax.set_xlim(-0.5, len(opt_count) - 0.5)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")


def avg_timing_per_optimizer(df, log_path):
    """
    Groups all experiments by tsdf optimizer (analytic/numerical/autodiff) and plots the average pose optimization time
    """
    save_path = os.path.join(log_path, f"avg_timing_per_optimizer.pdf")

    plot_df = df.copy()

    # count number of modalities in df
    opt_count = plot_df["optimizer"].unique()

    plot_df = plot_df.sort_values("optimizer")
    plot_df = plot_df.explode(
        "pose_timings"
    )  # turns list in column into separate cells

    # Create a boxplot of the pose_timings for each optimizer
    fig, ax = plt.subplots()
    sns.boxplot(
        x="optimizer",
        y="pose_timings",
        color=my_pal["pose_timings"],
        whis=1.0,
        data=plot_df,
        width=0.5,
        linewidth=1.5,
        showfliers=False,
        dodge=True,
        ax=ax,
    )
    ax.set_ylabel("Average pose optimizer time (secs)", color=my_pal["pose_timings"])

    plt.xlabel("", fontsize=25)

    ax.tick_params(axis="y", colors=my_pal["pose_timings"], labelsize=25)

    plt.axvspan(-0.5, len(opt_count) - 0.5, facecolor="gray", alpha=0.05)
    ax.set_ylim(bottom=0.0)

    ax.set_xlim(-0.5, len(opt_count) - 0.5)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")


def avg_timing_per_modality(df, log_path):
    """
    Groups all experiments by modality (vi/vitac) and plots the average pose optimization time
    """
    save_path = os.path.join(log_path, f"avg_timing_per_modality.pdf")

    plot_df = df.copy()

    # count number of modalities in df
    mod_count = plot_df["modality"].unique()

    # check if "map_timings" is in plot_df
    if "map_timings" not in plot_df:
        plot_type = "pose_timings"
    elif "pose_timings" not in plot_df:
        plot_type = "map_timings"
    else:
        plot_type = "both"

    if plot_type == "both":
        # Sort the dataframe by the mean pose_timings and map_timings
        plot_df = pd.melt(
            plot_df,
            id_vars=["modality"],
            value_vars=["pose_timings", "map_timings"],
            var_name="metric",
        )

        plot_df = plot_df.explode("value")  # turns list in column into separate cells

        # Create a boxplot of the pose_timings and map_timings for each optimizer
        fig, ax = plt.subplots()
        sns.boxplot(
            x="modality",
            y="value",
            hue="metric",
            palette=my_pal,
            whis=1.0,
            data=plot_df,
            width=0.5,
            linewidth=1.5,
            showfliers=False,
            dodge=True,
            ax=ax,
        )

        # print the mean pose_timings and map_timings ± std dev for each modality
        for mod in mod_count:
            pose_mean = plot_df[
                (plot_df["modality"] == mod) & (plot_df["metric"] == "pose_timings")
            ]["value"].mean()
            pose_std = plot_df[
                (plot_df["modality"] == mod) & (plot_df["metric"] == "pose_timings")
            ]["value"].std()
            map_mean = plot_df[
                (plot_df["modality"] == mod) & (plot_df["metric"] == "map_timings")
            ]["value"].mean()
            map_std = plot_df[
                (plot_df["modality"] == mod) & (plot_df["metric"] == "map_timings")
            ]["value"].std()
            print(
                f"Modality: {mod}, Pose timings: {pose_mean:.2f} ± {pose_std:.2f}, Map timings: {map_mean:.2f} ± {map_std:.2f}"
            )
        ax.set_ylabel("Avg. pose optimizer time (secs)", color=my_pal["pose_timings"])
        ax.tick_params(axis="y", colors=my_pal["pose_timings"], labelsize=25)
        plt.xlabel("", fontsize=25)

        ax.legend_.remove()

        ax2 = ax.twinx()
        ax2.set_ylabel("Avg. map optimizer time (secs)", color=my_pal["map_timings"])

        plt.axvspan(-0.5, len(mod_count) - 0.5, facecolor="gray", alpha=0.05)
        ax.set_ylim(bottom=0.0)
        ax2.set_ylim(ax.get_ylim())

        ax.set_xlim(-0.5, len(mod_count) - 0.5)
    else:
        mod_count = plot_df["modality"].unique()

        plot_df = plot_df.sort_values("modality")
        plot_df = plot_df.explode(plot_type)  # turns list in column into separate cells

        # Create a boxplot of the pose_timings and map_timings for each optimizer
        fig, ax = plt.subplots()
        sns.boxplot(
            x="modality",
            y=plot_type,
            color=my_pal[plot_type],
            whis=1.0,
            data=plot_df,
            width=0.5,
            linewidth=1.5,
            showfliers=False,
            dodge=True,
            ax=ax,
        )
        ax.set_ylabel("Average pose optimizer time (secs)", color=my_pal[plot_type])

        plt.xlabel("", fontsize=25)

        ax.tick_params(axis="y", colors=my_pal[plot_type], labelsize=25)

        plt.axvspan(-0.5, len(mod_count) - 0.5, facecolor="gray", alpha=0.05)
        ax.set_ylim(bottom=0.0)

        ax.set_xlim(-0.5, len(mod_count) - 0.5)

    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")


def avg_pose_error_per_camera_placement(df, log_path):
    """
    Groups all experiments by camera placement (front-left, top-down, back-right) and plots the average avg_3d_error for each optimizer
    """
    save_path = os.path.join(log_path, f"avg_pose_error_per_camera_placement.pdf")

    plot_df = df.copy()
    cam_count = plot_df["cameras"].unique()

    # Sort the dataframe by the mean avg_3d_error
    plot_df = pd.melt(
        plot_df,
        id_vars=["cameras"],
        value_vars=["avg_3d_error"],
        var_name="metric",
    )

    plot_df = plot_df.explode("value")  # turns list in column into separate cells
    # Create a boxplot of the avg_3d_error for each optimizer

    # sort camera order in plot as: back-right, front-left, top-down,
    cam_order = ["realsense_back_right", "realsense_front_left", "realsense_top_down"]
    plot_df["cameras"] = pd.Categorical(plot_df["cameras"], categories=cam_order)

    fig, ax = plt.subplots()
    sns.boxplot(
        x="cameras",
        y="value",
        whis=1.0,
        data=plot_df,
        width=0.5,
        linewidth=1.5,
        showfliers=False,
        dodge=True,
        ax=ax,
    )
    print(f"Mean avg_3d_error per camera placement:")
    for cam in cam_count:
        print(
            f"{cam} mean: {plot_df[plot_df['cameras'] == cam]['value'].mean():.2f} ± {plot_df[plot_df['cameras'] == cam]['value'].std():.2f}"
        )
    ax.tick_params(axis="y")
    plt.xlabel("", fontsize=25)

    ax.set_ylabel("Avg. pose error (mm)", fontsize=25)

    ax.tick_params(axis="y", labelsize=25)

    plt.axvspan(-0.5, len(cam_count) - 0.5, facecolor="gray", alpha=0.05)
    ax.set_ylim(bottom=0.0)

    ax.set_xlim(-0.5, len(cam_count) - 0.5)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")


def avg_pose_error_per_object(df, log_path):
    """
    Groups all experiments by object (rubiks cube, pear, cup, etc.) and plots the average avg_3d_error for each optimizer
    """
    save_path = os.path.join(log_path, f"avg_pose_error_per_object.pdf")

    plot_df = df.copy()

    object_names = plot_df["object"].unique()
    print(f"Object names: {object_names}")

    plot_df = pd.melt(
        plot_df,
        id_vars=["object", "modality"],
        value_vars=["avg_3d_error"],
        var_name="metric",
    )

    # keep only vision + touch modality
    plot_df = plot_df[plot_df["modality"] == "vision + touch"]
    # return if plot_df is empty
    if plot_df.empty:
        return

    median_df = plot_df.copy()

    # run p-test if "digit" and "binary" are in object_names
    if "digit" in object_names and "binary" in object_names:
        t_test_df = plot_df.copy()
        t_test_df["value"] = t_test_df["value"].apply(lambda x: np.nanmedian(x))
        digit = t_test_df[t_test_df["object"] == "digit"]["value"]
        binary = t_test_df[t_test_df["object"] == "binary"]["value"]
        # convert pandas obejcts to numpy arrays
        # digit = digit.to_numpy(dtype=np.float64)
        # binary = binary.to_numpy(dtype=np.float64)
        t_stat, p_value = ttest_ind(binary, digit, nan_policy="raise")
        cprint(
            f"Pose error T-statistic between digit and binary: {t_stat}, p-value: {p_value}",
            "green",
        )

    # combine the value arrays of each row based on object name
    median_df = (
        median_df.groupby(["object", "modality"])["value"].apply(list).reset_index()
    )
    # convert list of lists to a single list
    median_df["value"] = median_df["value"].apply(
        lambda x: [item for sublist in x for item in sublist]
    )
    median_df["median_value"] = median_df["value"].apply(lambda x: np.nanmean(x))
    median_df = median_df.sort_values("median_value", ascending=True)
    object_order = median_df["object"].tolist()

    plot_df = plot_df.explode("value")  # turns list in column into separate cells
    # remove all rows with nan values
    plot_df = plot_df[plot_df["value"].notna()]
    plot_df = plot_df.reset_index(drop=True)

    plot_df["object"] = pd.Categorical(plot_df["object"], categories=object_order)

    fig, ax = plt.subplots()
    sns.boxplot(
        x="object",
        y="value",
        palette="Set2",
        whis=1.0,
        data=plot_df,
        width=0.5,
        linewidth=1.5,
        showfliers=False,
        dodge=True,
        ax=ax,
    )
    plt.xlabel("", fontsize=25)

    # print mean and ± std dev for each object to 2 decimal places
    for object in object_names:
        print(
            f"{object} mean: {plot_df[plot_df['object'] == object]['value'].mean():.2f} ± {plot_df[plot_df['object'] == object]['value'].std():.2f}"
        )
    # ax.legend_.remove()

    ax.set_ylabel("Avg. pose error (mm)", fontsize=25)

    # rotate x-axis labels
    plt.xticks(rotation=60, ha="right", fontsize=5)
    ylim = (0, 8)
    y_plot_ticks = np.arange(ylim[0], ylim[1] + 1, 1)
    plt.yticks(y_plot_ticks)
    ax.set_ylim(bottom=ylim[0], top=ylim[1])

    # pose error plots
    # y_lim = (0, 4)
    # y_plot_ticks = np.arange(y_lim[0], y_lim[1] + 1, 2)

    ax.tick_params(axis="y", labelsize=25)

    plt.axvspan(-0.5, len(object_names) - 0.5, facecolor="gray", alpha=0.05)

    ax.set_xlim(-0.5, len(object_names) - 0.5)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")


def avg_pose_error_per_experiment(df, log_path):
    """
    Groups all experiments by object (rubiks cube, pear, cup, etc.) and plots the average avg_3d_error for each optimizer
    """
    save_path = os.path.join(log_path, f"avg_pose_error_per_experiment.pdf")

    plot_df = df.copy()
    # append df index to object name and create new column
    plot_df["object"] = plot_df["object"] + "_" + plot_df["expt"].astype(str)
    object_names = plot_df["object"].unique()

    plot_df = pd.melt(
        plot_df,
        id_vars=["object", "modality"],
        value_vars=["avg_3d_error"],
        var_name="metric",
    )

    plot_df = plot_df.explode("value")  # turns list in column into separate cells

    # sort object names in the order of ascending avg_3d_error values
    # plot_df = plot_df.sort_values("value", ascending=True)

    # drop index column
    plot_df = plot_df.reset_index(drop=True)

    # Create a boxplot of the avg_3d_error for each optimizer
    fig, ax = plt.subplots()
    sns.boxplot(
        x="object",
        y="value",
        hue="modality",
        palette=my_pal,
        whis=1.0,
        data=plot_df,
        width=0.5,
        linewidth=0.5,
        showfliers=False,
        dodge=True,
        ax=ax,
    )

    # print mean and ± std dev for each object to 2 decimal places
    for object in object_names:
        print(
            f"{object} mean: {plot_df[plot_df['object'] == object]['value'].mean():.3f} ± {plot_df[plot_df['object'] == object]['value'].std():.3f}"
        )
    plt.xlabel("", fontsize=25)

    ax.legend_.remove()

    ax.set_ylabel("Pose drift from ground-truth\n initialization (mm)", fontsize=18)

    # rotate x-axis labels
    plt.xticks(rotation=90, ha="right", fontsize=5)

    ax.tick_params(axis="y", labelsize=25)

    plt.axvspan(-0.5, len(object_names) - 0.5, facecolor="gray", alpha=0.05)
    ax.set_ylim(bottom=0.0)

    ax.set_xlim(-0.5, len(object_names) - 0.5)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")


def avg_pose_error_per_shape_res(df, log_path):
    """
    Groups all experiments by object (rubiks cube, pear, cup, etc.) and plots the average avg_3d_error for each optimizer
    """
    save_path = os.path.join(log_path, f"avg_pose_error_per_shape_res.pdf")

    # dict mapping expt number to resolution
    # 00, 01,02, 03, 04 --> 5e-4 1e-3 2e-3 5e-3 1e-2
    res_dict = {
        "00": 5e-4,
        "01": 1e-3,
        "02": 2e-3,
        "03": 5e-3,
        "04": 1e-2,
    }

    plot_df = df.copy()
    # append df index to object name and create new column
    plot_df["object"] = plot_df["expt"].astype(str)
    plot_df["object"] = plot_df["object"].map(res_dict)
    print(plot_df["object"])
    object_names = plot_df["object"].unique()

    plot_df = pd.melt(
        plot_df,
        id_vars=["object", "modality"],
        value_vars=["avg_3d_error"],
        var_name="metric",
    )

    plot_df = plot_df.explode("value")  # turns list in column into separate cells

    # sort object names in the order of ascending avg_3d_error values
    plot_df = plot_df.sort_values("value", ascending=True)

    # drop index column
    plot_df = plot_df.reset_index(drop=True)

    error_palette = {modality: "k" for modality in plot_df["modality"].unique()}

    # convert x value to mm
    plot_df["object"] = plot_df["object"] * 1000
    # Create a boxplot of the avg_3d_error for each optimizer
    fig, ax = plt.subplots()
    sns.lineplot(
        x="object",
        y="value",
        hue="modality",
        palette=error_palette,
        data=plot_df,
        ax=ax,
        errorbar="sd",
        estimator=np.nanmedian,
        marker="o",
        markersize=5,
        markeredgecolor="k",
        linewidth=2,
        alpha=0.7,
    )
    plt.xlabel("", fontsize=25)

    ax.legend_.remove()

    ax.set_ylabel("Pose drift from ground-truth\n initialization (mm)", fontsize=18)

    # make x axis log scale
    ax.tick_params(axis="y", labelsize=25)

    # plt.axvspan(-0.5, len(object_names) - 0.5, facecolor="gray", alpha=0.05)
    ax.set_ylim(bottom=0.0)

    # convert to log scale
    ax.set_xscale("log")

    # set x ticks to datapoints in plot_df["value"] only
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(plot_df["object"].unique())
    # crop x axis to only show data points
    ax.set_xlim(left=plot_df["object"].min(), right=plot_df["object"].max())
    plt.xlabel("Ground-truth shape resolution (mm) - logscale", fontsize=15)

    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")


def success_failure_stats(df, fail_thresh=10):
    """
    Use tracking metrics to find the number of successful and failed experiments for each object
    """

    plot_df = df.copy()
    # append df index to object name and create new column
    plot_df["object"] = plot_df["object"] + "_" + plot_df["expt"].astype(str)

    plot_df = pd.melt(
        plot_df,
        id_vars=["object", "modality"],
        value_vars=["avg_3d_error"],
        var_name="metric",
    )

    # combine results from different seeds of the same experiment
    plot_df = plot_df.groupby(["object", "modality"])["value"].apply(list).reset_index()
    # get the final value in each list
    plot_df["value"] = plot_df["value"].apply(lambda x: [sublist[-1] for sublist in x])

    # take average of value for each df index
    plot_df["value"] = plot_df["value"].apply(lambda x: np.nanmean(x))

    # print number of indexes with value > fail_thresh for each modality
    for modality in plot_df["modality"].unique():
        # plot min and max values for each modality
        print(
            f"{modality} min: {np.min(plot_df[plot_df['modality'] == modality]['value'])}"
        )
        print(
            f"{modality} max: {np.max(plot_df[plot_df['modality'] == modality]['value'])}"
        )
        failed_logs = plot_df[
            (plot_df["modality"] == modality) & (plot_df["value"] > fail_thresh)
        ]
        # find unique elements in failed_logs
        failed_logs = failed_logs["object"].unique()
        num_fail = len(failed_logs)

        num_total = len(plot_df[plot_df["modality"] == modality])
        print(f"# fail {modality} : {num_fail} / {num_total}")
        # print the experiment names of the failed experiments
        print(f"Failed experiments: {failed_logs}")


def avg_map_error_per_object(df, log_path):
    """
    Groups all experiments by object (rubiks cube, pear, cup, etc.) and plots the average f_score for each optimizer
    """

    save_path = os.path.join(log_path, f"avg_map_error_per_object.pdf")

    plot_df = df.copy()

    obj_count = plot_df["object"].unique()
    f_score_T = int(plot_df["f_score_T"].unique()[0])

    # only plot the final f-score
    plot_df["f_score"] = plot_df["f_score"].apply(lambda x: x[-1])

    plot_df = pd.melt(
        plot_df,
        id_vars=["object", "modality"],
        value_vars=["f_score"],
        var_name="metric",
    )

    # keep only vision + touch modality
    plot_df = plot_df[plot_df["modality"] == "vision + touch"]

    # run p-test if "digit" and "binary" are in object_names
    if "digit" in obj_count and "binary" in obj_count:
        t_test_df = plot_df.copy()
        t_test_df["value"] = t_test_df["value"].apply(lambda x: np.nanmedian(x))
        digit = t_test_df[t_test_df["object"] == "digit"]["value"]
        binary = t_test_df[t_test_df["object"] == "binary"]["value"]
        # convert pandas obejcts to numpy arrays
        # digit = digit.to_numpy(dtype=np.float64)
        # binary = binary.to_numpy(dtype=np.float64)
        t_stat, p_value = ttest_ind(binary, digit, nan_policy="raise")
        cprint(
            f"Map error T-statistic between digit and binary: {t_stat}, p-value: {p_value}",
            "green",
        )

    median_df = plot_df.copy()
    # combine the value arrays of each row based on object name
    median_df = (
        median_df.groupby(["object", "modality"])["value"].apply(list).reset_index()
    )
    median_df["median_value"] = median_df["value"].apply(lambda x: np.nanmedian(x))
    median_df = median_df.sort_values("median_value", ascending=False)
    object_order = median_df["object"].tolist()

    plot_df = plot_df.explode("value")  # turns list in column into separate cells
    # remove all rows with nan values
    plot_df = plot_df[plot_df["value"].notna()]
    plot_df = plot_df.reset_index(drop=True)
    plot_df["object"] = pd.Categorical(plot_df["object"], categories=object_order)

    # Create a boxplot of the f_score for each optimizer
    fig, ax = plt.subplots()
    sns.boxplot(
        x="object",
        y="value",
        color=".8",
        whis=1.0,
        data=plot_df,
        width=0.5,
        linewidth=1.5,
        showfliers=False,
        dodge=True,
        ax=ax,
    )
    plt.xlabel("", fontsize=25)

    # print mean and ± std dev for each object to 2 decimal places
    for object in obj_count:
        print(
            f"{object} mean: {plot_df[plot_df['object'] == object]['value'].mean():.2f} ± {plot_df[plot_df['object'] == object]['value'].std():.2f}"
        )
    # ax.legend_.remove()

    ax.set_ylabel(f"Avg. map F-score (< {f_score_T} mm)", fontsize=25)

    # rotate x-axis labels
    plt.xticks(rotation=60, ha="right", fontsize=5)
    ylim = (0.5, 1.05)
    # y_plot_ticks = np.arange(ylim[0], ylim[1] + 1, 5)
    # plt.yticks(y_plot_ticks)
    ax.set_ylim(bottom=ylim[0], top=ylim[1])

    ax.tick_params(axis="y", labelsize=25)

    plt.axvspan(-0.5, len(obj_count) - 0.5, facecolor="gray", alpha=0.05)

    ax.set_xlim(-0.5, len(obj_count) - 0.5)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")


def avg_map_error_per_experiment(df, log_path):
    """
    [RUN AFTER POSE ERROR]
    Groups all experiments by object (rubiks cube, pear, cup, etc.) and plots the average avg_3d_error for each optimizer
    """

    save_path = os.path.join(log_path, f"avg_map_error_per_experiment.pdf")

    plot_df = df.copy()
    plot_df["object"] = plot_df["object"] + "_" + plot_df["expt"].astype(str)

    f_score_T = int(plot_df["f_score_T"].unique()[0])

    obj_count = plot_df["object"].unique()

    plot_df = pd.melt(
        plot_df,
        id_vars=["object", "modality"],
        value_vars=["f_score"],
        var_name="metric",
    )

    # only plot the final f-score
    plot_df["value"] = plot_df["value"].apply(lambda x: x[-1])
    plot_df = plot_df.explode("value")  # turns list in column into separate cells

    # sort object names in the order of ascending f_score values
    # plot_df = plot_df.sort_values("value", ascending=False)

    # remove all those with f_score < 0.8
    # df = df[df["value"] > 0.8]

    # drop index column
    plot_df = plot_df.reset_index(drop=True)

    # Create a boxplot of the avg_3d_error for each optimizer
    fig, ax = plt.subplots()
    sns.boxplot(
        x="object",
        y="value",
        hue="modality",
        palette=my_pal,
        whis=1.0,
        data=plot_df,
        width=0.5,
        linewidth=0.5,
        showfliers=False,
        dodge=True,
        ax=ax,
    )
    plt.xlabel("", fontsize=25)

    # print mean and ± std dev for each object to 2 decimal places
    for object in obj_count:
        print(
            f"{object} mean (map): {plot_df[plot_df['object'] == object]['value'].mean():.3f} ± {plot_df[plot_df['object'] == object]['value'].std():.3f}"
        )

    ax.legend_.remove()

    ax.set_ylabel(f"Avg. map F-score (< {f_score_T} mm)", fontsize=25)
    # rotate x-axis labels
    plt.xticks(rotation=90, ha="right", fontsize=5)

    ax.tick_params(axis="y", labelsize=25)

    plt.axvspan(-0.5, len(obj_count) - 0.5, facecolor="gray", alpha=0.05)

    ax.set_xlim(-0.5, len(obj_count) - 0.5)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")


def pose_error_vs_noise(expt_paths, root_path):
    """
    Plot the pose error vs vision noise for all experiments
    """
    num_expts = len(expt_paths)
    print(f"Number of experiments: {num_expts}")
    from tqdm import tqdm

    df_list = []
    for i, expt_path in enumerate(tqdm(expt_paths)):
        pkl_path = os.path.join(root_path, expt_path, "stats.pkl")

        try:
            with open(pkl_path, "rb") as p:
                stats = pickle.load(p)
        except:
            continue

        pose_error_stats = stats["pose"]["errors"]
        # convert list of dicts into arrays
        avg_3d_error = np.array(
            [
                pose_error_stats[i]["avg_3d_error"] * 1000.0
                for i in range(len(pose_error_stats))
            ]
        )
        timestamps = np.array(
            [pose_error_stats[i]["time"] for i in range(len(pose_error_stats))]
        )
        # ignore the last frame (may be unoptimized)
        avg_3d_error, timestamps = avg_3d_error[:-1], timestamps[:-1]
        avg_3d_error[timestamps < 5] = np.nan

        metadata_df = load_metadata(expt_path, stats, i)
        noise_factor = float(expt_path.split("_")[-1])
        modality = metadata_df["modality"].iloc[-1]

        # skip if noise_factor > 50
        if noise_factor > 50:
            continue
        # create df with noise and avg_3d_error
        df = pd.DataFrame(
            {
                "avg_3d_error": avg_3d_error.tolist(),
                "noise_factor": [noise_factor] * len(avg_3d_error),
                "modality": [modality] * len(avg_3d_error),
            }
        )

        # explode df into separate rows
        df = df.explode("avg_3d_error")
        df_list = df_list + [df]

    df_pose_noise = pd.concat(df_list, axis=0)
    df_pose_noise = df_pose_noise.sort_values("noise_factor")

    # drop index column
    df_pose_noise = df_pose_noise.reset_index(drop=True)
    # turn nan into 0.0
    save_path = os.path.join(root_path, f"pose_error_vs_vision_noise_.pdf")
    # Create a boxplot of the avg_3d_error for each optimizer
    fig, ax = plt.subplots()
    print("Plotting...")

    # https://stackoverflow.com/questions/63156008/how-to-plot-lines-linking-medians-of-multiple-violin-distributions-in-seaborn
    df["noise_factor"] = [f"{x:.1f}" for x in df["noise_factor"]]

    # violinplot of avg_3d_error vs vision_noise, colored by modality
    ax = sns.violinplot(
        data=df_pose_noise,
        x="noise_factor",
        y="avg_3d_error",
        hue="modality",
        split=True,
        inner=None,
        cut=0,
        width=1.0,
        gridsize=500,
        linewidth=0.4,
        gap=0.1,
        palette=my_pal,
        saturation=0.75,
        ax=ax,
    )

    # convert vision_noise back to float
    df_pose_noise["noise_factor"] = df_pose_noise["noise_factor"].astype(float)
    # half each value
    df_pose_noise["noise_factor"] = df_pose_noise["noise_factor"] / 10
    # add matplotlib lineplot of median values of avg_3d_error vs vision_noise, colored by modality
    ax = sns.lineplot(
        y="avg_3d_error",
        x="noise_factor",
        data=df_pose_noise,
        hue="modality",
        palette=my_pal,
        errorbar=None,
        estimator=np.median,
        marker="o",
        markersize=5,
        markeredgecolor="k",
        linewidth=3,
        legend=False,
        ax=ax,
        zorder=100,
    )

    # draw vertical lines connecting markers of each hue in the lineplot
    ys = []
    for line in ax.lines:
        x, y = line.get_data()
        ys.append(y)

    # draw vertical lines connecting between the ys for each x value
    for i in range(len(ys[0])):
        ax.plot([x[i], x[i]], [ys[0][i], ys[1][i]], color="k", linewidth=1.5, alpha=1)

    plt.xlabel("Depth noise factor", fontsize=25)

    ax.legend_.remove()

    ax.set_ylabel("Avg. pose error (mm)", fontsize=25)

    ax.tick_params(axis="y", labelsize=25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # min_noise = (df_pose_noise["vision_noise"].astype(int)).min()
    # max_noise = (df_pose_noise["vision_noise"].astype(int)).max()
    # min_error, max_error = np.min(df_pose_noise["avg_3d_error"]), np.max(
    #     df_pose_noise["avg_3d_error"]
    # )
    # print(f"min_noise: {min_noise}, max_noise: {max_noise}")
    # print(f"min_error: {min_error}, max_error: {max_error}")

    # # set max noise to 15mm
    # max_noise = 20
    # x_plot_ticks = np.arange(0, max_noise, 10)
    # plt.xticks(x_plot_ticks)
    # # plt.axvline(x=5, color="k", linestyle="--", linewidth=0.5, alpha=0.2)
    ax.set_ylim(bottom=0.0, top=4)

    # plt.axvspan(x_plot_ticks[0], x_plot_ticks[-1], facecolor="gray", alpha=0.05)
    ax.set_xlim(-0.5, 5.5)
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")
    plt.close("all")


def pose_error_vs_occlusion(expt_paths, root_path, smoothing_factor=10):
    """
    Plot the pose error vs vision occlusion for all experiments
    """
    num_expts = len(expt_paths)
    print(f"Number of experiments: {num_expts}")
    from tqdm import tqdm

    modalities = ["vision", "vision + touch"]
    pose_errors_mean, pose_errors_std = {}, {}
    seg_areas = np.zeros(num_expts)
    colors = {}

    for modality in modalities:
        pose_errors_mean[modality] = np.zeros(num_expts)
        pose_errors_std[modality] = np.zeros(num_expts)
        colors[modality] = my_pal[modality]

    for i, expt_path in enumerate(tqdm(expt_paths)):
        pkl_path = os.path.join(root_path, expt_path, "stats.pkl")

        try:
            with open(pkl_path, "rb") as p:
                stats = pickle.load(p)
        except:
            continue

        pose_error_stats = stats["pose"]["errors"]
        seg_area = stats["seg_area"] * 100.0  # percentage of segmentation area
        # convert list of dicts into arrays
        avg_3d_error = np.array(
            [
                pose_error_stats[j]["avg_3d_error"] * 1000.0
                for j in range(len(pose_error_stats))
            ]
        )
        # ignore the last frame (may be unoptimized)
        avg_3d_error = avg_3d_error[:-1]

        metadata_df = load_metadata(expt_path, stats, i)
        modality = metadata_df["modality"].iloc[-1]
        cam_id = int(expt_path.split("/")[-3])

        pose_errors_mean[modality][cam_id] = np.mean(avg_3d_error)
        pose_errors_std[modality][cam_id] = np.std(avg_3d_error)
        seg_areas[cam_id] = seg_area

    # remove zero elements from seg_areas, pose_errors_mean, pose_errors_std
    seg_areas = seg_areas[seg_areas != 0]
    for modality in modalities:
        pose_errors_mean[modality] = pose_errors_mean[modality][
            pose_errors_mean[modality] != 0
        ]
        pose_errors_std[modality] = pose_errors_std[modality][
            pose_errors_std[modality] != 0
        ]

    # normalize all seg_areas to be between 0 and 1
    seg_areas = (seg_areas - np.min(seg_areas)) / (
        np.max(seg_areas) - np.min(seg_areas)
    )

    # find the elements in the array seg_areas that are closest to [0.25, 0.5, 0.75, 1.0] are print their positions
    for i in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # print the index of the element in seg_areas that is closest to i
        print(
            f"seg_area: {i}, index: {np.argmin(np.abs(seg_areas - i))}, value: {seg_areas[np.argmin(np.abs(seg_areas - i))]}"
        )

    # sort seg_areas in descending order, and store the sort indices
    sort_idx = np.argsort(seg_areas)[::-1]
    seg_areas = np.array(seg_areas)[sort_idx]

    # sort pose errors in ascending order
    for modality in modalities:
        pose_errors_mean[modality] = np.array(pose_errors_mean[modality])[sort_idx]
        pose_errors_std[modality] = np.array(pose_errors_std[modality])[sort_idx]

    # find p value for each range [1.0, 0.75], [0.75, 0.5], [0.5, 0.25], [0.25, 0.0]
    for i in range(4):
        # get all pose errors for seg_areas in the range [i*0.25, (i+1)*0.25]
        pose_errors_mean_range = {}
        for modality in modalities:
            pose_errors_mean_range[modality] = pose_errors_mean[modality][
                (seg_areas > i * 0.25) & (seg_areas <= (i + 1) * 0.25)
            ]
        t_stat, p_value = ttest_ind(
            pose_errors_mean_range["vision"],
            pose_errors_mean_range["vision + touch"],
            nan_policy="raise",
        )
        print(
            f"T-statistic range [{i*0.25}, {(i+1)*0.25}]: {t_stat}, p-value: {p_value}"
        )
        print(
            f"Avg gain in range [{i*0.25}, {(i+1)*0.25}]: {np.mean(pose_errors_mean_range['vision']) - np.mean(pose_errors_mean_range['vision + touch'])}"
        )
    # breakpoint()

    # smooth the pose errors
    for modality in modalities:
        pose_errors_mean[modality] = smooth_data(
            pose_errors_mean[modality], N=smoothing_factor
        )
        pose_errors_std[modality] = smooth_data(
            pose_errors_std[modality], N=smoothing_factor
        )

    save_path = os.path.join(root_path, f"pose_error_vs_vision_occlusion.pdf")
    # Create a boxplot of the avg_3d_error for each optimizer
    fig, ax = plt.subplots()

    # plot mean with standard deviation bars, colored by modality with matplotlib
    for modality in modalities:
        x, y_mean, y_std = (
            seg_areas,
            pose_errors_mean[modality],
            pose_errors_std[modality],
        )
        ax.plot(x, y_mean, color=colors[modality], label=modality)
        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std, alpha=0.1, color=colors[modality]
        )

    plt.xlabel("Occlusion score", fontsize=25)
    ax.set_ylabel(f"Avg. pose error (mm)", fontsize=25)

    ax.tick_params(axis="y", labelsize=25)
    ax.tick_params(axis="x", labelsize=25)

    # log scale
    ax.set_yscale("log")

    max_time, min_time = np.max(seg_areas), 0.01
    plt.axvspan(min_time, max_time, facecolor="gray", alpha=0.05)
    ax.set_ylim(bottom=1.0, top=20)

    # y ticks
    y_plot_ticks = [1, 5, 10, 15, 20]
    plt.yticks(y_plot_ticks)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xlim(min_time, max_time)

    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")
    plt.close("all")


def pose_errors_vs_camera_frustums(expt_paths, root_path):
    num_expts = len(expt_paths)
    print(f"Number of experiments: {num_expts}")
    from tqdm import tqdm

    df_list = []
    for i, expt_path in enumerate(tqdm(expt_paths)):
        pkl_path = os.path.join(root_path, expt_path, "stats.pkl")

        try:
            with open(pkl_path, "rb") as p:
                stats = pickle.load(p)
        except:
            continue

        pose_error_stats = stats["pose"]["errors"]
        seg_area = stats["seg_area"]
        # convert list of dicts into arrays
        avg_3d_error = np.array(
            [
                pose_error_stats[j]["avg_3d_error"] * 1000.0
                for j in range(len(pose_error_stats))
            ]
        )
        timestamps = np.array(
            [pose_error_stats[j]["time"] for j in range(len(pose_error_stats))]
        )
        # ignore the last frame (may be unoptimized)
        avg_3d_error, timestamps = avg_3d_error[:-1], timestamps[:-1]
        avg_3d_error[timestamps < 5] = np.nan

        metadata_df = load_metadata(expt_path, stats, i)

        # get the third string from the end of the expt_path divided by "/"
        cam_id = int(expt_path.split("/")[-3])
        modality = metadata_df["modality"].iloc[-1]

        # create df with noise and avg_3d_error
        df = pd.DataFrame(
            {
                "avg_3d_error": avg_3d_error.tolist(),
                "seg_area": [seg_area] * len(avg_3d_error),
                "cam_id": [cam_id] * len(avg_3d_error),
                "modality": [modality] * len(avg_3d_error),
            }
        )

        # explode df into separate rows
        df = df.explode("avg_3d_error")
        df_list = df_list + [df]

    df_pose_occlusion = pd.concat(df_list, axis=0)
    df_pose_occlusion = df_pose_occlusion.sort_values("seg_area")

    n_views = 200
    sphere_center = np.array([0.06, 0, 0])
    obj_pose = np.eye(4)
    obj_pose[:3, 3] = np.array([0.12, 0, 0])
    radius = 0.4
    poses = transform.look_at_on_sphere(
        n_views=n_views,
        radius=radius,
        sphere_center=sphere_center,
        look_at_noise=0.00,  # add 5cm noise to look at point
        cam_axis=[1, 0, 0],
    )

    # open3d vis window
    app = gui.Application.instance
    app.initialize()
    app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    window = gui.Application.instance.create_window(
        "Camera frustums", width=2000, height=2000
    )

    # init widget3d visualizer
    camera_visualizer = gui.SceneWidget()
    window.add_child(camera_visualizer)

    camera_visualizer.scene = rendering.Open3DScene(window.renderer)
    camera_visualizer.enable_scene_caching(True)

    # black background
    camera_visualizer.scene.set_background([0.0, 0.0, 0.0, 1.0])
    camera_visualizer.scene.scene.set_sun_light(
        [0.0, 0.0, -1.0], [1.0, 1.0, 1.0], 50000
    )
    camera_visualizer.scene.scene.enable_sun_light(True)
    camera_visualizer.scene.scene.set_indirect_light_intensity(20000.0)

    cam_pos = sphere_center + np.array([0.1, 1.2 * radius, 1.2 * radius])
    camera_visualizer.look_at(sphere_center, cam_pos, [1, 0, 0])

    # Address the white background issue: https://github.com/isl-org/Open3D/issues/6020
    cg_settings = rendering.ColorGrading(
        rendering.ColorGrading.Quality.ULTRA,
        rendering.ColorGrading.ToneMapping.LINEAR,
    )
    camera_visualizer.scene.view.set_color_grading(cg_settings)

    yfov = 45
    W, H = 640, 480
    fx = H * 0.5 / np.tan(np.deg2rad(yfov) * 0.5)
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        W,
        H,
        fx,
        fy,
        cx,
        cy,
    )

    import copy

    from neuralfeels.contrib.urdf import SceneGraph, URDFParser, URDFTree
    from neuralfeels.modules.allegro import Allegro

    allegro_urdf = os.path.join(
        root, "data/assets/allegro/allegro_digit_left_ball.urdf"
    )
    dataset_path = os.path.join(root, "data/feelsight_occlusion/077_rubiks_cube/00")
    urdf_parser = URDFParser(allegro_urdf)
    urdf_parser.parse()
    urdf_tree = URDFTree(urdf_parser.links, urdf_parser.joints)

    allegro = Allegro(
        dataset_path=dataset_path,
    )

    current_joint_state = allegro.joint_states[0].squeeze()
    init_joints = np.zeros(16)
    allegro_graph = SceneGraph(urdf_tree.root, init_joints)
    allegro_graph.updateJointAngles(current_joint_state)
    allegro_graph.updateState()
    allegro_mesh = allegro_graph.getMesh()

    mesh_mat = rendering.MaterialRecord()
    mesh_mat.shader = "defaultLitTransparency"
    mesh_mat.base_color = [0.467, 0.467, 0.467, 1.0]
    mesh_mat.base_roughness = 1.0
    mesh_mat.base_reflectance = 0.0
    mesh_mat.base_clearcoat = 0.0
    mesh_mat.thickness = 1.0
    mesh_mat.transmission = 0.1
    mesh_mat.absorption_distance = 10
    mesh_mat.absorption_color = [0.467, 0.467, 0.467]

    # Add allegro visualization
    for i, mesh in enumerate(allegro_mesh):
        if i == 16:
            continue  # skip visualizing joint=12.0 which has a parsing error
        # mesh = copy.deepcopy(mesh).transform(allegro.allegro_pose)
        camera_visualizer.scene.remove_geometry(f"allegro_{i}")
        camera_visualizer.scene.add_geometry(f"allegro_{i}", mesh, mesh_mat)

    gt_obj_file = os.path.join(root, "data/assets/gt_models/ycb/077_rubiks_cube.urdf")

    _, mesh_o3d = sdf_util.load_gt_mesh(gt_obj_file, color=True)

    obj_mesh = copy.deepcopy(mesh_o3d).transform(obj_pose)
    camera_visualizer.scene.add_geometry("gt_mesh", obj_mesh, mesh_mat)

    improvements, perc_improvement = [], []
    modalities = ["vision", "vision + touch"]
    for i in range(n_views):
        cam_error = {}
        for modality in modalities:
            # only select the modality vision + touch
            df_pose_occlusion_modality = df_pose_occlusion[
                df_pose_occlusion["modality"] == modality
            ]
            # get avg_3d_error for the camera with cam_id == i
            cam_error[modality] = df_pose_occlusion_modality[
                df_pose_occlusion_modality["cam_id"] == i
            ]["avg_3d_error"].median()
        improvements.append(cam_error["vision"] - cam_error["vision + touch"])
        # percentage improvement
        perc_improvement.append(improvements[-1] * 100 / cam_error["vision"])
        print(
            f" {i} Improvement: {improvements[-1] * 100 / cam_error['vision']:.2f}% {improvements[-1]:.2f} mm"
        )
    improvements = np.array(improvements)
    perc_improvement = np.array(perc_improvement)

    print(
        f"Min % improvement: {np.min(perc_improvement)}, max : {np.max(perc_improvement)}, mean: {np.mean(perc_improvement)}"
    )

    # log scale
    # improvements[improvements > 10] = 10  # cap the max error to 5mm
    improvements = np.log(1 + improvements)

    # find min and max across all modalities
    max_improvement = np.max(improvements)
    min_improvement = np.min(improvements)

    improvements = (improvements - min_improvement) / (
        max_improvement - min_improvement
    )

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"

    def save_viz(image, name: str, id: int):
        o3d.io.write_image(
            os.path.join(root_path, f"{name}_{str(id).zfill(6)}.jpg"),
            image,
            quality=100,
        )
        print(f"Saved {name}_{str(id).zfill(6)}.jpg")

    # Remove the middle 40% of the RdBu_r colormap
    # interval = np.hstack([np.linspace(0, 0.4, n_views//2), np.linspace(0.6, 1, n_views//2)])
    interval = np.hstack([np.linspace(0.0, 1.0, n_views)])
    colors = plt.cm.viridis(interval)
    cmap = LinearSegmentedColormap.from_list("name", colors)

    for i, pose in enumerate(poses):
        cam_frustum = o3d.geometry.LineSet.create_camera_visualization(
            intrinsics.width,
            intrinsics.height,
            intrinsics.intrinsic_matrix,
            np.linalg.inv(pose),
            scale=0.05,
        )

        # 3D sphere at the position of pose
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001, resolution=20)
        sphere.translate(pose[:3, 3])

        # set line width based on improvement percentatile
        if improvements[i] > np.percentile(improvements, 95):
            mat.line_width = 5.0
        elif improvements[i] > np.percentile(improvements, 75):
            mat.line_width = 4.0
        elif improvements[i] > np.percentile(improvements, 50):
            mat.line_width = 3.0
        else:
            mat.line_width = 1.0

        # Assign a colormap to the interpolated value. Use the reverse RdBu colormap
        if improvements[i] > np.percentile(improvements, 5):
            color = cmap(improvements[i])[:3]
        else:
            # white color for the worst 25% of improvements
            color = np.array([1.0, 1.0, 1.0])

        cam_frustum.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)

        # mat.line_width = (
        #     improvements[i] // 5 + 0.5
        # )  # note that this is scaled with respect to pixels,
        camera_visualizer.scene.add_geometry(f"cam_frustum_{i}", cam_frustum, mat)
        # camera_visualizer.scene.add_geometry(f"sphere_{i}", sphere, mesh_mat)

    __save_viz = functools.partial(save_viz, name=f"cam_frustums", id=0)
    gui.Application.instance.post_to_main_thread(
        window,
        lambda: camera_visualizer.scene.scene.render_to_image(__save_viz),
    )
    app.run()


def pose_error_vs_rays(expt_paths, root_path):
    df_pose_noise = pd.DataFrame()
    num_expts = len(expt_paths)
    print(f"Number of experiments: {num_expts}")
    for i, expt_path in enumerate(expt_paths):
        pkl_path = os.path.join(root_path, expt_path, "stats.pkl")
        with open(pkl_path, "rb") as p:
            stats = pickle.load(p)

        pose_error_stats = stats["pose"]["errors"]
        # convert list of dicts into arrays
        avg_3d_error = np.array(
            [
                pose_error_stats[i]["avg_3d_error"] * 1000.0
                for i in range(len(pose_error_stats))
            ]
        )
        avg_avg_3d_error = np.median(avg_3d_error, axis=0)
        rays = expt_path.split("_")[-1]  # number of rays of tactile sensing
        metadata_df = load_metadata(expt_path, stats, i)
        modality = metadata_df["modality"].iloc[-1]
        object_name = metadata_df["object"].iloc[-1]
        # create dataframe with avg_3d_error, noise, and modality
        df = pd.DataFrame(
            {
                "avg_3d_error": avg_avg_3d_error,
                "tactile_rays": rays,
                "modality": modality,
                "object": object_name,
            },
            index=[i],
        )
        df_pose_noise = pd.concat([df_pose_noise, df], axis=0)

    # sort dataframe by noise
    df_pose_noise = df_pose_noise.sort_values("tactile_rays")

    # drop index column
    df_pose_noise = df_pose_noise.reset_index(drop=True)
    # turn nan into 0.0
    save_path = os.path.join(root_path, f"pose_error_vs_tactile_rays_.pdf")
    # Create a boxplot of the avg_3d_error for each optimizer
    fig, ax = plt.subplots()
    print("Plotting...")
    # plot box plot of avg_3d_error vs tactile_rays
    sns.lineplot(
        data=df_pose_noise,
        x="tactile_rays",
        y="avg_3d_error",
        hue="modality",
        estimator=np.median,
        palette=my_pal,
        ax=ax,
    )
    print(df_pose_noise)
    plt.xlabel("Number of tactile rays", fontsize=25)

    ax.legend_.remove()

    ax.set_ylabel("Avg. pose error (mm)", fontsize=25)

    ax.tick_params(axis="y", labelsize=25)
    # x ticks show as integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    min_rays, max_rays = np.min(df_pose_noise["tactile_rays"]), np.max(
        df_pose_noise["tactile_rays"]
    )
    min_error, max_error = np.min(df_pose_noise["avg_3d_error"]), np.max(
        df_pose_noise["avg_3d_error"]
    )

    # x_plot_ticks = np.arange(0, max_rays + 1, 5)
    # plt.xticks(x_plot_ticks)
    # plt.axvline(x=5, color="k", linestyle="--", linewidth=0.5, alpha=0.2)
    ax.set_ylim(top=1.5)

    # plt.axvspan(x_plot_ticks[0], x_plot_ticks[-1], facecolor="gray", alpha=0.05)
    # ax.set_xlim(0, x_plot_ticks[-1])
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")
    plt.close("all")


def map_error_vs_thresh(expt_paths, root_path):
    """
    Plot the curve of f_score vs threshold averaged over all experiments
    """

    # make dataframe with two elements: final_f_score and f_score_T
    df_f_score = pd.DataFrame()
    for i, expt_path in enumerate(expt_paths):
        pkl_path = os.path.join(root_path, expt_path, "stats.pkl")
        with open(pkl_path, "rb") as p:
            stats = pickle.load(p)

        f_scores_Ts = stats["map"]["errors"][0]["f_score_T"]
        f_scores_Ts = np.array(f_scores_Ts) * 1000.0
        metadata_df = load_metadata(expt_path, stats, i)

        for num_f, f_score_T in enumerate(f_scores_Ts):
            map_df = load_map_stats(stats, i, which_f_score=num_f)
            final_f_score = map_df["f_score"].iloc[-1][-1]
            # add to dataframe
            df = pd.DataFrame(
                {
                    "object": metadata_df["object"].iloc[-1],
                    "modality": metadata_df["modality"].iloc[-1],
                    "f_score_T": f_score_T,
                    "final_f_score": final_f_score,
                },
                index=[0],
            )
            df_f_score = pd.concat([df_f_score, df], axis=0)

    # drop index column
    df_f_score = df_f_score.reset_index(drop=True)

    save_path = os.path.join(root_path, f"map_error_vs_threshold.pdf")
    # Create a boxplot of the avg_3d_error for each optimizer
    fig, ax = plt.subplots()
    if "sim" in root_path:
        linestyle = "--"
    else:
        linestyle = "-"
    sns.lineplot(
        data=df_f_score,
        x="f_score_T",
        y="final_f_score",
        hue="modality",
        palette=my_pal,
        linestyle=linestyle,
        linewidth=3,
        errorbar=None,
        ax=ax,
    )
    plt.xlabel("F-score thresholds (mm)", fontsize=25)

    ax.legend_.remove()

    ax.set_ylabel(f"Final F-score", fontsize=25)

    ax.tick_params(axis="y", labelsize=25)
    ax.tick_params(axis="x", labelsize=25)
    # x ticks show as integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0, top=1)

    # equally spaced x ticks between min(f_scores_Ts) and max(f_scores_Ts)
    x_plot_ticks = np.arange(0, max(f_scores_Ts) + 1, 5)
    plt.xticks(x_plot_ticks)
    plt.axvline(x=5, color="k", linestyle="--", linewidth=0.5, alpha=0.2)

    if "sim" in root_path:
        plt.axvspan(x_plot_ticks[0], x_plot_ticks[-1], facecolor="gray", alpha=0.05)
    ax.set_xlim(min(f_scores_Ts), x_plot_ticks[-1])
    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Saved {save_path}")
    plt.close("all")


def draw_pose_error(expt_path=".", slam_mode="slam", smoothing_factor=5):
    """
    ADD tracking error over time for a single trial.
    """

    stats_file = os.path.join(expt_path, "stats.pkl")
    with open(stats_file, "rb") as p:
        stats = pickle.load(p)
    pose_error_stats = stats["pose"]["errors"]
    save_path = os.path.join(expt_path, "track_error.pdf")
    # convert list of dicts into arrays
    avg_3d_error = np.array(
        [
            pose_error_stats[i]["avg_3d_error"] * 1000.0
            for i in range(len(pose_error_stats))
        ]
    )
    timestamps = np.array(
        [pose_error_stats[i]["time"] for i in range(len(pose_error_stats))]
    )
    # ignore the last frame (may be unoptimized)
    avg_3d_error, timestamps = avg_3d_error[:-1], timestamps[:-1]
    avg_3d_error = np.abs(avg_3d_error)

    # remove outliers due to ground-truth failures
    timestamps = timestamps[avg_3d_error < 20]
    avg_3d_error = avg_3d_error[avg_3d_error < 20]

    # if slam_mode == "slam":
    #     # set anything with timestamp < 5 to nan
    avg_3d_error[timestamps < 5] = np.nan

    avg_3d_error = smooth_data(avg_3d_error, N=smoothing_factor)

    # color_choice = "k"
    if "vitac" in expt_path:
        color_choice = my_pal["vision + touch"]
    else:
        color_choice = my_pal["vision"]

    dataset_length = timestamps[-1]
    # get checkpoints at equal intervals of timestamps
    checkpoints = np.arange(0, dataset_length + 5, 10, dtype=int)
    print(checkpoints)

    # ADD plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(
        timestamps,
        avg_3d_error,
        linestyle="-",
        linewidth=2,
        color=color_choice,
    )

    ax1.set_xlabel("Timestep", fontsize=25)
    # ax1.set_ylabel("Avg. pose error (mm)", fontsize=25)
    ax1.set_ylabel("Pose drift from ground-truth\n initialization (mm)", fontsize=18)
    print(f"Final pose drift: {avg_3d_error.iloc[-1]} mm")

    ax1.tick_params(axis="y", labelsize=25)

    plt.xticks(np.array(checkpoints), [str(x) for x in checkpoints])

    labels = [item.get_text() for item in ax1.get_xticklabels()]

    ax1.set_xticklabels(labels)
    ax1.tick_params(axis="x", labelsize=25)

    for xc in checkpoints:
        plt.axvline(x=xc, color="k", linestyle="--", linewidth=0.5, alpha=0.2)

    plt.axvline(x=5, color="k", linestyle="--", linewidth=0.5, alpha=0.2)

    plt.axvspan(0, dataset_length + 1, facecolor="gray", alpha=0.05)

    ax1.set_xlim(0.0, 30)
    ax1.set_ylim(bottom=0.0, top=10.0)

    plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Tracking graph saved at: {save_path}")
    plt.close("all")


def draw_map_error(expt_path=".", smoothing_factor=100):
    """
    f_score error over time for a single trial.
    """

    stats_file = os.path.join(expt_path, "stats.pkl")
    with open(stats_file, "rb") as p:
        stats = pickle.load(p)

    map_error_stats = stats["map"]["errors"]
    # convert list of dicts into arrays
    f_scores = np.array(
        [map_error_stats[i]["f_score"] for i in range(len(map_error_stats))]
    )
    precisions = np.array(
        [map_error_stats[i]["precision"] for i in range(len(map_error_stats))]
    )
    recalls = np.array(
        [map_error_stats[i]["recall"] for i in range(len(map_error_stats))]
    )
    timestamps = np.array(
        [map_error_stats[i]["time"] for i in range(len(map_error_stats))]
    )
    f_score_T = map_error_stats[0]["f_score_T"]

    for i, t in enumerate(f_score_T):
        t_mm = int(t * 1000)
        f_score, precision, recall = f_scores[:, i], precisions[:, i], recalls[:, i]
        save_path = os.path.join(expt_path, f"map_error_{t_mm}_mm.pdf")
        f_score = smooth_data(f_score, N=smoothing_factor)
        precision = smooth_data(precision, N=smoothing_factor)
        recall = smooth_data(recall, N=smoothing_factor)

        fig, ax = plt.subplots()
        dataset_length = timestamps[-1]
        # get checkpoints at equal intervals of timestamps
        checkpoints = np.arange(0, dataset_length + 5, 10, dtype=int)

        # f_score plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.plot(timestamps, f_score, linestyle="-", linewidth=2, color="k")
        plt.plot(
            timestamps,
            precision,
            linestyle="-",
            linewidth=1,
            color=my_pal["precision"],
            alpha=0.5,
        )
        plt.plot(
            timestamps,
            recall,
            linestyle="-",
            linewidth=1,
            color=my_pal["recall"],
            alpha=0.5,
        )

        ax1.set_xlabel("Timestep", fontsize=25)
        ax1.set_ylabel(
            f"Map F-score (< {t_mm} mm)",
            fontsize=25,
        )

        ax1.tick_params(axis="y", labelsize=25)
        plt.xticks(np.array(checkpoints), [str(x) for x in checkpoints])

        labels = [item.get_text() for item in ax1.get_xticklabels()]

        ax1.set_xticklabels(labels)
        ax1.tick_params(axis="x", labelsize=25)

        for xc in checkpoints:
            plt.axvline(x=xc, color="k", linestyle="--", linewidth=0.5, alpha=0.2)

        plt.axvspan(0, dataset_length + 1, facecolor="gray", alpha=0.05)

        ax1.set_xlim(0.0, 30)
        ax1.set_ylim(bottom=0.4, top=1.02)
        plt.savefig(save_path, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.close("all")
        print(f"Mapping graphs saved at: {save_path}")


def draw_grasp_error(pose_error, xcaption, ycaption, savepath):
    """
    Localization error for grasp trials, refer to grasp_error.py
    """
    # print items from pose_error_pd
    # get modalities and noise levels
    mods = list(pose_error.keys())
    noise_levels = list(pose_error[mods[0]].keys())
    cb_colors = sns.color_palette("colorblind")
    n_plots = len(mods)
    color_idxs = np.linspace(0, len(cb_colors), n_plots, dtype=int, endpoint=False)
    plotlays, plotcols = [n_plots], [cb_colors[i] for i in color_idxs]

    fig, ax = plt.subplots()

    for mod, plotcol in zip(mods, plotcols):
        pose_error_mod = pose_error[mod]
        # average results for all noise_i elements over number of trials
        # find the first number after underscore
        last_trial = list(pose_error_mod)[-1]  # get the last trial name
        num_trials = int(re.findall(r"\d+", last_trial)[0]) + 1

        pose_error_consolidated = {}
        for n in range(num_trials):
            noise_level = f"noise_{n}"
            matching_dists = [
                value["dist"]
                for key, value in pose_error_mod.items()
                if noise_level in key.lower()
            ]
            matching_times = [
                value["time"]
                for key, value in pose_error_mod.items()
                if noise_level in key.lower()
            ]
            min_length = len(min(matching_dists, key=len))
            matching_dists = [x[:min_length] for x in matching_dists]
            matching_times = [x[:min_length] for x in matching_times]
            matching_dists = np.median(np.vstack(matching_dists), axis=0)
            matching_times = np.median(np.vstack(matching_times), axis=0)

            pose_error_consolidated[noise_level] = {
                "dist": matching_dists,
                "time": matching_times,
            }  # average over all trials for noise level noise_n
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # get x and y data for modality
        x = np.arange(len(pose_error_consolidated.keys()))
        y = []
        for noise_level in pose_error_consolidated.keys():
            y.append(pose_error_consolidated[noise_level]["dist"][-1])
        y = np.stack(y)
        ax.plot(x, y, "-", c=plotcol, alpha=1.0, label=f"{mod}")

        print(f"{mod} : {(255 * np.array(plotcol)).astype(int)}")

    plt.legend(loc="upper left")

    logmap = True
    if logmap:
        plt.yscale("log")

    plt.xlabel(xcaption, fontsize=12)
    plt.ylabel(ycaption + (" (log scale)" if logmap else ""), fontsize=12)

    fig.savefig(savepath + ".pdf", transparent=True, bbox_inches="tight", pad_inches=0)
