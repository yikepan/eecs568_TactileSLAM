# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import pandas as pd


def smooth_data(y, N=5):
    # rolling avg. over N timesteps
    df = pd.DataFrame()
    df["y"] = y
    df_smooth = df.rolling(N).mean()
    df_smooth["y"][0 : N - 1] = y[0 : N - 1]  # first N readings are as-is
    return df_smooth["y"]


feelsight_sim_objects = [
    "contactdb_rubber_duck",
    "contactdb_elephant",
    "077_rubiks_cube",
    "large_dice",
    "016_pear",
    "015_peach",
    "010_potted_meat_can",
    "073-f_lego_duplo",
]

feelsight_sim_mesh_diag = {
    "contactdb_rubber_duck": 0.14833374114812853,
    "contactdb_elephant": 0.1850651169858869,
    "077_rubiks_cube": 0.12201651401757059,
    "large_dice": 0.08720458052763055,
    "016_pear": 0.13722709752814855,
    "015_peach": 0.10593046598594759,
    "010_potted_meat_can": 0.1449591345276316,
    "073-f_lego_duplo": 0.06760945759285457,
}

feelsight_real_objects = [
    "bell_pepper",
    "large_dice",
    "peach",
    "pear",
    "pepper_grinder",
    "rubiks_cube_small",
]

feelsight_real_mesh_diag = {
    "bell_pepper": 0.14895704905777368,
    "large_dice": 0.08720458052763055,
    "peach": 0.10578790231401698,
    "pear": 0.13838421462002087,
    "pepper_grinder": 0.14848234731441984,
    "rubiks_cube_small": 0.09042267417523107,
}
