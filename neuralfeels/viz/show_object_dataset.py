# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Viser visualization script for objects in the FeelSight dataset.
# pip install viser before running this script

import os
import time
from pathlib import Path

import git
import numpy as np
import trimesh
import viser

root = git.Repo(".", search_parent_directories=True).working_tree_dir


def main(
    model_path: Path = os.path.join(root, "data", "assets", "gt_models", "ycb")
) -> None:
    # get list of folders in model_path
    object_names = os.listdir(model_path)
    # remove urdf files

    if "ycb" in model_path:
        object_names = [
            "contactdb_rubber_duck",
            "contactdb_elephant",
            "077_rubiks_cube",
            "large_dice",
            "016_pear",
            "015_peach",
            "010_potted_meat_can",
            "073-f_lego_duplo",
        ]
    else:
        object_names = [
            x for x in object_names if not x.endswith(".urdf") and x != ".DS_Store"
        ]

    server = viser.ViserServer()

    def add_selectable_mesh(
        name: str, mesh: trimesh.Trimesh, x: float, y: float
    ) -> None:
        def add_mesh() -> None:
            handle = server.add_mesh_trimesh(
                "/" + name,
                mesh=mesh,
                # vertices=mesh.vertices,
                # faces=mesh.faces,
                position=(y, 0.0, x),
                # color=colorsys.hls_to_rgb(
                #     np.random.default_rng(
                #         np.frombuffer(
                #             hashlib.md5(name.encode("utf-8")).digest(),
                #             dtype="uint32",
                #         )
                #         + 5
                #     ).uniform(),
                #     0.6,
                #     0.9,
                # ),
            )

            # Requires the cmk/add_click branch of viser.
            # handle.clickable = True
            # @handle.on_click
            def _(_) -> None:
                add_mesh()

        add_mesh()

    nominal_column_width = len(object_names)
    rows_indices = np.array_split(
        np.arange(len(object_names)), np.rint(len(object_names) / nominal_column_width)
    )
    mesh_diags = []
    for row, row_indices in enumerate(rows_indices):
        for col, mesh_index in enumerate(row_indices):
            x = row * 0.12
            y = col * nominal_column_width * 0.12 / len(row_indices)
            mesh_path = os.path.join(
                model_path, object_names[mesh_index], "textured.obj"
            )
            # check if mesh_path exists
            if not os.path.exists(mesh_path):
                mesh_path = os.path.join(
                    model_path, object_names[mesh_index], "google_16k", "textured.obj"
                )
            if not os.path.exists(mesh_path):
                mesh_path = os.path.join(
                    model_path,
                    object_names[mesh_index],
                    f"{object_names[mesh_index]}.obj",
                )
            mesh = trimesh.load(
                mesh_path,
                force="mesh",
            )
            if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
                # TextureVisuals are not supported by viser yet
                mesh.visual = mesh.visual.to_color()

            # append mesh diagonal
            mesh_diags.append(mesh.scale)
            print(f"Added {object_names[mesh_index]} at ({x}, {y})")
            print(f"Object: {object_names[mesh_index]}, mesh diagonal: {mesh.scale}")
            add_selectable_mesh(object_names[mesh_index], mesh, x=x, y=y)

    # print min and max mesh diagonal
    mesh_diags = np.array(mesh_diags)
    print(f"Min mesh diagonal: {np.min(mesh_diags)}")
    print(f"Max mesh diagonal: {np.max(mesh_diags)}")
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    # main()
    main(
        model_path=os.path.join(root, "data", "assets", "gt_models", "ycb")
    )  # sim dataset
