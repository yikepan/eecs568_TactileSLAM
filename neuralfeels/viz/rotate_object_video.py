# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to rotate a sequence of meshes and save as a video. 
"""

import os
import pathlib
import time

import cv2
import ffmpeg
import git
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

root = git.Repo(".", search_parent_directories=True).working_tree_dir


def draw_rotating_geometry(mesh_dir, mesh_file):
    # create folder to save images
    image_path = os.path.join(mesh_dir, "images")

    def get_orbit(final_mesh, timsteps=400, num_orbits=1):
        diag = np.linalg.norm(
            np.asarray(final_mesh.get_max_bound())
            - np.asarray(final_mesh.get_min_bound())
        )
        radius = diag * 1.5
        # initialize camera at 45 degrees of circle
        orbit_size = timsteps // num_orbits
        theta = np.linspace(0, 2 * np.pi, orbit_size)
        z = np.zeros(orbit_size) + 0.33 * radius
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        traj = np.vstack((x, y, z)).transpose()
        center = final_mesh.get_center()
        offset_traj = traj + center

        offset_traj = np.tile(offset_traj, (num_orbits, 1))
        return offset_traj, center

    final_mesh = o3d.io.read_triangle_mesh(mesh_file)

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # delete existing images
    for file in os.listdir(image_path):
        os.remove(os.path.join(image_path, file))

    # 30 seconds of video, with 20*30 = 600 frames
    num_iters = 500
    orbit_path, center = get_orbit(final_mesh, timsteps=num_iters, num_orbits=1)

    render = rendering.OffscreenRenderer(1000, 1000)
    render.setup_camera(60.0, [0, 0, 0], [0, 10, 0], [0, 0, 1])
    render.scene.set_background([1, 1, 1, 1])
    # render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
    #                                  75000)
    # render.scene.scene.enable_sun_light(True)

    # Address the white background issue: https://github.com/isl-org/Open3D/issues/6020
    cg_settings = rendering.ColorGrading(
        rendering.ColorGrading.Quality.ULTRA,
        rendering.ColorGrading.ToneMapping.LINEAR,
    )

    obj_mat = rendering.MaterialRecord()
    mat_properties = {
        "metallic": 0.5,
        "roughness": 0.6,
        "reflectance": 0.2,
        "clearcoat": 0.0,
        "clearcoat_roughness": 0.0,
        "anisotropy": 0.3,
    }
    obj_mat.base_color = [0.9, 0.9, 0.9, 1.0]
    obj_mat.shader = "defaultLit"
    for key, val in mat_properties.items():
        setattr(obj_mat, "base_" + key, val)

    for i in range(num_iters):
        render.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
            -np.array(orbit_path[i, :] + [0.0, 0.0, 0.01]),
        )

        if i == 0:
            pcd = o3d.io.read_triangle_mesh(mesh_file, True)
            render.scene.add_geometry("pcd", pcd, obj_mat)
        render.setup_camera(60.0, center, orbit_path[i, :], [0, 0, 1])
        render.scene.view.set_color_grading(cg_settings)

        """capture images"""
        img = render.render_to_image()
        time_label = i
        o3d.io.write_image(os.path.join(image_path, f"{time_label:03d}.jpg"), img, 99)

    save_path = os.path.join(mesh_dir, "mesh_viz.mp4")
    create_video(image_path, save_path, 30, 20)


def get_int(file: str) -> int:
    """
    Extract numeric value from file name
    """
    return int(file.split(".")[0])


def create_video(path, save_path, length=30, fps=20):
    images = os.listdir(path)
    images = [im for im in images if im.endswith(".jpg")]

    images = sorted(images, key=get_int)

    interval = 1000.0 / fps

    # Execute FFmpeg sub-process, with stdin pipe as input, and jpeg_pipe input format
    process = (
        ffmpeg.input("pipe:", r=str(fps))
        .output(save_path, pix_fmt="yuv420p")
        .overwrite_output()
        .global_args("-loglevel", "warning")
        .global_args("-qscale", "0")
        .global_args("-y")
        .run_async(pipe_stdin=True)
    )

    for image in images:
        image_path = os.path.join(path, image)
        im = cv2.imread(image_path)
        success, encoded_image = cv2.imencode(".png", im)
        process.stdin.write(
            encoded_image.tobytes()
        )  # If broken pipe error, try mamba update ffmpeg

    # Close stdin pipe - FFmpeg fininsh encoding the output file.
    process.stdin.close()
    process.wait()


def get_last_folders(root_dir):
    """
    Recursively traverse down all directories until we reach the last folders, and store those in a list.
    """
    last_folders = []
    for path in root_dir.iterdir():
        if path.is_dir():
            # if only an obj file exists, then we have reached the last folder
            if len(list(path.glob("*.obj"))) == 1:
                last_folders.append(path)
            else:
                last_folders.extend(get_last_folders(path))

    if len(last_folders) == 0:
        last_folders = [root_dir]
    return last_folders


# define main function
if __name__ == "__main__":
    mesh_dir = pathlib.Path(root) / "data/results/mesh_trials/sim"
    all_mesh_dirs = get_last_folders(mesh_dir)
    for mesh_dir in all_mesh_dirs:
        # convert posix path to string
        print(f"Processing {mesh_dir}")
        # get all .obj files in mesh_dir
        mesh_files = list(mesh_dir.glob("*.obj"))
        final_mesh_path = None
        # check if final mesh exists
        if len(
            [
                x.name
                for x in mesh_files
                if (("final" in x.name) or (x.name == "textured.obj"))
            ]
        ):
            final_mesh_path = [
                x.name
                for x in mesh_files
                if (("final" in x.name) or (x.name == "textured.obj"))
            ][0]
            final_mesh_path = str(mesh_dir / final_mesh_path)

        if final_mesh_path is not None:
            draw_rotating_geometry(mesh_dir, final_mesh_path)
