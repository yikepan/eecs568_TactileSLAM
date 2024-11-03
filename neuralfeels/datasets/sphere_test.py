# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Generate synthetic data by sampling views in a sphere around an object

import argparse
import os
import pickle
from os import path as osp

import cv2
import imgviz
import matplotlib

matplotlib.use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc.
import matplotlib.pyplot as plt
import numpy as numpy
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pyrender
import trimesh
from PIL import Image

from neuralfeels.datasets import sdf_util
from neuralfeels.geometry import transform
from neuralfeels.modules.misc import remove_and_mkdir
from neuralfeels.viz.draw import draw_cams


class vision_sim:
    def __init__(self, scene_file, intrinsics, n_views=100):
        self.n_views = n_views
        self.scene_file = scene_file

        # Add noise with sigma
        self.color_noise = 5  # pixels
        self.depth_noise = 10e-3  # meters
        # depth noise: https://tinyurl.com/yufb43tp

        # Redwood Indoor LivingRoom1 (Augmented ICL-NUIM)
        # http://redwood-data.org/indoor/
        data = o3d.data.RedwoodIndoorOffice1()
        noise_model_path = data.noise_model_path
        im_src_path = data.depth_paths[0]
        self.simulator = o3d.t.io.DepthNoiseSimulator(noise_model_path)
        # Read clean depth image (uint16)
        self.test_depth_clean = o3d.t.io.read_image(im_src_path)

        # dense point cloud with vertice information
        fuze_trimesh, _ = sdf_util.load_gt_mesh(scene_file)
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        self.radius = 4.0 * fuze_trimesh.bounding_sphere.primitive.radius
        # print(f"Sphere radius: {radius}")

        # set camera params
        self.intrinsics = intrinsics
        print("Camera intrinsics for sphere sim:")
        print(self.intrinsics)

        self.scene = pyrender.Scene()
        self.scene.add(mesh)
        yfov = 2 * np.arctan(0.5 * intrinsics["h"] / intrinsics["fy"])
        camera = pyrender.PerspectiveCamera(yfov=yfov, znear=0.01)

        self.obj_center = mesh.centroid
        self.obj_pose = np.eye(4)

        # compute cam views and add single camera to pyrender scene
        poses_o3d = transform.look_at_on_sphere(
            self.n_views, self.radius, self.obj_center
        )
        poses_ogl = transform.to_trimesh(poses_o3d)
        self.cam_poses = poses_ogl  # in OpenGL/pyrender convention
        self.scene.add(camera, pose=self.cam_poses[0])

        # add lights on sphere
        self.n_lights = 4
        light_poses_o3d = transform.look_at_on_sphere(
            self.n_lights, self.radius * 1.2, self.obj_center
        )
        self.light_poses = transform.to_trimesh(
            light_poses_o3d
        )  # in OpenGL/pyrender convention
        for pose in self.light_poses:
            light = pyrender.SpotLight(
                color=np.ones(3),
                intensity=1e-2,
                innerConeAngle=np.pi / 16.0,
                outerConeAngle=np.pi / 6.0,
            )
            nl = pyrender.Node(light=light, matrix=pose)
            self.scene.add_node(nl)

        self.r = pyrender.OffscreenRenderer(intrinsics["w"], intrinsics["h"])

    def set_obj_pose(self, pose):
        self.obj_pose = pose

        obj_node = list(self.scene.mesh_nodes)[0]
        self.scene.set_pose(obj_node, self.obj_pose)

        self.obj_center = obj_node.mesh.centroid + pose[:3, 3]

        cam_poses_o3d = transform.look_at_on_sphere(
            self.n_views, self.radius, self.obj_center
        )
        self.cam_poses = transform.to_trimesh(cam_poses_o3d)

        light_poses_o3d = transform.look_at_on_sphere(
            self.n_lights, self.radius * 1.2, self.obj_center
        )
        self.light_poses = transform.to_trimesh(light_poses_o3d)
        for i, pose in enumerate(self.light_poses):
            light_node = list(self.scene.spot_light_nodes)[i]
            self.scene.set_pose(light_node, pose)

    def render(self, idx, noisy=True):
        # set camera pose
        pose = self.cam_poses[idx]
        self.scene.set_pose(self.scene.main_camera_node, pose)

        # render color, depth
        color, depth_clean = self.r.render(self.scene)

        # render segmentation
        nm = {node: i + 1 for i, node in enumerate(self.scene.mesh_nodes)}
        seg = self.r.render(self.scene, pyrender.RenderFlags.SEG, nm)[0][:, :, 0]
        seg = seg.astype(bool)

        # need to flip images to account for OpenGL to Open3D conversion
        color = np.flip(color, axis=1).copy()
        depth_clean = np.flip(depth_clean, axis=1).copy()
        seg = np.flip(seg, axis=1).copy()

        if not noisy:
            # self.simulator.enable_deterministic_debug_mode()
            depth = depth_clean

        else:
            color_noise = np.random.normal(0, self.color_noise, color.shape)
            color = np.clip(color + color_noise, 0, 255).astype(np.uint8)

            og_shape = depth_clean.shape
            depth_clean = cv2.resize(
                depth_clean, (640, 480), interpolation=cv2.INTER_NEAREST
            )
            depth = o3d.t.geometry.Image(o3c.Tensor(depth_clean))
            depth = self.simulator.simulate(depth, depth_scale=1.0)  # 0.00137 avg depth
            depth = np.array(depth).squeeze()
            depth = cv2.resize(
                depth, (og_shape[1], og_shape[0]), interpolation=cv2.INTER_NEAREST
            )

        pose_o3d = transform.to_trimesh(pose)
        return color, depth, seg, pose_o3d

    def viz_scene(self, pc=None, colors=None):
        fuze_trimesh, _ = sdf_util.load_gt_mesh(self.scene_file)
        scene = trimesh.Scene()
        scene.add_geometry(fuze_trimesh, transform=self.obj_pose)
        # convert to poses to trimesh
        cam_poses_tm = transform.to_trimesh(self.cam_poses)
        light_poses_tm = transform.to_trimesh(self.light_poses)
        draw_cams(self.n_views, cam_poses_tm, scene, cam_scale=0.02, latest_diff=False)
        draw_cams(
            self.n_lights,
            light_poses_tm,
            scene,
            cam_scale=0.02,
            latest_diff=False,
            color=[1, 0, 0],
        )
        if pc is not None:
            scene.add_geometry(trimesh.PointCloud(pc, colors=colors))
        scene.show()

    def plot_data(self, color, depth, seg):
        fig = plt.figure()
        ax = plt.subplot(1, 3, 1)
        # ax.set_title("Image rendering", size=10)
        plt.axis("off")
        plt.imshow(color)
        ax = plt.subplot(1, 3, 2)
        # ax.set_title("Depth image", size=10)
        plt.axis("off")
        depth[depth == 0] = np.nan
        plt.imshow(imgviz.depth2rgb(depth))
        plt.subplot(1, 3, 3)
        plt.axis("off")
        plt.imshow(seg.astype(np.float32), cmap=plt.cm.gray_r)
        fig.tight_layout()

        plt.show(block=False)
        plt.pause(15)
        plt.close()

    def pcd_from_depth(self, ix, depth, color, seg):
        depth = depth * seg  # apply object mask
        pc_cam = transform.backproject_pointclouds(
            depth[None, ...],
            self.intrinsics["fx"],
            self.intrinsics["fy"],
            self.intrinsics["cx"],
            self.intrinsics["cy"],
        ).squeeze()
        valid = ~np.isnan(pc_cam).any(axis=-1)
        pc_cam = pc_cam[valid]
        color = color.reshape(-1, 3)[valid]

        T_WC_ogl = self.cam_poses[ix]
        T_WC_o3d = transform.to_trimesh(T_WC_ogl)
        pc_world = transform.transform_points_np(pc_cam, T_WC_o3d)

        return pc_world, color

    def save_dataset(self, data_path, dummy_allegro_data):
        print(f"\nSaving sphere sim data with {self.n_views} views to: {data_path}")

        obj_name = self.scene_file.split("/")[-1].split(".")[0]

        # save many views as different time steps of single camera
        times = np.arange(1, self.n_views + 1) / 30.0
        obj_poses = self.obj_pose[None, ...].repeat(self.n_views, axis=0)
        dummy_allegro_data["finger_poses"] = (
            dummy_allegro_data["finger_poses"] * self.n_views
        )
        dummy_allegro_data["joint_state"] = (
            dummy_allegro_data["joint_state"] * self.n_views
        )
        data = {
            "time": times,
            "realsense": {},
            "object": {
                "name": obj_name,
                "mesh": self.scene_file,
                "pose": obj_poses,
            },
            "allegro": dummy_allegro_data,
        }

        realsense_path = os.path.join(data_path, "realsense")
        remove_and_mkdir(realsense_path)

        images, depths, segs, poses = [], [], [], []
        for idx in range(self.n_views):
            # render images
            color, depth, seg, pose = self.render(idx, noisy=False)
            images.append(color)
            depths.append(depth)
            segs.append(seg)
            poses.append(pose)

        cam_name = f"front-left"
        camera_path = os.path.join(data_path, "realsense", cam_name)

        # save images
        img_path = os.path.join(camera_path, "image")
        remove_and_mkdir(img_path)
        for i, img in enumerate(images):
            img = Image.fromarray(img.astype("uint8"), "RGB")
            img.save(f"{img_path}/{i}.jpg")

        # save depths
        np.savez_compressed(
            os.path.join(camera_path, "depth.npz"),
            depth=depths,
            depth_scale=np.array(1.0),
        )

        # save segs
        seg_path = os.path.join(camera_path, "seg")
        remove_and_mkdir(seg_path)
        for i, seg in enumerate(segs):
            cv2.imwrite(
                f"{seg_path}/{i}.jpg", (255 * seg / np.max(seg)).astype("uint8")
            )

        data["realsense"][cam_name] = {
            "depth_scale": 1.0,
            "pose": np.array(poses),
            "intrinsics": self.intrinsics,
        }

        file_path = os.path.join(data_path, "data.pkl")
        with open(file_path, "wb") as file:
            pickle.dump(data, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_name", type=str, default="077_rubiks_cube")
    parser.add_argument("--n_views", type=int, default=100)
    args = parser.parse_args()

    obj_name = args.obj_name
    n_views = args.n_views

    obj_path = f"data/assets/gt_models/ycb/{obj_name}.urdf"

    # load existing feelsight data as basis for generating new data
    existing_data_path = f"data/feelsight/{obj_name}/00/data.pkl"
    with open(existing_data_path, "rb") as p:
        data = pickle.load(p)

    intrinsics = data["realsense"]["front-left"]["intrinsics"]

    obj_pose = data["object"]["pose"][0]

    dummy_allegro_data = data["allegro"]
    dummy_allegro_data["finger_poses"] = [dummy_allegro_data["finger_poses"][0]]
    dummy_allegro_data["joint_state"] = [dummy_allegro_data["joint_state"][0]]

    generator = vision_sim(obj_path, intrinsics, n_views=n_views)
    generator.set_obj_pose(obj_pose)

    data_path = f"data/sphere_sim/{obj_name}/{n_views}_views/"
    # generator.save_dataset(
    #     data_path,
    #     dummy_allegro_data,
    # )

    """
    Example code to generate and visualize views and 3D scene
    """
    pcs = []
    colors = []
    for idx in range(n_views):
        color, depth, seg, pose = generator.render(idx, noisy=False)
        # generator.plot_data(color, depth, seg)
        pc, color = generator.pcd_from_depth(idx, depth, color, seg)
        pcs.append(pc)
        colors.append(color)

    pcs = np.concatenate(pcs)
    colors = np.concatenate(colors)
    generator.viz_scene(pcs, colors=colors)
