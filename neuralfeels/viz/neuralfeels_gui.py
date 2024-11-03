# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Open3D interactive GUI for NeuralFeels, loosely based on iSDF: https://github.com/facebookresearch/iSDF

import bisect
import copy
import functools
import logging
import os
import pathlib
import shutil
import threading
import time
import tkinter as tk

import cv2
import dill as pickle
import git
import matplotlib.pylab as plt
import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
import yappi
from hydra.utils import get_original_cwd
from termcolor import cprint

from neuralfeels import geometry
from neuralfeels.contrib.urdf import SceneGraph, URDFParser, URDFTree
from neuralfeels.datasets import sdf_util
from neuralfeels.eval.metrics import compute_f_score
from neuralfeels.viz.plot_metrics import draw_map_error, draw_pose_error

root = git.Repo(".", search_parent_directories=True).working_tree_dir


logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable


def _save_image(image, id: int):
    save_dir = "./images/open3d"
    o3d.io.write_image(
        os.path.join(save_dir, f"{str(id).zfill(6)}.jpg"), image, quality=100
    )


def _save_screenshot(image):
    save_dir = "./images/screenshots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get incrementing save id
    screenshot_id = 0
    while os.path.exists(f"{save_dir}/screenshot_{str(screenshot_id).zfill(6)}.jpg"):
        screenshot_id += 1
    screenshot_path = f"{save_dir}/screenshot_{str(screenshot_id).zfill(6)}.jpg"
    o3d.io.write_image(screenshot_path, image, quality=100)
    print(f"Saved screenshot to {screenshot_path}")


class GUI:
    def __init__(self, trainer, optim_iter, font_id, size_ratio, profile: bool = False):
        self.trainer = trainer  # trainer class
        self.optim_iter = optim_iter  # optimization funciton
        self.profile = profile  # profile the code
        self.intrinsic = {}
        for key, value in trainer.sensor.items():
            self.intrinsic[key] = o3d.camera.PinholeCameraIntrinsic(
                value.W,
                value.H,
                value.fx,
                value.fy,
                value.cx,
                value.cy,
            )

        tk_root = tk.Tk()
        tk_root.withdraw()
        w, h = tk_root.winfo_screenwidth(), tk_root.winfo_screenheight()

        # set minimum resolution to 1080p
        w = max(w, 1920)
        h = max(h, 1080)
        print(f"Screen size: {w} x {h}")

        font_sans_serif = gui.FontDescription(
            typeface="sans-serif", style=gui.FontStyle.NORMAL, point_size=15
        )
        font_id_sans_serif = gui.Application.instance.add_font(font_sans_serif)
        self.window = gui.Application.instance.create_window(
            "NeuralFeels viewer", int(w * size_ratio * 0.75), int(h * size_ratio)
        )

        mode = "incremental" if trainer.incremental else "batch"

        self.start_optimize = False if "grasp" in self.trainer.train_mode else True
        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.1 * em))
        vspacing = int(np.round(0.1 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)
        # Second panel
        self.stream_panel = gui.Vert(spacing, margins)

        ## Application control
        button_play_pause = gui.ToggleSwitch("Resume/Pause [space]")
        button_play_pause.set_on_clicked(self._on_switch)

        self.curr_sensor = None

        ### Info panel
        self.output_info = gui.Label("Output info")
        self.output_info.font_id = font_id

        ## Items in vis props
        self.vis_prop_grid = gui.VGrid(
            2, spacing, gui.Margins(0.1 * em, 0, 0.1 * em, 0)
        )

        self.vis_prop_grid.add_child(gui.Label(""))
        self.vis_prop_grid.add_child(gui.Label(""))

        self.write_all_meshes = self.trainer.cfg_viz.meshes.write_all_meshes
        self.save_rotate = self.trainer.cfg_viz.meshes.save_rotate
        self.save_neural_field = self.trainer.cfg_viz.meshes.save_neural_field
        self.render_open3d = self.trainer.cfg_viz.misc.render_open3d

        mesh_label = gui.Label("(m)esh reconstruction")
        self.mesh_box = gui.Checkbox("")
        self.mesh_box.checked = self.trainer.cfg_viz.meshes.mesh_rec
        self.mesh_box.set_on_checked(self._toggle_mesh)
        self.vis_prop_grid.add_child(mesh_label)
        self.vis_prop_grid.add_child(self.mesh_box)

        # modify fps of dataset
        train_fps_label = gui.Label("\t\tTrain FPS")
        self.fps_slider = gui.Slider(gui.Slider.DOUBLE)
        self.fps_slider.set_limits(0.01, 30)
        self.fps_slider.double_value = self.trainer.train_fps
        self.fps_slider.set_on_value_changed(self._change_train_fps)

        self.vis_prop_grid.add_child(train_fps_label)
        self.vis_prop_grid.add_child(self.fps_slider)

        voxel_dim_label = gui.Label("\t\tMeshing voxel grid dim")
        self.voxel_dim_slider = gui.Slider(gui.Slider.INT)
        self.voxel_dim_slider.set_limits(20, 1000)
        self.voxel_dim_slider.int_value = self.trainer.grid_dim
        self.voxel_dim_slider.set_on_value_changed(self._change_voxel_dim)
        self.vis_prop_grid.add_child(voxel_dim_label)
        self.vis_prop_grid.add_child(self.voxel_dim_slider)

        self.crop_box = gui.Checkbox("(c)op near pc")
        self.crop_box.checked = self.trainer.cfg_viz.meshes.mesh_rec_crop
        self.vis_prop_grid.add_child(self.crop_box)

        self.viz_ray = gui.Checkbox("(v)iz ray samples")
        self.viz_ray.checked = self.trainer.cfg_viz.debug.rays
        self.vis_prop_grid.add_child(self.viz_ray)

        self.scene_bb = o3d.geometry.OrientedBoundingBox(
            self.trainer.scene_center,
            self.trainer.p_WO_W_gt_np[:3, :3],
            self.trainer.obj_extents_np,
        )
        self.scene_bb.color = [1.0, 1.0, 1.0]
        self.scene_bb_box = gui.Checkbox("Scene (b)ounding box")
        self.scene_bb_box.checked = self.trainer.cfg_viz.debug.bbox
        self.vis_prop_grid.add_child(self.scene_bb_box)

        size = 0.06
        self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self.origin_box = gui.Checkbox("Origin")
        self.origin_box.checked = self.trainer.cfg_viz.debug.origin
        self.vis_prop_grid.add_child(self.origin_box)

        self.latest_frame = None
        self.latest_visual_frame = None

        self.slices_box = gui.Checkbox("SDF slices")
        self.slices_box.checked = False
        self.slices_box.set_on_checked(self._toggle_slices)
        self.vis_prop_grid.add_child(self.slices_box)

        allegro_urdf = os.path.join(
            root, "data/assets/allegro/allegro_digit_left_ball.urdf"
        )
        # Parse the URDF file
        urdf_parser = URDFParser(allegro_urdf)
        urdf_parser.parse()
        # Construct the URDF tree
        self.urdf_tree = URDFTree(urdf_parser.links, urdf_parser.joints)

        self.gt_mesh = None
        if self.trainer.cfg_viz.meshes.has_gt_object:
            self.gt_mesh = (
                self.trainer.gt_obj_mesh_o3d
            )  # o3d.io.read_triangle_mesh(self.trainer.scene_file)
            self.gt_mesh.compute_vertex_normals()

        # spacing
        for _ in range(3):
            self.vis_prop_grid.add_child(gui.Label(""))

        meshes_label = gui.Label("Meshes")
        self.vis_prop_grid.add_child(meshes_label)
        self.vis_prop_grid.add_child(gui.Label(""))

        self.allegro_box = gui.Checkbox("(a)llegro")
        self.allegro_box.checked = self.trainer.cfg_viz.meshes.allegro
        self.vis_prop_grid.add_child(self.allegro_box)

        self.gt_mesh_box = gui.Checkbox("object (g)round-truth")
        self.gt_mesh_box.checked = self.trainer.cfg_viz.meshes.show_gt_object
        self.vis_prop_grid.add_child(self.gt_mesh_box)

        self.digit_mesh = o3d.io.read_triangle_mesh(self.trainer.digit_file)
        self.digit_mesh.compute_vertex_normals()
        self.digit_mesh.paint_uniform_color(
            [210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0]
        )  # tan color

        self.realsense_mesh = o3d.io.read_triangle_mesh(self.trainer.realsense_file)
        self.realsense_mesh.compute_vertex_normals()
        self.realsense_mesh.paint_uniform_color(
            [210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0]
        )  # tan color
        self.digit_mesh_box = gui.Checkbox("(s)ensor est.")
        self.digit_mesh_box.checked = self.trainer.cfg_viz.meshes.sensors_est
        self.vis_prop_grid.add_child(self.digit_mesh_box)

        self.trans_box = gui.Checkbox("(t)ransparency")
        self.trans_box.checked = self.trainer.cfg_viz.meshes.transparent
        self.vis_prop_grid.add_child(self.trans_box)

        self.vis_prop_grid.add_child(gui.Label(""))
        self.vis_prop_grid.add_child(gui.Label(""))

        colormap_label = gui.Label("Colormap")
        colormap_label.font_id = font_id_sans_serif
        self.colormap_radio = gui.RadioButton(gui.RadioButton.HORIZ)
        colormap_list = ["Color", "Sensor", "Normals", "FScore", "n/a"]
        self.colormap_radio.set_items(colormap_list)
        self.colormap_radio.selected_index = [
            idx
            for idx, s in enumerate(colormap_list)
            if self.trainer.cfg_viz.layers.colormap in s
        ][0]

        self.vis_prop_grid.add_child(colormap_label)
        self.vis_prop_grid.add_child(self.colormap_radio)

        kf_label = gui.Label("Keyframes")
        kf_label.font_id = font_id_sans_serif
        self.kf_radio = gui.RadioButton(gui.RadioButton.HORIZ)
        kf_list = ["None ", "Latest", "All"]
        self.kf_radio.set_items(kf_list)
        self.kf_radio.selected_index = [
            idx
            for idx, s in enumerate(kf_list)
            if self.trainer.cfg_viz.layers.keyframes in s
        ][0]
        self.vis_prop_grid.add_child(kf_label)
        self.vis_prop_grid.add_child(self.kf_radio)

        pc_label = gui.Label("Pointcloud")
        pc_label.font_id = font_id_sans_serif
        self.pc_radio = gui.RadioButton(gui.RadioButton.HORIZ)
        pc_list = ["None  ", "Both", "Vision", "Touch"]
        self.pc_radio.set_items(pc_list)
        self.pc_radio.selected_index = [
            idx
            for idx, s in enumerate(pc_list)
            if self.trainer.cfg_viz.layers.pointcloud in s
        ][0]
        self.vis_prop_grid.add_child(pc_label)
        self.vis_prop_grid.add_child(self.pc_radio)

        ## Tabs for each sensor
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), em, em)
        self.tabs = gui.TabControl()

        self.visual_stream = gui.VGrid(
            1, spacing, gui.Margins(0.1 * em, 0, 0.1 * em, 0)
        )
        ### Tab for live visual stream
        tab_rgb = gui.Vert(0, tab_margins)
        tab_rgb.add_child(gui.Label("Visual stream"))
        self.visual_rgb = gui.ImageWidget()
        tab_rgb.add_child(self.visual_rgb)
        tab_rgb.add_fixed(vspacing)
        self.visual_stream.add_child(tab_rgb)

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_rgb_depth = gui.ImageWidget()
        self.render_normals_depth = gui.ImageWidget()
        tab1.add_child(gui.Label("Input (Keyframes : 0)"))
        tab1.add_child(self.input_rgb_depth)
        tab1.add_fixed(vspacing)

        sensor_one = list(self.trainer.sensor.values())[0]
        black_vis = np.full(
            [
                sensor_one.H,
                2 * sensor_one.W,
                3,
            ],
            [45, 45, 45],
            dtype=np.uint8,
        )
        self.no_render = o3d.geometry.Image(black_vis)
        self.render_box = gui.Checkbox("Render normals and depth")
        self.render_box.checked = self.trainer.cfg_viz.misc.render_stream
        self.render_box.set_on_checked(self._toggle_render)

        self.debug_frontend = gui.Checkbox("(d)ebug frontend")
        self.debug_frontend.checked = self.trainer.cfg_viz.debug.frontend

        render_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, 0, 0))
        render_grid.add_child(self.render_box)
        render_grid.add_child(self.debug_frontend)
        tab1.add_child(render_grid)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.render_normals_depth)
        tab1.add_fixed(vspacing)
        tab1.add_fixed(vspacing)
        self.tabs.add_tab("Latest frames", tab1)

        self.record = self.trainer.cfg_viz.misc.record

        ### Keyframes image tab
        tab2 = gui.Vert(0, tab_margins)
        self.n_panels = 10
        self.keyframe_panels = []
        for _ in range(self.n_panels):
            kf_panel = gui.ImageWidget()
            tab2.add_child(kf_panel)
            self.keyframe_panels.append(kf_panel)
        self.tabs.add_tab("Keyframes", tab2)

        set_enabled(self.vis_prop_grid, True)
        self.fps_slider.enabled = self.mesh_box.checked
        self.voxel_dim_slider.enabled = self.mesh_box.checked

        # self.panel.add_child(gui.Label("NeuralFeels controls"))
        self.panel.add_child(button_play_pause)

        self.panel.add_fixed(vspacing)
        # self.panel.add_child(gui.Label('Info'))
        self.panel.add_child(self.output_info)
        self.panel.add_fixed(vspacing)
        # self.panel.add_child(gui.Label("3D visualisation settings"))
        self.panel.add_child(self.vis_prop_grid)
        self.panel.add_child(self.visual_stream)

        self.create_combo_box()
        self.stream_panel.add_child(self.combobox)
        self.stream_panel.add_child(self.tabs)

        # Scene widget
        self.widget3d = gui.SceneWidget()
        self.widget3d.set_on_key(self._on_key)
        self.widget3d.enable_scene_caching(True)

        # timings panel
        self.timings_panel = gui.Vert(spacing, gui.Margins(em, 0.5 * em, em, em))
        self.timings = gui.Label("Compute balance in last 20s:")
        self.timings_panel.add_child(self.timings)

        # colorbar panel
        self.colorbar_panel = gui.Vert(spacing, gui.Margins(0, 0, 0, 0))
        self.sdf_colorbar = gui.ImageWidget()
        self.colorbar_panel.add_child(self.sdf_colorbar)

        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.stream_panel)

        w.add_child(self.widget3d)
        w.add_child(self.timings_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.enable_scene_caching(True)
        self.widget3d.scene.downsample_threshold = (
            self.trainer.cfg_viz.misc.downsample_threshold
        )
        self.widget3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.widget3d.scene.scene.set_sun_light(
            [0.0, 0.0, -1.0], [1.0, 1.0, 1.0], 50000
        )
        self.widget3d.scene.scene.enable_sun_light(True)
        self.widget3d.scene.scene.set_indirect_light_intensity(20000.0)

        # Address the white background issue: https://github.com/isl-org/Open3D/issues/6020
        self.cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(self.cg_settings)

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)

        self.is_done = False

        self.is_started = True
        self.is_running = True
        if self.is_started:
            gui.Application.instance.post_to_main_thread(self.window, self._on_start)

        self.kfs, self.T_WC_latest = {}, {}
        if mode == "batch":
            for sensor in self.trainer.sensor.keys():
                self.kfs[sensor] = []
                h = int(trainer.frames[sensor].im_batch_np[-1].shape[0] / 6)
                w = int(trainer.frames[sensor].im_batch_np[-1].shape[1] / 6)
                self.kfs[sensor] += [
                    cv2.resize(kf, (w, h)) for kf in trainer.frames[sensor].im_batch_np
                ]
        else:
            for sensor in self.trainer.sensor.keys():
                self.kfs[sensor] = []
                self.T_WC_latest[sensor] = None

        self.max_points = 1000000
        self.kf_panel_size = 3
        self.clear_kf_frustums = False
        self.sdf_grid_pc = None
        self.slice_ix = 0
        self.slice_step = 1
        self.colormap_fn = None
        self.vis_times = []
        self.optim_times = []
        self.prepend_text = ""
        self.latest_mesh = None
        self.latest_mesh_error = None
        self.latest_mesh_sampled = None
        self.latest_pcd = None
        self.latest_frustums = {}
        for sensor in self.trainer.sensor.keys():
            self.latest_frustums[sensor] = None

        self.lit_mat = rendering.MaterialRecord()
        self.lit_mat.shader = "defaultLit"

        self.unlit_mat = self._set_transparent_mesh(level=0.8)
        self.digit_mat = self._set_normal_mesh()

        if self.trans_box.checked:
            self.allegro_mat = self._set_transparent_mesh(level=0.8)
        else:
            self.allegro_mat = self._set_object_mesh_properties()
        self.object_mat = self._set_object_mesh_properties()
        self.gt_mat = self._set_transparent_mesh(level=0.5)

        self.bbox_mat = rendering.MaterialRecord()
        self.bbox_mat.shader = "unlitLine"
        self.bbox_mat.line_width = 0.5

        self.flip_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        r = 1 * self.scene_bb.get_max_bound().max()
        self.init_camera = np.array([0, r, 2 * r])
        self.orbit_size = 1500
        self.rotate_counter = 0
        self.step = 0
        self.save_id = 0

        self.window.set_focus_widget(self.widget3d)

        self.iter = 0
        self.step = 0

        self.last_save_time = 0.0

        self.save_fps = 20

        # make save fodlers and remove previous data
        save_dir = "./images/open3d"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        self.save_sensor_dirs = {}
        for sensor in self.trainer.sensor.keys():
            sensor_dirs = [
                f"./images/{sensor}/rgb",
                f"./images/{sensor}/depth",
            ]

            if self.trainer.cfg_viz.misc.render_stream:
                sensor_dirs += [
                    f"./images/{sensor}/rendered_normals",
                    f"./images/{sensor}/rendered_depth",
                ]

            for dir in sensor_dirs:
                if os.path.exists(dir):
                    shutil.rmtree(dir)
                os.makedirs(dir)
            self.save_sensor_dirs[sensor] = sensor_dirs

        self.all_threads = []
        # Start running
        self.all_threads.append(
            threading.Thread(name="UpdateMain", target=self.update_main)
        )

        self.all_threads.append(
            threading.Thread(name="ReconstructMesh", target=self.update_mesh)
        )
        if self.record:
            print("Starting record thread (visualization will be slower)")
            self.all_threads.append(
                threading.Thread(name="SaveImage", target=self.save_image)
            )

        for t in self.all_threads:
            t.start()

    def run_one_tick(self):
        app = o3d.visualization.gui.Application.instance
        tick_return = app.run_one_tick()
        print(f"tick_return: {tick_return}")
        if tick_return:
            self.window.post_redraw()
        return tick_return

    def _set_normal_mesh(self):
        standard_mat = rendering.MaterialRecord()
        standard_mat.shader = "defaultLit"
        return standard_mat

    def _set_transparent_mesh(self, level=0.5):
        trans_mat = rendering.MaterialRecord()
        trans_mat.shader = "defaultLitTransparency"
        trans_mat.base_color = [0.467, 0.467, 0.467, level]
        trans_mat.base_roughness = 1.0
        trans_mat.base_reflectance = 0.0
        trans_mat.base_clearcoat = 0.0
        trans_mat.thickness = 1.0
        trans_mat.transmission = 0.1
        trans_mat.absorption_distance = 10
        trans_mat.absorption_color = [0.467, 0.467, 0.467]
        return trans_mat

    def _set_object_mesh_properties(self):
        obj_mat = rendering.MaterialRecord()
        mat_properties = {
            "metallic": 0.8,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0,
        }

        obj_mat.base_color = [0.9, 0.9, 0.9, 1.0]
        obj_mat.shader = "defaultLit"
        for key, val in mat_properties.items():
            setattr(obj_mat, "base_" + key, val)
        return obj_mat

    def _capture_screenshot(self):
        """
        Capture a screenshot of the current scene on key press
        """
        __save_image = functools.partial(_save_screenshot)
        gui.Application.instance.post_to_main_thread(
            self.window,
            lambda: self.widget3d.scene.scene.render_to_image(__save_image),
        )

    def _on_key(self, e):
        if e.type == gui.KeyEvent.Type.DOWN:
            ret = gui.SceneWidget.EventCallbackResult.CONSUMED
            if e.key == 97:  # a
                self.allegro_box.checked = not self.allegro_box.checked
            if e.key == 109:  # m
                self.mesh_box.checked = not self.mesh_box.checked
            if e.key == 99:  # c
                self.crop_box.checked = not self.crop_box.checked
            if e.key == 98:  # b
                self.scene_bb_box.checked = not self.scene_bb_box.checked
            if e.key == 118:  # v
                self.viz_ray.checked = not self.viz_ray.checked
            if e.key == 103:  # g
                self.gt_mesh_box.checked = not self.gt_mesh_box.checked
            if e.key == 115:  # s
                self._capture_screenshot()
            if e.key == 116:  # t
                self.trans_box.checked = not self.trans_box.checked
            if e.key == 32:  # space
                self._on_switch()
            if e.key == 113:  # q
                self.is_done = True
            if (e.key > 47 and e.key < 58) or (
                e.key in [105, 111, 112]
            ):  # 0-9 or i,o,p for pose adjustments of realsense
                sensor = self.trainer.sensor["realsense"]
                sensor.minor_adjustment(e.key)
            else:
                ret = gui.SceneWidget.EventCallbackResult.IGNORED

            return ret

        return gui.SceneWidget.EventCallbackResult.IGNORED

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 50 * em  # width of left panel
        rect = self.window.content_rect

        # height of split panels
        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, (1 * rect.height / 2))
        self.stream_panel.frame = gui.Rect(
            rect.x,
            (1 * rect.height / 2),
            panel_width,
            (1 * rect.height / 2) - 1 * em,
        )

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y, rect.get_right() - x, rect.height)

        timings_panel_width = 15 * em
        timings_panel_height = 3 * em
        self.timings_panel.frame = gui.Rect(
            rect.get_right() - timings_panel_width,
            rect.y,
            timings_panel_width,
            timings_panel_height,
        )

    def create_combo_box(self, default=0):
        self.combobox = gui.Combobox()
        for sensor, _ in self.trainer.sensor.items():
            self.combobox.add_item(sensor)
        self.combobox.selected_index = default

    # Toggle callback: application's main controller
    def _on_switch(self, on_it=None):
        if not self.is_started:
            gui.Application.instance.post_to_main_thread(self.window, self._on_start)
        self.is_running = not self.is_running
        self.render_box.checked = not self.render_box.checked

    def _clear_keyframes(self):
        self.is_started = False
        time.sleep(0.3)
        self.output_info.text = "Clearing keyframes"
        self.trainer.clear_keyframes()
        self.iter = 0
        info, new_kf_set, end = self.optim_iter(self.trainer, self.iter)
        for sensor in self.trainer.sensor.keys():
            self.kfs[sensor] = [new_kf_set[sensor]]
        self.iter += 1
        self.output_info.text = f"Iteration {self.iter}\n" + info
        self.clear_kf_frustums = True
        self.is_started = True

    def _toggle_mesh(self, is_on):
        if self.mesh_box.checked:
            self.fps_slider.enabled = True
            self.voxel_dim_slider.enabled = True
        else:
            self.widget3d.scene.remove_geometry("rec_mesh")
            self.fps_slider.enabled = False
            self.voxel_dim_slider.enabled = False

    def _reset_view(self):
        while self.step <= 0:
            time.sleep(0.1)
        cam_center = self.T_WC_latest[self.curr_sensor]
        cam_look = np.eye(4)
        cam_look[2, -1] = 0.3
        cam_look = cam_center @ cam_look
        cam_up = cam_center[:3, :3] @ np.array([0, 1, 0])
        # self.output_info.text = f"cam_look: {cam_look[:3, 3]}, cam_center: {cam_center[:3, 3]}, cam_up: {cam_up}"
        self.widget3d.look_at(cam_center[:3, 3], cam_look[:3, 3], cam_up)

    def _toggle_slices(self, is_on):
        if self.slices_box.checked:
            self.sdf_colorbar.update_image(self.colorbar_img)
        else:
            self.widget3d.scene.remove_geometry("sdf_slice")
            self.sdf_colorbar.update_image(self.no_colorbar)

    def _toggle_render(self, is_on):
        if self.render_box.checked is False:
            self.render_normals_depth.update_image(self.no_render)

    def _change_voxel_dim(self, val):
        grid_dim = self.voxel_dim_slider.int_value
        grid_pc = geometry.transform.make_3D_grid(
            [-1.0, 1.0],
            grid_dim,
            self.trainer.device,
            transform=self.trainer.p_WO_W,
            scale=self.trainer.obj_scale,
        )
        self.trainer.new_grid_dim = grid_dim
        self.trainer.new_grid_pc = grid_pc.view(-1, 3).to(self.trainer.device)

    def _change_train_fps(self, val):
        self.trainer.train_fps = self.fps_slider.double_value

    # On start: point cloud buffer and model initialization.
    def _on_start(self):
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.sRGB_color = True
        self.mat.point_size = 4

        pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((self.max_points, 3), dtype=np.float32))
        )
        pcd_placeholder.point["colors"] = o3c.Tensor(
            np.repeat(
                np.array(
                    [[207.0 / 255.0, 180.0 / 255.0, 27.0 / 255.0]], dtype=np.float32
                ),
                self.max_points,
                axis=0,
            ),
        )  # yellow

        self.widget3d.scene.scene.add_geometry(
            "render_points", pcd_placeholder, self.mat
        )

        for i, sensor in enumerate(self.trainer.sensor.keys()):
            self.widget3d.scene.scene.add_geometry(
                f"ray_samples_{i}", pcd_placeholder, self.mat
            )
        self.widget3d.scene.scene.add_geometry("slice_pc", pcd_placeholder, self.mat)

        if self.origin_box.checked:
            self.widget3d.scene.add_geometry("origin", self.origin, self.unlit_mat)

        if self.scene_bb_box.checked:
            self.widget3d.scene.add_geometry("scene_bb", self.scene_bb, self.bbox_mat)

        init_joints = np.zeros(16)
        self.allegro_graph = SceneGraph(self.urdf_tree.root, init_joints)
        allegro_mesh = self.allegro_graph.getMesh()
        self.allegro_mesh_elements = len(allegro_mesh)

        self.is_started = True

    def _on_close(self):
        for t in self.all_threads:
            t.join()
        # self.save_data()
        print("Exiting")
        return True

    def save_data(self):
        self.save_stats()
        self.save_plots()
        self.save_baseline()
        if self.save_neural_field:
            self.save_neural_field_weights()

        if self.record:
            save_dir = "./videos"
            os.makedirs(save_dir)

            self.trainer.current_time = (
                self.trainer.tot_step_time
                if np.isinf(self.trainer.current_time)
                else self.trainer.current_time
            )  # for grasp data

            if (
                self.trainer.train_mode in ["map", "slam"]
                and self.latest_mesh is not None
            ):
                self.write_mesh(final_mesh=True)
                if self.save_rotate:
                    print(
                        f"Generating rotating mesh video of length {self.trainer.current_time}"
                    )
                    self.save_mesh_video()

            if self.render_open3d:
                print(
                    f"Saving open3d stream to video of length {self.trainer.current_time}"
                )
                images_to_video(
                    "./images/open3d",
                    "./videos/open3d.mp4",
                    fps=self.save_fps,
                )
            print(
                f"Saving image/rendering streams to video of length {self.trainer.current_time}"
            )
            for sensor in self.trainer.sensor.keys():
                dirs = self.save_sensor_dirs[sensor]
                for dir in dirs:
                    folder_name = dir.split("/")[-1]
                    images_to_video(
                        dir,
                        f"./videos/{sensor}_{folder_name}.mp4",
                        fps=self.save_fps,
                    )

        original_working_dir = get_original_cwd()
        save_path = os.getcwd()
        relative_save_path = os.path.relpath(save_path, original_working_dir)
        cprint(f"All data saved to {relative_save_path}", "green")

    def save_plots(self):
        """
        Generates the RMSE and F-score plots for the current run
        """
        print("Saving pose/map plots")
        if self.trainer.train_mode in ["pose", "slam"]:
            draw_pose_error()
        if self.trainer.train_mode in ["map", "slam"]:
            draw_map_error()

    def save_baseline(self):
        if self.trainer.is_baseline:
            print("Saving the baseline pose tracking as ground-truth")
            self.trainer.object.save_baseline()

    def save_stats(self):
        save_file = pathlib.Path(os.getcwd()) / "stats.pkl"
        print(f"Saving experiment stats in {save_file}")
        with open(save_file, "wb") as file:
            pickle.dump(self.trainer.save_stats, file)

    def get_orbit(self, final_mesh, timsteps=400, num_orbits=1):
        """
        Create an orbit which the open3d visualizer will follow for mesh video generation
        """
        diag = np.linalg.norm(
            np.asarray(final_mesh.get_max_bound())
            - np.asarray(final_mesh.get_min_bound())
        )
        radius = diag
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

    def save_mesh_video(self):
        """
        Generate a video of the evolving mesh over time rotating the open3d visualizer camera around the mesh, saves to ./videos/mesh_viz.mp4
        """

        mesh_dir = pathlib.Path(os.getcwd()) / "mesh"

        # get all .obj files in mesh_dir
        mesh_files = list(mesh_dir.glob("*.obj"))
        step_meshes = [x.name for x in mesh_files if "final" not in x.name]
        # sort step meshes by number
        step_meshes = sorted(
            step_meshes, key=lambda x: int(x.split(".")[0].split("/")[-1])
        )

        final_mesh_path = [x.name for x in mesh_files if "final" in x.name][0]
        final_mesh_path = str(mesh_dir / final_mesh_path)

        interval = 1000.0 / self.save_fps
        times = np.array([int(x.split(".")[0]) for x in step_meshes])

        print(f"Generating video from {len(times)} meshes")
        lbs = np.arange(0, times[-1], interval)
        last_frame_time = None
        all_meshes = []
        all_times = []
        for lb in lbs:
            ub = lb + interval
            if ub < times[0]:
                continue
            time = times[np.logical_and(times >= lb, times <= ub)]
            if len(time) == 0:
                time = last_frame_time
            else:
                time = time[0]
                last_frame_time = time

            mesh_path = os.path.join(mesh_dir, f"{time:06d}.obj")
            all_meshes.append(mesh_path)
            all_times.append(int(lb))

        image_path = os.path.join(mesh_dir, "images")
        final_mesh = o3d.io.read_triangle_mesh(final_mesh_path)
        final_mesh = final_mesh.transform(self.trainer.object.current_pose_offset)

        gt_mesh = self.gt_mesh
        gt_mesh_scale = 1.0 / self.trainer.obj_scale_np[0]
        gt_mesh = gt_mesh.scale(gt_mesh_scale, center=gt_mesh.get_center())

        orbit_path, center = self.get_orbit(
            final_mesh, timsteps=len(all_times), num_orbits=2
        )

        render = rendering.OffscreenRenderer(1000, 1000)
        render.setup_camera(60.0, [0, 0, 0], [0, 10, 0], [0, 0, 1])
        render.scene.set_background([1, 1, 1, 1])
        render.scene.view.set_color_grading(self.cg_settings)

        # define materials
        rotating_mesh_mat = rendering.MaterialRecord()
        mat_properties = {
            "metallic": 0.5,
            "roughness": 0.6,
            "reflectance": 0.2,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.3,
        }
        rotating_mesh_mat.base_color = [0.9, 0.9, 0.9, 1.0]
        rotating_mesh_mat.shader = "defaultLit"
        for key, val in mat_properties.items():
            setattr(rotating_mesh_mat, "base_" + key, val)
        rotating_bbox_mat = rendering.MaterialRecord()
        rotating_bbox_mat.shader = "unlitLine"
        rotating_bbox_mat.line_width = 0.5

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        bbox_gt = gt_mesh.get_axis_aligned_bounding_box()
        bbox_gt.color = (0, 0, 0)
        bbox_recon = final_mesh.get_axis_aligned_bounding_box()
        bbox_recon.color = (0, 0, 0)
        num_iters = len(orbit_path)

        for i in range(num_iters):
            # load each mesh and add to scene with lighting
            recon_mesh = o3d.io.read_triangle_mesh(all_meshes[i], True)
            recon_mesh = recon_mesh.transform(self.trainer.object.current_pose_offset)

            render.scene.set_lighting(
                rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
                -np.array(orbit_path[i, :] + [0.0, 0.0, 0.01]),
            )

            render.scene.add_geometry("recon_mesh", recon_mesh, rotating_mesh_mat)
            render.scene.add_geometry("bbox", bbox_recon, rotating_bbox_mat)
            render.setup_camera(60.0, center, orbit_path[i, :], [0, 0, 1])

            # save images to folder
            img_1 = render.render_to_image()
            time_label = all_times[i]
            render.scene.remove_geometry("recon_mesh")
            render.scene.remove_geometry("bbox")
            render.scene.add_geometry("gt_mesh", gt_mesh, rotating_mesh_mat)
            render.scene.add_geometry("bbox", bbox_gt, rotating_bbox_mat)
            # save images to folder
            img_2 = render.render_to_image()
            render.scene.remove_geometry("gt_mesh")
            # combine both images vertically
            img = np.vstack([np.asarray(img_1), np.asarray(img_2)])
            img = o3d.geometry.Image(img)
            o3d.io.write_image(
                os.path.join(image_path, f"{time_label:06d}.jpg"), img, 99
            )
            render.scene.remove_geometry("bbox")

        # exiting OffscreenRenderer

        # convert to mp4
        images_to_video(image_path, f"./videos/mesh_viz.mp4", fps=self.save_fps)
        # delete the .jpg files
        for file in os.listdir(image_path):
            if file.endswith(".jpg"):
                os.remove(os.path.join(image_path, file))
            # remove the image_path folder
        os.rmdir(image_path)

        if not self.write_all_meshes:
            # delete all .obj files except the final mesh
            for file in os.listdir(mesh_dir):
                if file.endswith(".obj") and "final" not in file:
                    os.remove(os.path.join(mesh_dir, file))

    def init_render(self):
        self.output_info.text = "\n\n\n"

        sensor_one = list(self.trainer.sensor.values())[0]
        blank = np.full(
            [
                sensor_one.H,
                2 * sensor_one.W,
                3,
            ],
            [45, 45, 45],
            dtype=np.uint8,
        )

        blank_im = o3d.geometry.Image(np.hstack([blank] * 2))
        self.no_colorbar = o3d.geometry.Image(blank)
        self.colorbar_img = self.no_colorbar
        self.input_rgb_depth.update_image(blank_im)
        self.visual_rgb.update_image(blank_im)

        self.render_normals_depth.update_image(self.no_render)

        kf_im = o3d.geometry.Image(np.hstack([blank] * self.kf_panel_size))
        for panel in self.keyframe_panels:
            panel.update_image(kf_im)

        self.window.set_needs_layout()

        # viewer mimics real-world camera pose looking at allegro
        bbox = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(
            45.0, bbox, bbox.get_center()
        )  # fov of visualizer camera
        pov_sensors = [
            value
            for key, value in self.trainer.sensor.items()
            if "realsense" in key.lower()
        ]
        if len(pov_sensors) > 0 and not self.trainer.cfg_main.occlusion:
            cam_target = self.trainer.allegro.allegro_pose[:3, 3] + np.array(
                [0.0, 0.0, 0.05]
            )  # 5cm above the hand pose
            cam_pos = pov_sensors[0].get_realsense_pose()[:3, 3]
            self.widget3d.look_at(cam_target, cam_pos, [0, 0, 1])
        else:
            cam_pos = np.array([-0.3, 0.3, 0.9])
            cam_target = self.trainer.allegro.allegro_pose[:3, 3]
            self.widget3d.look_at(cam_target, cam_pos, [0, 0, 1])

    def toggle_content(self, name, geometry, mat, show):
        if (self.widget3d.scene.has_geometry(name) is False) and show:
            self.widget3d.scene.add_geometry(name, geometry, mat)
        elif self.widget3d.scene.has_geometry(name) and (show is False):
            self.widget3d.scene.remove_geometry(name)

    def update_render(
        self,
        latest_frame,
        latest_visual_frame,
        render_frame,
        keyframes_vis,
        rec_mesh,
        slice_pcd,
        pcds,
        render_pcds,
        ray_sets,
        ray_pcds,
        latest_frustum,
        kf_frustums,
        allegro_mesh,
        current_object_state,
        current_object_state_gt,
    ):
        if latest_frame is not None:
            self.input_rgb_depth.update_image(latest_frame)
        if latest_visual_frame is not None:
            image_np = np.asarray(latest_visual_frame)
            resized_np = cv2.resize(
                image_np,
                (image_np.shape[1] // 2, image_np.shape[0] // 2),
                interpolation=cv2.INTER_LINEAR,
            )
            self.visual_rgb.update_image(o3d.geometry.Image(resized_np))

        if render_frame is not None:
            self.render_normals_depth.update_image(render_frame)
        if render_pcds is not None:
            self.widget3d.scene.scene.update_geometry(
                "render_points",
                render_pcds,
                rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG,
            )
        for im, kf_panel in zip(keyframes_vis, self.keyframe_panels):
            kf_panel.update_image(im)

        if self.trans_box.checked:
            self.allegro_mat = self._set_transparent_mesh(level=0.6)
        else:
            self.allegro_mat = self._set_object_mesh_properties()

        self.widget3d.scene.remove_geometry("points")
        if not "None" in self.pc_radio.selected_value:
            self.widget3d.scene.add_geometry("points", pcds, self.mat)

        if self.trainer.gt_scene:
            for i, sensor in enumerate(self.trainer.sensor.keys()):
                if self.trainer.sensor[sensor].end:
                    self.toggle_content(sensor, None, None, False)
                else:
                    latest_pose = self.trainer.latest_pose(sensor)
                    if "digit" in sensor:
                        mesh = copy.deepcopy(self.digit_mesh).transform(latest_pose)
                    else:
                        mesh = copy.deepcopy(self.realsense_mesh).transform(latest_pose)

                    self.widget3d.scene.remove_geometry(sensor)
                    self.widget3d.scene.add_geometry(sensor, mesh, self.digit_mat)

                    self.toggle_content(
                        sensor,
                        mesh,
                        self.digit_mat,
                        self.digit_mesh_box.checked,
                    )

            if self.scene_bb_box.checked:
                self.scene_bb = o3d.geometry.OrientedBoundingBox(
                    self.trainer.scene_center,
                    current_object_state[:3, :3],
                    self.trainer.obj_extents_np,
                )
                self.scene_bb.color = [1.0, 1.0, 1.0]
                self.widget3d.scene.remove_geometry("scene_bb")
                self.widget3d.scene.add_geometry(
                    "scene_bb", self.scene_bb, self.bbox_mat
                )

            obj_origin = copy.deepcopy(self.origin).transform(current_object_state)

            if self.origin_box.checked:
                self.widget3d.scene.remove_geometry("origin")
                self.widget3d.scene.add_geometry("origin", obj_origin, self.unlit_mat)

            self.widget3d.scene.remove_geometry("gt_mesh")
            # visualize ground truth mesh if baseline exists
            if (
                not np.allclose(current_object_state_gt, np.eye(4))
                and self.gt_mesh is not None
            ):
                obj_mesh = copy.deepcopy(self.gt_mesh).transform(
                    current_object_state_gt
                )
                self.widget3d.scene.add_geometry("gt_mesh", obj_mesh, self.gt_mat)
                self.toggle_content(
                    "gt_mesh", obj_mesh, self.gt_mat, self.gt_mesh_box.checked
                )

        for i, ray_pcd in enumerate(ray_pcds):
            self.widget3d.scene.scene.update_geometry(
                f"ray_samples_{i}",
                ray_pcd,
                rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG,
            )

        for i, ray_set in enumerate(ray_sets):
            self.widget3d.scene.remove_geometry(f"rays_{i}")
            if ray_set is not None:
                self.widget3d.scene.add_geometry(f"rays_{i}", ray_set, self.bbox_mat)

        if slice_pcd is not None:
            self.widget3d.scene.scene.update_geometry(
                "slice_pc",
                slice_pcd,
                rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG,
            )

        self.widget3d.scene.remove_geometry("rec_mesh")
        if rec_mesh and self.mesh_box.checked:
            plot_mesh = copy.deepcopy(rec_mesh)
            if self.trainer.train_mode in ["map", "slam"]:
                # scale the mesh if it is generated from SDF marching cubes
                plot_mesh = plot_mesh.scale(
                    self.trainer.obj_scale_np[0], center=(0, 0, 0)
                )
            plot_mesh = plot_mesh.transform(current_object_state)
            self.widget3d.scene.add_geometry("rec_mesh", plot_mesh, self.object_mat)
            self.start_optimize = (
                True  # Starts optimization when the rendering is loaded
            )

        if latest_frustum:
            for i, sensor in enumerate(self.trainer.sensor.keys()):
                frustum = latest_frustum[sensor]
                frustum.paint_uniform_color(
                    [210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0]
                )  # tan color
                self.widget3d.scene.remove_geometry(f"{sensor}_latest_frustum_{i}")
                if frustum is not None:
                    self.widget3d.scene.add_geometry(
                        f"{sensor}_latest_frustum_{i}", frustum, self.bbox_mat
                    )

        if kf_frustums:
            for sensor in self.trainer.sensor.keys():
                sensor_frustums = kf_frustums[sensor]
                for i, frustum in enumerate(sensor_frustums):
                    self.widget3d.scene.remove_geometry(f"{sensor}_frustum_{i}")
                    if frustum is not None:
                        self.widget3d.scene.add_geometry(
                            f"{sensor}_frustum_{i}", frustum, self.bbox_mat
                        )
        else:
            for sensor in self.trainer.sensor.keys():
                n_remove = len(self.trainer.frames[sensor].T_WC_batch_np[:-1])
                for i in range(n_remove):
                    self.widget3d.scene.remove_geometry(f"{sensor}_frustum_{i}")

        if allegro_mesh is not None:
            # Add allegro visualization
            for i, mesh in enumerate(allegro_mesh):
                if i == 16:
                    continue  # skip visualizing joint=12.0 which has a parsing error
                mesh = copy.deepcopy(mesh).transform(self.trainer.allegro.allegro_pose)
                self.widget3d.scene.remove_geometry(f"allegro_{i}")
                self.widget3d.scene.add_geometry(f"allegro_{i}", mesh, self.allegro_mat)
        else:
            for i in range(self.allegro_mesh_elements):
                self.widget3d.scene.remove_geometry(f"allegro_{i}")

        self.toggle_content(
            "origin", self.origin, self.unlit_mat, self.origin_box.checked
        )
        self.toggle_content(
            "scene_bb", self.scene_bb, self.bbox_mat, self.scene_bb_box.checked
        )
        self.widget3d.force_redraw()
        # self.window.renderer.update_renderer()
        # self.widget3d.scene.poll_events()

    def save_image(self):
        self.last_save_time = 0.0
        while not self.is_done:
            self.current_time_ms = (
                self.trainer.tot_step_time * 1000
                if np.isinf(self.trainer.current_time)
                else self.trainer.current_time * 1000
            )  # for grasp data
            if (self.current_time_ms - self.last_save_time) > 1000.0 / self.save_fps:
                save_id = int(self.current_time_ms)
                if self.render_open3d:
                    __save_image = functools.partial(_save_image, id=save_id)
                    gui.Application.instance.post_to_main_thread(
                        self.window,
                        lambda: self.widget3d.scene.scene.render_to_image(__save_image),
                    )
                do_render = False
                if self.render_box.checked:
                    do_render = True
                for sensor in self.trainer.sensor.keys():
                    do_render_sensor = do_render or "realsense" in sensor
                    self.update_latest_frames(
                        sensor, save_id, do_render_sensor, is_curr_sensor=False
                    )
                self.last_save_time = self.current_time_ms
            time.sleep(1.0 / self.trainer.train_fps)
        print("Terminating [save_image] thread")

    def _recon_fname(self, final_file: bool = False) -> str:
        fname = f"{self.trainer.cfg_data.object}"
        if final_file:
            fname = f"{fname}_final"
        elif self.trainer.cfg_eval.save_intermediate_recons:
            fname = f"{str(int(self.current_time_ms)).zfill(6)}"
        return fname

    def save_rendered_pc(self):
        render_pc_dir = pathlib.Path("rendered_pcs")
        render_pc_dir.mkdir(exist_ok=True)
        fname = self._recon_fname()
        pc = self.trainer.render_pc_from_sdf(
            n_points=self.trainer.cfg_eval.num_points_pcs
        )
        torch.save(pc, render_pc_dir / f"{fname}.pth")

    def update_mesh(self):
        if self.trainer.train_mode in ["pose", "pose_grasp"]:
            self.latest_mesh = self.gt_mesh
            print("Pose optimization, so no mesh reconstruction")
            return

        last_update = 0
        while not self.is_done:
            should_reconstruct_mesh = self.step > 0 and self.mesh_box.checked
            if self.is_running:
                should_reconstruct_mesh = (
                    should_reconstruct_mesh
                    and self.step != last_update
                    and (self.step % self.trainer.mesh_interval == 0 or self.step < 5)
                )

            if should_reconstruct_mesh:
                print(f"\n[t: {self.step}] Performing mesh reconstruction...\n")
                try:
                    self.reconstruct_mesh()
                    print(f"\n[t: {self.step}] Mesh reconstruction complete\n")
                    last_update = self.step
                except:
                    print("Error in mesh generation, continuing...")
                if self.trainer.cfg_eval.save_rendered_pcs:
                    self.save_rendered_pc()

            time.sleep(1)  # check every 1 second
        print("Terminating [update_mesh] thread")

    # Major loop
    def update_main(self):
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.init_render()
        )

        if self.profile:
            yappi.set_clock_type("wall")  # profiling
            yappi.start(builtins=True)

        while not self.is_done:
            t0 = time.time()
            if self.is_running:
                # -------------------------------------------- used to be in a loop
                # training steps ------------------------------
                info, new_kf_set, end = self.optim_iter(
                    trainer=self.trainer,
                    t=self.iter,
                    start_optimize=self.start_optimize,
                )
                for sensor in self.trainer.sensor.keys():
                    new_kf = new_kf_set[sensor]
                    if new_kf is not None:
                        self.kfs[sensor].append(new_kf)
                if (
                    end
                    and self.trainer.incremental
                    and not "grasp" in self.trainer.train_mode
                ):
                    # for SLAM, terminate when the sensor stream is over
                    self.prepend_text = "SEQUENCE ENDED - CONTINUING TRAINING\n"
                    self.is_done = True
                if (
                    end
                    and self.trainer.incremental
                    and "grasp" in self.trainer.train_mode
                ):
                    if self.iter >= self.trainer.grasp_threshold:
                        self.prepend_text = "POSE OPTIMIZATION COMPLETED\n"
                        self.is_done = True
                if not self.trainer.incremental:
                    print(
                        "iter",
                        self.iter,
                        "step",
                        self.step,
                        "tot_step_time",
                        self.trainer.tot_step_time,
                    )
                    if self.trainer.tot_step_time > self.trainer.train_time_min * 60:
                        self.prepend_text = "OFFLINE OPTIMIZATION COMPLETED\n"
                        self.is_done = True
                self.output_info.text = (
                    self.prepend_text
                    + f"Step {self.step} -- Iteration {self.iter}\n"
                    + info
                )
                self.iter += 1 * int(self.start_optimize)
                # --------------------------------------------

                self.curr_sensor = self.combobox.selected_text
                self.tabs.get_children()[0].get_children()[
                    0
                ].text = f"Input (Keyframes : {len(self.kfs[self.curr_sensor])})"

                # Drop reconstruction learning rate towards the end of the sequence to get better surfaces
                index = bisect.bisect(
                    self.trainer.map_schedule_times, self.trainer.current_time
                )
                if (
                    self.trainer.map_scheduler is not None
                    and index > self.trainer.map_schedule_idx
                ):
                    self.trainer.map_scheduler.step()
                    self.trainer.map_schedule_idx += 1
                    if (
                        self.trainer.current_lr
                        != self.trainer.map_scheduler.get_last_lr()
                    ):
                        self.trainer.current_lr = (
                            self.trainer.map_scheduler.get_last_lr()
                        )
                        print(
                            f"LR changed: {self.trainer.map_scheduler.get_last_lr()[0]}",
                            end="\n",
                        )
                t1 = time.time()
                self.optim_times.append(t1 - t0)

                # image vis -----------------------------------
                # keyframe vis
                kf_vis = []
                c = 0
                ims = []

                for im in self.kfs[self.curr_sensor]:
                    im = cv2.copyMakeBorder(
                        im,
                        0,
                        0,
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=[255, 255, 255],
                    )
                    ims.append(im)
                    c += 1
                    # once it reach width of the panel
                    if c == self.kf_panel_size:
                        kf_im = o3d.geometry.Image(np.hstack(ims))
                        kf_vis.append(kf_im)
                        ims = []
                        c = 0

                if len(ims) != 0:
                    blank = np.full(im.shape, [45, 45, 45], dtype=np.uint8)
                    for _ in range(c, 3):
                        ims.append(blank)
                    kf_im = o3d.geometry.Image(np.hstack(ims))
                    kf_vis.append(kf_im)

                render_pcds = None
                if self.trainer.render_samples["pc"] is not None:
                    sdf = self.trainer.render_samples["sdf"]
                    pc = self.trainer.render_samples["pc"]
                    render_pcds = o3d.t.geometry.PointCloud(o3c.Tensor(pc))
                    if sdf is not None:
                        cmap_fn = sdf_util.get_colormap([sdf.min(), sdf.max()], 1e-4)
                        sdf_cols = cmap_fn.to_rgba(sdf, bytes=False)
                        sdf_cols = sdf_cols[:, :3].astype(np.float32)
                        render_pcds.point["colors"] = o3c.Tensor(sdf_cols)

                # latest frame vis (rgbd and render) --------------------------------
                do_render = False
                if self.render_box.checked:
                    do_render = True
                self.update_latest_frames(self.curr_sensor, None, do_render)

                for sensor in self.trainer.sensor.keys():
                    self.T_WC_latest[sensor] = self.trainer.latest_pose(sensor)
                # 3D vis --------------------------------------

                # reconstructed mesh from marching cubes on zero level set
                rec_mesh = None
                if self.mesh_box.checked:
                    # self.reconstruct_mesh()
                    rec_mesh = self.latest_mesh

                # sdf slices
                if self.step % 20 == 0 and self.slices_box.checked:
                    self.compute_sdf_slices()

                # Save pkl stats every 100 steps
                if self.step % 100 == 0 and self.step > 0:
                    self.save_stats()

                slice_pcd = self.next_slice_pc()

                # point cloud from depth
                # o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))
                ray_pcds = [
                    o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))
                    for _ in self.trainer.sensor.keys()
                ]
                ray_sets = [o3d.geometry.LineSet() for _ in self.trainer.sensor.keys()]
                if not "None" in self.pc_radio.selected_value:
                    self.latest_pcd = self.trainer.get_pcd(
                        self.pc_radio.selected_value, None, self.max_points
                    )
                    pcds = self.latest_pcd
                    # pcds = o3d.t.geometry.PointCloud(
                    #     self.trainer.grid_pc.view(-1, 3).cpu().numpy()
                    # )
                else:
                    pcds = o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))
                if self.viz_ray.checked:
                    for i, sensor in enumerate(self.trainer.sensor.keys()):
                        ray_sets[i], ray_pcds[i] = self.get_ray_samples(sensor, skip=10)

                # keyframes
                kf_frustums = {}

                if "All" in self.kf_radio.selected_value:
                    for i, sensor in enumerate(self.trainer.sensor.keys()):
                        self.update_kf_frustums(sensor)
                        kf_frustums[sensor] = self.latest_frustums[sensor]

                # latest frame
                latest_frustum = None
                # if self.trainer.incremental or self.trainer.live:

                # TODO: fix
                self.latest_frustum = {}
                if "Latest" in self.kf_radio.selected_value:
                    for sensor in self.trainer.sensor.keys():
                        scale = 0.01 if "digit" in sensor else 0.1
                        latest_frustum = (
                            o3d.geometry.LineSet.create_camera_visualization(
                                self.intrinsic[sensor].width,
                                self.intrinsic[sensor].height,
                                self.intrinsic[sensor].intrinsic_matrix,
                                np.linalg.inv(
                                    self.T_WC_latest[sensor] @ self.flip_matrix
                                ),
                                scale=scale,
                            )
                        )
                        latest_frustum.paint_uniform_color([0.0, 0.0, 0.0])
                        self.latest_frustum[sensor] = latest_frustum

                allegro_mesh = None
                if self.allegro_box.checked:
                    current_joint_state = (
                        self.trainer.allegro.current_joint_state.squeeze()
                    )
                    self.allegro_graph.updateJointAngles(current_joint_state)
                    self.allegro_graph.updateState()
                    allegro_mesh = self.allegro_graph.getMesh()

                current_object_state = self.trainer.p_WO_W_np
                current_object_state_gt = self.trainer.p_WO_W_gt_np

                # update every second frame, else I recieve this segfault for long trials https://github.com/isl-org/Open3D/issues/4634
                if self.render_open3d and self.step % 2 == 0:
                    gui.Application.instance.post_to_main_thread(
                        self.window,
                        lambda: self.update_render(
                            self.latest_frame,
                            self.latest_visual_frame,
                            self.render_frame,
                            kf_vis,
                            rec_mesh,
                            slice_pcd,
                            pcds,
                            render_pcds,
                            ray_sets,
                            ray_pcds,
                            self.latest_frustum,
                            kf_frustums,
                            allegro_mesh,
                            current_object_state,
                            current_object_state_gt,
                        ),
                    )
                self.vis_times.append(time.time() - t1)
                t_vis = np.sum(self.vis_times)
                t_optim = np.sum(self.optim_times)
                t_tot = t_vis + t_optim
                prop_vis = int(np.round(100 * t_vis / t_tot))
                prop_optim = int(np.round(100 * t_optim / t_tot))
                while t_tot > 20:
                    self.vis_times.pop(0)
                    self.optim_times.pop(0)
                    t_tot = np.sum(self.vis_times) + np.sum(self.optim_times)
                self.timings.text = (
                    "Compute balance in last 20s:\n"
                    + f"training {prop_optim}% : visualisation {prop_vis}%"
                )

                self.step += 1

        if self.profile:
            stats = yappi.get_func_stats()
            stats.save(f"./thread_update_main.prof", type="pstat")
            print(f"Optimizer profile saved to ./thread_update_main.prof")

        print("Terminating [update_main] thread")
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.window.close()
        )

    def reconstruct_mesh(self):
        """
        Compute mesh reconstruction with marching cubes of SDF, and apply crop/coloring
        """
        self.output_info.text = (
            "Computing mesh reconstruction with marching cubes\n\n\n"
        )
        rec_mesh, full_rec_mesh, touch_ixs, crop_idxs = self.trainer.mesh_rec(
            self.crop_box.checked
        )  # gives trimesh

        if self.colormap_radio.selected_value in ["Color"]:
            rec_mesh_world = rec_mesh.copy().apply_scale(self.trainer.obj_scale_np)
            rec_mesh_world.apply_transform(self.trainer.p_WO_W_np)

        if self.colormap_radio.selected_value == "FScore":
            rec_mesh_object = rec_mesh.copy().apply_scale(self.trainer.obj_scale_np)
            rec_mesh_object.apply_transform(self.trainer.object.current_pose_offset)

            (
                _,
                _,
                _,
                self.latest_mesh_error,
                self.latest_mesh_sampled,
            ) = compute_f_score(
                self.trainer.gt_mesh_sampled,
                rec_mesh_object,
                num_mesh_samples=self.trainer.num_points_f_score,
                T=self.trainer.f_score_T,
            )  # Compute f-score for cropped mesh

        rec_mesh = rec_mesh.as_open3d
        rec_mesh.compute_vertex_normals()

        if (
            "Color" in self.colormap_radio.selected_value
            and self.trainer.train_mode in ["map", "slam"]
        ):
            pcds = self.trainer.get_pcd("Vision", None, self.max_points)
            mesh_colors = sdf_util.colorize_mesh(pcds, rec_mesh_world)
            rec_mesh.vertex_colors = mesh_colors

        elif "FScore" in self.colormap_radio.selected_value:
            # clip the max error_color to be < 1cm
            error_colors = np.clip(self.latest_mesh_error, 0, 0.01)
            error_colors = plt.get_cmap("plasma")(
                (error_colors - error_colors.min())
                / (error_colors.max() - error_colors.min())
            )
            error_colors = error_colors[:, :3]
            rec_mesh.vertex_colors = o3d.utility.Vector3dVector(
                error_colors[self.latest_mesh_sampled, :]
            )
        elif "Normals" in self.colormap_radio.selected_value:
            rec_mesh.vertex_colors = rec_mesh.vertex_normals
        elif "Sensor" in self.colormap_radio.selected_value:
            N = len(rec_mesh.vertices)
            color, color_touch, color_crop = (
                np.array([126.0, 171.0, 237.0]) / 255.0,
                np.array([100.0, 100.0, 170.0]) / 255.0,
                np.array([250.0, 170.0, 170.0]) / 255.0,
            )
            color = np.repeat(color[None, :], repeats=N, axis=0)
            color_touch = np.repeat(color_touch[None, :], repeats=N, axis=0)
            color_crop = np.repeat(color_crop[None, :], repeats=N, axis=0)

            crop_points = crop_idxs[:, None]
            vision_points = (~crop_idxs * ~touch_ixs)[:, None]
            touch_points = (~crop_idxs * touch_ixs)[:, None]
            vertex_colors = (
                vision_points * color
                + touch_points * color_touch
                + crop_points * color_crop
            )
            rec_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        elif "n/a" in self.colormap_radio.selected_value:
            N = len(rec_mesh.vertices)
            color = np.array([126.0, 171.0, 237.0]) / 255.0
            color = np.repeat(color[None, :], repeats=N, axis=0)
            rec_mesh.vertex_colors = o3d.utility.Vector3dVector(color)
        self.latest_mesh = rec_mesh

        if self.write_all_meshes or self.save_rotate:
            self.write_mesh()

    def write_mesh(self, final_mesh: bool = False):
        mesh_dir = pathlib.Path(os.getcwd()) / "mesh"
        mesh_dir.mkdir(exist_ok=True)
        fname = self._recon_fname(final_file=final_mesh)
        mesh_file = str(mesh_dir / f"{fname}.obj")
        save_mesh = copy.deepcopy(self.latest_mesh)
        # https://github.com/isl-org/Open3D/issues/2933
        save_mesh.triangle_normals = o3d.utility.Vector3dVector([])
        o3d.io.write_triangle_mesh(mesh_file, save_mesh)
        return mesh_file

    def save_neural_field_weights(self):
        if not "grasp" in self.trainer.train_mode:
            print("Saving neural field weights")
            save_dir = f"{self.trainer.cfg_data.object}.pth"
            save_dict = {}
            save_dict["step"] = self.step
            save_dict["model_state_dict"] = self.trainer.sdf_map.state_dict()
            save_dict["sdf_map_pose"] = self.trainer.sdf_map.sdf_map_pose
            if self.trainer.train_mode in ["pose", "pose_grasp"]:
                save_dict["sdf_map_scale"] = 1.0
            else:
                save_dict["sdf_map_scale"] = self.trainer.sdf_map.scale
            if self.trainer.map_optimizer is not None:
                save_dict["optimizer_state_dict"] = (
                    self.trainer.map_optimizer.state_dict()
                )
            torch.save(
                save_dict,
                save_dir,
            )

    def next_slice_pc(self, step_sz=1):
        if self.slices_box.checked and self.sdf_grid_pc is not None:
            slice_pc = self.sdf_grid_pc[:, :, self.slice_ix].reshape(-1, 4)
            slice_pcd = o3d.t.geometry.PointCloud(o3c.Tensor(slice_pc[:, :3]))
            slice_cols = self.colormap_fn.to_rgba(slice_pc[:, 3], bytes=False)
            slice_cols = slice_cols[:, :3].astype(np.float32)
            slice_pcd.point["colors"] = o3c.Tensor(slice_cols)
            # next slice
            if self.slice_ix >= self.sdf_grid_pc.shape[2] - step_sz:
                self.slice_step = -step_sz
            if self.slice_ix <= 0:
                self.slice_step = step_sz
            self.slice_ix += self.slice_step
            return slice_pcd
        else:
            return None

    def compute_sdf_slices(self):
        self.output_info.text = "Computing SDF slices\n\n\n"
        sdf_grid_pc = self.trainer.get_sdf_grid_pc()
        sdf_grid_pc = np.transpose(sdf_grid_pc, (2, 1, 0, 3))
        self.sdf_grid_pc = sdf_grid_pc
        sdf_range = [self.sdf_grid_pc[..., 3].min(), self.sdf_grid_pc[..., 3].max()]
        surface_cutoff = 1e-4
        self.colormap_fn = sdf_util.get_colormap(sdf_range, surface_cutoff)

        fig, ax = plt.subplots(figsize=(5, 2), tight_layout=True)
        plt.colorbar(self.colormap_fn, ax=ax, orientation="horizontal")
        ax.remove()
        fig.set_tight_layout(True)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data[120:]
        self.colorbar_img = o3d.geometry.Image(data)
        self.sdf_colorbar.update_image(self.colorbar_img)

    def get_ray_samples(self, sensor, skip: int = 1):
        """
        Visualize the vector field in world coordinates for a batch of images
        """
        if self.trainer.sample_pts[sensor] is None:
            return o3d.geometry.LineSet(), o3d.t.geometry.PointCloud(
                np.zeros((1, 3), dtype=np.float32)
            )

        z_vals = self.trainer.sample_pts[sensor]["z_vals"]
        T_WC = self.trainer.sample_pts[sensor]["T_WC_sample"]
        ray_samples_min = T_WC[:, :3, 3]

        if not torch.sum(z_vals > 0):
            # max is the minimum z value because of tactile coordinates
            _, max_inds = torch.min(z_vals, dim=1)
        else:
            _, max_inds = torch.max(z_vals, dim=1)
        max_inds = max_inds[0]

        ray_samples = self.trainer.sample_pts[sensor]["pc"]
        ray_samples_max = ray_samples[:, max_inds, :]
        ray_samples_min = ray_samples_min.detach().cpu().numpy()
        ray_samples_max = ray_samples_max.detach().cpu().numpy()
        ray_samples_min, ray_samples_max = (
            ray_samples_min[::skip, :],
            ray_samples_max[::skip, :],
        )
        ray_min_max = np.vstack([ray_samples_min, ray_samples_max])

        ray_samples = ray_samples.detach().cpu().numpy()
        colors = np.ones(ray_samples.shape) * np.array([0.0, 0.0, 1.0])
        object_rays = self.trainer.sample_pts[sensor]["object_rays"]
        colors[:object_rays, : self.trainer.sensor[sensor].n_surf_samples, :] = (
            np.array([1.0, 0.0, 0.0])
        )
        ray_samples = ray_samples.reshape(-1, ray_samples.shape[-1])
        colors = colors.reshape(ray_samples.shape)

        perm = np.random.permutation(ray_samples.shape[0])
        ray_samples, colors = ray_samples[perm, :], colors[perm, :]
        ray_samples, colors = ray_samples[::skip, :], colors[::skip, :]
        ray_pcd = o3d.t.geometry.PointCloud(o3c.Tensor(ray_samples))

        ray_pcd.point["colors"] = o3c.Tensor(colors.astype(np.float32))

        n_rays = ray_min_max.shape[0] // 2
        points = ray_min_max
        lines = np.hstack(
            [np.arange(0, n_rays)[:, None], n_rays + np.arange(0, n_rays)[:, None]]
        )
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0.000, 0.475, 0.900])
        return line_set, ray_pcd

    def update_kf_frustums(self, sensor_name: str = "digit"):
        """
        Update the latest poses of keyframe frustums for sensor_name
        """
        kf_frustums = []
        T_WC_batch = self.trainer.frames[sensor_name].T_WC_batch_np[:-1]

        for T_WC in T_WC_batch:
            scale = 0.01 if "digit" in sensor_name else 0.1
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.intrinsic[sensor_name].width,
                self.intrinsic[sensor_name].height,
                self.intrinsic[sensor_name].intrinsic_matrix,
                np.linalg.inv(T_WC @ self.flip_matrix),
                scale=scale,
            )
            frustum.paint_uniform_color([0.000, 0.475, 0.900])
            kf_frustums.append(frustum)
        self.latest_frustums[sensor_name] = kf_frustums

    def update_latest_frames(
        self, sensor: str, idx, do_render: bool = False, is_curr_sensor=True
    ):
        if len(self.trainer.frames[sensor]) == 0:
            return
        rgbd_vis, render_vis, _ = self.trainer.latest_frame_vis(
            sensor, do_render, debug=self.debug_frontend.checked
        )

        latest_frame = o3d.geometry.Image(rgbd_vis.astype(np.uint8))
        render_frame = None
        if render_vis is not None:
            render_frame = o3d.geometry.Image(render_vis.astype(np.uint8))

        if is_curr_sensor:  # update display variables
            self.latest_frame = latest_frame
            self.render_frame = render_frame

            sensor_list = self.trainer.sensor.keys()
            vision_sensor = [s for s in sensor_list if "realsense" in s]
            self.latest_visual_frame = None
            if len(vision_sensor):
                latest_visual_frame = self.trainer.latest_rgb_vis(vision_sensor[0])
                self.latest_visual_frame = o3d.geometry.Image(
                    latest_visual_frame.astype(np.uint8)
                )

        # save keyframe stream and renders
        if self.record and idx is not None:
            rgb, depth = np.split(
                rgbd_vis, 2, axis=1
            )  # split latest frame into two along width: rgb and depth
            rgb, depth = o3d.geometry.Image(rgb.astype(np.uint8)), o3d.geometry.Image(
                depth.astype(np.uint8)
            )
            save_dir_rgb = f"./images/{sensor}/rgb"
            save_dir_depth = f"./images/{sensor}/depth"
            o3d.io.write_image(
                os.path.join(save_dir_rgb, f"{str(idx).zfill(6)}.jpg"),
                rgb,
                quality=100,
            )
            o3d.io.write_image(
                os.path.join(save_dir_depth, f"{str(idx).zfill(6)}.jpg"),
                depth,
                quality=100,
            )
            if render_frame is not None:
                rendered_normals, rendered_depth = np.split(
                    render_vis, 2, axis=1
                )  # split latest frame into two along width: normals and depth render
                rendered_normals, rendered_depth = o3d.geometry.Image(
                    rendered_normals.astype(np.uint8)
                ), o3d.geometry.Image(rendered_depth.astype(np.uint8))
                save_dir_rendered_normals = f"./images/{sensor}/rendered_normals"
                save_dir_rendered_depth = f"./images/{sensor}/rendered_depth"
                o3d.io.write_image(
                    os.path.join(save_dir_rendered_normals, f"{str(idx).zfill(6)}.jpg"),
                    rendered_normals,
                    quality=100,
                )
                o3d.io.write_image(
                    os.path.join(save_dir_rendered_depth, f"{str(idx).zfill(6)}.jpg"),
                    rendered_depth,
                    quality=100,
                )


def get_int(file: str) -> int:
    """
    Extract numeric value from file name
    """
    return int(file.split(".")[0])


def images_to_video(path: str, save_path: str, fps: int = 20) -> None:
    import ffmpeg

    """
    adapted from https://stackoverflow.com/a/67152804 : list of images to .mp4
    """
    images = os.listdir(path)
    images = [im for im in images if im.endswith(".jpg")]

    images = sorted(images, key=get_int)
    times = np.array([int(name[:-4]) for name in images])

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

    lbs = np.arange(0, times[-1], interval)
    last_frame_time = None
    for lb in lbs:
        ub = lb + interval
        if ub < times[0]:
            continue
        time = times[np.logical_and(times >= lb, times <= ub)]
        if len(time) == 0:
            time = last_frame_time
        else:
            time = time[0]
            last_frame_time = time

        image_path = os.path.join(path, f"{time:06d}.jpg")
        im = cv2.imread(image_path)
        success, encoded_image = cv2.imencode(".png", im)
        process.stdin.write(
            encoded_image.tobytes()
        )  # If broken pipe error, try micromamba update ffmpeg

    # Close stdin pipe - FFmpeg fininsh encoding the output file.
    process.stdin.close()
    process.wait()
