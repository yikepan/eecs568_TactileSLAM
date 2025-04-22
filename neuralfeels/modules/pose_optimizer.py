# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Pose optimizer for optimizing SE(3) pose using theseus w.r.t. neural field + pose regularization + ICP constraint

import copy
from itertools import compress
from operator import itemgetter
from typing import List, Optional, Tuple, Type

import numpy as np
import open3d as o3d
import theseus as th
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from termcolor import cprint

from neuralfeels.geometry.align_utils import register, register_fmr, register_global
from neuralfeels.modules import loss

from thirdparty.fmr.demo import draw_registration_result, draw_registration_result_and_save

DTYPE = torch.float32


# Copied from theseus: https://github.com/facebookresearch/theseus/blob/main/theseus/utils/utils.py#L155
# but without random fill.
#
# Automatically checks the jacobians of the given cost function a number of times.
#
# Computes the manifold jacobians of the given cost function with respect to all
# optimization variables, evaluated at randomly sampled values
# of the optimization and auxiliary variable, and compares them with the corresponding
# ones computed by torch autograd. By default, only checks once, but more checks can
# be specified, with one set of sampled variables per each. The jacobians are
# compared using the infinity norm of the jacobian matrix, at the specified tolerance.
@torch.no_grad()
def check_jacobians(cf: th.CostFunction, tol: float = 1.0e-3):
    from theseus.core.cost_function import _tmp_tensors

    optim_vars: List[th.Manifold] = list(cf.optim_vars)
    aux_vars = list(cf.aux_vars)

    def autograd_fn(*optim_var_tensors):
        for v, t in zip(optim_vars, optim_var_tensors):
            v.update(t)
        return cf.error()

    with _tmp_tensors(optim_vars), _tmp_tensors(aux_vars):
        autograd_jac = torch.autograd.functional.jacobian(
            autograd_fn, tuple(v.tensor for v in optim_vars)
        )
        jac, _ = cf.jacobians()
        for idx, v in enumerate(optim_vars):
            j1 = jac[idx]
            j2 = autograd_jac[idx]
            # In some "unfriendly" cost functions, the error's batch size could
            # be different than the optim/aux vars batch size, if they save
            # tensors that are not exposed as Theseus variables. To avoid issues,
            # we just check the first element of the batch.
            j2_sparse = j2[:, :, 0, :]
            j2_sparse_manifold = v.project(j2_sparse, is_sparse=True)
            print(
                f"\nCost function {cf.name}. Jacobian shapes: ",
                j1.shape,
                j2_sparse_manifold.shape,
            )
            if (j1 - j2_sparse_manifold).abs().max() > tol:
                print(
                    RuntimeError(
                        f"Jacobian for variable {v.name} appears incorrect to the given tolerance."
                    )
                )
            else:
                print("Jacobians passed")


class PoseOptimizer(nn.Module):
    def __init__(self, sensors, cfg, train_mode, device):
        super(PoseOptimizer, self).__init__()
        self.pose_cfg = cfg

        self.pose_history = {}
        self.tsdf_method = (
            self.pose_cfg.second_order.tsdf_method
        )  # ["analytic", "numerical", autodiff]
        (self.sensor_pose_batch, self.depth_batch, self.frame_ids) = [
            {} for i in range(3)
        ]
        self.all_frame_ids = None
        self.sensors = sensors

        self.train_mode = train_mode
        if train_mode in ["pose"]:
            # if pose opt with known sdf, we reduce regularization
            self.pose_cfg.second_order.reg_w = 2e-3
            print(
                f"Pose optimizer in {train_mode} mode, reg_w: {self.pose_cfg.second_order.reg_w}"
            )
        self.w_vision = self.pose_cfg.w_vision
        self.w_tactile = self.pose_cfg.w_tactile
        self.lm_iters = self.pose_cfg.second_order.lm_iters
        self.num_iters = self.pose_cfg.second_order.num_iters

    def addVariables(
        self,
        depth_batch,
        sensor_pose_batch,
        frame_ids,
        sensor_name,
    ):
        self.sensor_pose_batch[sensor_name] = th.SE3(tensor=sensor_pose_batch[:, :3, :])
        self.depth_batch[sensor_name] = depth_batch
        self.frame_ids[sensor_name] = torch.tensor(frame_ids, device=depth_batch.device)
        self.optimized_pose_batch = None
        self.object_pcd, self.frame_pcd = None, None

    def addPointCloud(self, object_pcd, frame_pcd):
        self.object_pcd = object_pcd
        self.frame_pcd = frame_pcd

    def addPoses(self, object_pose_batch):
        self.object_pose_batch = th.SE3(
            tensor=object_pose_batch[:, :3, :], name="pose", dtype=DTYPE
        )
        self.all_frame_ids, _ = torch.sort(
            torch.unique(torch.cat(list(self.frame_ids.values())))
        )
        for i, frame_id in enumerate(self.all_frame_ids):
            self.pose_history[frame_id.item()] = self.object_pose_batch[[i]]

    def addSDF(self, frozen_sdf):
        self.frozen_sdf_map = frozen_sdf

    def startTimer(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()
        self.start_mem = torch.cuda.max_memory_allocated() / 1048576
        self.start_event.record()

    def stopTimer(self):
        self.end_event.record()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.end_mem = torch.cuda.max_memory_allocated() / 1048576
        forward_mem = self.end_mem - self.start_mem
        forward_time = self.start_event.elapsed_time(self.end_event)
        return forward_time, forward_mem

    def get_sample_stats(self, samples):
        """
        Returns the unique pose indices, max samples across all poses, and samples sorted by pose
        """
        indices_b = samples["indices_b"]
        pose_idxs = torch.unique(indices_b)
        per_pose_samples = [indices_b[indices_b == i] for i in pose_idxs]
        n_per_pose_samples = np.array([len(s) for s in per_pose_samples])
        return pose_idxs, n_per_pose_samples

    def expand_to_equal_indices(self, samples, max_samples):
        """
        The pose optimizer needs equal number of ray samples for each pose.
        This function identifies poses with fewer samples and expands them to have equal number of samples
        If number of samples is < num_expand we sample with repetition
        """
        stack_samples = copy.deepcopy(samples)
        stack_samples = self.stack_samples(stack_samples)
        pose_idxs, n_per_pose_samples = self.get_sample_stats(stack_samples)
        num_expand = max_samples - n_per_pose_samples

        new_samples = {
            "pc": [],
            "depth_sample": [],
            "z_vals": [],
            "dirs_C_sample": [],
            "T_WC_sample": [],
            "cost_mul": [],
            "indices_b": [],
        }
        for pose_idx, pose_expand in zip(pose_idxs, num_expand):
            if pose_expand == 0:
                continue
            rand_idxs = torch.argwhere(stack_samples["indices_b"] == pose_idx)
            rand_idxs = rand_idxs[torch.randint(len(rand_idxs), (pose_expand,))]
            rand_idxs = rand_idxs.squeeze()
            new_samples["pc"].append(stack_samples["pc"][rand_idxs])
            new_samples["depth_sample"].append(stack_samples["depth_sample"][rand_idxs])
            new_samples["z_vals"].append(stack_samples["z_vals"][rand_idxs])
            new_samples["dirs_C_sample"].append(
                stack_samples["dirs_C_sample"][rand_idxs]
            )
            new_samples["T_WC_sample"].append(stack_samples["T_WC_sample"][rand_idxs])
            new_samples["cost_mul"].append(stack_samples["cost_mul"][rand_idxs])
            new_samples["indices_b"].append(stack_samples["indices_b"][rand_idxs])
            # print(f"Added {num_expand} samples for pose {pose_idxs[i]}")

        if len(new_samples["pc"]) > 0:
            new_samples = self.stack_samples(new_samples)
            samples = self.expand_samples(samples, new_samples)
        samples = self.stack_samples(samples)  # stack all sample points
        assert torch.all(
            torch.unique(samples["indices_b"], return_counts=True)[1] == max_samples
        ), "Not all indices sampled equally"
        return samples

    def expand_samples(self, samples, new_sample):
        samples["pc"].append(new_sample["pc"])
        samples["depth_sample"].append(new_sample["depth_sample"])
        samples["z_vals"].append(new_sample["z_vals"])
        samples["dirs_C_sample"].append(new_sample["dirs_C_sample"])
        samples["T_WC_sample"].append(new_sample["T_WC_sample"])
        samples["cost_mul"].append(new_sample["cost_mul"])
        samples["indices_b"].append(new_sample["indices_b"])
        return samples

    def stack_samples(self, samples):
        if len(samples["pc"]) == 0:
            return samples
        samples["pc"] = torch.vstack(samples["pc"])
        samples["depth_sample"] = torch.hstack(samples["depth_sample"])
        samples["z_vals"] = torch.vstack(samples["z_vals"])
        samples["dirs_C_sample"] = torch.vstack(samples["dirs_C_sample"])
        samples["T_WC_sample"] = torch.vstack(samples["T_WC_sample"])
        samples["cost_mul"] = torch.hstack(samples["cost_mul"])
        samples["indices_b"] = torch.hstack(samples["indices_b"])
        return samples

    def get_icp_cost(self, pg_batch):
        min_fitness, max_inlier_rmse = (
            self.pose_cfg.second_order.icp_fitness,
            self.pose_cfg.second_order.icp_inlier_rmse,
        )
        icp_thresh = self.pose_cfg.second_order.icp_thresh
        icp_w = self.pose_cfg.second_order.icp_w
        between_cost_weight = th.DiagonalCostWeight(
            th.Variable(
                torch.tensor(
                    [
                        [
                            icp_w,
                            icp_w,
                            icp_w,
                            2e1 * icp_w,
                            2e1 * icp_w,
                            2e1 * icp_w,
                        ]
                    ],
                    dtype=DTYPE,
                    device=self.frozen_sdf_map.device,
                )
            )
        )
        between_constraint = pg_batch.copy(new_name="icp_relative_pose")

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(
            self.frame_pcd.astype(np.float32)
        )  # current frame

        # get the pointclouds
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(
            self.object_pcd.astype(np.float32)
        )  # previous frame

        # skip if either pointcloud is empty
        # breakpoint()
        if not (len(source_pcd.points) == 0 or len(target_pcd.points) == 0):
            # get the registration
            # T_reg, metrics_reg = register(
            #     points3d_1=source_pcd,
            #     points3d_2=target_pcd,
                # debug_vis=False,
            # )
            # o3d.io.write_point_cloud("C:\Workspace\Codebase\\neuralfeels\source_pcd.ply", source_pcd)
            # o3d.io.write_point_cloud("C:\Workspace\Codebase\\neuralfeels\\target_pcd.ply", target_pcd)
            # print(T_reg)
            # breakpoint()
            # metrics_reg = [1, 100, 0]
            voxel_size = 0.005
            
            T_est = register_fmr(points3d_1=source_pcd, points3d_2=target_pcd, voxel_size=voxel_size)
            T_reg = np.linalg.inv(T_est)
            # T_reg = register_global(source_pcd, target_pcd, voxel_size)
            
            # Visualization
            # p0 = source_pcd.voxel_down_sample(voxel_size=0.005)
            # p1 = target_pcd.voxel_down_sample(voxel_size=0.005)
            # draw_registration_result(p0, p1, T_reg)
            # draw_registration_result_and_save(p0, p1, T_reg)
            # breakpoint()
            
            # # skip if the registration is bad
            # if not (
            #     metrics_reg[0] == 0
            #     or metrics_reg[0] < min_fitness
            #     or metrics_reg[1] > max_inlier_rmse
            # ):
            if True:
                # convert 3x3 rotation matrix to euler angles T_reg[:3, :3]
                r = R.from_matrix(np.array(T_reg[:3, :3]))
                T_euler = r.as_euler("xyz", degrees=True)

                T_trans = T_reg[:3, 3]
                # check if rotation and translation in T_reg are in bounds
                if np.any(np.abs(T_euler) > icp_thresh[0]) or np.any(
                    np.abs(T_trans) > icp_thresh[1]
                ):
                    # print(
                    #     f"Rejecting ICP beyond bounds, rotation: {np.any(np.abs(T_euler) >  icp_thresh[0])} translation: {np.any(np.abs(T_trans) > icp_thresh[1])}"
                    # )
                    pass
                else:
                    # print(f"Fitness: {metrics_reg[0]}, Inlier RMSE: {metrics_reg[1]}")
                    T_reg = np.linalg.inv(
                        T_reg
                    )  # inverting for transformation from previous to current frame
                    T_reg = torch.tensor(T_reg[:3, :]).to(
                        self.frozen_sdf_map.device, dtype=DTYPE
                    )

                    pose_delta = th.SE3(
                        tensor=T_reg.unsqueeze(dim=0), name="pose_1", dtype=DTYPE
                    )
                    prev_pose = th.SE3(
                        tensor=pg_batch[[-2]], name="pose_1", dtype=DTYPE
                    )
                    between_constraint[-1] = pose_delta.compose(prev_pose).tensor
                    between_cost = th.Difference(
                        var=pg_batch,
                        cost_weight=between_cost_weight,
                        target=between_constraint,
                    )
                    return between_cost
            # else:
            #     print(
            #         f"Rejecting ICP due to bad fitness: {metrics_reg[0]}, RMSE: {metrics_reg[1]}"
            #     )
        return None

    def optimize_pose_theseus(self, opt_sensors, num_iters):
        """
        Optimize SE(3) pose using theseus w.r.t. neural field + pose regularization + ICP constraint
        """
        sample_pts = {
            "pc": [],
            "depth_sample": [],
            "z_vals": [],
            "dirs_C_sample": [],
            "T_WC_sample": [],
            "cost_mul": [],
            "indices_b": [],
        }

        def tsdf_loss(
            optim_vars: List[th.SE3],
            aux_vars=None,
        ):
            """
            TSDF SE(3) err_fn for theseus AutoDiffCostFunction
            Optimizes the pose of the object by minimizing the TSDF loss from point samples w.r.t. the neural field
            input: th.SE3 object pose batch of size (N, 4, 4)
            output: (6, N) tensor of pose errors w.r.t. the SDF
            """
            poses = optim_vars[0]
            # evaluate the SDF at the sample points [n_rays, n_samples]
            sdf = self.sdf_eval(sample_pts, poses.to_matrix())

            if self.tsdf_method in ["analytic", "numerical"]:
                # directly supervising the surface loss in the truncation region
                loss_mat = torch.abs((sdf) / self.pose_cfg.trunc_distance)
            else:
                # compute bounds from surface and supervise, https://joeaortiz.github.io/iSDF/
                pc = sample_pts["pc"]
                z_vals = sample_pts["z_vals"]
                depth_sample = sample_pts["depth_sample"]
                bounds, _ = loss.bounds_pc(
                    pc,
                    z_vals,
                    depth_sample,
                    self.pose_cfg.trunc_distance,
                    do_grad=False,
                )

                # compute losses
                sdf_loss_mat, free_space_ixs = loss.sdf_loss(
                    sdf,
                    bounds,
                    self.pose_cfg.trunc_distance,
                    loss_type=self.pose_cfg.loss_type,
                )

                total_loss, loss_mat, losses = loss.tot_loss(
                    sdf_loss_mat=sdf_loss_mat,
                    eik_loss_mat=None,
                    bounds=bounds,
                    free_space_ixs=free_space_ixs,
                    trunc_weight=self.pose_cfg.trunc_weight,
                    eik_weight=0,
                    vision_weights=None,
                )

            loss_mat = torch.mean(
                loss_mat * sample_pts["cost_mul"][:, None], dim=1
            )  # average the loss over the n_rays
            per_pose_losses = [
                loss_mat[sample_pts["indices_b"] == i]
                for i in torch.unique(sample_pts["indices_b"])
            ]  # sort the losses by pose
            per_pose_losses = torch.stack(per_pose_losses, dim=0)
            return per_pose_losses

        """
        Get neural field samples for each sensor
        """
        # get valid sensors with valid depth readings
        n_rays_per_sensor_vision = self.pose_cfg.n_rays_per_sensor_vision
        n_rays_per_sensor_tactile = self.pose_cfg.n_rays_per_sensor_tactile
        valid_vision_sensors = [
            torch.is_nonzero(torch.nansum(self.depth_batch[sensor.sensor_name]))
            for sensor in self.sensors
            if "digit" not in sensor.sensor_name
        ]
        valid_tactile_sensors = [
            torch.is_nonzero(torch.nansum(self.depth_batch[sensor.sensor_name]))
            for sensor in self.sensors
            if "digit" in sensor.sensor_name
        ]
        total_rays = (
            sum(valid_vision_sensors) * n_rays_per_sensor_vision
            + sum(valid_tactile_sensors) * n_rays_per_sensor_tactile
        )

        self.compute_sensor_dist()

        for sensor in self.sensors:
            sensor_name = sensor.sensor_name
            if sensor_name not in opt_sensors:
                continue
            cam_matrix = self.sensor_pose_batch[sensor_name].to_matrix()
            depth_batch = copy.deepcopy(self.depth_batch[sensor_name])

            if torch.nansum(depth_batch) == 0:
                continue  # skip if no depth in batch
            # sampling only around the surface with a small delta of surface_samples_offset
            sample_pts_ = sensor.sample_points(
                depth_batch,
                cam_matrix,
                n_rays=(
                    n_rays_per_sensor_vision
                    if "digit" not in sensor_name
                    else n_rays_per_sensor_tactile
                ),
                dist_behind_surf=0.0,
                n_strat_samples=0,
                n_surf_samples=1,
                surface_samples_offset=1e-10,
                free_space_ratio=0.0,
                grad=False,
            )
            weight = (
                self.w_vision * self.sensor_inv_dists[sensor_name]
                if "digit" not in sensor_name
                else self.w_tactile
            )
            sample_pts_["cost_mul"] = weight * torch.ones(
                sample_pts_["z_vals"].shape[0]
            ).to(self.frozen_sdf_map.device)

            # print(f"Sensor : {sensor_name}, Sample indices: {torch.unique(sample_pts_['indices_b'], return_counts=True)[1]}")
            sample_pts = self.expand_samples(sample_pts, sample_pts_)

        pg_batch = self.object_pose_batch
        # breakpoint()

        theseus_inputs = {}
        if self.pose_cfg.timer:
            self.startTimer()

        """
        Loop over the poses and add them to the pose graph
        """
        pg_batch = th.SE3(
            tensor=pg_batch,
            name=f"pose_batch",
        )
        pg_batch.to(dtype=DTYPE, device=self.frozen_sdf_map.device)
        theseus_inputs["pose_batch"] = pg_batch
        objective = th.Objective(dtype=DTYPE)
        objective.to(self.frozen_sdf_map.device)

        """
        Create cost function for TSDF loss
        """
        if len(sample_pts["pc"]):
            # expand to have equal samples/pose
            sample_pts = self.expand_to_equal_indices(sample_pts, total_rays)
            if self.tsdf_method in ["analytic", "numerical"]:
                sdf_eval = self.sdf_eval

                # jacobian fn for part of tsdf_loss after the SDF evaluation
                def jac_tsdf_loss(jac_pose, sdf):
                    # jac of: torch.abs((sdf) / self.pose_cfg.trunc_distance)
                    jac_pose[sdf < 0] *= -1.0
                    jac_pose = jac_pose / self.pose_cfg.trunc_distance

                    # jac of: torch.mean(loss_mat * sample_pts["cost_mul"][:, None], dim=1)
                    jac_pose = torch.mean(
                        jac_pose * sample_pts["cost_mul"][:, None, None], dim=1
                    )

                    # jac of sort the losses by pose
                    jac_pose = [
                        jac_pose[sample_pts["indices_b"] == i]
                        for i in torch.unique(sample_pts["indices_b"])
                    ]
                    per_pose_jac = torch.stack(jac_pose, dim=0)
                    return per_pose_jac.squeeze()[None, :]

                class TSDFCostFunction(th.CostFunction):
                    def __init__(
                        self,
                        pose: th.Vector,
                        frozen_sdf_map,
                        cost_weight: th.CostWeight,
                        name: Optional[str] = None,
                    ):
                        super().__init__(cost_weight, name=name)
                        if not isinstance(pose, th.SE3):
                            raise ValueError(
                                f"Pose must be type th.SE3, but got {type(pose)}."
                            )
                        self.pose = pose
                        self.frozen_sdf_map = frozen_sdf_map

                        self._dim = max(
                            [
                                sum(sample_pts["indices_b"] == i).item()
                                for i in torch.unique(sample_pts["indices_b"])
                            ]
                        )
                        self.register_optim_vars(["pose"])

                    def error(self) -> torch.Tensor:
                        return tsdf_loss([self.pose])

                    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
                        pose = self.pose.to_matrix()
                        jac_pose, sdf = sdf_eval(sample_pts, pose, return_jacobian=True)
                        per_pose_jac = jac_tsdf_loss(jac_pose, sdf)
                        return [per_pose_jac], self.error()

                    def dim(self) -> int:
                        return self._dim

                    def _copy_impl(
                        self, new_name: Optional[str] = None
                    ) -> "TSDFCostFunction":
                        return TSDFCostFunction(  # type: ignore
                            self.pose.copy(),
                            copy.deepcopy(self.frozen_sdf_map),
                            self.weight.copy(),
                            name=new_name,
                        )

                tsdf_w = self.pose_cfg.second_order.tsdf_w
                cost_function = TSDFCostFunction(
                    pg_batch,
                    self.frozen_sdf_map,
                    cost_weight=th.ScaleCostWeight(
                        torch.tensor(
                            tsdf_w,
                            dtype=DTYPE,
                            device=self.frozen_sdf_map.device,
                        )
                    ),
                )
                if self.pose_cfg.second_order.test_jacobians:
                    check_jacobians(cost_function)
                    # self.test_all_jacobians(
                    #     cost_function, sample_pts, pg_batch, tsdf_loss, jac_tsdf_loss
                    # )

            else:
                cf_dim = max(
                    [
                        sum(sample_pts["indices_b"] == i).item()
                        for i in torch.unique(sample_pts["indices_b"])
                    ]
                )
                cost_function = th.AutoDiffCostFunction(
                    optim_vars=[pg_batch],
                    err_fn=tsdf_loss,
                    dim=cf_dim,  # n_rays
                    cost_weight=th.ScaleCostWeight(
                        torch.tensor(
                            self.pose_cfg.second_order.tsdf_w,
                            dtype=DTYPE,
                            device=self.frozen_sdf_map.device,
                        )
                    ),
                    aux_vars=None,
                    autograd_create_graph=False,
                    autograd_vectorize=self.pose_cfg.second_order.vectorize,
                    autograd_strategy=self.pose_cfg.second_order.autograd_strategy,
                    autograd_mode=th.AutogradMode.DENSE,  # TODO: VMAP errors
                )

            objective.add(cost_function)

        """
        Create cost function for pose regularization
        """
        if self.pose_cfg.second_order.regularize:
            reg_w = self.pose_cfg.second_order.reg_w
            if sum(valid_vision_sensors) == 0:
                # if no vision sensors, we reduce regularization
                # print("No vision sensors, reducing regularization")
                reg_w *= 1e-2

            if pg_batch.shape[0] == 1:
                reg_w = 1e10  # prior on first pose
            pg_batch_init = th.SE3(
                tensor=torch.cat([pg_batch.tensor[[0]], pg_batch.tensor[:-1]], dim=0),
                name=pg_batch.name + "__PRIOR",
            )
            # regularize rotation tighter than translation
            pose_reg_cost = th.Difference(
                var=pg_batch,
                cost_weight=th.DiagonalCostWeight(
                    th.Variable(
                        torch.tensor(
                            [
                                [
                                    reg_w,
                                    reg_w,
                                    reg_w,
                                    5e1 * reg_w,
                                    5e1 * reg_w,
                                    5e1 * reg_w,
                                ]
                            ],
                            dtype=DTYPE,
                            device=self.frozen_sdf_map.device,
                        )
                    )
                ),
                target=pg_batch_init,
            )
            # cprint(f"pose_reg error: {pose_reg_cost.error().mean()}", color="red")
            objective.add(pose_reg_cost)

        """
        Add ICP constraint between the object point cloud and the frame point cloud
        """
        if self.pose_cfg.second_order.icp and self.frame_pcd is not None:
            between_cost = self.get_icp_cost(pg_batch)
            if between_cost is not None:
                # cprint(f"icp error: {between_cost.error().mean()}", color="green")
                objective.add(between_cost)

        """
        Define second order optimizer
        """
        optimizer_cls: Type[th.NonlinearLeastSquares] = getattr(
            th, self.pose_cfg.second_order.optimizer_cls
        )
        objective.update(theseus_inputs)
        optimizer = optimizer_cls(
            objective,
            max_iterations=num_iters,
            step_size=self.pose_cfg.second_order.step_size,
            linear_solver_cls=getattr(th, self.pose_cfg.second_order.linear_solver_cls),
            linearization_cls=getattr(th, self.pose_cfg.second_order.linearization_cls),
        )

        """
        Define theseus layer
        """
        theseus_optim = th.TheseusLayer(
            optimizer, vectorize=self.pose_cfg.second_order.vectorize
        )
        with torch.no_grad():
            updated_inputs, info = theseus_optim.forward(
                theseus_inputs,
                optimizer_kwargs={**self.pose_cfg.second_order.optimizer_kwargs},
            )
            if info.status[0] == th.NonlinearOptimizerStatus.FAIL:
                cprint(
                    f"WARNING: {info.status}", color="red"
                )  # check for convergence errors

        """
        Get optimized poses and update the pose history
        """
        self.optimized_pose_batch = info.best_solution["pose_batch"].detach()
        self.optimized_pose_batch = th.SE3(tensor=self.optimized_pose_batch)
        self.optimized_pose_batch.to(self.frozen_sdf_map.device)
        # breakpoint()
        pose_loss = info.last_err[-1].item()
        time_string = ""
        if self.pose_cfg.timer:
            fwd_time, fwd_mem = self.stopTimer()
            time_string = f"Forward pass took {fwd_time} ms"
            mem_string = f"{fwd_mem} MB"
            print(f"{time_string} / {mem_string}, loss: {info.last_err[-1].item():.6f}")
            print(
                "---------------------------------------------------------------"
                "---------------------------"
            )
        # gc.collect()
        # torch.cuda.empty_cache()
        if self.pose_cfg.show_samples:
            # stack all sample points
            return sample_pts["pc"].view(-1, 3), pose_loss
        else:
            return None, pose_loss

    def compute_sensor_dist(self):
        """
        Calculates the distance of each sensor from the current object pose for weighting loss (multi-cam)
        """
        sensor_inv_dists = {}
        for sensor in self.sensors:
            sensor_name = sensor.sensor_name
            if "digit" in sensor_name:
                continue
            cam_matrix = self.sensor_pose_batch[sensor_name].to_matrix()
            sensor_dist = torch.norm(cam_matrix[0, :3, 3]) - torch.norm(
                self.object_pose_batch[0, :3, 3]
            )
            inv_sensor_dist = 1.0 / sensor_dist
            sensor_inv_dists[sensor_name] = inv_sensor_dist
        # scale sensor_inv_dists [0, 1]
        self.sensor_inv_dists = {
            k: v / max(sensor_inv_dists.values()) for k, v in sensor_inv_dists.items()
        }

    def sdf_eval(self, sample, pose, return_jacobian=False):
        """
        Computes the SDF for a given pose and sample points
        if return_jacobian: returns the jacobian of the sdf evaluation wrt pose
        """
        pc = sample["pc"]
        indices_b = sample["indices_b"]

        # updates the pose batch of the object
        self.frozen_sdf_map.update_pose_and_scale(pose)
        # updates poses each sample is observed in
        self.frozen_sdf_map.update_frames(obs_frames=indices_b)

        if return_jacobian:
            is_numerical = "numerical" in self.tsdf_method
            sdf, jac_pose = self.frozen_sdf_map.jacobian(
                pc, numerical_jacobian=is_numerical
            )
            sdf = sdf.reshape(pc.shape[:-1])
            jac_pose = jac_pose.reshape([*pc.shape[:-1], 6])
            return jac_pose, sdf
        else:
            sdf = self.frozen_sdf_map(pc)  # evaluates the SDF grid at the sample points
            sdf = sdf.reshape(pc.shape[:-1])
            return sdf

    def test_all_jacobians(
        self, full_cf, sample_pts, pg_batch, tsdf_loss, jac_tsdf_loss
    ):
        """
        Check the full Jacobian and then each part of the Jacobian separately.
        """
        check_jacobians(full_cf)

        """
        Test each part of the Jacobian separately.
        """

        cost_weight = th.ScaleCostWeight(1.0)
        sdf_eval = self.sdf_eval

        """
        Check the SDF evaluation jacobian.
        """

        class SDFEvalCostFunction(th.CostFunction):
            def __init__(self, pose, cost_weight, name=None):
                super().__init__(cost_weight, name=name)
                self.pose = pose
                self.register_optim_vars(["pose"])

            def error(self) -> torch.Tensor:
                return sdf_eval(sample_pts, self.pose.to_matrix()).reshape(1, -1)

            def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
                jac_pose, sdf = sdf_eval(
                    sample_pts, self.pose.to_matrix(), return_jacobian=True
                )
                return [jac_pose.reshape(1, -1, 6)], sdf.reshape(1, -1)

            def dim(self) -> int:
                return int(sample_pts["pc"].numel() / 3)

            def _copy_impl(
                self, new_name: Optional[str] = None
            ) -> "SDFEvalCostFunction":
                return SDFEvalCostFunction(
                    self.pose.copy(), self.weight.copy(), name=new_name
                )

        cf_evalsdf_num = SDFEvalCostFunction(pg_batch, cost_weight)
        check_jacobians(cf_evalsdf_num)

        """
        Break down SDF evaluation into transform and trilinear interpolation.

        First check jacobian for transform.
        """
        frozen_sdf_map = self.frozen_sdf_map
        pc = sample_pts["pc"]

        class TransformCF(th.CostFunction):
            def __init__(self, pose, cost_weight, name=None):
                super().__init__(cost_weight, name=name)
                self.pose = pose
                self.register_optim_vars(["pose"])

            def error(self) -> torch.Tensor:
                frozen_sdf_map.update_pose_and_scale(self.pose.to_matrix())
                return frozen_sdf_map.transform(pc)

            def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
                frozen_sdf_map.update_pose_and_scale(self.pose.to_matrix())
                pts, jac_pose = frozen_sdf_map.transform(pc, return_jacobian=True)
                return [jac_pose], pts

            def dim(self) -> int:
                return 3

            def _copy_impl(self, new_name: Optional[str] = None) -> "TransformCF":
                return TransformCF(self.pose.copy(), self.weight.copy(), name=new_name)

        cf_transf_analytic = TransformCF(pg_batch, cost_weight)
        check_jacobians(cf_transf_analytic)

        """
        Check jacobian for trilinear interpolation.
        Note this is a numerical jacobian. 
        """
        pts_to_interp = cf_transf_analytic.error()
        frozen_sdf_map.update_pose_and_scale(pg_batch.to_matrix())

        class TrilinInterpNumCF(th.CostFunction):
            def __init__(self, pts, cost_weight, name=None):
                super().__init__(cost_weight, name=name)
                self.pts = pts
                self.register_optim_vars(["pts"])

            def error(self) -> torch.Tensor:
                (xx, yy, zz) = torch.tensor_split(self.pts.tensor, 3, dim=1)
                h = frozen_sdf_map.sdf_interp((xx, yy, zz))
                h = h.squeeze()
                return h[:, None]

            def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
                jac_trilin = frozen_sdf_map.sdf_interp.jacobian(self.pts.tensor)
                return [jac_trilin], self.error()

            def dim(self) -> int:
                return 1

            def _copy_impl(self, new_name: Optional[str] = None) -> "TrilinInterpNumCF":
                return TrilinInterpNumCF(
                    self.pose.copy(), self.weight.copy(), name=new_name
                )

        pts = th.Vector(tensor=pts_to_interp)
        cf_tri_interp_num = TrilinInterpNumCF(pts, cost_weight)
        check_jacobians(cf_tri_interp_num)

        class TrilinInterpAnalyticCF(th.CostFunction):
            def __init__(self, pts, cost_weight, name=None):
                super().__init__(cost_weight, name=name)
                self.pts = pts
                self.register_optim_vars(["pts"])

            def error(self) -> torch.Tensor:
                (xx, yy, zz) = torch.tensor_split(self.pts.tensor, 3, dim=1)
                h = frozen_sdf_map.sdf_interp((xx, yy, zz))
                h = h.squeeze()
                return h[:, None]

            def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
                (xx, yy, zz) = torch.tensor_split(self.pts.tensor, 3, dim=1)
                h, jac_trilin = frozen_sdf_map.sdf_interp(
                    (xx, yy, zz), return_jacobian=True
                )
                h = h.squeeze()
                return [jac_trilin], h[:, None]

            def dim(self) -> int:
                return 1

            def _copy_impl(
                self, new_name: Optional[str] = None
            ) -> "TrilinInterpAnalyticCF":
                return TrilinInterpAnalyticCF(
                    self.pose.copy(), self.weight.copy(), name=new_name
                )

        pts = th.Vector(tensor=pts_to_interp)
        cf_tri_interp_an = TrilinInterpAnalyticCF(pts, cost_weight)
        check_jacobians(cf_tri_interp_an)

        """
        Check the TSDF loss function jacobian.
        """
        # full autograd jacobian
        cf_dim = max(
            [
                sum(sample_pts["indices_b"] == i).item()
                for i in torch.unique(sample_pts["indices_b"])
            ]
        )
        cf_full_ad = th.AutoDiffCostFunction([pg_batch], tsdf_loss, cf_dim, cost_weight)
        jac_ad = cf_full_ad.jacobians()[0][0]

        # get autograd jacobians for SDF evaluation
        def autograd_fn(pose):
            return sdf_eval(sample_pts, pose)

        autograd_jac = torch.autograd.functional.jacobian(
            autograd_fn, pg_batch.to_matrix()
        )
        j2_sparse = autograd_jac[:, :, 0, :]
        jac_evalsdf_ad = pg_batch.project(j2_sparse[:, :, :3], is_sparse=True)

        # compute full jacobians using autograd for sdf_eval + analytic for tsdf_loss
        # and compare with full jacobians with autograd
        sdf = cf_evalsdf_num.error()
        pc_shape = sample_pts["pc"].shape
        sdf = sdf.reshape(pc_shape[:-1])
        jac_evalsdf_ad = jac_evalsdf_ad.reshape([*pc_shape[:-1], 6])
        jac_analytic = jac_tsdf_loss(jac_evalsdf_ad, sdf)

        tol = 1e-3
        print(
            f"\nCost function tsdf_loss. Jacobian shapes: ",
            jac_ad.shape,
            jac_analytic.shape,
        )
        if (jac_ad - jac_analytic).abs().max() > tol:
            print(
                RuntimeError(
                    f"Jacobian for variable {pg_batch.name} appears incorrect to the given tolerance."
                )
            )
        else:
            print("Jacobians passed")

    def getOptimizedPoses(self, matrix=False):
        assert self.optimized_pose_batch is not None, "Must optimize first"
        if matrix:
            return self.optimized_pose_batch.to_matrix(), self.all_frame_ids
        else:
            return self.optimized_pose_batch, self.all_frame_ids


def rot2euler(rot: torch.Tensor) -> torch.Tensor:
    """
    From: https://github.com/facebookresearch/MidasTouch
    Convert rotation matrix to euler angles
    Adapted from so3_rotation_angle() in  https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/so3.html
    """
    rot_trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    phi_cos = torch.acos((rot_trace - 1.0) * 0.5)
    return torch.rad2deg(phi_cos)

    # r = R.from_matrix(np.atleast_3d(rot.cpu().numpy()))
    # eul = r.as_euler('xyz', degrees = True)
    # return torch.tensor(eul, device= rot.device)


def wrap_angles(angles: torch.Tensor) -> torch.Tensor:
    """
    From: https://github.com/facebookresearch/MidasTouch
    angles : (N, 3) angles in degrees
    Wraps to [-np.pi, np.pi] or [-180, 180]
    """

    mask = angles > 180.0
    angles[mask] -= 2.0 * 180.0

    mask = angles < -180.0
    angles[mask] += 2.0 * 180.0
    return angles


def rot_rmse(gt_rot, target_rot):
    R_diff = torch.matmul(gt_rot, target_rot.permute(0, 2, 1))
    R_diff = torch.nan_to_num(rot2euler(R_diff))
    e_r = wrap_angles(R_diff)
    rmse_r = torch.sqrt(torch.mean((e_r) ** 2, dim=0))
    return rmse_r


def trans_rmse(target_trans, gt_trans):
    T_diff = gt_trans - target_trans
    e_t = torch.norm(T_diff, dim=1)
    rmse_t = torch.sqrt(torch.mean((e_t) ** 2))
    return rmse_t


def pose_rmse(target_pose, gt_pose):
    """
    From: https://github.com/facebookresearch/MidasTouch
    Returns RMSE of rotation and translation in degrees and meters respectively
    """
    rmse_t = trans_rmse(gt_pose[:, :3, 3], target_pose[:, :3, 3])
    rmse_r = rot_rmse(gt_pose[:, :3, :3], target_pose[:, :3, :3])
    return rmse_t, rmse_r
