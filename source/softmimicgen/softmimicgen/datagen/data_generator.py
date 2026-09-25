# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base class for data generator.
"""
import asyncio
import numpy as np
import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import (
    ManagerBasedRLMimicEnv,
    MimicEnvCfg,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaaclab.managers import TerminationTermCfg

from softmimicgen.datagen.datagen_info import DatagenInfo
from softmimicgen.datagen.selection_strategy import make_selection_strategy
from softmimicgen.datagen.waypoint import MultiWaypoint, Waypoint, WaypointSequence, WaypointTrajectory

from .datagen_info_pool import DataGenInfoPool

from rapprentice.tps import tps_fit2, tps_eval, tps_grad
from rapprentice.registration import tps_rpm_bij, unit_boxify, unscale_tps

def transform_source_data_segment_using_delta_object_pose(
    src_eef_poses: torch.Tensor,
    delta_obj_pose: torch.Tensor,
) -> torch.Tensor:
    """
    Transform a source data segment (object-centric subtask segment from source demonstration) using
    a delta object pose.

    Args:
        src_eef_poses: pose sequence (shape [T, 4, 4]) for the sequence of end effector control poses
            from the source demonstration
        delta_obj_pose: 4x4 delta object pose

    Returns:
        transformed_eef_poses: transformed pose sequence (shape [T, 4, 4])
    """
    return PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=delta_obj_pose[None],
    )


def transform_source_data_segment_using_object_pose(
    obj_pose: torch.Tensor,
    src_eef_poses: torch.Tensor,
    src_obj_pose: torch.Tensor,
) -> torch.Tensor:
    """
    Transform a source data segment (object-centric subtask segment from source demonstration) such that
    the relative poses between the target eef pose frame and the object frame are preserved. Recall that
    each object-centric subtask segment corresponds to one object, and consists of a sequence of
    target eef poses.

    Args:
        obj_pose: 4x4 object pose in current scene
        src_eef_poses: pose sequence (shape [T, 4, 4]) for the sequence of end effector control poses
            from the source demonstration
        src_obj_pose: 4x4 object pose from the source demonstration

    Returns:
        transformed_eef_poses: transformed pose sequence (shape [T, 4, 4])
    """

    # transform source end effector poses to be relative to source object frame
    src_eef_poses_rel_obj = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=PoseUtils.pose_inv(src_obj_pose[None]),
    )

    # apply relative poses to current object frame to obtain new target eef poses
    transformed_eef_poses = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses_rel_obj,
        pose_A_in_B=obj_pose[None],
    )
    return transformed_eef_poses


def get_delta_pose_with_scheme(
    src_obj_pose: torch.Tensor,
    cur_obj_pose: torch.Tensor,
    task_constraint: dict,
) -> torch.Tensor:
    """
    Get the delta pose with the given coordination scheme.

    Args:
        src_obj_pose: 4x4 object pose in source scene
        cur_obj_pose: 4x4 object pose in current scene
        task_constraint: task constraint dictionary

    Returns:
        delta_pose: 4x4 delta pose
    """
    coord_transform_scheme = task_constraint["coordination_scheme"]
    device = src_obj_pose.device
    if coord_transform_scheme == SubTaskConstraintCoordinationScheme.TRANSFORM:
        delta_pose = PoseUtils.get_delta_object_pose(cur_obj_pose, src_obj_pose)
        # add noise to delta pose position
    elif coord_transform_scheme == SubTaskConstraintCoordinationScheme.TRANSLATE:
        delta_pose = torch.eye(4, device=device)
        delta_pose[:3, 3] = cur_obj_pose[:3, 3] - src_obj_pose[:3, 3]
    elif coord_transform_scheme == SubTaskConstraintCoordinationScheme.REPLAY:
        delta_pose = torch.eye(4, device=device)
    else:
        raise ValueError(
            f"coordination coord_transform_scheme {coord_transform_scheme} not supported, only"
            f" {[e.value for e in SubTaskConstraintCoordinationScheme]} are supported"
        )

    pos_noise_scale = task_constraint["coordination_scheme_pos_noise_scale"]
    rot_noise_scale = task_constraint["coordination_scheme_rot_noise_scale"]
    if pos_noise_scale != 0.0 or rot_noise_scale != 0.0:
        pos = delta_pose[:3, 3]
        rot = delta_pose[:3, :3]
        pos_new, rot_new = PoseUtils.add_uniform_noise_to_pose(pos, rot, pos_noise_scale, rot_noise_scale)
        delta_pose = torch.eye(4, device=device)
        delta_pose[:3, 3] = pos_new
        delta_pose[:3, :3] = rot_new
    return delta_pose


def transform_source_data_segment_using_nodal_registration(
    src_eef_poses: torch.Tensor,
    src_obj_nodal_pos: torch.Tensor,
    tgt_obj_nodal_pos: torch.Tensor,
    use_rotation_transform: bool = True,
    bend_coef: float = 0.1,
    rot_coef: float = 1e-3,
    eef_warp_offset: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """
    Transform a source data segment using non-rigid registration between source and target object nodal positions.

    Args:
        src_eef_poses: pose sequence (shape [T, 4, 4]) for the sequence of end effector control poses
            from the source demonstration
        src_obj_nodal_pos: source object nodal positions (shape [N, 3])
        tgt_obj_nodal_pos: target object nodal positions (shape [N, 3])
        use_rotation_transform: whether to transform rotations using Jacobian (default: True)
        bend_coef: bending coefficient for TPS regularization (default: 0.1)
        rot_coef: rotation regularization coefficient for TPS (default: 1e-3)
        eef_warp_offset: fixed grasp-point offset in the recorded EEF's local axes, in metres.
            Warp this point and its local Jacobian, then return poses of the original control frame.
            None keeps the warp centered at the recorded EEF origin.

    Returns:
        transformed_eef_poses: transformed pose sequence (shape [T, 4, 4])
    """
    # Convert tensors to numpy for Rapprentice functions
    src_nodal_np = src_obj_nodal_pos.detach().cpu().numpy().astype(np.float64)
    tgt_nodal_np = tgt_obj_nodal_pos.detach().cpu().numpy().astype(np.float64)

    # Ensure we have the same number of nodal points
    assert src_nodal_np.shape[0] == tgt_nodal_np.shape[0], "Source and target must have same number of nodal points"
    assert src_nodal_np.shape[1] == 3, "Nodal positions must be 3D"

    # Similarity-anchored registration. Two earlier findings drive this design:
    # (1) A free-affine TPS between differently-crumpled mid-fold cloths produced a
    #     collapsing linear part (det ~0.5, Z-shear -0.44 on the YAM cell), shrinking
    #     the workspace and flattening the release sweep.
    # (2) A pure rigid anchor (rotation + translation) protects against (1) but cannot
    #     absorb uniform cloth scaling: a 0.7x cloth left residual dets of 0.04-0.49,
    #     tripping the guardrail on every cloth subtask and reverting to unadapted
    #     trajectories that grasped empty air.
    # The anchor is therefore a 2D-xy similarity transform: rotation + uniform xy
    # scale + translation fitted by Umeyama on the two clouds, with z treated
    # rigidly. Rationale: the cloth is a planar object on a fixed-height table, so
    # legitimate scale changes are in-plane; scaling z would also scale EE heights
    # (dive depth, sweep clearance) relative to the table, which is unphysical.
    # Rotation is fitted in 3D so cloth tilt still maps correctly. The TPS residual
    # then absorbs only local shape difference (crumple), with the same guardrail.
    src_mean = src_nodal_np.mean(axis=0)
    tgt_mean = tgt_nodal_np.mean(axis=0)
    src_centered = src_nodal_np - src_mean
    tgt_centered = tgt_nodal_np - tgt_mean
    cov = src_centered.T @ tgt_centered / len(src_centered)
    u, _, vt = np.linalg.svd(cov)
    # Kabsch reflection guard: correct the last singular direction (a column flip,
    # not a row flip) so rotation is always proper, det = +1. The source cloud at
    # the grasp boundary is nearly flat (z-extent ~0), so its third singular vector
    # is sign-ambiguous and this branch is hit routinely.
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    # Umeyama uniform scale in the SOURCE (object) frame. USD cloth scale acts on the
    # object's local axes BEFORE rotation, so the anchor is A = diag(s,s,1) @ R
    # (scale the source plane, then rotate). Row-convention proof: tgt-sc =
    # sc @ diag(s,s,1) @ Rk, so rotating the target back into the source frame
    # (tc @ Rk.T) recovers sc @ diag(s,s,1) exactly, and the scale is the
    # cross-term ratio in that frame. Kabsch Rk (with the reflection guard) equals
    # the true rotation in row convention; verified to 1e-7 on yaw/pitch/combined
    # synthetic cases. Guarded against degenerate source spread.
    y_src_frame = tgt_centered @ rotation.T
    s_est = float((y_src_frame[:, :2] * src_centered[:, :2]).sum() / (src_centered[:, :2] ** 2).sum())
    scale = float(np.clip(s_est, 0.25, 4.0))
    anchor_lin = np.diag([scale, scale, 1.0]) @ rotation
    anchor_trans = tgt_mean - src_mean @ anchor_lin
    aligned_src = src_centered @ anchor_lin + tgt_mean

    lin_ag, trans_g, w_ng = tps_fit2(
        x_na=aligned_src,
        y_ng=tgt_nodal_np,
        bend_coef=bend_coef,
        rot_coef=rot_coef,
    )

    # Guardrail: the residual affine part must stay near-identity. If a degenerate
    # fit still slips through, fall back to the similarity anchor alone.
    residual_det = float(np.linalg.det(lin_ag))
    if not 0.7 < residual_det < 1.4:
        print(
            f"[nodal_registration] WARNING: residual TPS affine det={residual_det:.3f} outside [0.7, 1.4];"
            " falling back to similarity-only warp for this subtask.",
            flush=True,
        )
        lin_ag = np.eye(3)
        trans_g = np.zeros(3)
        w_ng = np.zeros_like(w_ng)

    def apply_warp(points_np):
        aligned = points_np @ anchor_lin + anchor_trans
        return tps_eval(
            x_ma=aligned,
            lin_ag=lin_ag,
            trans_g=trans_g,
            w_ng=w_ng,
            x_na=aligned_src,
        )

    # Extract positions and rotations from source eef poses
    src_eef_pos, src_eef_rot = PoseUtils.unmake_pose(src_eef_poses)
    src_eef_pos_np = src_eef_pos.detach().cpu().numpy()
    src_eef_rot_np = src_eef_rot.detach().cpu().numpy()
    if eef_warp_offset is not None:
        # The cloth scales/deforms; the physical tool does not. Transfer the grasp
        # point rather than the wrist, including the Jacobian evaluation below.
        offset_np = np.asarray(eef_warp_offset, dtype=src_eef_pos_np.dtype)
        src_eef_pos_np = src_eef_pos_np + src_eef_rot_np @ offset_np

    # Apply the similarity-anchored TPS warp to the positions
    transformed_pos_np = apply_warp(src_eef_pos_np)
    # Handle rotation transformation based on the flag
    if use_rotation_transform:
        # The warped map is p -> tps(p @ anchor_lin + anchor_trans), so the full
        # column-convention Jacobian is J_tps @ anchor_lin.T (chain rule on the
        # row-vector affine). tps_grad returns exactly the column convention:
        # for a pure affine y = x @ A it returns A.T (verified against tps_fit2).
        grad_mga = tps_grad(
            x_ma=src_eef_pos_np @ anchor_lin + anchor_trans,
            lin_ag=lin_ag,
            _trans_g=trans_g,
            w_ng=w_ng,
            x_na=aligned_src
        )

        # Transform each rotation matrix using the local Jacobian
        transformed_rot_np = np.zeros_like(src_eef_rot_np)
        for i in range(len(src_eef_pos_np)):
            # R_transformed = (J_tps @ anchor_lin.T) @ R_original
            transformed_rot_np[i] = grad_mga[i] @ anchor_lin.T @ src_eef_rot_np[i]

            # Orthogonalize the result to ensure it's still a valid rotation matrix
            # Use SVD to get the closest rotation matrix
            U, _, Vt = np.linalg.svd(transformed_rot_np[i])
            transformed_rot_np[i] = U @ Vt
            # Ensure proper orientation (determinant = 1)
            if np.linalg.det(transformed_rot_np[i]) < 0:
                Vt[-1, :] *= -1
                transformed_rot_np[i] = U @ Vt
    else:
        # Keep original rotations without transformation
        transformed_rot_np = src_eef_rot_np

    if eef_warp_offset is not None:
        transformed_pos_np -= transformed_rot_np @ offset_np

    # Convert back to torch tensors
    transformed_pos = torch.from_numpy(transformed_pos_np).to(
        device=src_eef_poses.device,
        dtype=src_eef_poses.dtype
    )
    transformed_rot = torch.from_numpy(transformed_rot_np).to(
        device=src_eef_poses.device,
        dtype=src_eef_poses.dtype
    )

    # Reconstruct the transformed poses
    transformed_eef_poses = PoseUtils.make_pose(
        pos=transformed_pos,
        rot=transformed_rot
    )

    return transformed_eef_poses


def transform_source_data_segment_using_nodal_registration_scaled_tps_rpm_bij(
    src_eef_poses: torch.Tensor,
    src_obj_nodal_pos: torch.Tensor,
    tgt_obj_nodal_pos: torch.Tensor,
    use_rotation_transform: bool = True,
    n_iter: int = 50,
    reg_init: float = 10.0,
    reg_final: float = 0.1,
    rad_init: float = 0.1,
    rad_final: float = 0.005,
    rot_reg: float = 1e-3,
    eef_warp_offset: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """
    Transform a source data segment using TPS-RPM-BIJ registration with scaled point clouds.

    Args:
        src_eef_poses: pose sequence (shape [T, 4, 4]) for the sequence of end effector control poses
            from the source demonstration
        src_obj_nodal_pos: source object nodal positions (shape [N, 3])
        tgt_obj_nodal_pos: target object nodal positions (shape [M, 3])
        use_rotation_transform: whether to transform rotations using Jacobian (default: True)
        n_iter: number of RPM iterations (default: 50)
        reg_init: initial regularization coefficient (default: 10.0)
        reg_final: final regularization coefficient (default: 0.1)
        rad_init: initial radius for correspondence (default: 0.1)
        rad_final: final radius for correspondence (default: 0.005)
        rot_reg: rotation regularization coefficient (default: 1e-3)
        eef_warp_offset: fixed grasp-point offset in local EEF axes, in metres. Returned poses
            remain in the original control frame; None keeps registration centered at its origin.

    Returns:
        transformed_eef_poses: transformed pose sequence (shape [T, 4, 4])
    """

    # Convert tensors to numpy for Rapprentice functions
    src_nodal_np = src_obj_nodal_pos.detach().cpu().numpy()
    tgt_nodal_np = tgt_obj_nodal_pos.detach().cpu().numpy()

    # Ensure we have 3D points
    assert src_nodal_np.shape[1] == 3, "Source nodal positions must be 3D"
    assert tgt_nodal_np.shape[1] == 3, "Target nodal positions must be 3D"

    # Scale the point clouds using unit_boxify
    scaled_src_nodal, src_params = unit_boxify(src_nodal_np)
    scaled_tgt_nodal, tgt_params = unit_boxify(tgt_nodal_np)

    # Use TPS-RPM-BIJ for robust registration
    f_transform, g_transform = tps_rpm_bij(
        x_nd=scaled_src_nodal,
        y_md=scaled_tgt_nodal,
        n_iter=n_iter,
        reg_init=reg_init,
        reg_final=reg_final,
        rad_init=rad_init,
        rad_final=rad_final,
        rot_reg=rot_reg
    )

    # Unscale the transformation
    f_unscaled = unscale_tps(f_transform, src_params, tgt_params)

    # Extract positions and rotations from source eef poses
    src_eef_pos, src_eef_rot = PoseUtils.unmake_pose(src_eef_poses)
    src_eef_pos_np = src_eef_pos.detach().cpu().numpy()
    src_eef_rot_np = src_eef_rot.detach().cpu().numpy()
    if eef_warp_offset is not None:
        offset_np = np.asarray(eef_warp_offset, dtype=src_eef_pos_np.dtype)
        src_eef_pos_np = src_eef_pos_np + src_eef_rot_np @ offset_np

    # Apply the unscaled transformation to the end effector positions
    transformed_pos_np = f_unscaled.transform_points(src_eef_pos_np)

    # Handle rotation transformation based on the flag
    if use_rotation_transform:
        # Transform rotations using the transformation's Jacobian
        transformed_rot_np = np.zeros_like(src_eef_rot_np)

        for i in range(len(src_eef_pos_np)):
            # Get the Jacobian at this point
            J = f_unscaled.compute_jacobian(src_eef_pos_np[i:i+1])[0]  # Shape: (3, 3)

            # Apply the Jacobian to the rotation matrix
            transformed_rot_np[i] = J @ src_eef_rot_np[i]

            # Orthogonalize the result to ensure it's still a valid rotation matrix
            # Use SVD to get the closest rotation matrix
            U, _, Vt = np.linalg.svd(transformed_rot_np[i])
            transformed_rot_np[i] = U @ Vt

            # Ensure proper orientation (determinant = 1)
            if np.linalg.det(transformed_rot_np[i]) < 0:
                Vt[-1, :] *= -1
                transformed_rot_np[i] = U @ Vt
    else:
        # Keep original rotations without transformation
        transformed_rot_np = src_eef_rot_np

    if eef_warp_offset is not None:
        transformed_pos_np -= transformed_rot_np @ offset_np

    # Convert back to torch tensors
    transformed_pos = torch.from_numpy(transformed_pos_np).to(
        device=src_eef_poses.device,
        dtype=src_eef_poses.dtype
    )
    transformed_rot = torch.from_numpy(transformed_rot_np).to(
        device=src_eef_poses.device,
        dtype=src_eef_poses.dtype
    )

    # Reconstruct the transformed poses
    transformed_eef_poses = PoseUtils.make_pose(
        pos=transformed_pos,
        rot=transformed_rot
    )

    return transformed_eef_poses


class DataGenerator:
    """
    The main data generator class that generates new trajectories from source datasets.

    The data generator, inspired by the MimicGen, enables the generation of new datasets based on a few human
    collected source demonstrations.

    The data generator works by parsing demonstrations into object-centric subtask segments, stored in DataGenInfoPool.
    It then adapts these subtask segments to new scenes by transforming each segment according to the new scene’s context,
    stitching them into a coherent trajectory for a robotic end-effector to execute.
    """

    def __init__(
        self,
        env: ManagerBasedRLMimicEnv,
        src_demo_datagen_info_pool: DataGenInfoPool | None = None,
        dataset_path: str | None = None,
        demo_keys: list[str] | None = None,
    ):
        """
        Args:
            env: environment to use for data generation
            src_demo_datagen_info_pool: source demo datagen info pool
            dataset_path: path to hdf5 dataset to use for generation
            demo_keys: list of demonstration keys to use in file. If not provided, all demonstration keys
                will be used.
        """
        self.env = env
        self.env_cfg = env.cfg
        assert isinstance(self.env_cfg, MimicEnvCfg)
        self.dataset_path = dataset_path

        # sanity check on task spec offset ranges - final subtask should not have any offset randomization
        for subtask_configs in self.env_cfg.subtask_configs.values():
            assert subtask_configs[-1].subtask_term_offset_range[0] == 0
            assert subtask_configs[-1].subtask_term_offset_range[1] == 0

        self.demo_keys = demo_keys

        if src_demo_datagen_info_pool is not None:
            self.src_demo_datagen_info_pool = src_demo_datagen_info_pool
        elif dataset_path is not None:
            self.src_demo_datagen_info_pool = DataGenInfoPool(
                env=self.env, env_cfg=self.env_cfg, device=self.env.device
            )
            self.src_demo_datagen_info_pool.load_from_dataset_file(dataset_path, select_demo_keys=self.demo_keys)
        else:
            raise ValueError("Either src_demo_datagen_info_pool or dataset_path must be provided")

    def __repr__(self):
        """
        Pretty print this object.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tdataset_path={}\n\tdemo_keys={}\n)".format(
            self.dataset_path,
            self.demo_keys,
        )
        return msg

    def randomize_subtask_boundaries(self) -> dict[str, np.ndarray]:
        """
        Apply random offsets to sample subtask boundaries according to the task spec.
        Recall that each demonstration is segmented into a set of subtask segments, and the
        end index of each subtask can have a random offset.
        """

        randomized_subtask_boundaries = {}

        for eef_name, subtask_boundaries in self.src_demo_datagen_info_pool.subtask_boundaries.items():
            # initial subtask start and end indices - shape (N, S, 2)
            subtask_boundaries = np.array(subtask_boundaries)

            # Randomize the start of the first subtask
            first_subtask_start_offsets = np.random.randint(
                low=self.env_cfg.subtask_configs[eef_name][0].first_subtask_start_offset_range[0],
                high=self.env_cfg.subtask_configs[eef_name][0].first_subtask_start_offset_range[0] + 1,
                size=subtask_boundaries.shape[0],
            )
            subtask_boundaries[:, 0, 0] += first_subtask_start_offsets

            # for each subtask (except last one), sample all end offsets at once for each demonstration
            # add them to subtask end indices, and then set them as the start indices of next subtask too
            for i in range(subtask_boundaries.shape[1] - 1):
                end_offsets = np.random.randint(
                    low=self.env_cfg.subtask_configs[eef_name][i].subtask_term_offset_range[0],
                    high=self.env_cfg.subtask_configs[eef_name][i].subtask_term_offset_range[1] + 1,
                    size=subtask_boundaries.shape[0],
                )
                subtask_boundaries[:, i, 1] = subtask_boundaries[:, i, 1] + end_offsets
                # don't forget to set these as start indices for next subtask too
                subtask_boundaries[:, i + 1, 0] = subtask_boundaries[:, i, 1]

            # ensure non-empty subtasks
            assert np.all((subtask_boundaries[:, :, 1] - subtask_boundaries[:, :, 0]) > 0), "got empty subtasks!"

            # ensure subtask indices increase (both starts and ends)
            assert np.all(
                (subtask_boundaries[:, 1:, :] - subtask_boundaries[:, :-1, :]) > 0
            ), "subtask indices do not strictly increase"

            # ensure subtasks are in order
            subtask_inds_flat = subtask_boundaries.reshape(subtask_boundaries.shape[0], -1)
            assert np.all((subtask_inds_flat[:, 1:] - subtask_inds_flat[:, :-1]) >= 0), "subtask indices not in order"

            randomized_subtask_boundaries[eef_name] = subtask_boundaries

        return randomized_subtask_boundaries

    def select_source_demo(
        self,
        eef_name: str,
        eef_pose: np.ndarray,
        object_pose: np.ndarray,
        object_nodal_positions: np.ndarray,
        src_demo_current_subtask_boundaries: np.ndarray,
        subtask_object_name: str,
        subtask_object_soft: bool,
        selection_strategy_name: str,
        selection_strategy_kwargs: dict | None = None,
    ) -> int:
        """
        Helper method to run source subtask segment selection.

        Args:
            eef_name: name of end effector
            eef_pose: current end effector pose
            object_pose: current object pose for this subtask
            object_nodal_positions: current object nodal positions for this subtask
            src_demo_current_subtask_boundaries: start and end indices for subtask segment in source demonstrations of shape (N, 2)
            subtask_object_name: name of reference object for this subtask
            subtask_object_soft: whether the object is soft
            selection_strategy_name: name of selection strategy
            selection_strategy_kwargs: extra kwargs for running selection strategy

        Returns:
            selected_src_demo_ind: selected source demo index
        """
        if subtask_object_name is None:
            # no reference object - only random selection is supported
            assert selection_strategy_name == "random", selection_strategy_name

        # We need to collect the datagen info objects over the timesteps for the subtask segment in each source
        # demo, so that it can be used by the selection strategy.
        src_subtask_datagen_infos = []
        for i in range(len(self.src_demo_datagen_info_pool.datagen_infos)):
            # datagen info over all timesteps of the src trajectory
            src_ep_datagen_info = self.src_demo_datagen_info_pool.datagen_infos[i]

            # time indices for subtask
            subtask_start_ind = src_demo_current_subtask_boundaries[i][0]
            subtask_end_ind = src_demo_current_subtask_boundaries[i][1]

            # get subtask segment using indices
            src_subtask_datagen_infos.append(
                DatagenInfo(
                    eef_pose=src_ep_datagen_info.eef_pose[eef_name][subtask_start_ind:subtask_end_ind],
                    # only include object pose for relevant object in subtask if available
                    object_poses=(
                        {
                            subtask_object_name: src_ep_datagen_info.object_poses[subtask_object_name][
                                subtask_start_ind:subtask_end_ind
                            ]
                        }
                        if (subtask_object_name is not None and not subtask_object_soft)
                        else None
                    ),
                    # include nodal positions if available
                    object_nodal_positions=(
                        {
                            subtask_object_name: src_ep_datagen_info.object_nodal_positions[subtask_object_name][
                                subtask_start_ind:subtask_end_ind
                            ]
                        }
                        if (subtask_object_name is not None and subtask_object_soft)
                        else None
                    ),
                    # subtask termination signal is unused
                    subtask_term_signals=None,
                    target_eef_pose=src_ep_datagen_info.target_eef_pose[eef_name][subtask_start_ind:subtask_end_ind],
                    gripper_action=src_ep_datagen_info.gripper_action[eef_name][subtask_start_ind:subtask_end_ind],
                )
            )

        # make selection strategy object
        selection_strategy_obj = make_selection_strategy(selection_strategy_name)

        # run selection
        if selection_strategy_kwargs is None:
            selection_strategy_kwargs = dict()
        selected_src_demo_ind = selection_strategy_obj.select_source_demo(
            eef_pose=eef_pose,
            object_pose=object_pose,
            object_nodal_positions=object_nodal_positions,
            src_subtask_datagen_infos=src_subtask_datagen_infos,
            **selection_strategy_kwargs,
        )

        return selected_src_demo_ind

    def generate_trajectory(
        self,
        env_id: int,
        eef_name: str,
        subtask_ind: int,
        all_randomized_subtask_boundaries: dict[str, np.ndarray],
        runtime_subtask_constraints_dict: dict[tuple[str, int], dict],
        selected_src_demo_inds: dict[str, int | None],
        prev_executed_traj: dict[str, list[Waypoint] | None],
    ) -> list[Waypoint]:
        """
        Generate a trajectory for the given subtask.

        Args:
            env_id: environment index
            eef_name: name of end effector
            subtask_ind: index of subtask
            all_randomized_subtask_boundaries: randomized subtask boundaries
            runtime_subtask_constraints_dict: runtime subtask constraints
            selected_src_demo_inds: dictionary of selected source demo indices per eef, updated in place
            prev_executed_traj: dictionary of previously executed eef trajectories

        Returns:
            trajectory: generated trajectory
        """
        subtask_configs = self.env_cfg.subtask_configs[eef_name]
        # name of object for this subtask
        subtask_object_name = self.env_cfg.subtask_configs[eef_name][subtask_ind].object_ref
        subtask_object_soft = self.env_cfg.subtask_configs[eef_name][subtask_ind].object_soft
        subtask_object_pose = (
            self.env.get_object_poses(env_ids=[env_id])[subtask_object_name][0]
            if (subtask_object_name is not None and not subtask_object_soft)
            else None
        )
        subtask_nodal_positions = (
            self.env.get_object_nodal_positions(env_ids=[env_id])[subtask_object_name][0]
            if (subtask_object_name is not None and subtask_object_soft)
            else None
        )

        is_first_subtask = subtask_ind == 0

        need_source_demo_selection = is_first_subtask or self.env_cfg.datagen_config.generation_select_src_per_subtask

        if not self.env_cfg.datagen_config.generation_select_src_per_arm:
            need_source_demo_selection = need_source_demo_selection and selected_src_demo_inds[eef_name] is None

        use_delta_transform = None
        coord_transform_scheme = None
        if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
            if runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["type"] == SubTaskConstraintType.COORDINATION:
                # avoid selecting source demo if it has already been selected by the concurrent task
                concurrent_task_spec_key = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_task_spec_key"
                ]
                concurrent_subtask_ind = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_subtask_ind"
                ]
                concurrent_selected_src_ind = runtime_subtask_constraints_dict[
                    (concurrent_task_spec_key, concurrent_subtask_ind)
                ]["selected_src_demo_ind"]
                if concurrent_selected_src_ind is not None:
                    # the concurrent task has started, so we should use the same source demo
                    selected_src_demo_inds[eef_name] = concurrent_selected_src_ind
                    need_source_demo_selection = False
                    use_delta_transform = runtime_subtask_constraints_dict[
                        (concurrent_task_spec_key, concurrent_subtask_ind)
                    ]["transform"]
                else:
                    assert (
                        "transform" not in runtime_subtask_constraints_dict[(eef_name, subtask_ind)]
                    ), "transform should not be set for concurrent task"
                    # need to transform demo according to scheme
                    coord_transform_scheme = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                        "coordination_scheme"
                    ]
                    if coord_transform_scheme != SubTaskConstraintCoordinationScheme.REPLAY:
                        assert (
                            subtask_object_name is not None
                        ), f"object reference should not be None for {coord_transform_scheme} coordination scheme"

        if need_source_demo_selection:
            selected_src_demo_inds[eef_name] = self.select_source_demo(
                eef_name=eef_name,
                eef_pose=self.env.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0],
                object_pose=subtask_object_pose,
                object_nodal_positions=subtask_nodal_positions,
                src_demo_current_subtask_boundaries=all_randomized_subtask_boundaries[eef_name][:, subtask_ind],
                subtask_object_name=subtask_object_name,
                subtask_object_soft=subtask_object_soft,
                selection_strategy_name=self.env_cfg.subtask_configs[eef_name][subtask_ind].selection_strategy,
                selection_strategy_kwargs=self.env_cfg.subtask_configs[eef_name][subtask_ind].selection_strategy_kwargs,
            )

        assert selected_src_demo_inds[eef_name] is not None
        selected_src_demo_ind = selected_src_demo_inds[eef_name]

        if not self.env_cfg.datagen_config.generation_select_src_per_arm and need_source_demo_selection:
            for itrated_eef_name in self.env_cfg.subtask_configs.keys():
                selected_src_demo_inds[itrated_eef_name] = selected_src_demo_ind

        # selected subtask segment time indices
        selected_src_subtask_boundary = all_randomized_subtask_boundaries[eef_name][selected_src_demo_ind, subtask_ind]

        if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
            if runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["type"] == SubTaskConstraintType.COORDINATION:
                # store selected source demo ind for concurrent task
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "selected_src_demo_ind"
                ] = selected_src_demo_ind
                concurrent_task_spec_key = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_task_spec_key"
                ]
                concurrent_subtask_ind = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_subtask_ind"
                ]
                concurrent_src_subtask_inds = all_randomized_subtask_boundaries[concurrent_task_spec_key][
                    selected_src_demo_ind, concurrent_subtask_ind
                ]
                subtask_len = selected_src_subtask_boundary[1] - selected_src_subtask_boundary[0]
                concurrent_subtask_len = concurrent_src_subtask_inds[1] - concurrent_src_subtask_inds[0]
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["synchronous_steps"] = min(
                    subtask_len, concurrent_subtask_len
                )

        # TODO allow for different anchor selection strategies for each subtask

        # get subtask segment, consisting of the sequence of robot eef poses, target poses, gripper actions
        src_ep_datagen_info = self.src_demo_datagen_info_pool.datagen_infos[selected_src_demo_ind]
        src_subtask_eef_poses = src_ep_datagen_info.eef_pose[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]
        src_subtask_target_poses = src_ep_datagen_info.target_eef_pose[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]
        src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]

        # get reference object pose from source demo
        src_subtask_object_pose = (
            src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_boundary[0]]
            if (subtask_object_name is not None and not subtask_object_soft)
            else None
        )

        src_subtask_nodal_positions = (
            src_ep_datagen_info.object_nodal_positions[subtask_object_name][selected_src_subtask_boundary[0]]
            if (subtask_object_name is not None and subtask_object_soft)
            else None
        )

        if is_first_subtask or self.env_cfg.datagen_config.generation_transform_first_robot_pose:
            # Source segment consists of first robot eef pose and the target poses. This ensures that
            # we will interpolate to the first robot eef pose in this source segment, instead of the
            # first robot target pose.
            src_eef_poses = torch.cat([src_subtask_eef_poses[0:1], src_subtask_target_poses], dim=0)
            # account for extra timestep added to @src_eef_poses
            src_subtask_gripper_actions = torch.cat(
                [src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], dim=0
            )
        else:
            # Source segment consists of just the target poses.
            src_eef_poses = src_subtask_target_poses.clone()
            src_subtask_gripper_actions = src_subtask_gripper_actions.clone()

        if not subtask_object_soft:
            # Transform source demonstration segment using relevant object pose.
            if use_delta_transform is not None:
                # use delta transform from concurrent task
                transformed_eef_poses = transform_source_data_segment_using_delta_object_pose(
                    src_eef_poses, use_delta_transform
                )

                # TODO: Uncomment below to support case of temporal concurrent but NOT does not require coordination. Need to decide if this case is necessary
                # if subtask_object_name is not None:
                #     transformed_eef_poses = PoseUtils.transform_source_data_segment_using_object_pose(
                #         cur_object_poses[task_spec_idx],
                #         src_eef_poses,
                #         src_subtask_object_pose,
                #     )

            else:
                if coord_transform_scheme is not None:
                    delta_obj_pose = get_delta_pose_with_scheme(
                        src_subtask_object_pose,
                        subtask_object_pose,
                        runtime_subtask_constraints_dict[(eef_name, subtask_ind)],
                    )
                    transformed_eef_poses = transform_source_data_segment_using_delta_object_pose(
                        src_eef_poses, delta_obj_pose
                    )
                    runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["transform"] = delta_obj_pose
                else:
                    if subtask_object_name is not None:
                        transformed_eef_poses = transform_source_data_segment_using_object_pose(
                            subtask_object_pose,
                            src_eef_poses,
                            src_subtask_object_pose,
                        )
                    else:
                        print(f"skipping transformation for {subtask_object_name}")
                        # skip transformation if no reference object is provided
                        transformed_eef_poses = src_eef_poses

        if subtask_object_soft:
            if subtask_object_name is not None:
                eef_warp_offset = getattr(self.env_cfg, "eef_warp_offsets", {}).get(eef_name)
                # Choose between different nodal registration methods
                # Option 1: Use scaled TPS-RPM-BIJ
                # Option 2 - default: Use standard TPS method (requires same point counts)
                use_scaled_tps_rpm_bij = False

                if use_scaled_tps_rpm_bij:
                    transformed_eef_poses = transform_source_data_segment_using_nodal_registration_scaled_tps_rpm_bij(
                        tgt_obj_nodal_pos=subtask_nodal_positions,
                        src_eef_poses=src_eef_poses,
                        src_obj_nodal_pos=src_subtask_nodal_positions,
                        use_rotation_transform=True,
                        eef_warp_offset=eef_warp_offset,
                        n_iter=50,
                        reg_init=10.0,
                        reg_final=0.1,
                        rad_init=0.01,
                        rad_final=0.001,
                        rot_reg=1e-3
                    )
                else:
                    # Use standard TPS method (requires same number of points)
                    transformed_eef_poses = transform_source_data_segment_using_nodal_registration(
                        tgt_obj_nodal_pos=subtask_nodal_positions,
                        src_eef_poses=src_eef_poses,
                        src_obj_nodal_pos=src_subtask_nodal_positions,
                        use_rotation_transform=True,
                        eef_warp_offset=eef_warp_offset,
                        bend_coef=0.1,
                        rot_coef=1e-3
                    )
            else:
                print(f"skipping transformation for {subtask_object_name}")
                # skip transformation if no reference object is provided
                transformed_eef_poses = src_eef_poses
                
        # We will construct a WaypointTrajectory instance to keep track of robot control targets
        # that will be executed and then execute it.
        traj_to_execute = WaypointTrajectory()

        if self.env_cfg.datagen_config.generation_interpolate_from_last_target_pose and (not is_first_subtask):
            # Interpolation segment will start from last target pose (which may not have been achieved).
            assert prev_executed_traj[eef_name] is not None
            last_waypoint = prev_executed_traj[eef_name][-1]
            init_sequence = WaypointSequence(sequence=[last_waypoint])
        else:
            # Interpolation segment will start from current robot eef pose.
            init_sequence = WaypointSequence.from_poses(
                poses=self.env.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0].unsqueeze(0),
                gripper_actions=src_subtask_gripper_actions[0].unsqueeze(0),
                action_noise=subtask_configs[subtask_ind].action_noise,
            )
        traj_to_execute.add_waypoint_sequence(init_sequence)

        # Construct trajectory for the transformed segment.
        transformed_seq = WaypointSequence.from_poses(
            poses=transformed_eef_poses,
            gripper_actions=src_subtask_gripper_actions,
            action_noise=subtask_configs[subtask_ind].action_noise,
        )
        transformed_traj = WaypointTrajectory()
        transformed_traj.add_waypoint_sequence(transformed_seq)

        # Merge this trajectory into our trajectory using linear interpolation.
        # Interpolation will happen from the initial pose (@init_sequence) to the first element of @transformed_seq.
        traj_to_execute.merge(
            transformed_traj,
            num_steps_interp=self.env_cfg.subtask_configs[eef_name][subtask_ind].num_interpolation_steps,
            num_steps_fixed=self.env_cfg.subtask_configs[eef_name][subtask_ind].num_fixed_steps,
            action_noise=(
                float(self.env_cfg.subtask_configs[eef_name][subtask_ind].apply_noise_during_interpolation)
                * self.env_cfg.subtask_configs[eef_name][subtask_ind].action_noise
            ),
        )

        # We initialized @traj_to_execute with a pose to allow @merge to handle linear interpolation
        # for us. However, we can safely discard that first waypoint now, and just start by executing
        # the rest of the trajectory (interpolation segment and transformed subtask segment).
        traj_to_execute.pop_first()

        # Return the generated trajectory
        return traj_to_execute.get_full_sequence().sequence

    async def generate(
        self,
        env_id: int,
        success_term: TerminationTermCfg,
        env_reset_queue: asyncio.Queue | None = None,
        env_action_queue: asyncio.Queue | None = None,
        pause_subtask: bool = False,
        export_demo: bool = True,
    ) -> dict:
        """
        Attempt to generate a new demonstration.

        Args:
            env_id: environment index
            success_term: success function to check if the task is successful
            env_reset_queue: queue to store environment IDs for reset
            env_action_queue: queue to store actions for each environment
            pause_subtask: if True, pause after every subtask during generation, for debugging
            export_demo: if True, export the generated demonstration

        Returns:
            results: dictionary with the following items:
                initial_state (dict): initial simulator state for the executed trajectory
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (np.array): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
                src_demo_inds (list): list of selected source demonstration indices for each subtask
                src_demo_labels (np.array): same as @src_demo_inds, but repeated to have a label for each timestep of the trajectory
        """

        # reset the env to create a new task demo instance
        env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=self.env.device)
        self.env.recorder_manager.reset(env_ids=env_id_tensor)
        await env_reset_queue.put(env_id)
        await env_reset_queue.join()
        new_initial_state = self.env.scene.get_state(is_relative=True)

        # create runtime subtask constraint rules from subtask constraint configs
        runtime_subtask_constraints_dict = {}
        for subtask_constraint in self.env_cfg.task_constraint_configs:
            runtime_subtask_constraints_dict.update(subtask_constraint.generate_runtime_subtask_constraints())

        # save generated data in these variables
        generated_states = []
        generated_obs = []
        generated_actions = []
        generated_success = False

        # some eef-specific state variables used during generation
        current_eef_selected_src_demo_indices = {}
        current_eef_subtask_trajectories = {}
        current_eef_subtask_indices = {}
        current_eef_subtask_step_indices = {}
        eef_subtasks_done = {}
        for eef_name in self.env_cfg.subtask_configs.keys():
            current_eef_selected_src_demo_indices[eef_name] = None
            # prev_eef_executed_traj[eef_name] = None  # type of list of Waypoint
            current_eef_subtask_trajectories[eef_name] = None  # type of list of Waypoint
            current_eef_subtask_indices[eef_name] = 0
            current_eef_subtask_step_indices[eef_name] = None
            eef_subtasks_done[eef_name] = False

        prev_src_demo_datagen_info_pool_size = 0
        # While loop that runs per time step
        while True:
            async with self.src_demo_datagen_info_pool.asyncio_lock:
                if len(self.src_demo_datagen_info_pool.datagen_infos) > prev_src_demo_datagen_info_pool_size:
                    # src_demo_datagen_info_pool at this point may be updated with new demos,
                    # so we need to updaet subtask boundaries again
                    randomized_subtask_boundaries = (
                        self.randomize_subtask_boundaries()
                    )  # shape [N, S, 2], last dim is start and end action lengths
                    prev_src_demo_datagen_info_pool_size = len(self.src_demo_datagen_info_pool.datagen_infos)

                # generate trajectory for a subtask for the eef that is currently at the beginning of a subtask
                for eef_name, eef_subtask_step_index in current_eef_subtask_step_indices.items():
                    if eef_subtask_step_index is None:
                        # current_eef_selected_src_demo_indices will be updated in generate_trajectory
                        subtask_trajectory = self.generate_trajectory(
                            env_id,
                            eef_name,
                            current_eef_subtask_indices[eef_name],
                            randomized_subtask_boundaries,
                            runtime_subtask_constraints_dict,
                            current_eef_selected_src_demo_indices,
                            current_eef_subtask_trajectories,
                        )
                        current_eef_subtask_trajectories[eef_name] = subtask_trajectory
                        current_eef_subtask_step_indices[eef_name] = 0
                        # current_eef_selected_src_demo_indices[eef_name] = selected_src_demo_inds
                        # two_arm_trajectories[task_spec_idx] = subtask_trajectory
                        # prev_executed_traj[task_spec_idx] = subtask_trajectory

            # determine the next waypoint for each eef based on the current subtask constraints
            eef_waypoint_dict = {}
            for eef_name in sorted(self.env_cfg.subtask_configs.keys()):
                # handle constraints
                step_ind = current_eef_subtask_step_indices[eef_name]
                subtask_ind = current_eef_subtask_indices[eef_name]
                if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
                    task_constraint = runtime_subtask_constraints_dict[(eef_name, subtask_ind)]
                    if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_LATTER:
                        min_time_diff = task_constraint["min_time_diff"]
                        if not task_constraint["fulfilled"]:
                            if (
                                min_time_diff == -1
                                or step_ind >= len(current_eef_subtask_trajectories[eef_name]) - min_time_diff
                            ):
                                if step_ind > 0:
                                    # wait at the same step
                                    step_ind -= 1
                                    current_eef_subtask_step_indices[eef_name] = step_ind

                    elif task_constraint["type"] == SubTaskConstraintType.COORDINATION:
                        synchronous_steps = task_constraint["synchronous_steps"]
                        concurrent_task_spec_key = task_constraint["concurrent_task_spec_key"]
                        concurrent_subtask_ind = task_constraint["concurrent_subtask_ind"]
                        concurrent_task_fulfilled = runtime_subtask_constraints_dict[
                            (concurrent_task_spec_key, concurrent_subtask_ind)
                        ]["fulfilled"]

                        if (
                            task_constraint["coordination_synchronize_start"]
                            and current_eef_subtask_indices[concurrent_task_spec_key] < concurrent_subtask_ind
                        ):
                            # the concurrent eef is not yet at the concurrent subtask, so wait at the first action
                            # this also makes sure that the concurrent task starts at the same time as this task
                            step_ind = 0
                            current_eef_subtask_step_indices[eef_name] = 0
                        else:
                            if (
                                not concurrent_task_fulfilled
                                and step_ind >= len(current_eef_subtask_trajectories[eef_name]) - synchronous_steps
                            ):
                                # trigger concurrent task
                                runtime_subtask_constraints_dict[(concurrent_task_spec_key, concurrent_subtask_ind)][
                                    "fulfilled"
                                ] = True

                            if not task_constraint["fulfilled"]:
                                if step_ind >= len(current_eef_subtask_trajectories[eef_name]) - synchronous_steps:
                                    if step_ind > 0:
                                        step_ind -= 1
                                        current_eef_subtask_step_indices[eef_name] = step_ind  # wait here

                waypoint = current_eef_subtask_trajectories[eef_name][step_ind]
                eef_waypoint_dict[eef_name] = waypoint
            multi_waypoint = MultiWaypoint(eef_waypoint_dict)

            # execute the next waypoints for all eefs
            exec_results = await multi_waypoint.execute(
                env=self.env,
                success_term=success_term,
                env_id=env_id,
                env_action_queue=env_action_queue,
            )

            # update execution state buffers
            if len(exec_results["states"]) > 0:
                generated_states.extend(exec_results["states"])
                generated_obs.extend(exec_results["observations"])
                generated_actions.extend(exec_results["actions"])
                generated_success = generated_success or exec_results["success"]

            for eef_name in self.env_cfg.subtask_configs.keys():
                current_eef_subtask_step_indices[eef_name] += 1
                subtask_ind = current_eef_subtask_indices[eef_name]
                if current_eef_subtask_step_indices[eef_name] == len(
                    current_eef_subtask_trajectories[eef_name]
                ):  # subtask done
                    if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
                        task_constraint = runtime_subtask_constraints_dict[(eef_name, subtask_ind)]
                        if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_FORMER:
                            constrained_task_spec_key = task_constraint["constrained_task_spec_key"]
                            constrained_subtask_ind = task_constraint["constrained_subtask_ind"]
                            runtime_subtask_constraints_dict[(constrained_task_spec_key, constrained_subtask_ind)][
                                "fulfilled"
                            ] = True
                        elif task_constraint["type"] == SubTaskConstraintType.COORDINATION:
                            concurrent_task_spec_key = task_constraint["concurrent_task_spec_key"]
                            concurrent_subtask_ind = task_constraint["concurrent_subtask_ind"]
                            # concurrent_task_spec_idx = task_spec_keys.index(concurrent_task_spec_key)
                            task_constraint["finished"] = True
                            # check if concurrent task has been finished
                            assert (
                                runtime_subtask_constraints_dict[(concurrent_task_spec_key, concurrent_subtask_ind)][
                                    "finished"
                                ]
                                or current_eef_subtask_step_indices[concurrent_task_spec_key]
                                >= len(current_eef_subtask_trajectories[concurrent_task_spec_key]) - 1
                            )

                    if pause_subtask:
                        input(
                            f"Pausing after subtask {current_eef_subtask_indices[eef_name]} of {eef_name} execution."
                            " Press any key to continue..."
                        )
                    # This is a check to see if this arm has completed all the subtasks
                    if current_eef_subtask_indices[eef_name] == len(self.env_cfg.subtask_configs[eef_name]) - 1:
                        eef_subtasks_done[eef_name] = True
                        # If all subtasks done for this arm, repeat last waypoint to make sure this arm does not move
                        current_eef_subtask_trajectories[eef_name].append(
                            current_eef_subtask_trajectories[eef_name][-1]
                        )
                    else:
                        current_eef_subtask_step_indices[eef_name] = None
                        current_eef_subtask_indices[eef_name] += 1
            # Check if all eef_subtasks_done values are True
            if all(eef_subtasks_done.values()):
                break

        # merge numpy arrays
        if len(generated_actions) > 0:
            generated_actions = torch.cat(generated_actions, dim=0)

        # set success to the recorded episode data and export to file
        self.env.recorder_manager.set_success_to_episodes(
            env_id_tensor, torch.tensor([[generated_success]], dtype=torch.bool, device=self.env.device)
        )
        if export_demo:
            self.env.recorder_manager.export_episodes(env_id_tensor)

        results = dict(
            initial_state=new_initial_state,
            states=generated_states,
            observations=generated_obs,
            actions=generated_actions,
            success=generated_success,
        )
        return results
