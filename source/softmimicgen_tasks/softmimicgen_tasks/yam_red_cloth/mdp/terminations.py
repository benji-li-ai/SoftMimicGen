# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, Deformable
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_folded(
    env: ManagerBasedRLEnv,
    robot_1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot_2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    fold_threshold: float = 0.5,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    atol=0.01,
    rtol=0.01,
) -> torch.Tensor:
    """Fold detection using PCA-based projection area reduction.

    Computes the XY covariance of the object's nodal positions and uses the
    square root of the determinant (product of eigenvalues) as a rotation-invariant
    area measure. This works for both square and rectangular towels, since folding
    along any axis reduces the projection area. Success requires the area ratio to
    be below threshold and both grippers to be open.

    Args:
        env: The environment.
        robot_1_cfg: First robot configuration.
        robot_2_cfg: Second robot configuration.
        object_cfg: Deformable object configuration.
        fold_threshold: Ratio of current/initial area extent for success.
        gripper_open_val: Joint position when gripper is fully open.
        atol: Absolute tolerance for gripper position check.
        rtol: Relative tolerance for gripper position check.
    """
    robot_1: Articulation = env.scene[robot_1_cfg.name]
    robot_2: Articulation = env.scene[robot_2_cfg.name]
    object: Deformable = env.scene[object_cfg.name]

    nodal_pos_w = object.data.nodal_pos_w

    xy_pos = nodal_pos_w[..., :2]
    centered = xy_pos - xy_pos.mean(dim=1, keepdim=True)
    cov = torch.bmm(centered.transpose(1, 2), centered) / centered.shape[1]
    eigvals = torch.linalg.eigvalsh(cov)
    current_extent = torch.sqrt(eigvals.prod(dim=-1).clamp(min=1e-12))

    default_xy = object.data.default_nodal_state_w[..., :2]
    default_centered = default_xy - default_xy.mean(dim=1, keepdim=True)
    default_cov = torch.bmm(default_centered.transpose(1, 2), default_centered) / default_centered.shape[1]
    default_eigvals = torch.linalg.eigvalsh(default_cov)
    initial_extent = torch.sqrt(default_eigvals.prod(dim=-1).clamp(min=1e-12))

    fold_ratio = current_extent / initial_extent.clamp(min=1e-6)
    is_folded = fold_ratio < fold_threshold

    gripper_1_open = torch.isclose(
        robot_1.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=atol, rtol=rtol
    )

    gripper_2_open = torch.isclose(
        robot_2.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=atol, rtol=rtol
    )

    return is_folded & gripper_1_open & gripper_2_open
