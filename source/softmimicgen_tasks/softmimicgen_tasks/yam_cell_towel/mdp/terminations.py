# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, Deformable
from isaaclab.managers import SceneEntityCfg

from softmimicgen_tasks.yam_towel.mdp.terminations import object_folded as native_object_folded

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_folded(
    env: "ManagerBasedRLEnv",
    robot_1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot_2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    fold_threshold: float = 0.5,
    gripper_open_val: torch.Tensor = torch.tensor([0.04]),
    gripper_open_required: bool = True,
    atol: float = 0.01,
    rtol: float = 0.01,
) -> torch.Tensor:
    """Native fold gate with an optional open-gripper requirement.

    The recorded teleop demonstrations intentionally end without both grippers
    fully open, so collection uses a cloth-fold-only policy. When
    ``gripper_open_required`` is False, only the PCA XY area ratio must pass;
    otherwise this is exactly the native predicate.
    """
    folded = native_object_folded(
        env,
        robot_1_cfg=robot_1_cfg,
        robot_2_cfg=robot_2_cfg,
        object_cfg=object_cfg,
        fold_threshold=fold_threshold,
        gripper_open_val=gripper_open_val,
        atol=atol,
        rtol=rtol,
    )
    if gripper_open_required:
        return folded
    # Recompute the area clause alone with the same math as the native gate.
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
    return (current_extent / initial_extent.clamp(min=1e-6)) < fold_threshold
