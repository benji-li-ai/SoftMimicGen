# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from softmimicgen.envs.yam_towel_joint_mimic_env_cfg import YamTowelMimicEnvCfg

from softmimicgen_tasks.yam_cell_towel.config.joint_pos_env_cfg import YamCellTowelEnvCfg


@configclass
class YamCellTowelMimicEnvCfg(YamCellTowelEnvCfg, YamTowelMimicEnvCfg):
    """Native YAM Mimic subtasks and device sampling on the package-backed cell."""

    eef_warp_offsets: dict[str, tuple[float, float, float]] = {
        "robot0": (0.0, 0.0, 0.1347),
        "robot1": (0.0, 0.0, 0.1347),
    }
    """Local link_6-to-grasp offsets from the cell USD's sites/grasp_site/grasp_site.

    Nodal registration transfers the grasp point; recordings and IK remain in link_6.
    The asset's tcp_site is at the wrist origin and must not be used for this offset.
    """

    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "yam_cell_towel_task"
        if self.success_fold_only:
            # Mirror the collection policy's cloth-fold-only success: the recorded
            # teleop endings do not satisfy the native open-gripper clause.
            from softmimicgen_tasks.yam_cell_towel.mdp.terminations import object_folded
            self.terminations.success.func = object_folded
            self.terminations.success.params["gripper_open_required"] = False
