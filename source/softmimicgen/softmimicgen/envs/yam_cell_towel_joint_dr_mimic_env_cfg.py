# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic data-generation configuration for YAM red cloth with visual randomization."""

from isaaclab.utils import configclass

from softmimicgen.envs.yam_cell_towel_joint_mimic_env_cfg import YamCellTowelMimicEnvCfg
from softmimicgen_tasks.yam_cell_towel.config.dr_env_cfg import YamCellTowelDREnvCfg


@configclass
class YamCellTowelDRMimicEnvCfg(YamCellTowelDREnvCfg, YamCellTowelMimicEnvCfg):
    """YAM red-cloth Mimic environment with visual domain randomization."""

    pass
