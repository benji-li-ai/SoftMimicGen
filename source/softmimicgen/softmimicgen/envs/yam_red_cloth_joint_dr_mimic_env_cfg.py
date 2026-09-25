# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic data-generation configuration for YAM red cloth with visual randomization."""

from isaaclab.utils import configclass

from softmimicgen.envs.yam_red_cloth_joint_mimic_env_cfg import YamRedClothMimicEnvCfg
from softmimicgen_tasks.yam_red_cloth.config.dr_env_cfg import YamRedClothDREnvCfg


@configclass
class YamRedClothDRMimicEnvCfg(YamRedClothDREnvCfg, YamRedClothMimicEnvCfg):
    """YAM red-cloth Mimic environment with visual domain randomization."""

    pass
