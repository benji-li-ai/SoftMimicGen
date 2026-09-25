# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic data-generation configuration for YAM red cloth with depth and segmentation."""

from isaaclab.utils import configclass

from softmimicgen.envs.yam_red_cloth_joint_mimic_env_cfg import YamRedClothMimicEnvCfg
from softmimicgen_tasks.yam_red_cloth.config.perception_env_cfg import YamRedClothPerceptionEnvCfg


@configclass
class YamRedClothPerceptionMimicEnvCfg(YamRedClothPerceptionEnvCfg, YamRedClothMimicEnvCfg):
    """YAM red-cloth Mimic environment with top-camera depth and segmentation."""

    pass
