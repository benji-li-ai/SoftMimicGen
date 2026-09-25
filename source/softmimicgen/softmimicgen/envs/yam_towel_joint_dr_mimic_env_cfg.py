# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic data-generation config for the YAM towel task with visual domain randomization.

Data generation is identical to :class:`YamTowelMimicEnvCfg` -- same subtasks, same
registration-cost source selection -- but the recorded images carry per-frame camera pose jitter
and per-episode lighting variation. This is the environment the sim-to-real training corpus should
be generated from; generate the held-out evaluation set from the clean
``Isaac-Fold-Towel-Yam-Joint-Mimic-v0`` instead.

Camera pose and dome light intensity do not affect physics, so the generated trajectories are
statistically identical to the clean environment's -- only the pixels differ.
"""

from isaaclab.utils import configclass

from softmimicgen.envs.yam_towel_joint_mimic_env_cfg import YamTowelMimicEnvCfg
from softmimicgen_tasks.yam_towel.config.dr_env_cfg import YamTowelDREnvCfg


@configclass
class YamTowelDRMimicEnvCfg(YamTowelDREnvCfg, YamTowelMimicEnvCfg):
    """YAM towel Mimic environment with camera pose and lighting randomization."""

    pass
