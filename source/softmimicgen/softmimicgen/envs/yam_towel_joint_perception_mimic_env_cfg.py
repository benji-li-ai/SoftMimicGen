# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic data-generation config for the YAM towel task with depth and segmentation exported.

Identical data generation to :class:`YamTowelMimicEnvCfg` -- same subtask structure, same
registration-cost source selection -- but the recorded observations additionally carry the top
camera's depth and segmentation. Use it for the smaller diagnostic batches that need masks; the
extra modalities roughly triple dataset size, so the main training corpus should come from
``Isaac-Fold-Towel-Yam-Joint-Mimic-v0``.
"""

from isaaclab.utils import configclass

from softmimicgen.envs.yam_towel_joint_mimic_env_cfg import YamTowelMimicEnvCfg
from softmimicgen_tasks.yam_towel.config.perception_env_cfg import YamTowelPerceptionEnvCfg


@configclass
class YamTowelPerceptionMimicEnvCfg(YamTowelPerceptionEnvCfg, YamTowelMimicEnvCfg):
    """YAM towel Mimic environment that also records top-camera depth and segmentation."""

    pass
