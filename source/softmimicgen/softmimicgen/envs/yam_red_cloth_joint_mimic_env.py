# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic runtime wrapper for the YAM red-cloth task."""

from softmimicgen.envs.yam_towel_joint_mimic_env import YamTowelMimicEnv


class YamRedClothMimicEnv(YamTowelMimicEnv):
    """YAM red-cloth Mimic environment with joint-position control."""

    pass
