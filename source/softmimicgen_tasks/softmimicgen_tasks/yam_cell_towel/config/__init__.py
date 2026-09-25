# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Fold-Cell-Towel-Yam-Joint-v0",
    entry_point="softmimicgen.envs.soft_mimic_env:SoftManagerBasedRLMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:YamCellTowelEnvCfg",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image.json"),
        "robomimic_bc_crop224_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_image_crop224.json"
        ),
        "robomimic_diffusion_policy_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/diffusion_policy.json"
        ),
        "robomimic_diffusion_policy_smoke_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/diffusion_policy_smoke.json"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Cell-Towel-Yam-Joint-DR-v0",
    entry_point="softmimicgen.envs.soft_mimic_env:SoftManagerBasedRLMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dr_env_cfg:YamCellTowelDREnvCfg",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image.json"),
        "robomimic_bc_crop224_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_image_crop224.json"
        ),
        "robomimic_diffusion_policy_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/diffusion_policy.json"
        ),
        "robomimic_diffusion_policy_smoke_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/diffusion_policy_smoke.json"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Cell-Towel-Yam-Joint-Perception-v0",
    entry_point="softmimicgen.envs.soft_mimic_env:SoftManagerBasedRLMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perception_env_cfg:YamCellTowelPerceptionEnvCfg",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image.json"),
        "robomimic_bc_crop224_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_image_crop224.json"
        ),
        "robomimic_diffusion_policy_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/diffusion_policy.json"
        ),
        "robomimic_diffusion_policy_smoke_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/diffusion_policy_smoke.json"
        ),
    },
    disable_env_checker=True,
)
