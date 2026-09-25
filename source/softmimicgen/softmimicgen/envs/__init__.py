# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .franka_cube_ik_rel_mimic_env import FrankaCubeMimicEnv
from .franka_cube_ik_rel_mimic_env_cfg import FrankaCubeMimicEnvCfg
from .franka_jenga_ik_rel_mimic_env import FrankaJengaMimicEnv
from .franka_jenga_ik_rel_mimic_env_cfg import FrankaJengaMimicEnvCfg
from .franka_rope_ik_rel_mimic_env import FrankaRopeMimicEnv
from .franka_rope_ik_rel_mimic_env_cfg import FrankaRopeMimicEnvCfg
from .franka_towel_ik_rel_mimic_env import FrankaTowelMimicEnv
from .franka_towel_ik_rel_mimic_env_cfg import FrankaTowelMimicEnvCfg
from .surgical_threading_ik_rel_mimic_env import SurgicalThreadingMimicEnv
from .surgical_threading_ik_rel_mimic_env_cfg import SurgicalThreadingMimicEnvCfg
from .surgical_tissue_ik_rel_mimic_env import SurgicalTissueMimicEnv
from .surgical_tissue_ik_rel_mimic_env_cfg import SurgicalTissueMimicEnvCfg
from .yam_bag_joint_mimic_env import YamBagMimicEnv
from .yam_bag_joint_mimic_env_cfg import YamBagMimicEnvCfg
from .yam_cell_towel_joint_dr_mimic_env_cfg import YamCellTowelDRMimicEnvCfg
from .yam_cell_towel_joint_mimic_env_cfg import YamCellTowelMimicEnvCfg
from .yam_cell_towel_joint_perception_mimic_env_cfg import YamCellTowelPerceptionMimicEnvCfg
from .yam_red_cloth_joint_dr_mimic_env_cfg import YamRedClothDRMimicEnvCfg
from .yam_red_cloth_joint_mimic_env import YamRedClothMimicEnv
from .yam_red_cloth_joint_mimic_env_cfg import YamRedClothMimicEnvCfg
from .yam_red_cloth_joint_perception_mimic_env_cfg import YamRedClothPerceptionMimicEnvCfg
from .yam_towel_joint_dr_mimic_env_cfg import YamTowelDRMimicEnvCfg
from .yam_towel_joint_mimic_env import YamTowelMimicEnv
from .yam_towel_joint_mimic_env_cfg import YamTowelMimicEnvCfg
from .yam_towel_joint_perception_mimic_env_cfg import YamTowelPerceptionMimicEnvCfg

gym.register(
    id="Isaac-Stack-Soft-Cube-Franka-IK-Rel-Mimic-v0",
    entry_point="softmimicgen.envs:FrankaCubeMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_cube_ik_rel_mimic_env_cfg.FrankaCubeMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Jenga-Franka-IK-Rel-Mimic-v0",
    entry_point="softmimicgen.envs:FrankaJengaMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_jenga_ik_rel_mimic_env_cfg.FrankaJengaMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Rope-Franka-IK-Rel-Mimic-v0",
    entry_point="softmimicgen.envs:FrankaRopeMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_rope_ik_rel_mimic_env_cfg.FrankaRopeMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Towel-Franka-IK-Rel-Mimic-v0",
    entry_point="softmimicgen.envs:FrankaTowelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_towel_ik_rel_mimic_env_cfg.FrankaTowelMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Thread-PSM-IK-Rel-Mimic-v0",
    entry_point="softmimicgen.envs:SurgicalThreadingMimicEnv",
    kwargs={
        "env_cfg_entry_point": surgical_threading_ik_rel_mimic_env_cfg.SurgicalThreadingMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Tissue-PSM-IK-Rel-Mimic-v0",
    entry_point="softmimicgen.envs:SurgicalTissueMimicEnv",
    kwargs={
        "env_cfg_entry_point": surgical_tissue_ik_rel_mimic_env_cfg.SurgicalTissueMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Bag-Yam-Joint-Mimic-v0",
    entry_point="softmimicgen.envs:YamBagMimicEnv",
    kwargs={
        "env_cfg_entry_point": yam_bag_joint_mimic_env_cfg.YamBagMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Towel-Yam-Joint-Mimic-v0",
    entry_point="softmimicgen.envs:YamTowelMimicEnv",
    kwargs={
        "env_cfg_entry_point": yam_towel_joint_mimic_env_cfg.YamTowelMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Towel-Yam-Joint-DR-Mimic-v0",
    entry_point="softmimicgen.envs:YamTowelMimicEnv",
    kwargs={
        "env_cfg_entry_point": yam_towel_joint_dr_mimic_env_cfg.YamTowelDRMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Towel-Yam-Joint-Perception-Mimic-v0",
    entry_point="softmimicgen.envs:YamTowelMimicEnv",
    kwargs={
        "env_cfg_entry_point": yam_towel_joint_perception_mimic_env_cfg.YamTowelPerceptionMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Red-Cloth-Yam-Joint-Mimic-v0",
    entry_point="softmimicgen.envs:YamRedClothMimicEnv",
    kwargs={
        "env_cfg_entry_point": yam_red_cloth_joint_mimic_env_cfg.YamRedClothMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Red-Cloth-Yam-Joint-DR-Mimic-v0",
    entry_point="softmimicgen.envs:YamRedClothMimicEnv",
    kwargs={
        "env_cfg_entry_point": yam_red_cloth_joint_dr_mimic_env_cfg.YamRedClothDRMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Red-Cloth-Yam-Joint-Perception-Mimic-v0",
    entry_point="softmimicgen.envs:YamRedClothMimicEnv",
    kwargs={
        "env_cfg_entry_point": yam_red_cloth_joint_perception_mimic_env_cfg.YamRedClothPerceptionMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Cell-Towel-Yam-Joint-Mimic-v0",
    entry_point="softmimicgen.envs:YamTowelMimicEnv",
    kwargs={
        "env_cfg_entry_point": yam_cell_towel_joint_mimic_env_cfg.YamCellTowelMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Cell-Towel-Yam-Joint-DR-Mimic-v0",
    entry_point="softmimicgen.envs:YamTowelMimicEnv",
    kwargs={
        "env_cfg_entry_point": yam_cell_towel_joint_dr_mimic_env_cfg.YamCellTowelDRMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Fold-Cell-Towel-Yam-Joint-Perception-Mimic-v0",
    entry_point="softmimicgen.envs:YamTowelMimicEnv",
    kwargs={
        "env_cfg_entry_point": yam_cell_towel_joint_perception_mimic_env_cfg.YamCellTowelPerceptionMimicEnvCfg,
    },
    disable_env_checker=True,
)
