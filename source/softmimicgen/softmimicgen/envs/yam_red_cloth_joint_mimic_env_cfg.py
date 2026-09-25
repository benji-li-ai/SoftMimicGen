# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.utils import configclass

from softmimicgen.envs.soft_mimic_env_cfg import SoftSubTaskConfig as SubTaskConfig
from softmimicgen_tasks.yam_red_cloth.config.joint_pos_env_cfg import YamRedClothEnvCfg


@configclass
class YamRedClothMimicEnvCfg(YamRedClothEnvCfg, MimicEnvCfg):
    """Isaac Lab Mimic configuration for YAM red-cloth joint-position control."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "yam_red_cloth_task"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        self.subtask_configs["robot0"] = [
            SubTaskConfig(
                object_ref=None,
                object_soft=True,
                subtask_term_signal="robot0_base_pose",
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="object",
                object_soft=True,
                subtask_term_signal="grasp",
                subtask_term_offset_range=(5, 10),
                selection_strategy="registration_cost",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=20,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="object",
                object_soft=True,
                subtask_term_signal="release_above",
                subtask_term_offset_range=(5, 10),
                selection_strategy="registration_cost",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref=None,
                object_soft=True,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
        self.subtask_configs["robot1"] = [
            SubTaskConfig(
                object_ref=None,
                object_soft=True,
                subtask_term_signal="robot1_base_pose",
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="object",
                object_soft=True,
                subtask_term_signal="grasp",
                subtask_term_offset_range=(5, 10),
                selection_strategy="registration_cost",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=20,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref="object",
                object_soft=True,
                subtask_term_signal="release_above",
                subtask_term_offset_range=(5, 10),
                selection_strategy="registration_cost",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref=None,
                object_soft=True,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
