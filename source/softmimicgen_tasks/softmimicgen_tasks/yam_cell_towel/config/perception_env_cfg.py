# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAM towel folding with the top camera's depth and segmentation exported alongside RGB.

The top camera already renders ``distance_to_image_plane`` and ``semantic_segmentation`` every
step (see ``ObjectTableSceneCfg.top_camera``), but the base task declares no observation term
that reads them, so they never reach the recorded dataset -- they are pure rendering cost.

These modalities roughly triple the on-disk size of a demonstration, so they are deliberately
kept out of the default task and exposed through this variant instead. Use it to generate the
smaller diagnostic batches that need masks -- verifying that cloth appearance randomization lands
on the towel and nothing else -- or to keep a point-cloud based policy open as a fallback. Train
the RGB policy on the default task.

This mirrors how Isaac Lab separates ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`` from
its ``-Cosmos-`` counterpart, which exists solely to add these same extra modalities.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from softmimicgen_tasks.yam_cell_towel.config.joint_pos_env_cfg import YamCellTowelEnvCfg
from softmimicgen_tasks.yam_towel import mdp


@configclass
class YamCellTowelPerceptionEnvCfg(YamCellTowelEnvCfg):
    """YAM red-cloth folding that additionally records top-camera depth and segmentation."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Without semantic tags every prim lands in a single class and the segmentation AOV comes
        # back as one uniform label, which is useless for isolating the towel. Tags follow the
        # Replicator (type, data) convention.
        self.scene.object.spawn.semantic_tags = [("class", "cloth")]
        self.scene.workstation.spawn.semantic_tags = [("class", "workstation")]
        self.scene.robot_1.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot_2.spawn.semantic_tags = [("class", "robot")]

        # Match the RGB top view so depth, segmentation and colour stay pixel-aligned.
        rgb_params = self.observations.policy.top.params
        target_height = rgb_params.get("target_height", 240)
        target_width = rgb_params.get("target_width", 320)

        # normalize=True is what zeroes the infinities the depth AOV returns for empty pixels;
        # for depth data types it applies no other scaling.
        self.observations.policy.top_depth = ObsTerm(
            func=mdp.image_cropped,
            params={
                "sensor_cfg": SceneEntityCfg("top_camera"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
                "target_height": target_height,
                "target_width": target_width,
            },
        )

        self.observations.policy.top_segmentation = ObsTerm(
            func=mdp.image_cropped,
            params={
                "sensor_cfg": SceneEntityCfg("top_camera"),
                "data_type": "semantic_segmentation",
                "normalize": False,
                "target_height": target_height,
                "target_width": target_width,
            },
        )

        # Deliberately left out of image_obs_list: consumers of that list assume 3-channel uint8
        # RGB, and these two are single-channel float32 and uint32 respectively.
