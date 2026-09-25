# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.assets import DeformableCfg
from isaaclab.utils import configclass

from softmimicgen_tasks.yam_towel.config.joint_pos_env_cfg import YamTowelEnvCfg

from ..env_cfg import ObjectTableSceneCfg
from ..spawners import CellArmCfg, CellClothCfg


@configclass
class YamCellTowelEnvCfg(YamTowelEnvCfg):
    """Native YAM control and MDP on the current package's workcell and red cloth."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    mount_height_offset: float = 0.0304
    """Arm base offset above the 0.755 m cell tabletop (nominal base z = 0.7854 m)."""

    def __post_init__(self):
        from yam_assets import scene as package_scene
        from yam_assets.robots.yam import YAM_CELL_SOURCE_CFG

        super().__post_init__()
        self.sim.render_interval = self.decimation
        self.scene.workstation.spawn.usd_path = package_scene.CELL_USD
        self.events.reset_object_position.params["pose_range"].update(
            x=(-0.03, 0.03), y=(-0.03, 0.03), yaw=(-math.pi / 2, math.pi / 2)
        )
        self.success_fold_only: bool = True
        """Fold-area gate without the native open-gripper clause."""
        self.cloth_scale: float = 1.0
        """Uniform cloth spawn scale; 1.0 keeps the recorded 0.292 m footprint."""

        for name, side_path, y in (
            ("robot_1", package_scene.LEFT_YAM_PATH, 0.31),
            ("robot_2", package_scene.RIGHT_YAM_PATH, -0.31),
        ):
            robot = YAM_CELL_SOURCE_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/{name.capitalize()}")
            robot.spawn = CellArmCfg(usd_path=package_scene.CELL_USD, source_prim_path=side_path)
            robot.articulation_root_prim_path = "/joints/world_weld"
            robot.init_state.pos = (0.2525, y, 0.755 + self.mount_height_offset)
            robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
            setattr(self.scene, name, robot)

        self.scene.object = DeformableCfg(
            prim_path="{ENV_REGEX_NS}/Cloth",
            init_state=DeformableCfg.InitialStateCfg(
                pos=package_scene.CLOTH_CUSTOM_RED_SPAWN, rot=(1.0, 0.0, 0.0, 0.0)
            ),
            spawn=CellClothCfg(usd_path=package_scene.CLOTH_CUSTOM_RED_USD, scale=(self.cloth_scale,) * 3),
        )
