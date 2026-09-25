# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.assets import DeformableCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from softmimicgen_assets import SOFTMIMICGEN_ASSETS_DATA_DIR

from softmimicgen_tasks.yam_red_cloth import mdp
from softmimicgen_tasks.yam_red_cloth.env_cfg import EnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from softmimicgen_assets.robots.yam import YAM_CONFIG_HIGH_PD_CFG  # isort: skip


@configclass
class YamRedClothEnvCfg(EnvCfg):

    mount_height_offset: float = 0.02
    """Vertical correction applied to both arm mounts, in metres.

    Measured at roughly 2 cm on the target cell: the real arms sit above the tabletop rather than
    flush with it as the shipped workstation asset has them.

    The shipped workstation asset mounts both arms flush with the tabletop at z = 0.755. On real
    YAM cells the arms sit higher, and that mismatch is a sim-to-real gap best closed by measuring
    the real offset and setting it here -- not by randomizing around a value known to be wrong.
    Applies to the clean environment as well as the randomized one, so evaluation and training
    share the same geometry.

    Note that raising the arms changes their reach to the cloth on the table. The source
    demonstrations were teleoperated at offset 0.0, so a large correction may make the recorded
    trajectories unreachable; watch the generation success rate after changing it.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set robot
        mount_z = 0.755 + self.mount_height_offset
        self.scene.robot_1 = YAM_CONFIG_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.init_state.pos = (0.2525, 0.28, mount_z)
        self.scene.robot_1.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot_2 = YAM_CONFIG_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.init_state.pos = (0.2525, -0.33, mount_z)
        self.scene.robot_2.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # Set actions for the specific robot type
        self.actions.arm_1_action = mdp.JointPositionActionCfg(
            asset_name="robot_1", joint_names=["joint.*"], scale=1, use_default_offset=True
        )
        self.actions.arm_2_action = mdp.JointPositionActionCfg(
            asset_name="robot_2", joint_names=["joint.*"], scale=1, use_default_offset=True
        )

        # Gripper actions: single continuous action [0, 1] per gripper
        # Action value controls both fingers: left_finger scales positive, right_finger mirrors negative
        self.actions.gripper_1_action = mdp.CoupledGripperPositionActionCfg(
            asset_name="robot_1",
            joint_names=["left_finger", "right_finger"],
            open_command_expr={"left_finger": 0.04, "right_finger": -0.04},
            close_command_expr={"left_finger": 0.0, "right_finger": 0.0},
        )
        self.actions.gripper_2_action = mdp.CoupledGripperPositionActionCfg(
            asset_name="robot_2",
            joint_names=["left_finger", "right_finger"],
            open_command_expr={"left_finger": 0.04, "right_finger": -0.04},
            close_command_expr={"left_finger": 0.0, "right_finger": 0.0},
        )

        self.scene.object = DeformableCfg(
            prim_path="{ENV_REGEX_NS}/Cloth",
            init_state=DeformableCfg.InitialStateCfg(pos=[0.495, -0.015, 0.755], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Props/Cloth/Cloth_square_11_5in_red_1_5mm_vis_2_5mm_contact.usd",
            )
        )
        
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_1_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/arm/arm",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/arm/link_6",
                    name="end_effector",
                ),
            ],
        )
        self.scene.ee_2_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_2/arm/arm",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_2/arm/link_6",
                    name="end_effector",
                ),
            ],
        )
