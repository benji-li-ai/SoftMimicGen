# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package-backed cell geometry with the native YAM task's calibrated sensors."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from softmimicgen_tasks.yam_towel.env_cfg import ObjectTableSceneCfg as YamTowelSceneCfg

from .spawners import CellWorkstationCfg


@configclass
class ObjectTableSceneCfg(YamTowelSceneCfg):
    """Keep native lighting/top camera and attach wrist views to the cell link frames."""

    # PhysX replication does not support the package's surface-deformable body,
    # simulation mesh or material. Clone USD and parse each environment's physics.
    replicate_physics: bool = False

    # The spawner keeps the backdrop at the env datum; this state is the nested
    # TableSurface rigid body, matching the package collector's workstation frame.
    workstation = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workstation",
        spawn=CellWorkstationCfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.655, 0.0, 0.745)),
    )

    def __post_init__(self):
        super().__post_init__()
        # The package includes camera mounts, not render sensors. These calibrated
        # OpenGL offsets in link_6 preserve Rx(35 degrees) @ Ry(180 degrees).
        for robot, camera in (
            ("Robot_1", self.left_wrist_camera),
            ("Robot_2", self.right_wrist_camera),
        ):
            camera.prim_path = f"{{ENV_REGEX_NS}}/{robot}/arm/link_6/wrist_camera"
            camera.offset = CameraCfg.OffsetCfg(
                pos=(-0.0107, 0.079729, 0.066021),
                rot=(0.0, 0.0, 0.953717, 0.300706),
                convention="opengl",
            )
