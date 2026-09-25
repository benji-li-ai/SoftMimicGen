# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native YAM visual randomization with the cell's red-fabric appearance settings."""

from pathlib import Path

from isaaclab.utils import configclass

from softmimicgen_tasks.yam_towel.config.dr_env_cfg import YamTowelDREnvCfg

from .joint_pos_env_cfg import YamCellTowelEnvCfg


@configclass
class YamCellTowelDREnvCfg(YamCellTowelEnvCfg, YamTowelDREnvCfg):
    """Keep the cell layout while inheriting camera, light and mount randomization."""

    def __post_init__(self):
        super().__post_init__()
        if self.randomize_cloth_appearance:
            from yam_assets.scene import CLOTH_CUSTOM_RED_USD

            # The 6x6 authored UV tiling keeps the cloth weave on the deforming surface.
            self.events.randomize_cloth_texture.params.update(
                texture_paths=[str(Path(CLOTH_CUSTOM_RED_USD).with_name("red_cloth.jpg"))],
                project_uvw=False,
                reflection_roughness_range=(0.82, 0.9),
                metallic_range=(0.0, 0.0),
                specular_level_range=(0.15, 0.25),
                texture_rotation=(0.0, 0.0),
            )
