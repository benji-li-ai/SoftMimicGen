# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAM towel folding with visual domain randomization, for sim-to-real policy training.

Kept as a variant rather than folded into the base task so the clean environment stays available
as a control -- the held-out evaluation that actually predicts transfer needs to compare a policy
on randomized versus nominal conditions.

**What is deliberately NOT randomized:** the three walls, the extrusion frame and the cell layout.
The real workspace is an enclosure with three white walls, so those are stable, informative cues.
Randomizing them would train the policy to discard information it can rely on.

Axis ordering follows the empirical finding that spatial randomization transfers better than
appearance randomization for vision-language-action policies, and that per-frame camera
perturbation beats per-episode. Camera pose does not affect physics, so per-frame jitter is free
of any effect on the generated trajectories.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from softmimicgen_tasks.yam_towel import mdp
from softmimicgen_tasks.yam_towel.config.joint_pos_env_cfg import YamTowelEnvCfg

# One environment step at 120 Hz simulation with decimation 4, i.e. the 30 Hz control period.
# Used as the interval so camera jitter is resampled every frame.
_CONTROL_PERIOD_S = 4.0 / 120.0

# Fabric base-colour textures for the towel. Real cloth-folding cells see different towels, so
# unlike the walls this is a genuine source of variation rather than a stable cue.
#
# The Base/Textiles entries are clean single-purpose albedo maps. The vMaterials entries are
# restricted to the handful whose textures are plain diffuse -- most vMaterials fabrics are
# channel-packed (e.g. denim_R_diff_G_mask.jpg) and would render as nonsense if used as a
# straight colour map.
_FABRIC_TEXTURES = [
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Textiles/Cloth_Black/Cloth_Black_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Textiles/Cloth_Gray/Cloth_Gray_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Textiles/Linen_Beige/Linen_Beige_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Textiles/Linen_Blue/Linen_Blue_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Textiles/Linen_White/Linen_White_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Fabric/textures/cotton_roughly_woven_diff.jpg",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Fabric/textures/fine_woven_cotton_diff.jpg",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Fabric/textures/pique_weave_diff.jpg",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Fabric/textures/felt_white_diff.jpg",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Fabric/textures/tweed_fibers_diff.jpg",
]


@configclass
class YamTowelDREnvCfg(YamTowelEnvCfg):
    """YAM towel folding with camera pose, lighting and cloth appearance randomization."""

    mount_height_range: tuple[float, float] | None = (-0.015, 0.015)
    """Per-episode vertical jitter of both arm mounts, in metres, or ``None`` to disable.

    Centred on ``mount_height_offset`` (inherited from the base config, currently 2 cm), so the
    sampled mount height spans roughly 0.5 cm to 3.5 cm above the tabletop. The band is
    deliberately wide enough to absorb the "roughly 2 cm" measurement being off, but does not
    reach back down to flush-with-the-tabletop, which is known not to match the real cell.

    This is the only randomization axis here that perturbs physics -- it changes each arm's reach
    to the cloth -- so widen it only as far as the generation success rate tolerates.
    """

    randomize_cloth_appearance: bool = True
    """Whether to randomize the towel's fabric texture.

    Enabling it forces ``replicate_physics=False``, because the underlying Replicator-based term
    refuses to run with scene replication enabled. That slows multi-environment generation, so it
    is exposed as a switch: turn it off to generate the bulk corpus with replication and recover
    appearance variety from a re-render pass instead.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # -- Camera pose. The highest-value axis, and absent from Isaac Lab entirely.
        #
        # Split into the two physically distinct error sources rather than one big per-frame term:
        #
        #   calibration  fixed per episode, large  -- the real mount differs from the sim nominal
        #   vibration    per frame, small          -- the mount shakes a little
        #
        # A single large per-frame term (the first version of this) shifted the image 7.6 px on
        # average and up to 17.6 px between consecutive frames -- 2.4% of image width at 30 Hz,
        # which no rigidly-mounted camera does. It also made every observation independently
        # mis-localized instead of consistently so. The dataset sees the same range of viewpoints
        # either way; this just distributes them the way reality does.
        self.events.resample_top_camera_calibration = EventTerm(
            func=mdp.resample_camera_calibration,
            mode="reset",
            params={
                "sensor_cfg": SceneEntityCfg("top_camera"),
                "pos_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.02, 0.02)},
                "rot_range": {
                    "x": (-math.radians(2.5), math.radians(2.5)),
                    "y": (-math.radians(2.5), math.radians(2.5)),
                    "z": (-math.radians(2.5), math.radians(2.5)),
                },
            },
        )

        self.events.randomize_top_camera_pose = EventTerm(
            func=mdp.randomize_camera_pose,
            mode="interval",
            interval_range_s=(_CONTROL_PERIOD_S, _CONTROL_PERIOD_S),
            is_global_time=False,
            params={
                "sensor_cfg": SceneEntityCfg("top_camera"),
                # ~0.6 px of image motion at the table, versus 7.6 px before.
                "pos_range": {"x": (-0.0015, 0.0015), "y": (-0.0015, 0.0015), "z": (-0.001, 0.001)},
                "rot_range": {
                    "x": (-math.radians(0.1), math.radians(0.1)),
                    "y": (-math.radians(0.1), math.radians(0.1)),
                    "z": (-math.radians(0.1), math.radians(0.1)),
                },
                "convention": "opengl",
            },
        )

        # The wrist cameras are deliberately NOT randomized. They are parented to link_6, so their
        # configured offset is relative to that link; `randomize_camera_pose` writes an absolute
        # world pose and would detach them from the arm, leaving both wrist views staring at the
        # ground plane and the sky for the whole episode. Mounting-tolerance jitter for a
        # link-parented camera needs a local-transform perturbation instead, which is not
        # implemented -- and it is a low-value axis compared with the top camera, since a
        # bolted-on wrist camera barely moves relative to its mount in reality.

        # -- Lighting, per episode. Wide intensity range, mild colour drift, no HDR sky: an
        # enclosed white cell reads mostly as ambient, and sky textures would tint the walls in
        # ways the real workspace never does.
        # The overhead fixture inside the cell is what actually lights the workspace, so it
        # carries the randomization. Randomizing the dome instead is nearly a no-op once the
        # ceiling is in place -- that mistake collapsed brightness variation from 16% to 6%.
        self.events.randomize_ceiling_light = EventTerm(
            func=mdp.randomize_dome_light,
            mode="reset",
            params={
                "intensity_range": (10000.0, 45000.0),
                "color_variation": 0.06,
                "base_color": (1.0, 0.98, 0.95),
                "textures": None,
                "asset_cfg": SceneEntityCfg("ceiling_light"),
            },
        )

        # The dome is now only ambient spill through the open sides; vary it mildly so the
        # overall ambient level is not perfectly constant.
        self.events.randomize_dome_light = EventTerm(
            func=mdp.randomize_dome_light,
            mode="reset",
            params={
                "intensity_range": (1500.0, 4500.0),
                "color_variation": 0.08,
                "base_color": (0.75, 0.75, 0.75),
                "textures": None,
                "asset_cfg": SceneEntityCfg("light"),
            },
        )

        # -- Arm mount height, per episode. This is calibration tolerance around the nominal set
        # by `mount_height_offset`, not a substitute for measuring it: randomizing around a wrong
        # centre is worse than centring correctly and randomizing narrowly.
        #
        # Both arms move by the same offset because they share one workstation frame. This is the
        # only axis here that perturbs physics, so widen it only as far as the generation success
        # rate tolerates.
        if self.mount_height_range is not None:
            self.events.randomize_robot_mount_height = EventTerm(
                func=mdp.randomize_robot_mount_height,
                mode="reset",
                params={
                    "height_range": self.mount_height_range,
                    "asset_cfgs": [SceneEntityCfg("robot_1"), SceneEntityCfg("robot_2")],
                },
            )

        # -- Cloth appearance, per episode. Scoped to the towel only; the walls, frame and
        # tabletop keep their real-world appearance.
        if self.randomize_cloth_appearance:
            self.events.randomize_cloth_texture = EventTerm(
                func=mdp.randomize_visual_texture_material,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "texture_paths": _FABRIC_TEXTURES,
                    "event_name": "randomize_cloth_texture",
                    "texture_rotation": (0.0, math.pi / 2.0),
                },
            )
            # Replicator-backed texture randomization refuses to run with scene replication on.
            # This is the throughput cost of this one axis: camera and lighting randomization do
            # not need it. Set randomize_cloth_appearance=False to generate with replication.
            self.scene.replicate_physics = False
