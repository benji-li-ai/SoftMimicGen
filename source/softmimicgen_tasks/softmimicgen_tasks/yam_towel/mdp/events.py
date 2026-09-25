# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Event terms for domain randomization of the YAM towel folding task.

Isaac Lab ships texture, colour and lighting randomization but has no camera pose randomization
term, which empirical work on sim-to-real transfer of vision-language-action policies finds to be
one of the highest-value axes -- ahead of appearance randomization. This module fills that gap.

Camera pose does not affect physics, so unlike most randomization these terms are safe to apply at
every step ("frame-wise"), which transfers better than perturbing once per episode.
"""

from __future__ import annotations

import random
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Cache of nominal camera poses, keyed by sensor name. The nominal must be captured before any
# randomization is applied, otherwise perturbations would compound across calls and the camera
# would random-walk away from its configured pose.
_NOMINAL_CAMERA_POSES: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _sample_uniform(
    ranges: dict[str, tuple[float, float]], keys: tuple[str, ...], num: int, device: str | torch.device
) -> torch.Tensor:
    """Sample a (num, len(keys)) tensor from per-axis uniform ranges, defaulting missing axes to 0."""
    samples = torch.zeros((num, len(keys)), device=device)
    for index, key in enumerate(keys):
        low, high = ranges.get(key, (0.0, 0.0))
        if low != high:
            samples[:, index] = torch.empty(num, device=device).uniform_(low, high)
    return samples


def reset_camera_pose_cache(env: ManagerBasedEnv, env_ids: torch.Tensor | None = None):
    """Clear the cached nominal camera poses.

    Only needed when the configured camera offsets change within a single process, which normal
    training and generation runs never do.
    """
    _NOMINAL_CAMERA_POSES.clear()


# Per-episode calibration offsets, keyed by (sensor name, env index). Held constant for the whole
# episode so that a camera is mis-calibrated in a *fixed* way, the way a real mount is.
_CAMERA_EPISODE_OFFSETS: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}


def resample_camera_calibration(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    pos_range: dict[str, tuple[float, float]] | None = None,
    rot_range: dict[str, tuple[float, float]] | None = None,
):
    """Draw a new fixed calibration offset for a camera, to be held for the whole episode.

    Pair this (``mode="reset"``) with :func:`randomize_camera_pose` (``mode="interval"``) to model
    the two physically distinct sources of camera-pose error separately:

    * **calibration** -- the real mount differs from the sim nominal by an unknown but *constant*
      amount. Large, resampled once per episode.
    * **vibration** -- the mount shakes slightly. Small, resampled every frame.

    Collapsing both into one large per-frame term produces a camera that visibly jitters several
    pixels between consecutive frames, which no real rigidly-mounted camera does. It also makes
    every observation independently mis-localized rather than consistently so, which is a harder
    and less realistic learning problem.

    Args:
        env: The environment.
        env_ids: Environments to resample. ``None`` means all of them.
        sensor_cfg: Camera sensor whose calibration offset is being drawn.
        pos_range: Per-axis translation ranges in metres for the calibration offset.
        rot_range: Per-axis rotation ranges in radians for the calibration offset.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    num = len(env_ids)
    pos = _sample_uniform(pos_range or {}, ("x", "y", "z"), num, env.device)
    euler = _sample_uniform(rot_range or {}, ("x", "y", "z"), num, env.device)
    rot = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

    for row, env_id in enumerate(env_ids.tolist()):
        _CAMERA_EPISODE_OFFSETS[(sensor_cfg.name, env_id)] = (pos[row], rot[row])


def randomize_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    pos_range: dict[str, tuple[float, float]] | None = None,
    rot_range: dict[str, tuple[float, float]] | None = None,
    convention: str = "opengl",
):
    """Perturb a camera's pose around its configured nominal pose.

    The perturbation is always measured from the nominal pose captured on the first call, never
    from the camera's current pose, so repeated application does not accumulate drift.

    Any per-episode calibration offset registered by :func:`resample_camera_calibration` is added
    on top of the nominal pose before this term's own offset is applied, so the two compose into
    "fixed mis-calibration plus small vibration". With no calibration offset registered this term
    behaves exactly as before.

    Rotation offsets are applied in the camera's own frame, so ``x``/``y``/``z`` read as
    tilt/pan/roll of the view rather than rotations about world axes.

    Args:
        env: The environment.
        env_ids: Environments to randomize. ``None`` means all of them.
        sensor_cfg: Camera sensor to perturb.
        pos_range: Per-axis translation ranges in metres, e.g. ``{"x": (-0.02, 0.02)}``. Axes that
            are absent are not perturbed.
        rot_range: Per-axis rotation ranges in radians, keyed ``"x"``, ``"y"``, ``"z"``.
        convention: Camera convention the configured offset is expressed in. The YAM cameras are
            configured in ``"opengl"``.
    """
    sensor = env.scene.sensors[sensor_cfg.name]

    # This term writes an absolute world pose, so it is only valid for a camera whose configured
    # offset is expressed relative to the environment. A camera parented to a moving link (the
    # wrist cameras hang off link_6) has an offset relative to that link, and writing it as a
    # world pose detaches the camera from the arm and pins it near the environment origin -- the
    # rendered view then shows the ground plane and empty sky instead of the workspace.
    prim_path = sensor.cfg.prim_path
    if "{ENV_REGEX_NS}/" in prim_path and prim_path.split("{ENV_REGEX_NS}/")[-1].count("/") > 0:
        raise ValueError(
            f"randomize_camera_pose cannot be used on '{sensor_cfg.name}' (prim path"
            f" '{prim_path}'): it is parented to a link rather than mounted in the environment"
            " frame, so an absolute world pose would detach it from its mount. Randomize a"
            " link-parented camera by perturbing its local transform instead."
        )

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    num_envs = len(env_ids)
    if num_envs == 0:
        return

    if sensor_cfg.name not in _NOMINAL_CAMERA_POSES:
        offset = sensor.cfg.offset
        nominal_pos = torch.tensor(offset.pos, device=env.device, dtype=torch.float32)
        nominal_rot = torch.tensor(offset.rot, device=env.device, dtype=torch.float32)
        _NOMINAL_CAMERA_POSES[sensor_cfg.name] = (nominal_pos, nominal_rot)

    nominal_pos, nominal_rot = _NOMINAL_CAMERA_POSES[sensor_cfg.name]

    # The configured offset is relative to the environment, the setter expects world coordinates.
    positions = nominal_pos.unsqueeze(0).repeat(num_envs, 1) + env.scene.env_origins[env_ids]
    orientations = nominal_rot.unsqueeze(0).repeat(num_envs, 1)

    # Fixed per-episode calibration offset, if one has been drawn for these environments.
    if _CAMERA_EPISODE_OFFSETS:
        calib_pos = torch.zeros_like(positions)
        calib_rot = torch.zeros_like(orientations)
        calib_rot[:, 0] = 1.0  # identity quaternion (w, x, y, z)
        for row, env_id in enumerate(env_ids.tolist()):
            entry = _CAMERA_EPISODE_OFFSETS.get((sensor_cfg.name, env_id))
            if entry is not None:
                calib_pos[row], calib_rot[row] = entry
        positions = positions + calib_pos
        orientations = math_utils.quat_mul(orientations, calib_rot)

    # Small per-frame vibration on top.
    if pos_range:
        positions = positions + _sample_uniform(pos_range, ("x", "y", "z"), num_envs, env.device)
    if rot_range:
        euler = _sample_uniform(rot_range, ("x", "y", "z"), num_envs, env.device)
        delta = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        orientations = math_utils.quat_mul(orientations, delta)

    sensor.set_world_poses(positions, orientations, env_ids=env_ids, convention=convention)


def randomize_robot_mount_height(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    height_range: tuple[float, float],
    asset_cfgs: list[SceneEntityCfg],
):
    """Shift the mounting height of several robots together by a single sampled offset.

    Both YAM arms are bolted to one workstation frame, so their mount height is a single physical
    quantity -- perturbing the arms independently would model a cell that cannot exist. One offset
    is sampled per environment and applied to every listed asset.

    Unlike camera pose, this *does* change physics: raising the arms changes their reach to the
    cloth lying on the table. Widen ``height_range`` only as far as generation success rate
    tolerates.

    Args:
        env: The environment.
        env_ids: Environments to randomize. ``None`` means all of them.
        height_range: Range in metres to sample the shared vertical offset from.
        asset_cfgs: The robot assets to move together.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    offsets = torch.empty(len(env_ids), device=env.device).uniform_(*height_range)

    for asset_cfg in asset_cfgs:
        asset = env.scene[asset_cfg.name]
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] += env.scene.env_origins[env_ids]
        root_state[:, 2] += offsets
        asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def _balanced_color(
    base: tuple[float, float, float], variation: float, generator: random.Random
) -> tuple[float, float, float]:
    """Perturb an RGB colour while preserving overall brightness.

    Offsets are forced to sum to zero so the light shifts in hue without also getting brighter or
    dimmer, which keeps colour and intensity independent randomization axes.
    """
    offsets = [generator.uniform(-variation, variation) for _ in range(3)]
    mean_offset = sum(offsets) / 3
    return tuple(min(1.0, max(0.0, channel + offset - mean_offset)) for channel, offset in zip(base, offsets))


def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    intensity_range: tuple[float, float],
    color_variation: float = 0.0,
    base_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    textures: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    """Randomize the dome light's intensity, colour and optionally its HDR texture.

    Isaac Lab's equivalent term in the Franka stacking task is gated behind ``env.cfg.eval_mode``
    and silently does nothing during data generation, so this is a plain ungated implementation.

    The YAM workspace is an enclosed cell with white walls, so intensity is the axis that carries
    most of the realistic variation. ``textures`` is left empty by default: HDR skies tint white
    walls in ways the real cell never does, which would randomize away a cue that is genuinely
    stable.

    Args:
        env: The environment.
        env_ids: Unused -- the dome light is a single global prim, not one per environment.
        intensity_range: Range to sample light intensity from.
        color_variation: Maximum per-channel deviation, brightness-preserving. 0 disables.
        base_color: Colour to perturb around.
        textures: Optional HDR texture paths to sample from. Empty means no dome texture.
        asset_cfg: The light asset.
    """
    light_prim = env.scene[asset_cfg.name].prims[0]
    generator = random.Random(torch.randint(0, 2**31 - 1, (1,)).item())

    light_prim.GetAttribute("inputs:intensity").Set(generator.uniform(*intensity_range))

    if color_variation > 0.0:
        light_prim.GetAttribute("inputs:color").Set(_balanced_color(base_color, color_variation, generator))
    else:
        light_prim.GetAttribute("inputs:color").Set(base_color)

    # Only dome lights carry an HDR texture; disk/sphere fixtures have no such attribute.
    texture_attr = light_prim.GetAttribute("inputs:texture:file")
    if texture_attr.IsValid():
        texture_attr.Set(generator.choice(textures) if textures else "")
