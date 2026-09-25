# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cv2
import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

GRIPPER_QPOS_SCALE = 0.04


def state(
    env: ManagerBasedRLEnv,
    robot_1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot_2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
) -> torch.Tensor:
    """Combined state observation for both robots including joint positions and gripper positions."""
    robot_1: Articulation = env.scene[robot_1_cfg.name]
    robot_1_joints = robot_1.data.joint_pos[:, robot_1_cfg.joint_ids][..., :-2]
    robot_1_finger_2 = robot_1.data.joint_pos[:, -2].unsqueeze(1) / GRIPPER_QPOS_SCALE

    robot_2: Articulation = env.scene[robot_2_cfg.name]
    robot_2_joints = robot_2.data.joint_pos[:, robot_2_cfg.joint_ids][..., :-2]
    robot_2_finger_2 = robot_2.data.joint_pos[:, -2].unsqueeze(1) / GRIPPER_QPOS_SCALE

    return torch.cat(
        (robot_1_joints, robot_1_finger_2, robot_2_joints, robot_2_finger_2),
        dim=1,
    )


def get_left_eef_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get left (robot_1) end effector position relative to env origin."""
    body_pos_w = env.scene["robot_1"].data.body_pos_w
    idx = env.scene["robot_1"].data.body_names.index("link_6")
    return body_pos_w[:, idx] - env.scene.env_origins


def get_left_eef_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get left (robot_1) end effector quaternion."""
    body_quat_w = env.scene["robot_1"].data.body_quat_w
    idx = env.scene["robot_1"].data.body_names.index("link_6")
    return body_quat_w[:, idx]


def get_right_eef_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get right (robot_2) end effector position relative to env origin."""
    body_pos_w = env.scene["robot_2"].data.body_pos_w
    idx = env.scene["robot_2"].data.body_names.index("link_6")
    return body_pos_w[:, idx] - env.scene.env_origins


def get_right_eef_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get right (robot_2) end effector quaternion."""
    body_quat_w = env.scene["robot_2"].data.body_quat_w
    idx = env.scene["robot_2"].data.body_names.index("link_6")
    return body_quat_w[:, idx]


def actions_ee(
    env: ManagerBasedRLEnv,
    robot_1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot_2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
) -> torch.Tensor:
    """Combined EE pose and gripper state for both robots."""
    left_eef_pos = get_left_eef_pos(env)
    left_eef_quat = get_left_eef_quat(env)
    left_eef_finger_2 = env.scene["robot_1"].data.joint_pos[:, -2].unsqueeze(1) / GRIPPER_QPOS_SCALE
    right_eef_pos = get_right_eef_pos(env)
    right_eef_quat = get_right_eef_quat(env)
    right_eef_finger_2 = env.scene["robot_2"].data.joint_pos[:, -2].unsqueeze(1) / GRIPPER_QPOS_SCALE

    return torch.cat(
        (left_eef_pos, left_eef_quat, left_eef_finger_2, right_eef_pos, right_eef_quat, right_eef_finger_2),
        dim=1,
    )


# dtypes cv2.resize accepts; anything else (e.g. uint32 segmentation ids) takes the numpy path
_CV2_RESIZE_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def _nearest_resize(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Nearest-neighbour resize that preserves exact values and works for any dtype."""
    rows = (np.arange(target_height) * frame.shape[0] // target_height).clip(0, frame.shape[0] - 1)
    cols = (np.arange(target_width) * frame.shape[1] // target_width).clip(0, frame.shape[1] - 1)
    return frame[rows[:, None], cols[None, :]]


def _resize_with_aspect_ratio(
    frame: np.ndarray, target_width: int, target_height: int, nearest: bool = False
) -> np.ndarray:
    """Resize frame to target resolution, applying center crop if aspect ratios don't match.

    Args:
        frame: Image of shape (H, W, C).
        target_width: Target width.
        target_height: Target height.
        nearest: Use nearest-neighbour sampling. Required for label images such as semantic
            segmentation, where interpolating between ids would invent ids that do not exist.
    """
    src_height, src_width = frame.shape[:2]
    src_aspect = src_width / src_height
    target_aspect = target_width / target_height

    if abs(src_aspect - target_aspect) > 1e-6:
        if src_aspect > target_aspect:
            new_width = int(src_height * target_aspect)
            crop_left = (src_width - new_width) // 2
            frame = frame[:, crop_left : crop_left + new_width, :]
        else:
            new_height = int(src_width / target_aspect)
            crop_top = (src_height - new_height) // 2
            frame = frame[crop_top : crop_top + new_height, :, :]

    if nearest or frame.dtype not in _CV2_RESIZE_DTYPES:
        return _nearest_resize(frame, target_width, target_height)

    resized = cv2.resize(frame, (target_width, target_height))
    # cv2.resize drops a trailing singleton channel; restore it so shapes stay consistent
    return resized[..., None] if resized.ndim == 2 and frame.ndim == 3 else resized


def image_cropped(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    target_height: int = 240,
    target_width: int = 320,
) -> torch.Tensor:
    """Camera images with aspect-ratio-aware center cropping and resizing.

    Args:
        env: The environment.
        sensor_cfg: Camera sensor configuration.
        data_type: Image data type to read.
        convert_perspective_to_orthogonal: Convert perspective depth to orthogonal.
        normalize: Whether to normalize the images.
        target_height: Target height for resized images.
        target_width: Target width for resized images.
    """
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type]

    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
        elif "normals" in data_type:
            images = (images + 1.0) * 0.5

    # label images must not be interpolated, or resizing invents segmentation ids
    nearest = "segmentation" in data_type

    original_device = images.device
    original_dtype = images.dtype
    im = images.cpu().numpy()

    if im.ndim == 4:
        batch_size = im.shape[0]
        channels = im.shape[3]
        resized_images = np.zeros((batch_size, target_height, target_width, channels), dtype=im.dtype)
        for i in range(batch_size):
            resized_images[i] = _resize_with_aspect_ratio(im[i], target_width, target_height, nearest)
    else:
        resized_images = _resize_with_aspect_ratio(im, target_width, target_height, nearest)

    return torch.from_numpy(resized_images).to(device=original_device, dtype=original_dtype)
