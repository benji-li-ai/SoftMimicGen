# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render still images of a SoftMimicGen scene from several viewpoints.

Spawns the task environment, lets the deformable object settle, then captures
RGB from a free-flying "beauty" camera placed at a number of eye/target pairs,
plus whatever cameras the task itself defines (top / wrist views).

Example:
    python scripts/tools/render_scene.py \
        --task Isaac-Fold-Towel-Yam-Joint-v0 --enable_cameras --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render still images of a SoftMimicGen scene.")
parser.add_argument("--task", type=str, default="Isaac-Fold-Towel-Yam-Joint-v0", help="Name of the task.")
parser.add_argument("--output_dir", type=str, default="renders", help="Directory to write PNGs into.")
parser.add_argument("--settle_steps", type=int, default=60, help="Sim steps to run before capturing.")
parser.add_argument("--width", type=int, default=1280, help="Beauty camera width.")
parser.add_argument("--height", type=int, default=960, help="Beauty camera height.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# rendering the scene is the whole point of this script
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import torch

import gymnasium as gym

from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

import softmimicgen_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# (name, eye, target) triples for the free-flying camera. Coordinates are in the
# env frame: the workstation table top sits at z = 0.755, the cloth near
# (0.495, -0.015), and the two YAM arms at y = +0.28 and y = -0.33.
VIEWPOINTS = [
    ("01_hero_three_quarter", (-1.05, -0.52, 1.32), (0.45, -0.02, 0.84)),
    ("02_front", (-1.30, -0.02, 1.25), (0.50, -0.02, 0.85)),
    ("03_operator_side", (-0.95, 0.46, 1.28), (0.45, -0.02, 0.84)),
    ("04_top_down", (0.40, -0.02, 2.15), (0.42, -0.02, 0.76)),
    ("05_close_grippers", (-0.50, -0.30, 1.05), (0.45, -0.02, 0.80)),
    ("06_low_front", (-1.15, -0.02, 0.95), (0.45, -0.02, 0.80)),
]


def save_png(rgb: torch.Tensor, path: str) -> None:
    """Write an HxWx3 or HxWx4 uint8 tensor to ``path`` as a PNG."""
    from PIL import Image

    arr = rgb.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(arr[..., :3]).save(path)


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # add a free-flying camera to the scene; InteractiveScene picks up any
    # sensor cfg attribute on the scene cfg, including one added here.
    env_cfg.scene.beauty_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/BeautyCamera",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0, focus_distance=1.5, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    # nicer stills than the training config, which turns AA off for throughput
    env_cfg.sim.render.antialiasing_mode = "DLAA"

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    # let the cloth drape and the arms reach their reset pose
    zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    with torch.inference_mode():
        for _ in range(args_cli.settle_steps):
            env.step(zero_actions)

    scene = env.unwrapped.scene
    beauty = scene["beauty_camera"]
    origin = scene.env_origins[0]

    written = []
    with torch.inference_mode():
        for name, eye, target in VIEWPOINTS:
            eyes = (torch.tensor([eye], device=env.unwrapped.device) + origin).float()
            targets = (torch.tensor([target], device=env.unwrapped.device) + origin).float()
            beauty.set_world_poses_from_view(eyes, targets)
            # a couple of render ticks so the new pose is actually shaded
            for _ in range(3):
                env.unwrapped.sim.render()
            beauty.update(dt=0.0, force_recompute=True)
            path = os.path.join(args_cli.output_dir, f"{name}.png")
            save_png(beauty.data.output["rgb"][0], path)
            written.append(path)
            print(f"[render] wrote {path}")

        # also dump the cameras the task itself defines
        for cam_name in ("top_camera", "left_wrist_camera", "right_wrist_camera"):
            if cam_name not in scene.sensors:
                continue
            cam = scene[cam_name]
            cam.update(dt=0.0, force_recompute=True)
            path = os.path.join(args_cli.output_dir, f"task_{cam_name}.png")
            save_png(cam.data.output["rgb"][0], path)
            written.append(path)
            print(f"[render] wrote {path}")

    print(f"[render] {len(written)} images written to {os.path.abspath(args_cli.output_dir)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
