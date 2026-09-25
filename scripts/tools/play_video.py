# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Roll out a trained robomimic policy and record the episode to MP4.

Like scripts/imitation_learning/robomimic/play.py, but writes a video per rollout
so failure modes can be inspected. Each frame tiles a free-flying third-person
camera with the three camera views the policy actually consumes.

Example:
    python scripts/tools/play_video.py \
        --task Isaac-Fold-Towel-Yam-Joint-v0 \
        --checkpoint logs/.../model_epoch_200.pth \
        --norm_factor_min -3.222730875015259 --norm_factor_max 5.869629383087158 \
        --num_rollouts 3 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Roll out a robomimic policy and record video.")
parser.add_argument("--task", type=str, default="Isaac-Fold-Towel-Yam-Joint-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=1000, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=3, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument("--out_dir", type=str, default="outputs/policy_videos", help="Directory for the MP4s.")
parser.add_argument("--fps", type=int, default=30, help="Frame rate of the output video.")
parser.add_argument(
    "--render_interval",
    type=int,
    default=4,
    help=(
        "Physics steps per render. The task config ships 1, which renders on every physics"
        " substep -- 4 render passes per control step with decimation=4. Matching decimation"
        " renders once per control step, which is all the policy observes, and is ~4x faster."
    ),
)
parser.add_argument("--norm_factor_min", type=float, default=None, help="Minimum action normalization factor.")
parser.add_argument("--norm_factor_max", type=float, default=None, help="Maximum action normalization factor.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# recording the scene requires the renderer
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import cv2
import gymnasium as gym
import numpy as np
import os
import random
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

import softmimicgen_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# third-person view through the workstation's open side (the enclosure only opens toward -x)
BEAUTY_EYE = (-1.05, -0.52, 1.32)
BEAUTY_TARGET = (0.45, -0.02, 0.84)

PANEL_H = 240
"""Height every tile is scaled to before being concatenated."""


def to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """Squeeze a camera tensor to an HxWx3 uint8 array."""
    arr = torch.squeeze(t).detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return arr[..., :3]


def panel(img: np.ndarray, label: str) -> np.ndarray:
    """Scale an image to PANEL_H and caption it."""
    scale = PANEL_H / img.shape[0]
    out = cv2.resize(img, (int(img.shape[1] * scale), PANEL_H))
    cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def compose(beauty: np.ndarray, obs_policy: dict) -> np.ndarray:
    """Tile the third-person view beside the policy's own camera views."""
    tiles = [panel(beauty, "third-person")]
    for key in ("top", "left", "right"):
        if key in obs_policy:
            tiles.append(panel(to_uint8_hwc(obs_policy[key]), key))
    return np.concatenate(tiles, axis=1)


def rollout(policy, env, success_term, horizon, device, beauty, writer, action_log=None):
    """Roll out the policy, writing one video frame per step. Returns success flag."""
    policy.start_episode()
    obs_dict, _ = env.reset()

    for _ in range(horizon):
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        # robomimic expects chw normalized float for image observations
        if hasattr(env.cfg, "image_obs_list"):
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"]:
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float() / 255.0
                    obs[image_name] = image.clip(0.0, 1.0)

        actions = policy(obs)

        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = ((actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        if action_log is not None:
            action_log.append(actions.squeeze(0).detach().cpu().numpy().copy())

        # capture before stepping so the frame matches the observation the policy acted on
        beauty.update(dt=0.0, force_recompute=True)
        frame = compose(to_uint8_hwc(beauty.data.output["rgb"][0]), obs_dict["policy"])
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        obs_dict, _, terminated, truncated, _ = env.step(actions)

        if bool(success_term.func(env, **success_term.params)[0]):
            return True
        if terminated or truncated:
            return False

    return False


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.sim.render_interval = args_cli.render_interval
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # free-flying camera for the third-person panel; InteractiveScene picks up any
    # sensor cfg attribute on the scene cfg, including one added here
    env_cfg.scene.beauty_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/BeautyCamera",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0, focus_distance=1.5, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    env.reset()

    beauty = env.scene["beauty_camera"]
    origin = env.scene.env_origins[0]
    eyes = (torch.tensor([BEAUTY_EYE], device=env.device) + origin).float()
    targets = (torch.tensor([BEAUTY_TARGET], device=env.device) + origin).float()

    tag = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    results = []

    for trial in range(args_cli.num_rollouts):
        print(f"[play_video] starting trial {trial}")
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)

        # env.reset() inside rollout() re-runs events, so re-aim the camera each trial
        beauty.set_world_poses_from_view(eyes, targets)

        path = os.path.join(args_cli.out_dir, f"{tag}_trial{trial}.mp4")
        # frame size is fixed by the tiling, so probe it with one composed frame
        beauty.update(dt=0.0, force_recompute=True)
        probe = compose(to_uint8_hwc(beauty.data.output["rgb"][0]), env.obs_buf["policy"])
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), args_cli.fps, (probe.shape[1], probe.shape[0])
        )

        action_log = []
        success = rollout(policy, env, success_term, args_cli.horizon, device, beauty, writer, action_log)
        np.save(os.path.join(args_cli.out_dir, f"{tag}_trial{trial}_actions.npy"), np.array(action_log))
        writer.release()
        results.append(success)
        print(f"[play_video] trial {trial}: {'SUCCESS' if success else 'failure'} -> {path}")

    print(f"\n[play_video] success rate: {results.count(True)}/{len(results)}")
    print(f"[play_video] videos in {os.path.abspath(args_cli.out_dir)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
