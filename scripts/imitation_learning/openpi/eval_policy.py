# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate an external action-chunking policy (openpi / pi0.5) in an Isaac Lab environment.

This is a fork of ``scripts/imitation_learning/robomimic/play.py``. The rollout loop, success
checking and environment setup are deliberately kept identical so that success rates are
comparable between a robomimic checkpoint and a remote policy server; only the policy itself is
swapped out.

The policy is served out of process -- ``openpi`` runs its own JAX stack, which does not coexist
with Isaac Sim in one interpreter. Start the server separately, e.g.::

    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_yam_towel --policy.dir=/path/to/checkpoint

then point this script at it::

    python scripts/imitation_learning/openpi/eval_policy.py \
        --task Isaac-Fold-Towel-Yam-Joint-v0 --enable_cameras --headless \
        --host 0.0.0.0 --port 8000 --num_rollouts 50

Args:
    task: Name of the environment.
    policy: Which policy to evaluate -- ``remote`` (a policy server), or ``zero``/``random`` which
        are local debug policies used to sanity check the harness itself.
    host / port: Address of the policy server.
    prompt: Language instruction sent with every observation.
    replan_steps: How many actions to consume from each predicted chunk before re-querying.
    horizon: Step horizon of each rollout.
    num_rollouts: Number of rollouts.
    seed: Random seed.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an external policy in an Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--policy",
    type=str,
    default="remote",
    choices=["remote", "zero", "random"],
    help="Policy to evaluate. 'zero' and 'random' are local debug policies for validating the harness.",
)
parser.add_argument("--host", type=str, default="0.0.0.0", help="Policy server host.")
parser.add_argument("--port", type=int, default=8000, help="Policy server port.")
parser.add_argument("--prompt", type=str, default="fold the towel", help="Language instruction for the policy.")
parser.add_argument(
    "--replan_steps",
    type=int,
    default=8,
    help="Actions consumed from each predicted chunk before re-querying the server.",
)
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")
parser.add_argument(
    "--video_dir", type=str, default=None, help="If set, record each trial's camera views to an mp4 in this directory."
)
parser.add_argument(
    "--video_cameras", type=str, nargs="+", default=["top", "left", "right"], help="Camera obs keys to tile in the video."
)
parser.add_argument("--video_fps", type=int, default=30, help="Frame rate of recorded videos.")
parser.add_argument(
    "--cloth_usd_path",
    type=str,
    default=None,
    help="Override the cloth asset's USD file, e.g. a variant with a different surfaceThickness material property.",
)
parser.add_argument(
    "--cloth_scale",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override the cloth spawn scale (the mesh is 0.4m x 0.7m in its local x/y before scaling).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import random
import torch
import cv2

if args_cli.enable_pinocchio:
    import softmimicgen_tasks.humanoid  # noqa: F401

import softmimicgen_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Camera observation key in the environment -> image key the policy server expects.
# This must match the RepackTransform of the openpi TrainConfig the checkpoint was trained with.
CAMERA_KEY_MAP = {
    "top": "cam_high",
    "left": "cam_left_wrist",
    "right": "cam_right_wrist",
}

ACTION_DIM = 14

# Log the observation shapes once, so a mismatch is visible immediately rather than as a
# PIL error inside the policy server.
_SHAPES_LOGGED = False


def build_observation(obs_dict: dict, prompt: str) -> dict:
    """Convert an Isaac Lab policy observation group into an openpi observation.

    Images are sent as uint8 CHW, matching what LeRobotDataset yields during training --
    `AlohaInputs` rearranges "c h w -> h w c" itself. Isaac reports images as HWC, so they are
    transposed here. Normalization and resizing happen server side, so robomimic's /255 is not
    applied.
    """
    global _SHAPES_LOGGED

    policy_obs = obs_dict["policy"]
    images = {}
    for env_key, policy_key in CAMERA_KEY_MAP.items():
        if env_key in policy_obs:
            # CHW, not HWC. `AlohaInputs` documents its input as [channel, height, width] and
            # unconditionally applies `einops.rearrange(img, "c h w -> h w c")`
            # (aloha_policy.py:29,171), so an HWC image is rearranged into (W, C, H) and PIL
            # later dies with "Cannot handle this data type: (1, 1, 240)". Training fed CHW
            # because that is what LeRobotDataset yields, so inference has to match.
            #
            # uint8 is fine: convert_image scales only floating-point inputs by 255.
            image = torch.squeeze(policy_obs[env_key], dim=0)
            array = image.detach().cpu().numpy()
            if array.ndim == 3 and array.shape[2] in (1, 3):
                array = np.transpose(array, (2, 0, 1))
            array = np.ascontiguousarray(array, dtype=np.uint8)
            if not _SHAPES_LOGGED:
                print(f"[obs] {env_key}: env tensor {tuple(policy_obs[env_key].shape)}"
                      f" -> sent {array.shape} {array.dtype} (CHW)", flush=True)
            images[policy_key] = array

    _SHAPES_LOGGED = True

    state = torch.squeeze(policy_obs["state"], dim=0).detach().cpu().numpy().astype(np.float32)

    return {"images": images, "state": state, "prompt": prompt}


class ChunkedPolicy:
    """Wraps a chunk-predicting policy so the rollout loop can pull one action per step.

    Action-chunking policies emit a horizon of actions per inference call. Querying once per
    environment step would pay a round trip at every one of the 30 control steps per second, so
    instead each chunk is consumed for ``replan_steps`` steps before re-querying.
    """

    def __init__(self, infer_fn, replan_steps: int):
        self._infer_fn = infer_fn
        self._replan_steps = replan_steps
        self._chunk: np.ndarray | None = None
        self._index = 0

    def start_episode(self):
        """Drop any buffered actions so a new episode starts from a fresh inference."""
        self._chunk = None
        self._index = 0

    def __call__(self, obs: dict) -> np.ndarray:
        if self._chunk is None or self._index >= min(self._replan_steps, len(self._chunk)):
            self._chunk = np.asarray(self._infer_fn(obs), dtype=np.float32)
            if self._chunk.ndim == 1:
                self._chunk = self._chunk[None, :]
            self._index = 0

        action = self._chunk[self._index]
        self._index += 1
        return action


def make_infer_fn():
    """Return an inference callable for the selected policy."""
    if args_cli.policy == "zero":
        # Deliberately inert: a harness that reports successes for this policy is not measuring
        # the task, it is measuring a success term that is trivially satisfiable.
        return lambda obs: np.zeros((1, ACTION_DIM), dtype=np.float32)

    if args_cli.policy == "random":
        rng = np.random.default_rng(args_cli.seed)
        return lambda obs: rng.uniform(-0.5, 0.5, size=(1, ACTION_DIM)).astype(np.float32)

    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise ImportError(
            "openpi_client is required for --policy remote. Install openpi in this environment, or"
            " use --policy zero / --policy random to exercise the harness without a server."
        ) from exc

    client = websocket_client_policy.WebsocketClientPolicy(host=args_cli.host, port=args_cli.port)
    print(f"[INFO] Connected to policy server at {args_cli.host}:{args_cli.port}")
    print(f"[INFO] Server metadata: {client.get_server_metadata()}")

    def infer(obs: dict) -> np.ndarray:
        result = client.infer(obs)
        return result["actions"]

    return infer


def tile_cameras(obs_dict: dict, cameras: list[str]) -> np.ndarray | None:
    """Tile the requested camera observations side by side into one uint8 HWC frame."""
    policy_obs = obs_dict["policy"]
    frames = [torch.squeeze(policy_obs[cam], dim=0).detach().cpu().numpy() for cam in cameras if cam in policy_obs]
    if not frames:
        return None
    height = max(frame.shape[0] for frame in frames)
    scaled = [
        frame if frame.shape[0] == height else cv2.resize(frame, (round(frame.shape[1] * height / frame.shape[0]), height))
        for frame in frames
    ]
    return np.ascontiguousarray(np.concatenate(scaled, axis=1)[..., :3])


def annotate(frame: np.ndarray, text: str) -> None:
    """Draw an outlined caption onto the frame in place."""
    for color, thickness in ((0, 0, 0), 3), ((255, 255, 255), 1):
        cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness, cv2.LINE_AA)


def rollout(policy, env, success_term, horizon, device, video_writer=None, video_cameras=None):
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The policy to play. Must expose ``start_episode()`` and ``__call__(obs)``.
        env: The environment to play in.
        success_term: Termination term used to check task success.
        horizon: The step horizon of each rollout.
        device: The device to place actions on.
        video_writer: If given, an imageio writer that each step's tiled camera frame is appended to.
        video_cameras: Camera obs keys to tile into the recorded frame.

    Returns:
        success: Whether the rollout succeeded.
        num_steps: Number of environment steps taken.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()

    for step in range(horizon):
        if video_writer is not None:
            frame = tile_cameras(obs_dict, video_cameras)
            if frame is not None:
                annotate(frame, f"t={step}/{horizon}")
                video_writer.append_data(frame)

        action = policy(build_observation(obs_dict, args_cli.prompt))

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != env.action_space.shape[1]:
            raise ValueError(
                f"Policy returned {action.shape[0]} action dimensions, environment expects"
                f" {env.action_space.shape[1]}."
            )

        actions = torch.from_numpy(action).to(device=device).view(1, env.action_space.shape[1])
        obs_dict, _, terminated, truncated, _ = env.step(actions)

        if bool(success_term.func(env, **success_term.params)[0]):
            return True, step + 1
        if terminated or truncated:
            return False, step + 1

    return False, horizon


def main():
    """Evaluate an external policy in an Isaac Lab environment."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Observations must stay a dict so individual cameras can be addressed by name.
    env_cfg.observations.policy.concatenate_terms = False

    # The rollout horizon is the only time limit; success is checked explicitly each step.
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    if args_cli.cloth_usd_path is not None:
        env_cfg.scene.object.spawn.usd_path = args_cli.cloth_usd_path
    if args_cli.cloth_scale is not None:
        env_cfg.scene.object.spawn.scale = tuple(args_cli.cloth_scale)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    device = torch.device(args_cli.device)
    policy = ChunkedPolicy(make_infer_fn(), args_cli.replan_steps)

    if args_cli.video_dir is not None:
        os.makedirs(args_cli.video_dir, exist_ok=True)

    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")

        video_writer = None
        if args_cli.video_dir is not None:
            video_path = os.path.join(args_cli.video_dir, f"trial{trial}.mp4")
            video_writer = imageio.get_writer(video_path, fps=args_cli.video_fps, macro_block_size=1, quality=8)

        try:
            success, num_steps = rollout(
                policy, env, success_term, args_cli.horizon, device, video_writer, args_cli.video_cameras
            )
        finally:
            if video_writer is not None:
                video_writer.close()
                print(f"[INFO] Trial {trial}: video saved to {video_path}")

        results.append(success)
        print(f"[INFO] Trial {trial}: success={success} after {num_steps} steps\n")

    num_success = results.count(True)
    print(f"\nPolicy: {args_cli.policy}")
    print(f"Successful trials: {num_success}, out of {len(results)} trials")
    print(f"Success rate: {num_success / len(results)}")
    print(f"Trial results: {results}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
