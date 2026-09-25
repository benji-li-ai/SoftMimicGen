# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""openpi training config for fine-tuning pi0.5 on SoftMimicGen YAM towel-folding data.

This file is **not** importable from this repository -- it is a snippet to paste into the
``_CONFIGS`` list in ``openpi/src/openpi/training/config.py`` (or to import from there). It lives
here so the three places that must agree on key names stay next to each other:

1. ``convert_to_lerobot.py``    -- writes ``observation.images.cam_high`` / ``cam_left_wrist`` /
                                   ``cam_right_wrist``, ``observation.state``, ``action``
2. this config                 -- consumes those keys
3. ``eval_policy.py``          -- ``CAMERA_KEY_MAP`` sends the same names at inference time

Changing a camera name requires changing all three, or training silently sees black images where
a camera should be.

Why the ALOHA data config is reused: pi0.5's ALOHA path already expects a bimanual 14-dimensional
robot with exactly ``cam_high`` / ``cam_left_wrist`` / ``cam_right_wrist``, which is why the
converter emits those names. The one thing that must change is ``adapt_to_pi``.

**adapt_to_pi MUST be False.** With the default ``True``, ``AlohaInputs``/``AlohaOutputs`` apply
Trossen-specific joint flipping and a gripper angle conversion. YAM joints are plain absolute
radians and its gripper is already normalized to [0, 1] by
``CoupledGripperPositionAction``, so those transforms would corrupt both the training targets and
the actions sent back to the simulator.

Workflow::

    # 1. convert (in the openpi environment)
    uv run python .../convert_to_lerobot.py --input 'datasets/generated_dataset/*.hdf5' \
        --repo_id local/yam_towel_sim

    # 2. compute normalization statistics -- required before training
    uv run scripts/compute_norm_stats.py --config-name pi05_yam_towel

    # 3. train
    uv run scripts/train.py pi05_yam_towel --exp-name yam_towel_dr_v1

    # 4. serve for evaluation, then point eval_policy.py at it
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_yam_towel --policy.dir=checkpoints/pi05_yam_towel/yam_towel_dr_v1/29999

VERIFY BEFORE RUNNING: openpi's config API moves between releases. Check the argument names of
``LeRobotAlohaDataConfig`` and ``Pi0Config`` against the openpi revision you have checked out --
in particular whether ``repo_id`` is still passed to the data config, and whether ``action_dim``
is still exposed on the model config.
"""

# ruff: noqa
# The imports below resolve inside the openpi package, not this repository.
from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.training.config import (
    AssetsConfig,
    LeRobotAlohaDataConfig,
    TrainConfig,
)
from openpi.training import weight_loaders

# The LeRobot dataset produced by convert_to_lerobot.py.
YAM_TOWEL_REPO_ID = "local/yam_towel_sim"

YAM_TOWEL_TRAIN_CONFIG = TrainConfig(
    name="pi05_yam_towel",
    # pi05=True selects the pi0.5 architecture. 14 action dimensions is well under the model's
    # 32-dimensional limit, so no max_action_dim change is needed -- the remainder is padded.
    # action_horizon 16 at the environment's 30 Hz is roughly half a second of open-loop motion,
    # which pairs with --replan_steps 8 in eval_policy.py.
    model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16),
    data=LeRobotAlohaDataConfig(
        repo_id=YAM_TOWEL_REPO_ID,
        # See the module docstring: True would apply Trossen joint flipping to YAM joints.
        adapt_to_pi=False,
        default_prompt="fold the towel",
        # MUST be set explicitly. The default maps a single camera from a key we do not write
        # ({"cam_high": "observation.images.top"}) and omits both wrist cameras entirely.
        # AlohaInputs substitutes black images with mask=False for any camera it cannot find, so
        # the wrong mapping trains silently on blank inputs rather than raising.
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        ),
        # Our actions are ABSOLUTE joint position targets in radians, straight from
        # JointPositionActionCfg with scale=1 and a zero offset. The default True would convert
        # them to deltas against the current state and train the policy in the wrong space.
        use_delta_joint_actions=False,
        # Norm stats are computed from this dataset by scripts/compute_norm_stats.py. Do NOT point
        # this at pi05_base's bundled Trossen assets -- those describe a different robot's joint
        # ranges and would normalize YAM states and actions incorrectly.
        assets=AssetsConfig(asset_id=YAM_TOWEL_REPO_ID),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
    batch_size=64,
)

# After computing norm stats, inspect q01/q99 for the two gripper dimensions (indices 6 and 13).
# They are bounded in [0, 1] and are frequently near-constant early in an episode, so a degenerate
# spread there shows up as exploding normalized actions rather than as an obvious error.
