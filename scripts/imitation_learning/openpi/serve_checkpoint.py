# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a fine-tuned YAM towel checkpoint over openpi's websocket policy server.

openpi's ``scripts/serve_policy.py`` resolves a training config **by name** out of openpi's own
``_CONFIGS_DICT``. ``pi05_yam_towel`` lives in the robotics repo
(``policies/pi05/softmimicgen_configs.py``), which openpi has never heard of, so this wrapper
registers it and then hands off to openpi's own serving entrypoint. Nothing is duplicated.

**Run this in the openpi environment**, not the ``softmimicgen`` conda environment -- it needs
JAX and openpi, and deliberately imports nothing from Isaac Lab. The evaluation half
(``eval_policy.py``) runs in the Isaac environment and talks to this over a websocket, which is
also how the two get to hold GPU memory independently.

Normalization statistics are read from ``<checkpoint>/assets/<asset_id>/norm_stats.json``, which
the trainer writes into every checkpoint, so a downloaded checkpoint serves standalone.

Only ``params/`` and ``assets/`` are needed to serve -- ``train_state/`` is optimizer state and
is roughly two thirds of a checkpoint's ~45 GB. ``fetch_checkpoint.sh`` downloads just the two.

Typical use, alongside a running Isaac evaluation::

    # terminal 1 (openpi env)
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python serve_checkpoint.py --checkpoint_dir /tmp/ckpt/5000 --port 8000

    # terminal 2 (softmimicgen env)
    python scripts/imitation_learning/openpi/eval_policy.py \
        --task Isaac-Fold-Towel-Yam-Joint-v0 --policy remote --port 8000 \
        --num_rollouts 50 --enable_cameras --headless

``XLA_PYTHON_CLIENT_PREALLOCATE=false`` is not optional when Isaac Sim shares the GPU: JAX
otherwise grabs ~75% of VRAM at import, Isaac then fails to initialize, and the error it reports
("no supported devices found for platform CUDA", or an opaque physics backend failure) names
neither memory nor JAX.
"""

import argparse
import logging
import pathlib
import sys

# The robotics checkout that owns policies/pi05/softmimicgen_configs.py.
DEFAULT_ROBOTICS_ROOT = "/home/benjamin.li/src/robotics"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Serve a fine-tuned YAM towel policy.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Checkpoint step directory, containing params/ and assets/.")
    parser.add_argument("--config_name", type=str, default="pi05_yam_towel",
                        help="Registered openpi TrainConfig name.")
    parser.add_argument("--port", type=int, default=8000, help="Websocket port.")
    parser.add_argument("--robotics_root", type=str, default=DEFAULT_ROBOTICS_ROOT,
                        help="Checkout providing policies/pi05/softmimicgen_configs.py.")
    parser.add_argument("--default_prompt", type=str, default="fold the towel",
                        help="Used when an observation carries no prompt of its own.")
    args = parser.parse_args()

    checkpoint = pathlib.Path(args.checkpoint_dir)
    for required in ("params", "assets"):
        if not (checkpoint / required).exists():
            raise FileNotFoundError(
                f"{checkpoint / required} is missing. Serving needs both params/ and assets/ "
                "(the latter carries norm_stats.json); see fetch_checkpoint.sh."
            )

    # Register our config into openpi's registry before serve_policy resolves it by name.
    sys.path.insert(0, args.robotics_root)
    from policies.pi05 import softmimicgen_configs as smg_configs

    # assets_base_dir is irrelevant here: create_trained_policy loads norm stats from the
    # checkpoint's own assets/ directory, not from this path.
    added = smg_configs.register(assets_base_dir=str(checkpoint / "assets"))
    logger.info("registered configs: %s", sorted(added))

    from openpi.training import config as _config

    if args.config_name not in _config._CONFIGS_DICT:  # noqa: SLF001
        raise ValueError(f"{args.config_name!r} did not register; have {sorted(_config._CONFIGS_DICT)[:5]}...")

    import openpi.serving.websocket_policy_server as _server
    from openpi.policies import policy_config as _policy_config

    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config_name), str(checkpoint), default_prompt=args.default_prompt
    )
    logger.info("serving %s from %s on port %d", args.config_name, checkpoint, args.port)
    _server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port).serve_forever()


if __name__ == "__main__":
    main()
