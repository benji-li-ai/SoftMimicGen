# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert SoftMimicGen HDF5 demonstrations into a LeRobot dataset for openpi fine-tuning.

**Run this in the openpi environment, not the ``softmimicgen`` conda environment.** openpi pins a
specific ``lerobot`` commit whose dependencies conflict with Isaac Sim's, so installing lerobot
alongside Isaac Sim risks breaking the simulator. This script deliberately imports nothing from
Isaac Lab -- it only needs ``h5py``, ``numpy``, ``tqdm`` and ``lerobot``::

    uv run python /path/to/SoftMimicGen/scripts/imitation_learning/openpi/convert_to_lerobot.py \
        --input datasets/generated_dataset/'yam_towel_*.hdf5' \
        --repo_id your-org/yam_towel_sim

Camera key names must match ``CAMERA_KEY_MAP`` in ``eval_policy.py`` and the repack transform of
the openpi ``TrainConfig``; all three are listed in ``yam_towel_train_config.py``.

Real demonstrations can be appended into the same dataset later with ``--resume``, which is what
the sim-real co-training setting needs -- there is no separate path for them.
"""

import argparse
import glob
import importlib
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Environment observation key -> LeRobot feature key. The wrist views are square while the top
# view keeps the camera's native 4:3; openpi resizes every camera to 224x224 at train time, so the
# shapes need not match each other.
CAMERA_KEYS = {
    "top": "cam_high",
    "left": "cam_left_wrist",
    "right": "cam_right_wrist",
}

# 6 arm joints + 1 coupled gripper, per arm.
JOINT_NAMES = [f"{side}_{name}" for side in ("left", "right") for name in [f"joint{i}" for i in range(1, 7)] + ["gripper"]]

CONTROL_HZ = 30


def build_features(hdf5_path: Path, image_dtype: str) -> dict:
    """Derive the LeRobot feature spec from an actual demo, rather than hard-coding shapes."""
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data"][next(iter(handle["data"].keys()))]
        obs = demo["obs"]

        features = {}
        for env_key, feature_key in CAMERA_KEYS.items():
            if env_key not in obs:
                raise KeyError(
                    f"'{env_key}' is missing from {hdf5_path}. The dataset was generated without"
                    " --enable_cameras, so it cannot train a visuomotor policy."
                )
            height, width, channels = obs[env_key].shape[1:]
            features[f"observation.images.{feature_key}"] = {
                "dtype": image_dtype,
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            }

        state_dim = obs["state"].shape[1]
        action_dim = demo["actions"].shape[1]
        names = JOINT_NAMES if state_dim == len(JOINT_NAMES) else [f"dim_{i}" for i in range(state_dim)]

        features["observation.state"] = {"dtype": "float32", "shape": (state_dim,), "names": names}
        features["action"] = {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": names if action_dim == len(names) else [f"dim_{i}" for i in range(action_dim)],
        }
    return features


def import_lerobot():
    """Import LeRobotDataset across lerobot's package reorganisations.

    Two independent axes vary between the versions this project has to support -- lerobot 0.4 in
    the analysis environment and the older pinned revision openpi depends on:

    * the module moved from ``lerobot.common.datasets`` to ``lerobot.datasets``
    * the dataset-root constant is named ``HF_LEROBOT_HOME`` on both recent revisions but
      ``LEROBOT_HOME`` on older ones

    Datasets written by one version are not necessarily readable by the other, so run this
    converter in the same environment that will consume the output.
    """
    module = None
    for name in ("lerobot.datasets.lerobot_dataset", "lerobot.common.datasets.lerobot_dataset"):
        try:
            module = importlib.import_module(name)
            break
        except ImportError:
            continue
    if module is None:
        raise ImportError("Could not import LeRobotDataset from either lerobot layout.")

    for const in ("HF_LEROBOT_HOME", "LEROBOT_HOME"):
        home = getattr(module, const, None)
        if home is not None:
            break
    else:
        raise ImportError(f"{module.__name__} exports neither HF_LEROBOT_HOME nor LEROBOT_HOME.")

    return module.LeRobotDataset, home


def save_episode(dataset, task: str):
    """Finish an episode across lerobot versions that moved ``task`` between calls.

    In lerobot 0.4 the task is supplied per frame via ``add_frame`` and ``save_episode`` takes no
    task argument; older versions took it here.
    """
    try:
        dataset.save_episode()
    except TypeError:
        dataset.save_episode(task=task)


def convert(args):
    paths = sorted(Path(p) for pattern in args.input for p in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No HDF5 files matched {args.input}")
    print(f"Converting {len(paths)} file(s): {[p.name for p in paths]}")

    LeRobotDataset, lerobot_home = import_lerobot()

    output_dir = Path(lerobot_home) / args.repo_id
    if args.resume:
        dataset = LeRobotDataset(args.repo_id)
    else:
        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"{output_dir} exists. Pass --overwrite to replace or --resume to append.")
            shutil.rmtree(output_dir)
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            robot_type=args.robot_type,
            fps=args.fps,
            features=build_features(paths[0], args.image_dtype),
        )

    num_episodes = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
            for demo_key in tqdm(demo_keys, desc=path.name):
                demo = data[demo_key]
                # Generated datasets are exported successes-only, but a dataset produced with
                # generation_keep_failed=True would mix failures in; never train on those silently.
                if not args.include_failures and not bool(demo.attrs.get("success", True)):
                    continue
                if args.max_episodes is not None and num_episodes >= args.max_episodes:
                    break

                obs = demo["obs"]
                images = {feature: np.asarray(obs[env_key]) for env_key, feature in CAMERA_KEYS.items()}
                states = np.asarray(obs["state"], dtype=np.float32)
                actions = np.asarray(demo["actions"], dtype=np.float32)

                for step in range(len(actions)):
                    frame = {f"observation.images.{k}": images[k][step] for k in images}
                    frame["observation.state"] = states[step]
                    frame["action"] = actions[step]
                    # lerobot 0.4 consumes the task per frame and pops it off before validation.
                    frame["task"] = args.task
                    dataset.add_frame(frame)

                save_episode(dataset, args.task)
                num_episodes += 1

    print(f"Wrote {num_episodes} episodes to {output_dir}")
    if args.push_to_hub:
        dataset.push_to_hub(tags=["softmimicgen", "yam", "towel", "sim"], private=True)
        print(f"Pushed to hub as {args.repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert SoftMimicGen HDF5 demos to a LeRobot dataset.")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="HDF5 path(s) or glob(s).")
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset id, e.g. org/yam_towel_sim.")
    parser.add_argument("--task", type=str, default="fold the towel", help="Language instruction for every episode.")
    parser.add_argument("--robot_type", type=str, default="yam_bimanual", help="Robot type recorded in the dataset.")
    parser.add_argument("--fps", type=int, default=CONTROL_HZ, help="Control rate of the demonstrations.")
    parser.add_argument(
        "--image_dtype",
        type=str,
        default="video",
        choices=["video", "image"],
        help="'video' encodes frames to mp4 and is far smaller; 'image' stores PNGs.",
    )
    parser.add_argument("--max_episodes", type=int, default=None, help="Stop after this many episodes.")
    parser.add_argument("--include_failures", action="store_true", help="Also convert episodes marked unsuccessful.")
    parser.add_argument("--resume", action="store_true", help="Append to an existing dataset, e.g. to add real demos.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing dataset of the same name.")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset to the Hugging Face Hub.")
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
