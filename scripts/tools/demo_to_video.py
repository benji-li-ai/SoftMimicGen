# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the recorded camera streams of a demo dataset to an mp4, without launching the simulator.

Datasets recorded or generated with ``--enable_cameras`` store their RGB observations directly in the
HDF5 file, so a rollout can be reviewed by re-encoding those frames instead of replaying the episode.

Args:
    --dataset_file            Dataset file to read. (required)
    --demo                    Demo key to render, e.g. demo_3. (default: the first demo)
    --cameras                 Image observation keys to tile side by side. (default: top left right)
    --fps                     Frame rate of the output video. (default: 30, the control rate)
    --out                     Output mp4 path. (default: outputs/videos/<dataset>_<demo>.mp4)
    --list                    Print the demos and observation keys, then exit.
"""

import argparse
import cv2
import h5py
import numpy as np
import os

import imageio.v2 as imageio

parser = argparse.ArgumentParser(description="Render demo camera streams to an mp4.")
parser.add_argument("--dataset_file", type=str, required=True, help="Dataset file to read.")
parser.add_argument("--demo", type=str, default=None, help="Demo key to render. Defaults to the first demo.")
parser.add_argument(
    "--cameras", type=str, nargs="+", default=["top", "left", "right"], help="Image observation keys to tile."
)
parser.add_argument("--fps", type=int, default=30, help="Frame rate of the output video.")
parser.add_argument("--out", type=str, default=None, help="Output mp4 path.")
parser.add_argument("--list", action="store_true", default=False, help="List demos and observation keys, then exit.")
args_cli = parser.parse_args()


def sorted_demo_keys(data: h5py.Group) -> list[str]:
    """Return the demo keys in recorded order rather than lexicographic order."""
    return sorted(data.keys(), key=lambda key: int(key.split("_")[-1]))


def subtask_boundaries(obs: h5py.Group) -> dict[str, int]:
    """Return the first timestep at which each subtask termination signal fires."""
    boundaries = {}
    if "datagen_info" in obs and "subtask_term_signals" in obs["datagen_info"]:
        for signal, values in obs["datagen_info"]["subtask_term_signals"].items():
            fired = np.flatnonzero(np.asarray(values).reshape(-1))
            if fired.size:
                boundaries[signal] = int(fired[0])
    return dict(sorted(boundaries.items(), key=lambda item: item[1]))


def tile(frames: list[np.ndarray]) -> np.ndarray:
    """Tile frames side by side, scaling each to a common height.

    Cameras in a dataset need not share a resolution -- the top view keeps its native 4:3 while
    the wrist views are square -- so pad-free horizontal concatenation requires a common height.
    """
    height = max(frame.shape[0] for frame in frames)
    scaled = [
        (
            frame
            if frame.shape[0] == height
            else cv2.resize(frame, (round(frame.shape[1] * height / frame.shape[0]), height))
        )
        for frame in frames
    ]
    return np.ascontiguousarray(np.concatenate(scaled, axis=1))


def annotate(frame: np.ndarray, text: str) -> None:
    """Draw an outlined caption onto the frame in place."""
    for color, thickness in ((0, 0, 0), 3), ((255, 255, 255), 1):
        cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness, cv2.LINE_AA)


def main():
    with h5py.File(args_cli.dataset_file, "r") as dataset_file:
        data = dataset_file["data"]
        demo_keys = sorted_demo_keys(data)

        if args_cli.list:
            print(f"{len(demo_keys)} demos in {args_cli.dataset_file}")
            print(f"observation keys: {list(data[demo_keys[0]]['obs'].keys())}")
            for key in demo_keys:
                attrs = data[key].attrs
                print(f"  {key}: {attrs.get('num_samples')} steps, success={attrs.get('success')}")
            return

        demo_key = args_cli.demo or demo_keys[0]
        if demo_key not in data:
            raise ValueError(f"Demo '{demo_key}' not in dataset. Available demos: {demo_keys}")
        obs = data[demo_key]["obs"]

        cameras = [camera for camera in args_cli.cameras if camera in obs and obs[camera].ndim == 4]
        if not cameras:
            raise ValueError(
                f"None of {args_cli.cameras} are image observations in this dataset. The dataset was likely recorded"
                f" without --enable_cameras. Available keys: {list(obs.keys())}"
            )

        num_steps = obs[cameras[0]].shape[0]
        boundaries = subtask_boundaries(obs)

        out_path = args_cli.out
        if out_path is None:
            dataset_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
            out_path = os.path.join("outputs", "videos", f"{dataset_name}_{demo_key}.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        print(f"Rendering {demo_key}: {num_steps} steps, cameras={cameras}, subtask ends={boundaries}")
        writer = imageio.get_writer(out_path, fps=args_cli.fps, macro_block_size=1, quality=8)
        # Native HDF5 camera chunks span many timesteps. Reading one frame at a
        # time repeatedly inflates those chunks beyond h5py's small default cache.
        # Bounded contiguous batches keep memory modest and decode each chunk far
        # fewer times, without changing frame order or playback timing.
        for start in range(0, num_steps, 64):
            stop = min(start + 64, num_steps)
            batches = [np.asarray(obs[camera][start:stop])[..., :3] for camera in cameras]
            for offset in range(stop - start):
                step = start + offset
                frame = tile([batch[offset] for batch in batches])
                caption = " | ".join(cameras) + f"   t={step}/{num_steps}"
                completed = [signal for signal, boundary in boundaries.items() if step >= boundary]
                if completed:
                    caption += "   done: " + ", ".join(completed)
                annotate(frame, caption)
                writer.append_data(frame)
        writer.close()

    print(f"Video saved to: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
