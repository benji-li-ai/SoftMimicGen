# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot the low-dimensional datastreams of a demo, annotated with its subtask boundaries.

Reads the HDF5 directly, so it works on recorded, annotated, and generated datasets alike and does not
require the simulator. Panels that are absent from a given dataset are skipped, which keeps the script
usable across the single-arm and bimanual tasks.

Args:
    --dataset_file            Dataset file to read. (required)
    --demo                    Demo key to plot, e.g. demo_3. (default: the first demo)
    --out                     Output image path. (default: outputs/plots/<dataset>_<demo>.png)
"""

import argparse
import os

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # isort: skip


parser = argparse.ArgumentParser(description="Plot the datastreams of a demo.")
parser.add_argument("--dataset_file", type=str, required=True, help="Dataset file to read.")
parser.add_argument("--demo", type=str, default=None, help="Demo key to plot. Defaults to the first demo.")
parser.add_argument("--out", type=str, default=None, help="Output image path.")
args_cli = parser.parse_args()


def subtask_boundaries(obs: h5py.Group) -> dict[str, int]:
    """Return the first timestep at which each subtask termination signal fires."""
    boundaries = {}
    if "datagen_info" in obs and "subtask_term_signals" in obs["datagen_info"]:
        for signal, values in obs["datagen_info"]["subtask_term_signals"].items():
            fired = np.flatnonzero(np.asarray(values).reshape(-1))
            if fired.size:
                boundaries[signal] = int(fired[0])
    return dict(sorted(boundaries.items(), key=lambda item: item[1]))


def collect_panels(demo: h5py.Group) -> list[tuple[str, np.ndarray, str | None]]:
    """Build the list of (title, array, per-column labels) panels available in this demo."""
    obs = demo["obs"]
    panels = []

    for robot in ("robot0", "robot1"):
        if f"{robot}_eef_pos" in obs:
            panels.append((f"{robot} eef pos (xyz)", np.asarray(obs[f"{robot}_eef_pos"]), "xyz"))

    grippers = [np.asarray(obs[key]) for key in (f"{r}_gripper_qpos" for r in ("robot0", "robot1")) if key in obs]
    if grippers:
        panels.append(("gripper qpos", np.concatenate(grippers, axis=1), None))

    actions = np.asarray(demo["actions"])
    panels.append((f"actions ({actions.shape[1]})", actions, None))

    nodal = obs.get("datagen_info", {}).get("object_nodal_position") if "datagen_info" in obs else None
    if nodal is not None:
        for name in nodal.keys():
            panels.append((f"{name} nodal centroid (xyz)", np.asarray(nodal[name]).mean(axis=1), "xyz"))

    return panels


def main():
    with h5py.File(args_cli.dataset_file, "r") as dataset_file:
        data = dataset_file["data"]
        demo_keys = sorted(data.keys(), key=lambda key: int(key.split("_")[-1]))
        demo_key = args_cli.demo or demo_keys[0]
        if demo_key not in data:
            raise ValueError(f"Demo '{demo_key}' not in dataset. Available demos: {demo_keys}")

        demo = data[demo_key]
        num_steps = demo.attrs["num_samples"]
        boundaries = subtask_boundaries(demo["obs"])
        panels = collect_panels(demo)

        figure, axes = plt.subplots(len(panels), 1, figsize=(13, 2.4 * len(panels)), sharex=True, squeeze=False)
        axes = axes[:, 0]
        for axis, (title, values, labels) in zip(axes, panels):
            for column in range(values.shape[1]):
                axis.plot(values[:, column], lw=1, label=labels[column] if labels else None)
            axis.set_title(title, fontsize=9, loc="left")
            axis.grid(alpha=0.25)
            if labels:
                axis.legend(fontsize=7, ncol=3, loc="upper right")
            for start in boundaries.values():
                axis.axvline(start, color="crimson", ls="--", lw=0.8)
        for signal, start in boundaries.items():
            axes[0].annotate(
                signal, (start, axes[0].get_ylim()[1]), fontsize=7, color="crimson", rotation=90, va="top", ha="right"
            )
        axes[-1].set_xlabel(f"timestep (of {num_steps})")
        figure.suptitle(
            f"{os.path.basename(args_cli.dataset_file)} :: {demo_key}  (success={demo.attrs.get('success')})",
            fontsize=10,
        )
        figure.tight_layout()

        out_path = args_cli.out
        if out_path is None:
            dataset_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
            out_path = os.path.join("outputs", "plots", f"{dataset_name}_{demo_key}.png")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        figure.savefig(out_path, dpi=120)

    print(f"Plot saved to: {out_path}")
    print(f"Subtask ends: {boundaries}")


if __name__ == "__main__":
    main()
