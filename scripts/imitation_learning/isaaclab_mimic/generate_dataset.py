# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""


"""Launch Isaac Sim Simulator first."""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--generation_num_trials",
    type=positive_int,
    default=None,
    help="Task target: successful demos if generation_guarantee is enabled, otherwise completed attempts.",
)
parser.add_argument(
    "--num_envs", type=positive_int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--max_attempts",
    type=positive_int,
    default=None,
    help="Optional cap on attempts started across all environments, regardless of success.",
)
parser.add_argument(
    "--max_steps",
    type=positive_int,
    default=None,
    help="Optional total vector env.step tick cap for the run (not per-attempt steps or physics substeps).",
)
parser.add_argument(
    "--timeout_seconds",
    type=positive_seconds,
    default=None,
    help=(
        "Optional generation-loop wall-time cap, including queue waits but excluding Kit startup/source loading. "
        "Checked between synchronous operations; does not preempt a running simulator call."
    ),
)
parser.add_argument(
    "--keep_failed",
    action="store_true",
    help="Export completed failed attempts to the native *_failed.hdf5 file. Interrupted attempts are not exported.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if not Path(args_cli.input_file).is_file():
    parser.error(f"Source dataset does not exist: {args_cli.input_file}")

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import inspect
import json
import numpy as np
import random
import torch
import traceback
from dataclasses import asdict

import omni

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils.dict import class_to_dict, replace_slices_with_strings

import softmimicgen.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import softmimicgen.envs.pinocchio_envs  # noqa: F401

from softmimicgen.datagen.generation import (
    GenerationLimits,
    env_loop,
    setup_async_generation,
    setup_env_config,
    success_term_metadata,
)
from softmimicgen.datagen.utils import get_env_name_from_dataset, setup_output_paths

import softmimicgen_tasks  # noqa: F401


def json_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize generation metadata of type {type(value).__name__}")


def report_json(report):
    return json.dumps(replace_slices_with_strings(class_to_dict(report)), default=json_array, indent=2)


def write_report(path, report):
    path.write_text(report_json(report) + "\n")


def main():
    limits = GenerationLimits(args_cli.max_attempts, args_cli.max_steps, args_cli.timeout_seconds)
    output_dir, output_file_name = setup_output_paths(str(Path(args_cli.output_file).resolve()))
    task_name = args_cli.task.split(":")[-1] if args_cli.task else None
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
        keep_failed=args_cli.keep_failed,
    )
    report_path = Path(output_dir) / f"{output_file_name}.generation.json"
    report = None
    async_components = None
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    try:
        if not isinstance(env, ManagerBasedRLMimicEnv):
            raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")
        if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
            omni.log.warn(
                f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
                "is deprecated. Please update the API to take action_noise_dict instead."
            )

        report = {
            "task": env_name,
            "input_file": str(Path(args_cli.input_file).resolve()),
            "output_file": str(Path(output_dir) / f"{output_file_name}.hdf5"),
            "failed_output_file": (
                str(Path(output_dir) / f"{output_file_name}_failed.hdf5")
                if env.cfg.datagen_config.generation_keep_failed
                else None
            ),
            "num_envs": env.num_envs,
            "enable_cameras": args_cli.enable_cameras,
            "seed": env.cfg.datagen_config.seed,
            "success_term": success_term_metadata(success_term),
            "physics_dt": env.physics_dt,
            "step_dt": env.step_dt,
            "decimation": env.cfg.decimation,
            "render_interval": env.cfg.sim.render_interval,
            "generation_num_trials": env.cfg.datagen_config.generation_num_trials,
            "generation_guarantee": env.cfg.datagen_config.generation_guarantee,
            "generation_keep_failed": env.cfg.datagen_config.generation_keep_failed,
            "limits": asdict(limits),
            "limit_semantics": {
                "max_attempts": "Attempts started across all environments, regardless of success.",
                "max_steps": "Total vector env.step ticks; not per-attempt steps or physics substeps.",
                "timeout_seconds": (
                    "Generation-loop wall time, including queue waits; excludes Kit startup and source "
                    "loading. Checked between synchronous operations, which cannot be preempted."
                ),
                "interrupted_attempts": "Incomplete; excluded from completed-attempt counters and not exported.",
            },
            "summary": None,
        }
        write_report(report_path, report)
        print("GENERATION_CONFIG " + report_json(report), flush=True)

        random.seed(env.cfg.datagen_config.seed)
        np.random.seed(env.cfg.datagen_config.seed)
        torch.manual_seed(env.cfg.datagen_config.seed)
        env.reset()
        async_components = setup_async_generation(
            env=env,
            num_envs=args_cli.num_envs,
            input_file=args_cli.input_file,
            success_term=success_term,
            pause_subtask=args_cli.pause_subtask,
            limits=limits,
        )
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["event_loop"],
            generation_tasks=async_components["tasks"],
            state=async_components["state"],
        )
    except BaseException as error:
        if report is not None:
            report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            if report is not None:
                if async_components is not None:
                    report["summary"] = async_components["state"].summary()
                write_report(report_path, report)
                print("GENERATION_SUMMARY " + report_json(report["summary"]), flush=True)
                print(f"Generation report: {report_path}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except BaseException:
        # Kit shutdown can exit before an escaping exception gets printed.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
