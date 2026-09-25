# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import math
import time
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLMimicEnv
    from isaaclab.managers import TerminationTermCfg

    from softmimicgen.datagen.data_generator import DataGenerator


@dataclass(frozen=True)
class GenerationLimits:
    """Optional run-wide bounds, independent of the task's success/attempt target.

    ``max_attempts`` caps attempts started across all environments. ``max_steps``
    counts vector ``env.step`` calls, not physics substeps or per-environment steps.
    ``timeout_seconds`` measures wall time in ``env_loop``, including queue waits,
    but excluding simulator startup and source loading. It is checked between
    synchronous simulator/generator operations, which cannot be preempted.
    """

    max_attempts: int | None = None
    max_steps: int | None = None
    timeout_seconds: float | None = None

    def __post_init__(self):
        for name in ("max_attempts", "max_steps"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.timeout_seconds is not None and (not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")


@dataclass
class GenerationState:
    """Run-local counters shared by the native workers and simulation loop."""

    generation_num_trials: int
    generation_guarantee: bool
    limits: GenerationLimits = field(default_factory=GenerationLimits)
    num_success: int = 0
    num_failures: int = 0
    num_attempts: int = 0
    num_started: int = 0
    num_steps: int = 0
    stop_reason: str | None = None
    started_at: float | None = None
    elapsed_seconds: float = 0.0
    active_env_ids: set[int] = field(default_factory=set)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)

    def check_stop(self) -> str | None:
        """Latch the first normal or bounded stop condition."""
        if self.stop_reason is None:
            target_count = self.num_success if self.generation_guarantee else self.num_attempts
            if target_count >= self.generation_num_trials:
                self.stop_reason = "generation_num_trials"
            elif self.limits.max_attempts is not None and self.num_attempts >= self.limits.max_attempts:
                self.stop_reason = "max_attempts"
            elif self.limits.max_steps is not None and self.num_steps >= self.limits.max_steps:
                self.stop_reason = "max_steps"
            elif (
                self.limits.timeout_seconds is not None
                and self.started_at is not None
                and time.monotonic() - self.started_at >= self.limits.timeout_seconds
            ):
                self.stop_reason = "timeout_seconds"
        if self.stop_reason is not None:
            self.stopped.set()
        return self.stop_reason

    def can_start_attempt(self) -> bool:
        """Reserve no more than the attempt target/cap, even with multiple workers."""
        if self.limits.max_attempts is not None and self.num_started >= self.limits.max_attempts:
            return False
        return self.generation_guarantee or self.num_started < self.generation_num_trials

    def summary(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "num_success": self.num_success,
            "num_failures": self.num_failures,
            "num_attempts": self.num_attempts,
            "num_started": self.num_started,
            "num_interrupted": self.num_started - self.num_attempts,
            "num_steps": self.num_steps,
            "elapsed_seconds": self.elapsed_seconds,
        }


def success_term_metadata(success_term: TerminationTermCfg) -> dict[str, Any]:
    """Describe the actual success callable, including its effective defaults."""
    signature = inspect.signature(success_term.func)
    bound = signature.bind(None, **success_term.params)
    bound.apply_defaults()
    parameters = dict(bound.arguments)
    parameters.pop(next(iter(signature.parameters)))  # The environment is supplied at evaluation time.
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            parameters.update(parameters.pop(name, {}))
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL and not parameters.get(name):
            parameters.pop(name, None)
    return {
        "function": f"{success_term.func.__module__}.{success_term.func.__qualname__}",
        "parameters": parameters,
    }


async def run_data_generator(
    env: ManagerBasedRLMimicEnv,
    env_id: int,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    data_generator: DataGenerator,
    success_term: TerminationTermCfg,
    state: GenerationState,
    pause_subtask: bool = False,
):
    """Run native generation, checking completion before starting another attempt."""
    while state.check_stop() is None:
        if not state.can_start_attempt():
            # Other workers own the remaining attempts. Do not complete normally
            # until the shared target is reached, or enqueue an extra reset/action.
            await state.stopped.wait()
            return
        state.num_started += 1
        state.active_env_ids.add(env_id)
        try:
            results = await data_generator.generate(
                env_id=env_id,
                success_term=success_term,
                env_reset_queue=env_reset_queue,
                env_action_queue=env_action_queue,
                pause_subtask=pause_subtask,
            )
        finally:
            state.active_env_ids.remove(env_id)
        if bool(results["success"]):
            state.num_success += 1
        else:
            state.num_failures += 1
        state.num_attempts += 1


def env_loop(
    env: ManagerBasedRLMimicEnv,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    asyncio_event_loop: asyncio.AbstractEventLoop,
    generation_tasks: Sequence[asyncio.Task],
    state: GenerationState,
) -> dict[str, Any]:
    """Drive one native vector environment and own cancellation of its workers.

    Bounds and task results are checked on every queue-wait iteration, before
    resetting or stepping. A final step is acknowledged to the workers before
    stopping, so a completed episode is exported by ``DataGenerator`` without
    starting another attempt. Interrupted attempts are not completed demos and
    are not exported. The caller owns closing the environment and Kit; the
    existing asyncio loop is borrowed, never replaced or closed here.
    """
    prev_num_attempts = 0
    state.started_at = time.monotonic()
    try:
        env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
        actions = torch.zeros(env.action_space.shape, device=env.device)
        with torch.inference_mode():
            while True:
                # Let generators consume the last step and publish their result
                # before examining limits, resets, or the next action batch.
                asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                stop_reason = state.check_stop()
                for task in generation_tasks:
                    if task.done():
                        task.result()  # Exceptions always win over a simultaneous stop.
                        if stop_reason is None:
                            raise RuntimeError(f"Data generation task {task.get_name()!r} completed unexpectedly.")

                if prev_num_attempts != state.num_attempts:
                    prev_num_attempts = state.num_attempts
                    success_rate = 100 * state.num_success / state.num_attempts
                    print(
                        f"{state.num_success}/{state.num_attempts} ({success_rate:.1f}%) successful demos generated"
                        " by mimic",
                        flush=True,
                    )
                if stop_reason is not None:
                    print(f"Generation stopped: {stop_reason}", flush=True)
                    break
                if env.sim.is_stopped():
                    state.stop_reason = "simulation_stopped"
                    break

                while not env_reset_queue.empty() and state.check_stop() is None:
                    env_id_tensor[0] = env_reset_queue.get_nowait()
                    env.reset(env_ids=env_id_tensor)
                    env_reset_queue.task_done()

                if state.check_stop() is not None:
                    continue
                if not state.active_env_ids or env_action_queue.qsize() != len(state.active_env_ids):
                    continue

                # Workers with no remaining attempt reservation are idle. They do
                # not participate in the action barrier of the vector environment.
                if len(state.active_env_ids) != env.num_envs:
                    actions.zero_()
                num_actions = env_action_queue.qsize()
                for _ in range(num_actions):
                    env_id, action = env_action_queue.get_nowait()
                    actions[env_id] = action
                if state.check_stop() is not None:
                    continue
                env.step(actions)
                state.num_steps += 1
                for _ in range(num_actions):
                    env_action_queue.task_done()
    except KeyboardInterrupt:
        state.stop_reason = "keyboard_interrupt"
        raise
    except BaseException:
        state.stop_reason = "error"
        raise
    finally:
        for task in generation_tasks:
            if not task.done():
                task.cancel()
        if generation_tasks:
            asyncio_event_loop.run_until_complete(asyncio.gather(*generation_tasks, return_exceptions=True))
        state.elapsed_seconds = time.monotonic() - state.started_at
    return state.summary()


def setup_env_config(
    env_name: str,
    output_dir: str,
    output_file_name: str,
    num_envs: int,
    device: str,
    generation_num_trials: int | None = None,
    keep_failed: bool = False,
) -> tuple[Any, Any]:
    """Configure the registered task and its native action/state recorders.

    ``generation_num_trials`` preserves the task's success/attempt semantics.
    ``keep_failed`` enables failed exports without disabling a task's own setting.
    """
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)
    if generation_num_trials is not None:
        env_cfg.datagen_config.generation_num_trials = generation_num_trials
    if keep_failed:
        env_cfg.datagen_config.generation_keep_failed = True
    env_cfg.env_name = env_name

    # Extract the registered success predicate before disabling terminations.
    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is None:
        raise NotImplementedError("No success termination term was found in the environment.")
    env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    if env_cfg.datagen_config.generation_keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    return env_cfg, success_term


def setup_async_generation(
    env: Any,
    num_envs: int,
    input_file: str,
    success_term: Any,
    pause_subtask: bool = False,
    limits: GenerationLimits | None = None,
) -> dict[str, Any]:
    """Load the native source pool and schedule workers on the existing event loop."""
    from softmimicgen.datagen.data_generator import DataGenerator
    from softmimicgen.datagen.datagen_info_pool import DataGenInfoPool

    asyncio_event_loop = asyncio.get_event_loop()
    env_reset_queue = asyncio.Queue()
    env_action_queue = asyncio.Queue()
    shared_datagen_info_pool_lock = asyncio.Lock()
    shared_datagen_info_pool = DataGenInfoPool(env, env.cfg, env.device, asyncio_lock=shared_datagen_info_pool_lock)
    shared_datagen_info_pool.load_from_dataset_file(input_file)
    print(f"Loaded {shared_datagen_info_pool.num_datagen_infos} to datagen info pool")

    state = GenerationState(
        generation_num_trials=env.cfg.datagen_config.generation_num_trials,
        generation_guarantee=env.cfg.datagen_config.generation_guarantee,
        limits=limits if limits is not None else GenerationLimits(),
    )
    data_generator = DataGenerator(env=env, src_demo_datagen_info_pool=shared_datagen_info_pool)
    tasks = [
        asyncio_event_loop.create_task(
            run_data_generator(
                env,
                i,
                env_reset_queue,
                env_action_queue,
                data_generator,
                success_term,
                state,
                pause_subtask=pause_subtask,
            ),
            name=f"data_generator_{i}",
        )
        for i in range(num_envs)
    ]
    return {
        "tasks": tasks,
        "event_loop": asyncio_event_loop,
        "reset_queue": env_reset_queue,
        "action_queue": env_action_queue,
        "info_pool": shared_datagen_info_pool,
        "state": state,
    }
