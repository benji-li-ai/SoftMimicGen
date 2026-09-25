# Native YAM cell recording and generation

## Current implementation

The registered cell task now owns the package-backed scene. Recording, annotation and
generation use the native SoftMimicGen/Isaac Lab interfaces; there is no separate cell
smoke runner, CPU-sampling adapter, or copied cell MDP implementation.

Reference implementation: NVlabs/SoftMimicGen `c9d146b` and its Isaac Lab fork `25c69e0`.
The genuine environment-origin and async-failure fixes remain in place.

- Recording task: `Isaac-Fold-Cell-Towel-Yam-Joint-v0`
- Annotation/generation task: `Isaac-Fold-Cell-Towel-Yam-Joint-Mimic-v0`
- DR and perception variants are registered for both task and Mimic use.
- Assets come from the installed `yam_assets` package. Live spawners apply its physics
  helpers and preserve original material bindings without saving into source USD files.
- The cell retains its physical mounts, cloth, package actuators and calibrated cameras.
  Optional extra pad boxes are not enabled; this is the validated package-default setup.
- Surface-deformable physics is cloned normally with `replicate_physics=False`; the
  installed PhysX replication shortcut does not support this cloth's FEM schemas.

Install the supplied package in the same environment before using a cell task:

```bash
conda activate softmimicgen
python -m pip install --no-deps --no-build-isolation -e ../yam_assets
```

## Shared recording contract

Use native recorder and observation helpers rather than constructing lookalike states.

| Field/setting | Contract |
|---|---|
| Physics/control | `dt=1/120`, `decimation=4`: 30 Hz control |
| Rendering | Once per control step; three 640x480 sensors, native saved observation crops |
| Actions | 14 values: left arm 6, left gripper 1, right arm 6, right gripper 1 |
| Gripper commands | 0 closed, 1 open; targets +0.04/-0.04 m; native /0.04 observation normalization |
| Measured articulation state | All 8 actual joints per arm, real joint velocities, valid root pose/quaternion |
| Cloth state | All 5,989 nodes and solver-readback nodal velocities |
| Sampled cloth annotation | Native 100-node `randperm(seed=42)` on the points' device; CUDA for GPU generation |
| EE annotations | Native YAM link_6 convention, including measured FK reused as target_eef_pose |
| Workstation | The registered rigid body is the package's TableSurface frame |
| Success | Native fold ratio <0.5 plus both grippers open; actual parameters are reported |

The workstation configuration's outer reference is `{ENV_REGEX_NS}/Workstation`;
its actual rigid body is `{ENV_REGEX_NS}/Workstation/Workstation/TableSurface`.
Its nominal environment-relative position is `(0.655, 0, 0.745)` with identity rotation,
matching collection readout from `/World/Workstation/TableSurface`. Walls and mounting
deck remain static; resetting the table does not move the outer backdrop.

Native recording order:

1. Reset/restore, settle if configured, and forward the simulator; record live `initial_state`.
2. Process a control action; record pre-step `actions`, cached `obs`, and annotation data.
3. Execute four physics steps; record post-step `states` and `processed_actions`.
4. Record the completed episode before resetting it.

A stored `nodal_subsample_indices` attribute is useful for checking compatibility, but
native `DataGenInfoPool` does not consume it. New annotations must use the same IDs as
native CUDA sampling. Finite-difference velocities or reconstructed finger values are
approximations, not native measured state.

A reference array from this GPU environment is saved at
`outputs/native_cutover_20260915/native_cuda_node_indices.json`, including node count,
seed and PyTorch version. A CPU-side collector can apply those same indices directly;
the native generator still computes its own device permutation, so verify parity after
changing the PyTorch runtime or cloth node ordering.

## Grasp-centered cloth trajectory transfer

Cell Mimic configurations provide `eef_warp_offsets` for both arms: `(0, 0, 0.1347)`
metres in the recorded `link_6` axes. This is the cell asset's authored
`sites/grasp_site/grasp_site` position, not its zero-offset `tcp_site`.

Both nodal-registration implementations warp `p + R d`, evaluate the warp Jacobian
there, and return the wrist target `F(p + R d) - R_new d`. The physical tool offset
does not shrink with the cloth. Recordings, waypoint interpolation and native IK
retain their `link_6` convention. Only cell Mimic tasks (including their DR and
perception variants) enable the offsets; tasks without the mapping retain their
existing warp origin. Rigid transforms remain equivalent, but nonrigid TPS can
change both grasp and folding segments even at 1.0x cloth scale.

Focused CPU coverage exercises scale/rotation geometry, both registration methods,
grasp-point Jacobians, disabled rotation, zero offsets, dtype/device preservation,
and input nonmutation. Run with the existing generation/export regressions:

```bash
python -m pytest tests/test_nodal_registration.py tests/test_generation.py tests/test_demo_to_video.py -q
```

Native validation used `yam_towel_demo_0917-native1.hdf5`, one bounded attempt per
condition, the existing fold-only gate, and 219 steps per rollout:

| Cloth condition | Native success | Minimum / final fold ratio |
|---|---|---|
| 1.0x, zero yaw | Yes | 0.260 / 0.398 |
| 0.7x, zero yaw | No | 0.776 / 0.853 |
| 0.7x, +30 degrees yaw | Yes, transient only | 0.461 / 0.608 |

The corrected zero-yaw 0.7x grasp centers descended outside the live cloth edge
by 15.9 / 21.6 mm (left/right), rather than inside by 24.2 / 13.5 mm in the previous
saved run, measured at the first downward crossing of grasp-center z = 0.785 m.
This verifies the frame correction, not reliable small-cloth folding. The rotated
run does not retain a passing final fold; its left arm also crosses that height
later, during inward motion, so that height is not a universal contact-time marker.
Fixed finger geometry, feedforward contact timing, and the any-step success gate
are unchanged. These three trials are smoke checks, not a success-rate estimate.

Reports, geometry measurements, contact sheets and native camera videos are under
`outputs/grasp_frame_validation/{scale100,scale070,scale070_yaw30}/`. Temporary test
perturbations were restored to the original 0.7x / zero-yaw cell configuration.

## Canonical bounded generation

This command exercises the registered task with the shipped native annotated reference:

```bash
OMNI_KIT_ACCEPT_EULA=YES timeout --kill-after=5s 240s \
python -u scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-Fold-Cell-Towel-Yam-Joint-Mimic-v0 \
    --input_file datasets/annotated_dataset/annotated_dataset_yam_towel.hdf5 \
    --output_file outputs/cell_native_smoke/rollouts.hdf5 \
    --generation_num_trials 1 --max_attempts 1 --max_steps 1000 \
    --timeout_seconds 180 --keep_failed \
    --num_envs 1 --headless --enable_cameras --device cuda
```

Replace the input with a fresh, native-compatible annotated cell recording for the real
collection acceptance check. The shipped reference uses the stock scene and is an
integration check, not a benchmark of cell folding success.

- `--max_attempts` caps attempts started across all environments, regardless of success.
- `--max_steps` caps total vector `env.step` calls, not physics substeps or per-attempt steps.
- `--timeout_seconds` includes generation queue waits but excludes startup/source loading;
  synchronous operations cannot be preempted. The external timeout also bounds startup.
- `--keep_failed` exports completed failures to `rollouts_failed.hdf5` using the native recorder.
- Interrupted attempts are reported separately and are not exported as completed demos.
- `rollouts.generation.json` records effective success parameters, timing, limits and counters.

Video conversion uses saved native observations and does not launch the simulator.
Choose the populated successful or failed HDF5 file:

```bash
python scripts/tools/demo_to_video.py \
    --dataset_file outputs/cell_native_smoke/rollouts_failed.hdf5 \
    --demo demo_0 --fps 30 --out outputs/cell_native_smoke/rollout_00.mp4
```

Camera arrays are read in bounded batches to avoid repeatedly decompressing temporal
HDF5 chunks. Frame order and native 30 Hz playback are preserved.

## Verified locally

- 52 focused CPU tests pass: nodal grasp-frame transfer, generation limits, queue/error
  handling, effective success defaults, and compressed camera export frame continuity.
- Two-environment reset/step check: correct relative arm mounts, TableSurface state and
  object-pose agreement, native CUDA sampling, 14 actions/8 joints and three nonblack cameras.
- All six base/Mimic/DR/perception configurations resolve to the package-backed assets.
- Canonical reference generation completed exactly one attempt: 773 steps, zero interrupted
  attempts, stopped at `max_attempts`. The native 0.5 predicate classified it as a failure.
- Native exporter saved finite measured articulation/deformable states and camera observations.
- Video: `outputs/native_cutover_20260915/rollout_00.mp4`, 773 frames at 30 Hz.
- Detailed outcome: `outputs/native_cutover_20260915/rollouts.generation.json`.

## Before substantial new collection

Capture one short new episode and verify native recording -> annotation -> canonical
generation end to end. Confirm real state readback, hook ordering, 30 Hz timing, native
gripper conventions and CUDA node identity on the collection machine. That fresh-recording
acceptance remains pending; the local reference smoke cannot substitute for it.

Old teleop files and previous outputs are retained as archives. Their CPU sampling,
60 Hz timing and incomplete state fields are not supported by a legacy runtime adapter,
and no migration or in-place dataset repair is performed by this cutover.
