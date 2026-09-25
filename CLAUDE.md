# Working notes for this repository

## Isaac Sim processes do not reliably exit — always verify, then clean up

An Isaac Sim run that raises an exception, or whose wrapper is killed by a timeout, frequently
leaves the Python process alive and hung in Omniverse shutdown while still holding several GB of
VRAM. A tool reporting "background command completed" is **not** evidence the process died.

This has already caused a confusing failure: three leaked runs held 11.5 GB of VRAM between them,
and a later run died with `Exception: Failed to get DOF velocities from backend` during
articulation initialization — a resource-contention error that looks nothing like its cause.

After every simulation job, and before drawing conclusions from a failure:

```bash
pgrep -af generate_dataset.py            # or eval_policy.py, replay_demos.py, ...
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

`kill -TERM` is usually not enough; leaked Isaac processes need `kill -9`. When several runs are
in flight, identify each by its command line before killing anything:

```bash
tr '\0' ' ' < /proc/<pid>/cmdline
```

## Check GPU occupancy before launching a simulation job

This machine has a single RTX 5090 (32 GB) and simulation jobs may be launched by more than one
person or agent at a time. A single `generate_dataset.py` run uses roughly 5–17 GB of VRAM
depending on `--num_envs` and camera count, so two runs can fit but three usually cannot, and the
failure mode is an obscure backend error rather than a clean out-of-memory message.

Check free memory first, and never kill a process you did not start without matching its command
line — another agent's run may represent hours of generation.

## Other environment notes

- Isaac Sim blocks on an interactive EULA prompt unless `OMNI_KIT_ACCEPT_EULA=YES` is exported.
- Python buffers stdout when piped, so `python script.py | tail -20` can discard a script's own
  prints entirely. Use `python -u` and redirect to a file when the output matters.
- `[Error] omni.physicsschema.plugin` messages about `lf_rot` / `rf_rot` joints referencing
  non-existent prims are pre-existing in the YAM asset and unrelated to whatever you changed. The
  grippers work because `left_finger` / `right_finger` are the actuated joints.
- Fixes to `third_party/` do not survive reinstall — that tree is gitignored and re-cloned by
  `softmimicgen.sh`. Record them in `patches/` instead; see `patches/README.md`.
