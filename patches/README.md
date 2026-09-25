# Patches to vendored third-party code

`third_party/` is gitignored and re-cloned by `softmimicgen.sh`, so any fix applied there is lost
on a fresh install. Patches that must survive live here and are re-applied with:

```bash
cd third_party/IsaacLab && git apply ../../patches/0001-fix-deformable-env-origin-broadcast.patch
```

Check whether a patch is already applied before re-running (`git diff --stat` in
`third_party/IsaacLab` should show the file as modified).

---

## 0001 — deformable nodal positions ignore environment origins

**Symptom.** Any task with a deformable asset crashes on the first `env.reset()` as soon as
`--num_envs > 1`:

```
File ".../isaaclab/scene/interactive_scene.py", line 661, in get_state
    asset_state["nodal_position"][:, :3] -= self.env_origins
RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1
```

**Cause.** `nodal_position` has shape `(num_envs, num_nodes, 3)`. The expression `[:, :3]` slices
the first three *nodes* rather than the three xyz columns, producing `(num_envs, 3, 3)`. Subtracting
`env_origins` of shape `(num_envs, 3)` then broadcasts against the wrong axis.

At `num_envs == 1` the shapes happen to broadcast **and** `env_origins` is all zeros, so the
operation is a silent no-op and the bug is invisible. At `num_envs == 4` the trailing dimensions
are `(3, 3)` against `(4, 3)` and it raises. This is why every deformable example in SoftMimicGen
runs with `--num_envs 1`, while the rigid-body cube-stacking examples happily use `--num_envs 10`.

**Fix.** Operate on the whole array and broadcast the origin over the node axis, in all four
places (`get_state` and `reset_to`, each for `deformable` and `deformable_object`):

```python
asset_state["nodal_position"] -= self.env_origins.unsqueeze(1)   # get_state
nodal_position += self.env_origins[env_ids].unsqueeze(1)         # reset_to
```

**Effect on existing datasets: none.** Everything generated before this patch used `num_envs=1`,
where `env_origins` is zero and both the old and new code are no-ops. Old and new recordings store
nodal positions in the same frame and stay directly comparable.

**Why it matters.** It unblocks multi-environment data generation for the deformable tasks, which
is the difference between a corpus that generates overnight and one that takes days. It is also a
prerequisite for any state-injection re-render pass at `num_envs > 1`.

Worth upstreaming to the IsaacLab fork this repo clones.
