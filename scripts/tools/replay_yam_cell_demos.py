# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay the yam_towel_demo_new2.hdf5 teleop demos on the yam_assets cell stage.

The demos were recorded in the yam_isaac_cell frame (not the SoftMimicGen task env):
both YAM bases at (0.2525, ±0.31, 0.7854), table top at z=0.760, cloth footprint
11.5 in square centred at (0.55, 0). The recorded control stream decodes as:

- ``actions[t, 0:6]``   robot0 (left)  arm ABSOLUTE joint targets (rad)
- ``actions[t, 6]``     left gripper scalar in [0, 1]; left_finger target = 0.0475 * a
- ``actions[t, 7:13]``  robot1 (right) arm ABSOLUTE joint targets (rad)
- ``actions[t, 13]``    right gripper scalar (identically 0 in these demos)
- ``initial_state/deformable/object/nodal_position`` the 5,989-node cloth pose at
  t=0; every per-tick ``states`` array re-records the full nodal stream so replay
  fidelity is checkable node-for-node.

The state stream lags the target stream by 0-3 control steps (the 28/10 N·m
drive caps saturate during fast swings), so replay fidelity is measured as
joint-space tracking error and end-effector offset, not exact equality.

Usage (inside the softmimicgen conda env, from anywhere):

    python replay_yam_cell_demos.py \
        --dataset /home/benjamin.li/src/yam_towel_demo_new2.hdf5 \
        --select 0 --video outputs/videos/yam_cell_replay_demo_0.mp4

Requires a free GPU; check `nvidia-smi` first (see SoftMimicGen/CLAUDE.md).
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Replay YAM cell demos with cloth write-back and video capture.")
parser.add_argument("--dataset", type=str, default="/home/benjamin.li/src/yam_towel_demo_new2.hdf5")
parser.add_argument("--select", type=int, nargs="+", default=[0], help="Demo indices to replay in order.")
parser.add_argument("--video", type=str, default=None, help="Output MP4 path; omit to skip video.")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument(
    "--metrics_csv", type=str, default=None, help="Optional CSV path for per-step joint tracking error."
)
parser.add_argument("--settle_steps", type=int, default=60, help="Physics steps to settle the cloth at reset.")
parser.add_argument("--arm_stiffness", type=float, default=1500.0)
parser.add_argument("--arm_damping", type=float, default=80.0)
parser.add_argument("--max_steps", type=int, default=10_000, help="Safety cap on total physics steps.")
# Fingertip contact config. A/B measured on demo_0 (fold ratio, success < 0.5, recorded 0.266):
#   pads ON  -> grasp holds at the right place, fold does NOT complete (ratio 0.74)
#   pads OFF -> fold completes by area metric (0.39) but the cloth crumples off-position
# Neither is bit-faithful to the recording; the teleop's exact fingertip contact setup at
# recording time is unknown (the FEM cook pipeline it used lives in yam_teleop_860, which is
# not on this machine). See the investigation log in the session notes.
parser.add_argument("--show_pads", action="store_true", default=True,
                    help="Author MJCF high-friction (4.0) fingertip pad boxes on the cell arms.")
parser.add_argument("--no_pads", dest="show_pads", action="store_false")
parser.add_argument(
    "--finger_friction", type=float, default=None,
    help="Bind a friction material with this mu to the finger colliders (no-pad mode)."
)
parser.add_argument(
    "--sim_states_out", type=str, default=None,
    help="HDF5 path to dump the simulated cloth nodal stream + final joint states for comparison."
)
args_cli = parser.parse_args()

# --- Isaac boot -----------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

# yam_assets: keep ~/Workspace on the path (uninstalled use), per README.
for candidate in (
    os.path.expanduser("~/Workspace"),
    "/home/benjamin.li/src/yam_assets",  # repo checkout: package dir IS yam_assets/yam_assets
):
    if os.path.isfile(os.path.join(candidate, "yam_assets", "scene.py")) and candidate not in sys.path:
        sys.path.insert(0, candidate)

import yam_assets.scene as yscene  # noqa: E402
from yam_assets.scene import load_cell, apply_scene_physics, setup_arms, spawn_cloth_custom_red  # noqa: E402

FINGER_OPEN = 0.0475  # recorded full-open left_finger joint value at gripper scalar 1.0

WORLD_DT = 1.0 / 120.0


def main():
    import h5py
    import torch
    from isaacsim.core.utils.types import ArticulationActions

    cv2 = None
    if args_cli.video:
        import cv2 as _cv2
        cv2 = _cv2
    ds = h5py.File(args_cli.dataset, "r")
    demos = sorted(ds["data"].keys(), key=lambda k: int(k.split("_")[-1]))

    # ---- stage ----
    assert load_cell(), "failed to open the bundled cell USD"
    from isaacsim.core.api import World

    world = World(stage_units_in_meters=1.0, physics_dt=WORLD_DT, rendering_dt=1.0 / 60.0,
                  device="cuda:0")  # tensor pipeline MUST be cuda:0: the CPU backend does not
                                    # implement deformable nodal get/set (see CLAUDE.md notes)
    apply_scene_physics(world)
    arms = setup_arms(world, source="cell", arms="bimanual",
                      arm_stiffness=args_cli.arm_stiffness, arm_damping=args_cli.arm_damping,
                      show_pads=args_cli.show_pads)
    if args_cli.show_pads:
        # The pads are collision proxies (friction 4.0 boxes): they must exist in physics
        # but NOT render — apply_yam_physics authors them visible=True, which painted grey
        # boxes over the grippers (the "zebra stripes" from v2 onwards). Hide the pad scopes.
        import omni.usd as _ou0
        from pxr import UsdGeom as _Ug0
        _st0 = _ou0.get_context().get_stage()
        _hid_pads = 0
        for _yam in ("/World/envs/env_0/LeftYam", "/World/envs/env_0/RightYam"):
            for _link in ("link_left_finger", "link_right_finger"):
                _pad_scope = _st0.GetPrimAtPath(f"{_yam}/arm/{_link}/mjcf_pads")
                if _pad_scope.IsValid():
                    _Ug0.Imageable(_pad_scope).MakeInvisible()
                    _hid_pads += 1
        print(f"  Hid {_hid_pads} mjcf_pads scopes (render-only fix; physics keeps the pads)")
    if args_cli.finger_friction is not None and not args_cli.show_pads:
        # Optional high-friction material bound to the finger colliders (no pad boxes).
        # NOTE: currently binds to 0 prims — the colliders live on mesh descendants of the
        # link prims, not the links themselves. Kept for future debugging of the
        # pads-on/pads-off contact mismatch.
        from yam_assets import physics as _yphys
        import omni.usd as _ou2
        _st = _ou2.get_context().get_stage()
        _yphys.physics_material(_st, "/World/PhysicsMaterials/FingerGrip",
                                args_cli.finger_friction, combine="max")
        from pxr import UsdPhysics, UsdShade
        _bound = 0
        for _yam in ("/World/envs/env_0/LeftYam", "/World/envs/env_0/RightYam"):
            _links = _st.GetPrimAtPath(f"{_yam}/arm")
            for _prim in _links.GetAllChildren():
                _col = UsdPhysics.CollisionAPI(_prim)
                if not _col or not _prim.GetAttribute("physics:collisionEnabled").Get():
                    continue
                _matbind = UsdShade.MaterialBindingAPI(_prim)
                _matbind.Bind(
                    UsdShade.Material(_st.GetPrimAtPath("/World/PhysicsMaterials/FingerGrip")),
                    UsdShade.Tokens.physics,
                )
                _bound += 1
        print(f"  Finger friction {args_cli.finger_friction} bound to {_bound} colliders")
    # The cell USD authors no solver-iteration counts (PhysX default 4/1). Under cloth contact
    # at the fold-press the elbow joint diverges to NaN; SMG's YAM uses 8/1 — match it.
    from pxr import PhysxSchema
    import omni.usd as _omni_usd
    _stage = _omni_usd.get_context().get_stage()
    for _yam in ("/World/envs/env_0/LeftYam", "/World/envs/env_0/RightYam"):
        _prim = _stage.GetPrimAtPath(f"{_yam}/joints/world_weld")
        if _prim.IsValid():
            _api = PhysxSchema.PhysxArticulationAPI(_prim)
            _api.CreateSolverPositionIterationCountAttr().Set(16)
            _api.CreateSolverVelocityIterationCountAttr().Set(1)
            print(f"  Solver iterations 16/1 on {_yam}")
    # --- finger visual cleanup (the "zebra stripes") ---
    # Each finger's visual prototype carries MJCF-import geoms bound to material_rgba_1,
    # whose diffuse channels are swapped on render (renders salmon/orange instead of the
    # authored blue), plus six 0.6 mm site-marker spheres. The alternating orange segments
    # and shading bands read as zebra stripes down the gripper. Rebinding per-prim to YamDark
    # did not take (the instanced-visual material resolution ignores the direct override), so
    # instead re-author the rgba materials' diffuse color in place — YamDark near-black, the
    # real gripper's color.
    from pxr import Usd, UsdShade as _UsdShade
    _fixed = 0
    for _mat_name in ("material_rgba", "material_rgba_0", "material_rgba_1"):
        _mat = _stage.GetPrimAtPath(f"/World/envs/env_0/LeftYam/Looks/{_mat_name}")
        if not _mat.IsValid():
            continue
        for _sub in Usd.PrimRange(_mat):
            if _sub.GetTypeName() != "Shader":
                continue
            _sh = _UsdShade.Shader(_sub)
            _attr = _sub.GetAttribute("inputs:diffuse_color_constant")
            if _attr.IsValid():
                _attr.Set((0.045, 0.052, 0.055))
                _fixed += 1
    print(f"  Re-colored {_fixed} rgba material shaders to YamDark (zebra fix)")
    sim_mesh_path = spawn_cloth_custom_red()
    assert sim_mesh_path, "cloth sim mesh did not resolve — nodal write-back has no target"
    left, right = arms["left"], arms["right"]
    world.reset()

    # joint index maps (cell MJCF order: joint1..6, left_finger, right_finger)
    lnames = list(left.dof_names)
    rnames = list(right.dof_names)
    l_arm = [lnames.index(f"joint{i}") for i in range(1, 7)]
    l_lf = lnames.index("left_finger")
    l_rf = lnames.index("right_finger")
    r_arm = [rnames.index(f"joint{i}") for i in range(1, 7)]
    r_lf = rnames.index("left_finger")
    r_rf = rnames.index("right_finger")

    # The World(device="cuda:0") call makes SimulationManager create a torch-frontend,
    # GPU-bound tensor pipeline; the CPU backend does not implement deformable nodal
    # get/set ("CpuSimulationView::getSimulationNodalPositions is not implemented yet").
    # Pattern the BODY prim (DeformableBodyAPI xform), not the sim mesh.
    from isaacsim.core.simulation_manager import SimulationManager

    body_prim_path = sim_mesh_path.rsplit("/", 1)[0]
    sim_view = SimulationManager.get_physics_sim_view().create_surface_deformable_body_view(
        body_prim_path
    )
    if sim_view is None or sim_view.count == 0:
        raise RuntimeError(f"surface deformable view matched 0 bodies for {body_prim_path}")
    print(f"  Cloth view: {sim_view.count} body(s) via {body_prim_path}, "
          f"max_sim_nodes={sim_view.max_simulation_nodes_per_body}")

    video_writer = None
    metrics_rows = []
    sim_nodal_samples = []   # (demo, t, nodal[5989,3], q_left[8], q_right[8]) snapshots

    def open_video(frame_shape):
        nonlocal video_writer
        if args_cli.video and video_writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.video)), exist_ok=True)
            video_writer = cv2.VideoWriter(
                args_cli.video, cv2.VideoWriter_fourcc(*"mp4v"), args_cli.fps,
                (frame_shape[1], frame_shape[0]),
            )
            if not video_writer.isOpened():
                raise RuntimeError(f"Failed to open video output: {args_cli.video}")

    def make_camera():
        """Stage a 640x480 RGB camera looking at the table (ego-view geometry from
        scene.toml's simgen_top_camera: bit-exact with the recorded top D405)."""
        import omni.usd as _ou
        from pxr import Gf, UsdGeom

        stage = _ou.get_context().get_stage()
        cam_path = "/World/ReplayCamera"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.GetFocalLengthAttr().Set(12.8406 * 10.0)      # mm -> tenths of a scene unit
        cam.GetHorizontalApertureAttr().Set(20.955 * 10.0)
        cam.GetVerticalApertureAttr().Set(20.9265 * 10.0)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
        xform = UsdGeom.Xformable(cam.GetPrim())
        xform.ClearXformOpOrder()
        # eye (0.086, -0.009, 1.7043), look target (0.6085, -0.0164, 0.8517), up
        # (0.8526, -0.0042, 0.5225) — scene.toml [[render.views]] simgen_top_camera.
        eye = Gf.Vec3d(0.086, -0.009, 1.7043)
        target = Gf.Vec3d(0.60849420, -0.01644726, 0.85168968)
        up = Gf.Vec3d(0.85261040, -0.00415907, 0.52253058)
        forward = (target - eye).GetNormalized()
        right = Gf.Cross(forward, up).GetNormalized()
        up2 = Gf.Cross(right, forward)
        # USD camera looks down -Z with +Y up: rows are (right, up, -forward)
        m = Gf.Matrix4d(
            right[0], right[1], right[2], 0.0,
            up2[0], up2[1], up2[2], 0.0,
            -forward[0], -forward[1], -forward[2], 0.0,
            eye[0], eye[1], eye[2], 1.0,
        )
        xform.AddTransformOp().Set(m)
        return cam_path

    _camera_path = make_camera() if args_cli.video else None
    if args_cli.video:
        # Initialize the render product and annotator ONCE here, before the run: creating
        # them lazily on the first rendered frame raced with the Replicator orchestrator
        # and crashed Kit mid-run (CreateGraphAsNodeCommand segfault at t≈310 in one run).
        import omni.replicator.core as rep

        _render_product = rep.create.render_product(_camera_path, (640, 480))
        _rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        _rgb_annotator.attach(_render_product)

    def render_frame():
        """Render one frame via Replicator from the replay camera and return it as BGR."""
        if not args_cli.video:
            return None
        rep.orchestrator.step(rt_subframes=4)
        data = _rgb_annotator.get_data()
        if data is None or data.size == 0:
            return None
        rgb = np.asarray(data)[..., :3]
        return rgb[..., ::-1]  # RGB -> BGR for cv2.VideoWriter

    def set_cloth_state(nodal, nodal_vel):
        # tensors must live on cuda:0 — the backend rejects host-resident data
        idx = torch.zeros(1, dtype=torch.int32, device="cuda:0")
        sim_view.set_simulation_nodal_positions(
            torch.tensor(nodal, dtype=torch.float32, device="cuda:0").unsqueeze(0), idx
        )
        sim_view.set_simulation_nodal_velocities(
            torch.tensor(nodal_vel, dtype=torch.float32, device="cuda:0").unsqueeze(0), idx
        )

    def drive_arm(art, arm_ids, lf_id, rf_id, arm_target, grip_scalar, finger_target=None):
        # Drive via position TARGETS on the PhysX articulation drive. Do NOT use
        # set_joint_positions: that teleports the DOF state each tick, which injects
        # unbounded joint velocity into the solver and diverges at the first cloth
        # contact (observed: wrist joint spinning to 1000+ rad, then NaN).
        tgt = np.zeros(len(art.dof_names), dtype=np.float32)
        tgt[arm_ids] = np.asarray(arm_target, dtype=np.float32)
        if finger_target is not None:
            # Feedback replay for the right arm: the demo export's right-gripper command
            # channel (a13) is identically 0, but the recorded right finger sits ~0.034 rad
            # open (pushed open against the cloth by the teleop follower). Commanding 0
            # (closed) makes our right gripper squeeze the cloth the whole episode. Track
            # the recorded finger state instead — self-correcting on the lost channel.
            tgt[lf_id] = float(finger_target)
            tgt[rf_id] = 0.0
        else:
            # Left arm: recorded one-sided pinch — left_finger sweeps to 0.0475*a6 against
            # the static right pad (right_finger parked at 0 in all demos).
            tgt[lf_id] = FINGER_OPEN * float(grip_scalar)
            tgt[rf_id] = 0.0
        art._articulation_view.apply_action(
            ArticulationActions(joint_positions=torch.tensor(tgt))
        )

    total_steps = 0
    for demo_name in [demos[i] for i in args_cli.select]:
        ep = ds[f"data/{demo_name}"]
        actions = np.asarray(ep["actions"])
        init_state = {}

        def _flatten(group, prefix=""):
            for k in group:
                if isinstance(group[k], h5py.Group):
                    _flatten(group[k], prefix + k + "/")
                else:
                    init_state[prefix + k] = np.asarray(group[k])

        _flatten(ep["initial_state"])
        nodal_init = init_state["deformable/object/nodal_position"][0]
        nodal_vel_init = init_state["deformable/object/nodal_velocity"][0]
        rec_states = np.asarray(ep["states/articulation/robot_1/joint_position"])
        rec_states_r = np.asarray(ep["states/articulation/robot_2/joint_position"])
        rec_nodal = np.asarray(ep["states/deformable/object/nodal_position"])  # (T, 5989, 3)
        print(f"[{demo_name}] steps={len(actions)} cloth spawn "
              f"({nodal_init[:, 0].min():.3f}..{nodal_init[:, 0].max():.3f} x, z̄={nodal_init[:, 2].mean():.4f})")

        # ---- write cloth to recorded initial pose and settle ----
        init_j1 = init_state["articulation/robot_1/joint_position"][0]
        init_j2 = init_state["articulation/robot_2/joint_position"][0]
        set_cloth_state(nodal_init, nodal_vel_init)
        left._articulation_view.set_joint_positions(
            positions=torch.tensor(init_j1, dtype=torch.float32))
        right._articulation_view.set_joint_positions(
            positions=torch.tensor(init_j2, dtype=torch.float32))
        drive_arm(left, l_arm, l_lf, l_rf, init_j1[:6], init_j1[6])
        drive_arm(right, r_arm, r_lf, r_rf, init_j2[:6], init_j2[6])
        for _ in range(args_cli.settle_steps):
            world.step(render=False)
            total_steps += 1

        # ---- drive the arms through the recorded targets ----
        for t in range(len(actions)):
            a = actions[t]
            drive_arm(left, l_arm, l_lf, l_rf, a[0:6], a[6])
            drive_arm(right, r_arm, r_lf, r_rf, a[7:13], a[13],
                      finger_target=rec_states_r[t, 6])
            for _ in range(2):  # demo env_args: decimation=2 → 60 Hz control at 1/120 sim
                world.step(render=False)
                total_steps += 1
                if total_steps > args_cli.max_steps:
                    raise RuntimeError("step cap hit")

            if t % 10 == 0 or t == len(actions) - 1:
                got = left._articulation_view.get_joint_positions()
                got = got.cpu().numpy() if hasattr(got, "cpu") else np.asarray(got)
                per_joint = np.abs(got[0][l_arm] - rec_states[t, :6])
                err = per_joint.max()
                got2 = right._articulation_view.get_joint_positions()
                got2 = got2.cpu().numpy() if hasattr(got2, "cpu") else np.asarray(got2)
                err2 = np.abs(got2[0][r_arm] - rec_states_r[t, :6]).max()
                metrics_rows.append((demo_name, t, err, err2))
                if err > 0.1 and not np.isnan(err):
                    j = int(per_joint.argmax())
                    print(f"  [{demo_name}] t={t:4d} WARN joint{j+1} err={per_joint[j]:.4f} "
                          f"q={got[0][l_arm][j]:.3f} rec={rec_states[t, j]:.3f}")
                if np.isnan(err):
                    print(f"  [{demo_name}] t={t:4d} DIVERGED: q={got[0][l_arm]}")
                if t % 50 == 0 or t == len(actions) - 1:
                    print(f"  [{demo_name}] t={t:4d} |q_err| L={err:.4f} R={err2:.4f} rad")
                # cloth fidelity: mean nodal offset vs the recorded stream (the replay
                # contract's second half — node-for-node comparison)
                if t % 50 == 0 or t == len(actions) - 1:
                    cur = sim_view.get_simulation_nodal_positions()
                    cur = cur.cpu().numpy() if hasattr(cur, "cpu") else np.asarray(cur)
                    sim_nodal_samples.append((demo_name, t, cur[0].copy(),
                                              got[0].copy(), got2[0].copy()))
                    nd = np.linalg.norm(cur[0] - rec_nodal[t], axis=1)
                    print(f"  [{demo_name}] t={t:4d} cloth nodal err mean={nd.mean():.4f} "
                          f"p95={np.percentile(nd, 95):.4f} max={nd.max():.4f} m")

            if args_cli.video:
                frame = render_frame()
                if frame is not None:
                    open_video(frame.shape)
                    video_writer.write(frame)  # already BGR from imread

    if video_writer is not None:
        video_writer.release()
        print(f"Wrote replay video to {os.path.abspath(args_cli.video)}")

    if args_cli.metrics_csv and metrics_rows:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.metrics_csv)), exist_ok=True)
        with open(args_cli.metrics_csv, "w") as fh:
            fh.write("demo,step,err_left_rad,err_right_rad\n")
            for row in metrics_rows:
                fh.write(",".join(str(x) for x in row) + "\n")
        print(f"Wrote tracking metrics to {args_cli.metrics_csv}")

    if args_cli.sim_states_out and sim_nodal_samples:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.sim_states_out)), exist_ok=True)
        with h5py.File(args_cli.sim_states_out, "w") as fh:
            for demo_name, t, nodal, q_l, q_r in sim_nodal_samples:
                g = fh.require_group(f"{demo_name}")
                g.create_dataset(f"t{t:04d}/nodal", data=nodal)
                g.create_dataset(f"t{t:04d}/q_left", data=q_l)
                g.create_dataset(f"t{t:04d}/q_right", data=q_r)
        print(f"Wrote sim cloth state snapshots to {args_cli.sim_states_out}")

    # summary
    for demo_name in [demos[i] for i in args_cli.select]:
        rows = [r for r in metrics_rows if r[0] == demo_name]
        if rows:
            e = np.array([r[2] for r in rows]); e2 = np.array([r[3] for r in rows])
            print(f"[{demo_name}] final tracking: left median {np.median(e):.4f} max {e.max():.4f} rad | "
                  f"right median {np.median(e2):.4f} max {e2.max():.4f} rad")

    world.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
