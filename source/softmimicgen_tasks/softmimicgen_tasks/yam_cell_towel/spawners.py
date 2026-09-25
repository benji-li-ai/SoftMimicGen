# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live USD composition for the cell task; supplied package layers stay read-only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass


def _spawn_reference(stage, prim_path, usd_path, source_prim_path, translation, orientation):
    """Reference an explicit source subtree at the task's environment-local pose."""
    root = stage.DefinePrim(prim_path, "Xform")
    root.GetReferences().AddReference(usd_path, source_prim_path)
    UsdGeom.Xformable(root).MakeMatrixXform().Set(
        Gf.Matrix4d(
            Gf.Rotation(Gf.Quatd(*(orientation if orientation is not None else (1.0, 0.0, 0.0, 0.0)))),
            Gf.Vec3d(*(translation if translation is not None else (0.0, 0.0, 0.0))),
        )
    )
    return root


def _relocate_material_bindings(stage, root, usd_path):
    """Retain original materials whose source bindings escape the arm reference."""
    materials = {}
    for prim in list(Usd.PrimRange(root)):
        binding = prim.GetRelationship("material:binding")
        if not binding:
            continue
        for spec in binding.GetPropertyStack():
            authored = list(spec.targetPathList.GetAppliedItems())
            if len(authored) != 1 or "/Looks/" not in str(authored[0]):
                continue
            source_target = str(authored[0])
            target = materials.get(source_target)
            if target is None:
                target = f"{root.GetPath()}/ResolvedMaterials/material_{len(materials)}"
                material = stage.DefinePrim(target, "Material")
                material.GetReferences().AddReference(usd_path, source_target)
                materials[source_target] = target
            binding.SetTargets([target])
            break


@clone
def spawn_cell_arm(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Apply the package's validated cell physics before Isaac Lab clones the arm."""
    from yam_assets import physics

    stage = get_current_stage()
    with Usd.EditContext(stage, stage.GetRootLayer()):
        root = _spawn_reference(stage, prim_path, cfg.usd_path, cfg.source_prim_path, translation, orientation)
        weld_path = physics.apply_yam_physics(stage=stage, prim_path=prim_path, links_path=f"{prim_path}/arm")
        physics.disable_gravity_under(prim_path, stage=stage)
        _relocate_material_bindings(stage, root, cfg.usd_path)

        # USD permits a non-rigid Xform as the static side of a joint. Keep an
        # empty frame so default joint collision filtering cannot hide arm contacts.
        # Both targets are local to this asset, so cloning remaps them together and
        # each environment's transform supplies its own world anchor at parse time.
        anchor = UsdGeom.Xform.Define(stage, f"{prim_path}/WorldAnchor")
        base = stage.GetPrimAtPath(f"{prim_path}/arm/arm")
        transforms = UsdGeom.XformCache()
        base_in_anchor = (
            transforms.GetLocalToWorldTransform(base)
            * transforms.GetLocalToWorldTransform(anchor.GetPrim()).GetInverse()
        ).RemoveScaleShear()
        weld = UsdPhysics.FixedJoint.Get(stage, weld_path)
        weld.CreateBody0Rel().SetTargets([anchor.GetPath()])
        weld.CreateLocalPos0Attr().Set(Gf.Vec3f(base_in_anchor.ExtractTranslation()))
        weld.CreateLocalRot0Attr().Set(Gf.Quatf(base_in_anchor.ExtractRotationQuat()))
    return root


@clone
def spawn_cell_workstation(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Expose the recorded TableSurface frame without moving the cell backdrop.

    The RigidObject initial state describes the nested TableSurface body, not
    the outer reference root. Keep the backdrop at the environment datum and
    retain the source-authored table pose (.655, 0, .745); native resets write
    that same pose to the table body while the deck, frame and room stay static.
    """
    stage = get_current_stage()
    with Usd.EditContext(stage, stage.GetRootLayer()):
        root = _spawn_reference(stage, prim_path, cfg.usd_path, "/World", None, None)
        # The arms have their own articulation assets. Physics scenes and the
        # source SDG pipeline must not compete with the task's scene/recorders.
        stage.OverridePrim(f"{prim_path}/envs").SetActive(False)
        for prim in list(Usd.PrimRange(root)):
            if prim.IsA(UsdPhysics.Scene) or "SDGPipeline" in str(prim.GetPath()):
                prim.SetActive(False)

        # Source TableSurface has colliders but no rigid body. Add its kinematic
        # body without changing geometry, materials, or other fixture components.
        # It is the sole body here, so native rigid-state and object-pose readout
        # both resolve the same frame as /World/Workstation/TableSurface recording.
        table = stage.GetPrimAtPath(f"{prim_path}/Workstation/TableSurface")
        rigid = UsdPhysics.RigidBodyAPI.Apply(table)
        rigid.CreateKinematicEnabledAttr().Set(True)
    return root


@clone
def spawn_cell_cloth(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Preserve native USD placement and the package's validated cloth contact setup."""
    from yam_assets import scene

    stage = get_current_stage()
    with Usd.EditContext(stage, stage.GetRootLayer()):
        root = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
        scene._ensure_deformable_apis(prim_path, stage)
        scene._tune_cloth_pinch_contact(prim_path, stage)
    return root


@configclass
class CellArmCfg(SpawnerCfg):
    """A side-specific cell arm using package physics and actuators, not an export."""

    func: Callable = spawn_cell_arm
    usd_path: str = MISSING
    source_prim_path: str = MISSING


@configclass
class CellWorkstationCfg(SpawnerCfg):
    """Package backdrop with its TableSurface frame exposed as a kinematic rigid body."""

    func: Callable = spawn_cell_workstation
    usd_path: str = MISSING


@configclass
class CellClothCfg(UsdFileCfg):
    """Native cloth USD spawner with the package's deformable/contact authoring pass."""

    func: Callable = spawn_cell_cloth
