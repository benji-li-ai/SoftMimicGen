# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for grasp-frame registration using the native warp and TPS math."""

import ast
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
GRASP_OFFSET = (0.0, 0.0, 0.1347)
CENTER = np.array([0.4, -0.12, 0.755])
TRANSLATION = np.array([0.025, -0.035, 0.012])
HALF_EXTENTS = np.array([0.15, 0.18])


def _extract_functions(path, names, namespace):
    """Load only native functions, without importing simulator package initializers."""
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture(scope="module")
def native():
    # This is the real vendored package, not a sys.modules stub. Restore the
    # search path immediately; no Isaac/Kit modules are imported or replaced.
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "third_party/rapprentice"))
        from rapprentice.registration import tps_rpm_bij, unit_boxify, unscale_tps
        from rapprentice.tps import tps_eval, tps_fit2, tps_grad

    pose_namespace = {"torch": torch}
    _extract_functions(
        ROOT / "third_party/IsaacLab/source/isaaclab/isaaclab/utils/math.py",
        ("make_pose", "unmake_pose"),
        pose_namespace,
    )
    pose_utils = SimpleNamespace(
        make_pose=pose_namespace["make_pose"], unmake_pose=pose_namespace["unmake_pose"]
    )
    warp_namespace = {
        "np": np,
        "torch": torch,
        "PoseUtils": pose_utils,
        "tps_eval": tps_eval,
        "tps_fit2": tps_fit2,
        "tps_grad": tps_grad,
        "tps_rpm_bij": tps_rpm_bij,
        "unit_boxify": unit_boxify,
        "unscale_tps": unscale_tps,
    }
    name = "transform_source_data_segment_using_nodal_registration"
    rpm_name = name + "_scaled_tps_rpm_bij"
    _extract_functions(
        ROOT / "source/softmimicgen/softmimicgen/datagen/data_generator.py",
        (name, rpm_name),
        warp_namespace,
    )
    return SimpleNamespace(
        warp=warp_namespace[name], rpm_warp=warp_namespace[rpm_name], make_pose=pose_utils.make_pose
    )


@pytest.fixture(params=["nodal", "rpm_bij"])
def warp(native, request):
    if request.param == "nodal":
        return partial(native.warp, bend_coef=1e-4, rot_coef=1e-3)
    return partial(
        native.rpm_warp, n_iter=12, reg_init=1e-2, reg_final=1e-4,
        rad_init=0.02, rad_final=0.005, rot_reg=1e-3,
    )


@pytest.fixture
def cloud():
    x, y = np.meshgrid(np.linspace(-0.15, 0.15, 7), np.linspace(-0.18, 0.18, 7))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))) + CENTER
    return torch.tensor(points, dtype=torch.float64)


def _grasp_positions(poses, offset=GRASP_OFFSET):
    return poses[:, :3, 3] + poses[:, :3, :3] @ poses.new_tensor(offset)


@pytest.fixture
def poses(native):
    rotations = torch.tensor(
        Rotation.from_euler(
            "xyz",
            [[90, 0, 0], [-90, 0, 0], [0, -90, 0], [0, 90, 0], [25, 40, 65], [-35, -50, -20]],
            degrees=True,
        ).as_matrix(),
        dtype=torch.float64,
    )
    # Every grasp starts 10 mm outside one cloth edge. The first four tools
    # point inward, so scaling the wrist alone would move the grasp inside.
    grasp = torch.tensor(
        CENTER
        + np.array(
            [
                [0, 0.19, 0.01],
                [0, -0.19, 0.01],
                [0.16, 0, 0.015],
                [-0.16, 0, 0.015],
                [0.08, 0.19, 0.025],
                [-0.08, -0.19, 0.025],
            ]
        ),
        dtype=torch.float64,
    )
    return native.make_pose(grasp - rotations @ grasp.new_tensor(GRASP_OFFSET), rotations)


@pytest.mark.parametrize(
    "angles", [(0, 0, 0), (0, 0, 30), (0, 30, 0), (20, 30, 30)],
    ids=["unrotated", "yaw", "pitch", "combined"],
)
def test_scaled_cloud_warps_grasp_and_keeps_outside_margin(native, cloud, poses, angles):
    rotation = torch.tensor(Rotation.from_euler("xyz", angles, degrees=True).as_matrix())
    linear = torch.diag(torch.tensor([0.7, 0.7, 1.0], dtype=cloud.dtype)) @ rotation.T
    center, translation = cloud.new_tensor(CENTER), cloud.new_tensor(TRANSLATION)
    target = (cloud - center) @ linear + center + translation

    actual = native.warp(poses, cloud, target, eef_warp_offset=GRASP_OFFSET)

    expected_grasp = (_grasp_positions(poses) - center) @ linear + center + translation
    expected_rotations = rotation @ poses[:, :3, :3]
    expected = native.make_pose(
        expected_grasp - expected_rotations @ poses.new_tensor(GRASP_OFFSET), expected_rotations
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)

    # Undo target rotation to compare signed margins in the cloth's own plane.
    axes = torch.tensor([1, 1, 0, 0, 1, 1])
    signs = poses.new_tensor([1, -1, 1, -1, 1, -1])
    rows = torch.arange(len(poses))
    source_local = _grasp_positions(poses) - center
    target_local = (_grasp_positions(actual) - center - translation) @ rotation
    edge = poses.new_tensor(HALF_EXTENTS)[axes]
    source_margin = signs * source_local[rows, axes] - edge
    target_margin = signs * target_local[rows, axes] - 0.7 * edge
    assert torch.all(source_margin > 0)
    assert torch.all(target_margin > 0)
    torch.testing.assert_close(target_margin, 0.7 * source_margin, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    "angles", [(0, 0, 0), (0, 0, 35), (0, 25, 0)], ids=["translation", "yaw", "pitch"]
)
def test_rigid_mapping_is_unchanged_by_grasp_offset(native, cloud, poses, angles):
    rotation = torch.tensor(Rotation.from_euler("xyz", angles, degrees=True).as_matrix())
    center, translation = cloud.new_tensor(CENTER), cloud.new_tensor(TRANSLATION)
    target = (cloud - center) @ rotation.T + center + translation

    actual = native.warp(poses, cloud, target, eef_warp_offset=GRASP_OFFSET)
    control_frame = native.warp(poses, cloud, target)
    expected = native.make_pose(
        (poses[:, :3, 3] - center) @ rotation.T + center + translation,
        rotation @ poses[:, :3, :3],
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(actual, control_frame, rtol=0, atol=1e-7)


@pytest.mark.parametrize("offset", [None, (0.0, 0.0, 0.0)], ids=["none", "zero"])
@pytest.mark.parametrize("use_rotation", [False, True])
def test_no_offset_preserves_original_control_frame_warp(native, cloud, poses, offset, use_rotation):
    rotation = torch.tensor(Rotation.from_euler("xyz", [20, 30, 35], degrees=True).as_matrix())
    linear = torch.diag(cloud.new_tensor([0.7, 0.7, 1.0])) @ rotation.T
    center, translation = cloud.new_tensor(CENTER), cloud.new_tensor(TRANSLATION)
    target = (cloud - center) @ linear + center + translation

    actual = native.warp(
        poses, cloud, target, use_rotation_transform=use_rotation, eef_warp_offset=offset
    )
    omitted = native.warp(poses, cloud, target, use_rotation_transform=use_rotation)
    expected = native.make_pose(
        (poses[:, :3, 3] - center) @ linear + center + translation,
        rotation @ poses[:, :3, :3] if use_rotation else poses[:, :3, :3],
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(actual, omitted, rtol=0, atol=1e-12)


@pytest.mark.parametrize("offset", [GRASP_OFFSET, (0.02, -0.01, 0.1347)], ids=["yam", "all_axes"])
def test_rotation_disabled_still_converts_through_grasp_frame(native, cloud, poses, offset):
    rotation = torch.tensor(Rotation.from_euler("xyz", [20, 30, 35], degrees=True).as_matrix())
    linear = torch.diag(cloud.new_tensor([0.7, 0.7, 1.0])) @ rotation.T
    center, translation = cloud.new_tensor(CENTER), cloud.new_tensor(TRANSLATION)
    target = (cloud - center) @ linear + center + translation

    actual = native.warp(
        poses, cloud, target, use_rotation_transform=False, eef_warp_offset=offset
    )
    expected_grasp = (_grasp_positions(poses, offset) - center) @ linear + center + translation
    torch.testing.assert_close(actual[:, :3, :3], poses[:, :3, :3], rtol=0, atol=0)
    torch.testing.assert_close(_grasp_positions(actual, offset), expected_grasp, rtol=0, atol=1e-7)


@pytest.mark.parametrize("offset", [GRASP_OFFSET, (0.02, -0.01, 0.1347)], ids=["yam", "all_axes"])
@pytest.mark.parametrize("use_rotation", [False, True])
def test_nonlinear_warp_matches_explicit_grasp_conversion(native, warp, cloud, offset, use_rotation):
    target = cloud.clone()
    target[:, 2] += 0.04 * ((cloud[:, 0] - CENTER[0]) / HALF_EXTENTS[0]) ** 2
    rotations = torch.tensor(
        Rotation.from_euler("xyz", [[0, 90, 0], [0, 90, 20], [0, 65, 20]], degrees=True).as_matrix()
    )
    wrist = cloud.new_tensor(CENTER) + cloud.new_tensor(
        [[-0.09, -0.04, 0.01], [-0.06, 0.02, 0.015], [-0.03, 0.05, 0.02]]
    )
    poses = native.make_pose(wrist, rotations)

    # Independent reference: change only the rigid pose origin, then execute the
    # real zero-offset warp. No registration or Jacobian math is duplicated here.
    grasp_poses = poses.clone()
    grasp_poses[:, :3, 3] = _grasp_positions(poses, offset)
    expected = warp(
        grasp_poses, cloud, target, use_rotation_transform=use_rotation,
        eef_warp_offset=(0.0, 0.0, 0.0),
    )
    expected[:, :3, 3] -= expected[:, :3, :3] @ expected.new_tensor(offset)
    wrist_reference = warp(poses, cloud, target, use_rotation_transform=use_rotation)
    actual = warp(poses, cloud, target, use_rotation_transform=use_rotation, eef_warp_offset=offset)

    if use_rotation:
        # Distinguish the Jacobian evaluation sites: an affine map or a guardrail
        # fallback cannot pass this nonlinearity check vacuously.
        assert torch.max(torch.abs(expected[:, :3, :3] - wrist_reference[:, :3, :3])) > 1e-3
    else:
        torch.testing.assert_close(actual[:, :3, :3], poses[:, :3, :3], rtol=0, atol=0)
    assert torch.max(torch.abs(expected[:, :3, 3] - wrist_reference[:, :3, 3])) > 1e-4
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("use_rotation", [False, True])
def test_inputs_are_not_mutated_and_output_preserves_dtype_device(warp, cloud, poses, dtype, use_rotation):
    poses = poses.to(dtype=dtype)
    source = cloud.to(dtype=dtype)
    target = (source - source.new_tensor(CENTER)) * source.new_tensor([0.7, 0.7, 1.0])
    target += source.new_tensor(CENTER + TRANSLATION)
    snapshots = [value.clone() for value in (poses, source, target)]

    actual = warp(
        poses, source, target, use_rotation_transform=use_rotation, eef_warp_offset=GRASP_OFFSET
    )

    assert actual.dtype == dtype
    assert actual.device == poses.device
    assert actual.shape == poses.shape
    for value, snapshot in zip((poses, source, target), snapshots):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
