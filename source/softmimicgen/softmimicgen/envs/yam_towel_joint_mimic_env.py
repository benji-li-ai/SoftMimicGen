# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.controllers import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg

from softmimicgen.envs.soft_mimic_env import SoftManagerBasedRLMimicEnv as ManagerBasedRLMimicEnv


class YamTowelMimicEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic environment wrapper class for YAM Towel with Joint Position control."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.1},  # Higher damping for stability (default is 0.01)
        )

        self._ik_controller_robot0 = DifferentialIKController(cfg=ik_cfg, num_envs=1, device=self.device)
        self._ik_controller_robot1 = DifferentialIKController(cfg=ik_cfg, num_envs=1, device=self.device)

        self._robot0_arm_joint_ids = None
        self._robot1_arm_joint_ids = None
        self._robot0_ee_body_idx = None
        self._robot1_ee_body_idx = None
        self._robot0_joint_limits = None
        self._robot1_joint_limits = None

    def _initialize_joint_info(self):
        """Lazily initialize joint and body indices using SceneEntityCfg."""
        if self._robot0_arm_joint_ids is None:
            robot0 = self.scene["robot_1"]
            robot1 = self.scene["robot_2"]

            robot0_entity_cfg = SceneEntityCfg("robot_1", joint_names=["joint.*"], body_names=["link_6"])
            robot1_entity_cfg = SceneEntityCfg("robot_2", joint_names=["joint.*"], body_names=["link_6"])
            robot0_entity_cfg.resolve(self.scene)
            robot1_entity_cfg.resolve(self.scene)

            self._robot0_arm_joint_ids = robot0_entity_cfg.joint_ids
            self._robot1_arm_joint_ids = robot1_entity_cfg.joint_ids
            self._robot0_ee_body_idx = robot0_entity_cfg.body_ids[0]
            self._robot1_ee_body_idx = robot1_entity_cfg.body_ids[0]

            # For a fixed-base robot, the Jacobian body index is one less than the body index
            # because the root body is not included in the returned Jacobians.
            if robot0.is_fixed_base:
                self._robot0_ee_jacobi_idx = self._robot0_ee_body_idx - 1
            else:
                self._robot0_ee_jacobi_idx = self._robot0_ee_body_idx

            if robot1.is_fixed_base:
                self._robot1_ee_jacobi_idx = self._robot1_ee_body_idx - 1
            else:
                self._robot1_ee_jacobi_idx = self._robot1_ee_body_idx

            self._num_arm_joints = len(self._robot0_arm_joint_ids)
            self._robot0_joint_limits = robot0.data.soft_joint_pos_limits[0, self._robot0_arm_joint_ids, :]
            self._robot1_joint_limits = robot1.data.soft_joint_pos_limits[0, self._robot1_arm_joint_ids, :]

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        eef_pos = self.obs_buf["policy"][f"{eef_name}_eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"][f"{eef_name}_eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def _compute_jacobian(self, robot_name: str) -> torch.Tensor:
        """Compute the geometric Jacobian for the end effector in the robot's base frame.

        Args:
            robot_name: "robot_1" or "robot_2"

        Returns:
            Jacobian tensor of shape (num_envs, 6, num_arm_joints) in base frame.
        """
        self._initialize_joint_info()

        robot = self.scene[robot_name]

        if robot_name == "robot_1":
            ee_jacobi_idx = self._robot0_ee_jacobi_idx
            arm_joint_ids = self._robot0_arm_joint_ids
        else:
            ee_jacobi_idx = self._robot1_ee_jacobi_idx
            arm_joint_ids = self._robot1_arm_joint_ids

        jacobian_full = robot.root_physx_view.get_jacobians()
        jacobian_w = jacobian_full[:, ee_jacobi_idx, :, arm_joint_ids]

        # Rotate from world frame to base frame
        base_rot = robot.data.root_quat_w
        base_rot_matrix = PoseUtils.matrix_from_quat(PoseUtils.quat_inv(base_rot))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        return jacobian_b

    def target_eef_pose_to_action(
        self, target_eef_pose_dict: dict, gripper_action_dict: dict, action_noise_dict: dict | None = None, env_id: int = 0
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (joint position targets) to try and achieve that target pose using differential IK.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            action_noise_dict: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        self._initialize_joint_info()

        robot0 = self.scene["robot_1"]
        robot1 = self.scene["robot_2"]

        robot0_joint_pos = robot0.data.joint_pos[env_id, self._robot0_arm_joint_ids]
        robot1_joint_pos = robot1.data.joint_pos[env_id, self._robot1_arm_joint_ids]

        robot0_ee_pos_w = robot0.data.body_pos_w[env_id, self._robot0_ee_body_idx]
        robot0_ee_quat_w = robot0.data.body_quat_w[env_id, self._robot0_ee_body_idx]
        robot1_ee_pos_w = robot1.data.body_pos_w[env_id, self._robot1_ee_body_idx]
        robot1_ee_quat_w = robot1.data.body_quat_w[env_id, self._robot1_ee_body_idx]

        robot0_root_pos_w = robot0.data.root_pos_w[env_id]
        robot0_root_quat_w = robot0.data.root_quat_w[env_id]
        robot1_root_pos_w = robot1.data.root_pos_w[env_id]
        robot1_root_quat_w = robot1.data.root_quat_w[env_id]

        # Compute current EEF poses in base frame
        robot0_ee_pos_b, robot0_ee_quat_b = subtract_frame_transforms(
            robot0_root_pos_w.unsqueeze(0), robot0_root_quat_w.unsqueeze(0),
            robot0_ee_pos_w.unsqueeze(0), robot0_ee_quat_w.unsqueeze(0)
        )
        robot1_ee_pos_b, robot1_ee_quat_b = subtract_frame_transforms(
            robot1_root_pos_w.unsqueeze(0), robot1_root_quat_w.unsqueeze(0),
            robot1_ee_pos_w.unsqueeze(0), robot1_ee_quat_w.unsqueeze(0)
        )

        target_robot0_pos_w, target_robot0_rot_w = PoseUtils.unmake_pose(target_eef_pose_dict["robot0"])
        target_robot0_quat_w = PoseUtils.quat_from_matrix(target_robot0_rot_w)
        target_robot1_pos_w, target_robot1_rot_w = PoseUtils.unmake_pose(target_eef_pose_dict["robot1"])
        target_robot1_quat_w = PoseUtils.quat_from_matrix(target_robot1_rot_w)

        # The target poses arrive in the ENVIRONMENT frame, not the world frame: they are produced
        # by warping source poses that came from `get_robot_eef_pose` -> the `robot*_eef_pos`
        # observations, and `mdp.ee_frame_pos` subtracts `env_origins` from the world position.
        # The root poses below are true world poses, so the environment origin has to be added
        # back before converting into the robot base frame.
        #
        # Omitting this is a no-op for env 0, whose origin is zero, which is why single-environment
        # generation always worked. For env 1 with the default 2.5 m spacing it commands the IK to
        # a pose 2.5 m from the intended one; the arm saturates and the simulation diverges.
        env_origin = self.scene.env_origins[env_id]
        target_robot0_pos_w = target_robot0_pos_w + env_origin
        target_robot1_pos_w = target_robot1_pos_w + env_origin

        # Convert target poses to base frame
        target_robot0_pos_b, target_robot0_quat_b = subtract_frame_transforms(
            robot0_root_pos_w.unsqueeze(0), robot0_root_quat_w.unsqueeze(0),
            target_robot0_pos_w.unsqueeze(0) if target_robot0_pos_w.dim() == 1 else target_robot0_pos_w,
            target_robot0_quat_w.unsqueeze(0) if target_robot0_quat_w.dim() == 1 else target_robot0_quat_w
        )
        target_robot1_pos_b, target_robot1_quat_b = subtract_frame_transforms(
            robot1_root_pos_w.unsqueeze(0), robot1_root_quat_w.unsqueeze(0),
            target_robot1_pos_w.unsqueeze(0) if target_robot1_pos_w.dim() == 1 else target_robot1_pos_w,
            target_robot1_quat_w.unsqueeze(0) if target_robot1_quat_w.dim() == 1 else target_robot1_quat_w
        )

        jacobian_robot0 = self._compute_jacobian("robot_1")[env_id:env_id + 1]
        jacobian_robot1 = self._compute_jacobian("robot_2")[env_id:env_id + 1]

        self._ik_controller_robot0.reset()
        self._ik_controller_robot1.reset()

        self._ik_controller_robot0.set_command(torch.cat([target_robot0_pos_b, target_robot0_quat_b], dim=-1))
        self._ik_controller_robot1.set_command(torch.cat([target_robot1_pos_b, target_robot1_quat_b], dim=-1))

        target_robot0_joints = self._ik_controller_robot0.compute(
            ee_pos=robot0_ee_pos_b, ee_quat=robot0_ee_quat_b,
            jacobian=jacobian_robot0, joint_pos=robot0_joint_pos.unsqueeze(0)
        ).squeeze(0)

        target_robot1_joints = self._ik_controller_robot1.compute(
            ee_pos=robot1_ee_pos_b, ee_quat=robot1_ee_quat_b,
            jacobian=jacobian_robot1, joint_pos=robot1_joint_pos.unsqueeze(0)
        ).squeeze(0)

        target_robot0_joints = torch.clamp(
            target_robot0_joints, min=self._robot0_joint_limits[:, 0], max=self._robot0_joint_limits[:, 1]
        )
        target_robot1_joints = torch.clamp(
            target_robot1_joints, min=self._robot1_joint_limits[:, 0], max=self._robot1_joint_limits[:, 1]
        )

        robot0_gripper_action = gripper_action_dict["robot0"]
        robot1_gripper_action = gripper_action_dict["robot1"]

        if action_noise_dict is not None:
            target_robot0_joints += action_noise_dict["robot0"] * torch.randn_like(target_robot0_joints)
            target_robot1_joints += action_noise_dict["robot1"] * torch.randn_like(target_robot1_joints)

        return torch.cat(
            (target_robot0_joints, robot0_gripper_action, target_robot1_joints, robot1_gripper_action),
            dim=0,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        For joint position control, the joint actions have already been applied by the simulator,
        so the current EEF poses are read directly via FK.

        Args:
            action: Environment action. Shape is (num_envs, action_dim).

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to.
        """
        self._initialize_joint_info()

        robot0 = self.scene["robot_1"]
        robot1 = self.scene["robot_2"]

        robot0_ee_pos_w = robot0.data.body_pos_w[:, self._robot0_ee_body_idx]
        robot0_ee_quat_w = robot0.data.body_quat_w[:, self._robot0_ee_body_idx]
        robot1_ee_pos_w = robot1.data.body_pos_w[:, self._robot1_ee_body_idx]
        robot1_ee_quat_w = robot1.data.body_quat_w[:, self._robot1_ee_body_idx]

        return {
            "robot0": PoseUtils.make_pose(robot0_ee_pos_w, PoseUtils.matrix_from_quat(robot0_ee_quat_w)),
            "robot1": PoseUtils.make_pose(robot1_ee_pos_w, PoseUtils.matrix_from_quat(robot1_ee_quat_w)),
        }

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        self._initialize_joint_info()

        gripper0_idx = self._num_arm_joints
        gripper1_idx = 2 * self._num_arm_joints + 1

        return {
            "robot0": actions[:, gripper0_idx:gripper0_idx + 1],
            "robot1": actions[:, gripper1_idx:gripper1_idx + 1]
        }

    def _sample_random_points(self, points: torch.Tensor, num_samples: int, seed: int = 42) -> torch.Tensor:
        """Deterministic random sampling of points.

        Args:
            points: Tensor of shape (batch_size, num_points, 3).
            num_samples: Number of points to sample.
            seed: Random seed for reproducible sampling.

        Returns:
            Tensor of shape (batch_size, num_samples) with indices of sampled points.
        """
        batch_size, num_points, _ = points.shape
        device = points.device
        generator = torch.Generator(device=device).manual_seed(seed)
        indices = torch.randperm(num_points, device=device, generator=generator)[:num_samples].unsqueeze(0).expand(batch_size, -1)
        return indices

    def get_object_nodal_positions(self, env_ids: Sequence[int] | None = None, num_samples: int = 100):
        """
        Gets the nodal positions of each deformable object relevant to Isaac Lab Mimic data generation in the current scene.

        Args:
            env_ids: Environment indices to get the nodal positions for. If None, all envs are considered.
            num_samples: Number of nodal points to sample.

        Returns:
            A dictionary that maps object names to nodal position tensors.
            Shape is (len(env_ids), num_samples, 3) for each object.
        """
        if env_ids is None:
            env_ids = slice(None)

        object_nodal_position = dict()

        deformable_object_states = self.scene.get_state(is_relative=True)["deformable"]

        if deformable_object_states:
            for obj_name, obj_state in deformable_object_states.items():
                all_positions = obj_state["nodal_position"][env_ids]

                if all_positions.shape[1] > num_samples:
                    indices = self._sample_random_points(all_positions, num_samples)
                    sampled_positions = all_positions.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))
                else:
                    sampled_positions = all_positions[:, :num_samples, :]

                object_nodal_position[obj_name] = sampled_positions

        return object_nodal_position
