# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CoupledGripperAction(ActionTerm):
    """Coupled gripper action term that maps a single continuous action to multiple gripper joints.

    This action term takes a single continuous action value in the range [0, 1] and maps it to
    multiple gripper joints with different scales. This is useful for parallel-jaw grippers where
    both fingers should move symmetrically with a single control input.

    The mapping is:
    - 0.0 close configuration (uses close_command_expr)
    - 1.0 open configuration (uses open_command_expr)

    Example:
        For a gripper with two fingers (left and right):
        - Action = 0.0 left_finger = 0.0, right_finger = 0.0 (closed)
        - Action = 1.0 left_finger = 0.04, right_finger = -0.04 (open)
        - Action = 0.5 left_finger = 0.02, right_finger = -0.02 (half-open)
    """

    cfg: CoupledGripperActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: CoupledGripperActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # parse open command (target positions when action = 1.0)
        self._open_command = torch.zeros(self._num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for open command. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._open_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse close command (target positions when action = 0.0)
        self._close_command = torch.zeros_like(self._open_command)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for close command. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._close_command[index_list] = torch.tensor(value_list, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions

        # clamp action to [0, 1] range
        actions_clamped = torch.clamp(actions, 0.0, 1.0)

        # linear interpolation between close (0) and open (1) positions
        # action = 0 close_command
        # action = 1 open_command
        self._processed_actions = self._close_command + actions_clamped * (self._open_command - self._close_command)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class CoupledGripperPositionAction(CoupledGripperAction):
    """Coupled gripper action that sets the continuous action into joint position targets."""

    cfg: CoupledGripperPositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


##
# Coupled gripper action configs.
##


@configclass
class CoupledGripperActionCfg(ActionTermCfg):
    """Configuration for the coupled gripper action term.

    This action term takes a single continuous action in [0, 1] and maps it to multiple gripper joints.
    - 0.0 close configuration
    - 1.0 open configuration

    See :class:`CoupledGripperAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    open_command_expr: dict[str, float] = MISSING
    """The joint command to move to *open* configuration (action = 1.0)."""
    close_command_expr: dict[str, float] = MISSING
    """The joint command to move to *close* configuration (action = 0.0)."""


@configclass
class CoupledGripperPositionActionCfg(CoupledGripperActionCfg):
    """Configuration for the coupled gripper position action term.

    See :class:`CoupledGripperPositionAction` for more details.
    """

    class_type: type[ActionTerm] = CoupledGripperPositionAction
