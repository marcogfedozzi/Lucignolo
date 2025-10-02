"""
Inverse dynamics controllers for operational space control in MuJoCo environments.

This module implements hierarchical operational space control using inverse dynamics.
The controllers can handle underactuated systems and multiple simultaneous tasks
with proper priority handling through null-space projections.

Key concepts:
- Operational Space Control: Control in task space (e.g., end-effector position/orientation)
- Inverse Dynamics: Compute required joint torques to achieve desired accelerations
- Null-space Projection: Handle task priorities and underactuation constraints
- Dynamically Consistent Inverse: Proper handling of robot dynamics in task space

References:
[1] Khatib, Oussama. "A unified approach for motion and force control of robot manipulators: 
    The operational space formulation." IEEE Journal on Robotics and Automation 3.1 (1987): 43-53.
[2] Mistry, Michael, and Ludovic Righetti. "Operational space control of constrained and 
    underactuated systems." (2012).
"""

import numpy as np

import mujoco
from gymnasium.envs.mujoco import MujocoEnv

from core.eef_point import EEFPoint
from core.utils import IndexGetter
from src.controllers.jnt_ctrl import ConstraintJointController
from core.utils import JointGroup

from typing import Dict, List
from numpy.typing import NDArray
from functools import cached_property



def crux(J, M_x, M_inv):
	"""
	Compute the dynamically consistent generalized inverse of a Jacobian matrix.
	
	This function computes J^(T+) := M_x @ J @ M^(-1), which is the "dynamically consistent 
	generalized inverse" used in operational space control. This inverse properly accounts 
	for the robot's inertial properties when projecting forces/accelerations between 
	joint space and task space.
	
	The dynamically consistent inverse ensures that:
	1. Task space accelerations are properly mapped to joint torques
	2. Null-space projections preserve dynamic consistency
	3. Multiple tasks can be composed hierarchically
	
	Mathematical formulation for null-space projection:
	Q_0 = I
	Q_i = Q_{i-1} * (I - (J_i^T Q_{i-1})*(J_i^T Q_{i-1})^(T+))
	
	Args:
		J (NDArray): Task Jacobian matrix (task_dim x joint_dim)
		M_x (NDArray): Task space mass matrix (task_dim x task_dim)  
		M_inv (NDArray): Inverse of joint space mass matrix (joint_dim x joint_dim)
		
	Returns:
		NDArray: Dynamically consistent generalized inverse (joint_dim x task_dim)
		
	References:
		[1] Khatib, Oussama. "A unified approach for motion and force control of robot manipulators: 
		    The operational space formulation." IEEE Journal on Robotics and Automation 3.1 (1987): 43-53.
		[2] Mistry, Michael, and Ludovic Righetti. "Operational space control of constrained and 
		    underactuated systems." (2012).
	"""
	return M_x @ J @ M_inv

dynamically_consistent_generalized_inverse = crux

class Controller:
	"""
	Base class for inverse dynamics controllers.
	
	Provides common functionality for controllers that operate on specific subtrees
	of the robot (e.g., arms, legs, torso). Handles indexing and basic setup for
	interacting with MuJoCo models.
	
	Attributes:
		data: MuJoCo data structure
		model: MuJoCo model structure  
		actuators: Environment's actuator configuration
		subtree_type: Type of kinematic subtree being controlled
		_indexes: Joint/DOF indices for the controlled subtree
	"""
	
	def __init__(self, env: MujocoEnv, subtree_type: str, *args, **kwargs):
		"""
		Initialize base controller.
		
		Args:
			env: MuJoCo environment instance
			subtree_type: String identifying the kinematic subtree (e.g., 'left_arm', 'torso')
		"""

		self.data = env.data
		self.model = env.model

		self.actuators = env.mimo_actuators

		self.subtree_type = subtree_type
		self._indexes = IndexGetter(self.model)(subtree_type=self.subtree_type)

	def step(self, qact = None):
		"""Compute control signals for one time step. Must be implemented by subclasses."""
		raise NotImplementedError


class IDController(Controller):
	"""
	Inverse Dynamics Controller for operational space control of robotic systems.
	
	This controller implements hierarchical operational space control using inverse dynamics.
	It can handle underactuated systems and multiple simultaneous tasks with proper priority
	handling through null-space projections.
	
	Control Philosophy:
	- Operates in task space (e.g., end-effector position/orientation) rather than joint space
	- Uses inverse dynamics to compute required joint torques for desired task accelerations
	- Handles underactuation by incorporating gravity/bias forces and uncontrolled DOFs
	- Uses dynamically consistent null-space projections for task hierarchy
	
	Key Features:
	1. **Underactuation Handling**: Accounts for passive/unactuated joints and gravity compensation
	2. **Task Space Control**: Controls end-effector motion directly in Cartesian space
	3. **Dynamic Consistency**: Proper inertial coupling between joint and task spaces
	4. **Hierarchical Tasks**: Can compose multiple tasks with proper priority ordering
	
	Control Pipeline:
	1. Compute dynamic matrices (mass matrix, inverse, task space mass matrix)
	2. Handle underactuation (incorporate bias forces, identify uncontrolled DOFs)
	3. Apply field effects (end-effector task control with null-space projection)
	4. Output joint torques that achieve desired task space behavior
	
	Mathematical Foundation:
	- Uses operational space formulation from Khatib (1987)
	- Implements null-space projection method from Mistry & Righetti (2012)
	- Computes dynamically consistent generalized inverses for proper force mapping
	
	Typical Use Cases:
	- End-effector position/orientation control
	- Cartesian impedance control
	- Multi-task control (e.g., reaching while maintaining posture)
	- Humanoid robot control with underactuated floating base
	"""

	def __init__(self, env: MujocoEnv, eef: EEFPoint, subtree_type: str, *args, **kwargs):
		"""
		Initialize the inverse dynamics controller.
		
		Args:
			env: MuJoCo environment instance
			eef: End-effector point defining the task (position, orientation, velocity fields)
			subtree_type: String identifying the kinematic subtree being controlled
		"""

		super().__init__(env, subtree_type)
		
		# Task definition
		self.eef = eef
		self.err = np.empty((0))           # Task error vector
		self.task_jac = np.empty((0))      # Task Jacobian matrix
		
		# Underactuation matrix: identifies unactuated DOFs
		# UA[i,i] = 1 for unactuated DOFs, 0 for actuated DOFs
		self.UA = np.eye(self.model.nv)
		self.UA[self._indexes['dof_ids'],self._indexes['dof_ids']] = 0

		# Gear ratios for torque scaling
		# Unactuated joints have gear ratio = 1 (no scaling)
		self.gears = np.ones(self.model.nv) 
		self.gears[self._indexes['dof_ids']] = self.model.actuator_gear[self._indexes["actuator_ids"], 0]

		# Pre-allocated dynamic matrices for efficiency
		self.M_inv = np.zeros((self._indexes['dof_ids'].shape[0], self._indexes['dof_ids'].shape[0]))
		self.M = np.zeros((self.model.nv, self.model.nv))          # Joint space mass matrix
		self.M_inv_tmp = np.zeros((self.model.nv, self.model.nv))  # Full inverse mass matrix


	def step(self, qact = None):	
		"""
		Compute inverse dynamics control signals for one time step.
		
		This method implements the complete operational space control pipeline:
		1. Compute dynamic matrices (mass matrix and inverses)
		2. Handle underactuation (incorporate bias forces and uncontrolled DOFs)
		3. Apply task space control (end-effector control with null-space projection)
		4. Map joint space accelerations to actuator torques
		
		The algorithm follows the hierarchical structure:
		- Priority 1: Handle underactuation constraints (gravity, unactuated DOFs)
		- Priority 2: Achieve end-effector task objectives in remaining DOF space
		
		Args:
			qact: Optional external joint forces/torques. If None, uses bias forces 
				  (gravity, Coriolis, centrifugal) scaled by gear ratios.
				  
		Returns:
			NDArray: Control torques for actuated joints (size = number of actuators)
		"""

		y = np.zeros((self.model.nv), dtype=np.float64) # Joint space control signal
		Q = np.eye(self.model.nv)                        # Null-space projection matrix

		## ---- Dynamic Matrices Computation ---- ##
		# Compute full joint space mass matrix
		mujoco.mj_fullM(self.model, self.M, self.data.qM)
		
		# Compute mass matrix inverse and task space mass matrix
		self._set_M_inv()
		self._set_Mx(self.eef.point.jac)

		## ---- Underactuation Handling ---- ##
		# Handle unactuated DOFs and gravity compensation
		qact = self.data.qfrc_bias / self.gears if qact is None else qact
		y, Q = self.underactuation(qact)

		## ---- Task Space Control ---- ##
		# Apply end-effector control in null-space of underactuation constraints
		y, Q = self.field_effects(y, Q)

		# Map joint space signals to actuator space
		ctrl = np.zeros((self.model.nu), dtype=np.float64)
		ctrl[self._indexes['actuator_ids']] = y[self._indexes['dof_ids']]

		return ctrl
		
	def underactuation(self, qfrc):
		"""
		Handle underactuation constraints and gravity compensation.
		
		This method deals with the first priority in the control hierarchy:
		handling unactuated degrees of freedom and incorporating bias forces
		(gravity, Coriolis, centrifugal forces).
		
		For underactuated systems (e.g., floating base robots), some DOFs cannot
		be directly controlled. This method:
		1. Applies external forces to unactuated DOFs
		2. Computes the null-space for controlled DOFs
		
		Args:
			qfrc: Joint forces/torques to apply (typically bias forces)
			
		Returns:
			tuple: (y, Q) where:
				y: Joint space control signal with underactuation handled
				Q: Null-space projection matrix for remaining control DOFs
		"""

		y = self.UA @ qfrc  # Apply forces to unactuated DOFs
		Q = np.eye(self.model.nv) - self.UA  # Null-space for actuated DOFs

		return y, Q
	
	def field_effects(self, y, Q):
		"""
		Apply end-effector task control in the null-space of underactuation constraints.
		
		This method implements the second priority in the control hierarchy:
		achieving end-effector task objectives without interfering with 
		underactuation constraints handled in the first priority.
		
		The method:
		1. Computes desired task space velocity from the end-effector field
		2. Projects the task Jacobian into the available null-space
		3. Computes required joint velocities using dynamically consistent inverse
		4. Updates the null-space projection for potential additional tasks
		
		Mathematical formulation:
		- J_s = J @ Q^T  (projected task Jacobian)
		- y += Q @ J_s^T @ M_x @ (x_dot_desired - J @ M^(-1) @ y)
		- Q = Q @ (I - J_s^T @ J_s^(T+))  (updated null-space)
		
		Args:
			y: Current joint space control signal
			Q: Current null-space projection matrix
			
		Returns:
			tuple: (y, Q) where:
				y: Updated joint space control signal with task control added
				Q: Updated null-space projection matrix for additional tasks
		"""
		
		# Get desired task space velocity from end-effector field
		xdot = self.eef.compute_effect(self.data.qvel)
		J = self.eef.point.jac

		# Project task Jacobian into available null-space
		J_s = J @ Q.T

		# Add task control in null-space using dynamically consistent inverse
		y = y + Q @ J_s.T @ self.Mx @ (xdot - J @ self.M_inv @ y)
		
		# Update null-space projection for potential additional tasks
		Q = Q @ (np.eye(self.model.nv) - J_s.T @ crux(J_s, self.Mx, self.M_inv))

		return y, Q
	
	def _set_Mx(self, J):
		"""
		Compute the task space mass matrix from the Jacobian.
		
		The task space mass matrix M_x represents the apparent inertia
		of the robot as seen from the task space (e.g., end-effector).
		It's computed as: M_x = (J @ M^(-1) @ J^T)^(-1)
		
		Uses robust matrix inversion with pseudo-inverse fallback for
		near-singular matrices.
		
		Args:
			J: Task Jacobian matrix
		"""
		Mx_inv = J @ self.M_inv @ J.T 
		
		if abs(np.linalg.det(Mx_inv)) >= 1e-6:
			self.Mx = np.linalg.inv(Mx_inv)
		else:
			self.Mx = np.linalg.pinv(Mx_inv, rcond=1e-6)

	def _set_M_inv(self):
		"""
		Compute the inverse of the joint space mass matrix.
		
		Uses robust matrix inversion with pseudo-inverse fallback for
		near-singular matrices (e.g., at kinematic singularities).
		"""

		if abs(np.linalg.det(self.M)) >= 1e-6:
			self.M_inv = np.linalg.inv(self.M)
		else:
			self.M_inv = np.linalg.pinv(self.M, rcond=1e-6)
	
	def __repr__(self):
		"""String representation showing the controller type, end-effector, and controlled subtree."""
		return f"IDController({self.eef}) || subtree: {self.subtree_type}"