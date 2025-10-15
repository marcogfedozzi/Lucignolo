
import numpy as np

import mujoco
from gymnasium.envs.mujoco import MujocoEnv

from lucignolo.core.eef_point import EEFPoint
from .base import Controller, crux, NoiseGenerator
from typing import Optional

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

	def __init__(self, env: MujocoEnv, eef: EEFPoint, subtree_type: str, actuators_prefix: Optional[str] = None, *args, **kwargs):
		"""
		Initialize the inverse dynamics controller.
		
		Args:
			env: MuJoCo environment instance
			eef: End-effector point defining the task (position, orientation, velocity fields)
			subtree_type: String identifying the kinematic subtree being controlled
		"""

		super().__init__(env, subtree_type, actuators_prefix)

		print("#############")
		print(self._indexes)
		print("#############")

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
	
class NoisyIDController(IDController):

	def __init__(self, env: MujocoEnv, eef: EEFPoint, subtree_type: str, noise_var: float, noise_lambda: float = 1.0, *args, **kwargs):
		super().__init__(env, eef, subtree_type)
		self.noise_gen = NoiseGenerator(self.model.actuator_ctrlrange[self._indexes['actuator_ids']], noise_var, noise_lambda)

	def step(self, qact = None):
		ctrl = super().step(qact)
		ctrl[self._indexes['actuator_ids']] += self.noise_gen()
		return ctrl