"""
Joint space controllers for MuJoCo environments.

This module provides two different approaches for controlling joint positions:

1. JointController: Uses torque/force control with customizable gain functions
2. ConstraintJointController: Uses MuJoCo's equality constraints for precise position control

The choice between them depends on your needs:
- Use JointController for compliant, force-based control with tunable dynamics
- Use ConstraintJointController for rigid, precise position holding with constraint-based control
"""

import numpy as np
from gymnasium.envs.mujoco import MujocoEnv

from src.core.utils import IndexGetter

from typing import Dict

from numpy.typing import NDArray

class JointController:
	"""
	A PD-like controller in joint space that steers towards preferred joint angles.
	
	This controller computes control torques/forces based on the error between current 
	joint positions and target positions. It uses a gain function that can be customized
	with power laws, scaling, and clipping parameters.
	
	Control Method:
	- Computes joint position errors: error = target_pose - current_pose
	- Applies a gain function: control = k * sign(error) * (|error|/s)^pow
	- Returns control signals through actuators
	
	Key Features:
	- Direct torque/force control through actuators
	- Customizable gain function with power law scaling
	- Optional min/max clipping of control signals
	- Threshold-based dead zone to avoid micro-corrections
	
	Parameters:
		env: MujocoEnv instance
		target_posture: Dictionary mapping joint names to target angles
		control_params: Dictionary of control parameters (k, pow, s, min, max, thresh)
	"""

	DEFAULT_CTRL_PARAMS = {
		'k': 1.0,
		'pow': 1.0,
		's': 1.0,
		'min': None,
		'max': None,
		'thresh': 1e-3
	}

	def __init__(self, env: MujocoEnv, target_posture: Dict[str, float], control_params: Dict[str, Dict[str, float]], *args, **kwargs):

		self.data = env.data
		self.model = env.model

		self.actuators = env.mimo_actuators

		self.target_posture = target_posture
		self.inv_map = None


		self.set_pose(target_posture)

		assert self._indexes['dof_ids'].shape[0] == self.tgt_pose.shape[0], \
			f"Mismatch between target pose {self.tgt_pose.shape[0]} and number of joints {self._indexes['dof_ids'].shape[0]}."

		self.ctrl_params = self.DEFAULT_CTRL_PARAMS
		self.set_control_params(control_params)


		self.gain_func = None
		self._update_ctrl()

		#self.gears = self.model.actuator_gear[self._indexes["actuator_ids"], 0]

	
	def set_target(self, target_posture: Dict[str, float], control_params: Dict[str, Dict[str, float]] | Dict[str, NDArray]):
		self.set_pose(target_posture)
		self.set_control_params(control_params)
		self._update_ctrl()

	def set_pose(self, target_posture: Dict[str, float]):
		"""Set the target pose."""

		self.target_posture = target_posture

		self._indexes = IndexGetter(self.model)(joint_list=target_posture.keys())
		self.tgt_pose = np.empty(self._indexes['dof_ids'].shape[0], dtype=np.float64)
		self.inv_map = {}

		# always make sure to keep order consistent between tgt_pose and IndexGetter
		for i, jnt_name in enumerate(self._indexes['joint_names']):
			self.tgt_pose[i] = target_posture[jnt_name]
			self.inv_map[jnt_name] = i


	def set_control_params(self, params: Dict):
		"""Set the control parameters that determines the response to the joint distance from the target."""
		
		for key in self.ctrl_params.keys():
			if key in params:
				self.ctrl_params[key] = params[key]

	def _update_ctrl(self):
		k = self.ctrl_params['k']
		pow = self.ctrl_params['pow']
		s = self.ctrl_params['s']
		min = self.ctrl_params['min']
		max = self.ctrl_params['max']
		thresh = self.ctrl_params['thresh']


		if min is None and max is None:
			magnitude = lambda x: (x/s)**pow
		else:
			if max is None: max = np.inf
			if min is None: min = 0.0

			magnitude = lambda x: np.clip((x/s)**pow, min, max)
		
		_f = (lambda x : (-1)**(x<0)) if pow % 2 == 0 else (lambda x: 1)

		def func(x: NDArray) -> NDArray:

			if np.linalg.norm(x) < thresh:
				return np.zeros_like(x)

			return k * _f(x) *  magnitude(x)
		
		self.gain_func = func

	def step(self, *args, **kwargs):
		"""
		Take a step in the control loop.
		
		Computes joint position errors and applies the gain function to generate
		control torques/forces that are sent to the actuators.
		
		Returns:
			NDArray: Control signals for all actuators (non-zero only for controlled joints)
		"""
		
		err = self.tgt_pose - self.data.qpos[self._indexes['q_ids']]

		# control signal
		ctrl = np.zeros((self.model.nu), dtype=np.float64)
		ctrl[self._indexes['actuator_ids']] = self.gain_func(err) 

		return ctrl
	
	def __repr__(self) -> str:
		return f"JointController || subtree: {self._indexes['joint_names']})"

class ConstraintJointController(JointController):
	"""
	A constraint-based controller that uses MuJoCo's equality constraints to maintain joint positions.
	
	Unlike JointController which applies control torques/forces, this controller works by 
	activating/deactivating MuJoCo equality constraints that directly enforce joint positions.
	This approach can be more stable and precise for maintaining specific postures.
	
	Control Method:
	- Sets equality constraints in MuJoCo model (eq_data, eq_solimp, eq_solref)
	- Activates constraints when needed using eq_active flags
	- MuJoCo's constraint solver handles the actual position enforcement
	- Returns zero control signals (constraints do the work)
	
	Key Differences from JointController:
	- Uses constraint-based approach instead of torque control
	- More stable for maintaining exact positions
	- Can handle kinematic constraints that torque control cannot
	- Requires equality constraints to be defined in the MuJoCo model
	- Control parameters are SOLIMP parameters (5-element arrays) not gain parameters
	
	Parameters:
		env: MujocoEnv instance  
		target_posture: Dictionary mapping joint names to target angles
		control_params: Dictionary mapping subtree types to 5-element SOLIMP parameter arrays
	"""
	
	DEFAULT_CTRL_PARAMS = [0.001, 0.01, 0.99, 0.999, 10] # Almost inactive

	def __init__(self, env: MujocoEnv, target_posture: Dict[str, float], control_params: Dict[str, Dict[str, float]], *args, **kwargs):
		super().__init__(env, target_posture, control_params, *args, **kwargs)

		self.step_foo = self._switch_on

	def set_control_params(self, params: Dict):
		"""
		Set MuJoCo SOLIMP constraint parameters.
		
		Unlike JointController which uses gain parameters, this method expects
		SOLIMP parameters that control constraint solver behavior.
		
		Args:
			params: Dictionary mapping subtree types to 5-element SOLIMP parameter arrays
					[dmin, dmax, width, midpoint, power]
		"""
		# params keys are supposed to be entries for IndexGetter
		self.ctrl_params = np.zeros((self.tgt_pose.shape[0], 5), dtype=np.float64)
		self.ctrl_params[:] = self.DEFAULT_CTRL_PARAMS

		for key in params.keys():
			_idxs = IndexGetter(self.model)(subtree_type=key)
			if _idxs is None: # deal with potentially extra parameters
				continue

			for jnt_name in _idxs['joint_names']:
				_i = self.inv_map.get(jnt_name, None)
				if _i is None: continue

				assert len(params[key]) == 5, f"Expected 5 SOLIMP control parameters, got {len(params[key])} for {key=}."

				self.ctrl_params[_i] = params[key]
				r = self.model.joint(jnt_name).range
				self.ctrl_params[_i, 2] *= r[1] - r[0]

	def set_pose(self, target_posture: Dict[str, float]):
		super().set_pose(target_posture)

		self.eq_ids = np.zeros_like(self.tgt_pose, dtype=np.int32)
		# get all equality ids
		for i, jnt_name in enumerate(self._indexes['joint_names']):
			self.eq_ids[i] = self.model.eq(jnt_name).id
		
	def _get_polycoef(self, eq_ids: NDArray):
		return self.model.eq_data[eq_ids, :5]
	
	def _update_ctrl(self):
		self.model.eq_data[self.eq_ids, 0] 	= self.tgt_pose
		self.model.eq_data[self.eq_ids, 1:5] = 0
		self.model.eq_solimp[self.eq_ids] = self.ctrl_params
		self.model.eq_solref[self.eq_ids] = np.array([0.05, 0.6])
	
	def step(self, *args, **kwargs):
		"""
		Take a step in the control loop.
		
		Activates equality constraints on first call, then does nothing on subsequent calls.
		The constraints handle position control automatically through MuJoCo's solver.
		
		Returns:
			NDArray: Zero control signals (constraints do the actual control work)
		"""
		
		self.step_foo()
		return np.zeros((self.model.nu), dtype=np.float64)

	def _activate_constraints(self):
		"""Activate the equality constraints."""
		
		self.model.eq_active[self.eq_ids] = 1


	def _deactivate_constraints(self):
		"""Activate the equality constraints."""
		
		self.model.eq_active[self.eq_ids] = 0

	def _switch_on(self):
		self._activate_constraints()
		self.step_foo = lambda : None
	
	def _switch_off(self):
		self._deactivate_constraints()
		self.step_foo = lambda : None
