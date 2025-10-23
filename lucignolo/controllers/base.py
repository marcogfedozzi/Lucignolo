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

from gymnasium.envs.mujoco import MujocoEnv

from lucignolo.core.utils import IndexGetter, get_model_data
from numpy.typing import NDArray
import numpy as np
from typing import Optional
import gymnasium as gym
import mujoco

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


def get_actuators(env: gym.Wrapper, actuators_prefix: Optional[str] = None):

	actuators = []

	model = env.get_wrapper_attr("model")

	for i in range(model.nu):
		actuator_name = model.actuator(i).name
		if actuators_prefix is None or actuator_name.startswith(actuators_prefix):
			actuators.append(i)

	return np.asarray(actuators)

class _iController:
	def __init__(self, env: Optional[gym.Env] = None, 
			  model: mujoco.MjModel = None, data: mujoco.MjData = None, 
			  *args, **kwargs):
		self.model, self.data = get_model_data(env, model, data)


class Controller(_iController):
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

	def __init__(self, env: Optional[gym.Env] = None, subtree_type: str = "", actuators_prefix: Optional[str] = None, model: mujoco.MjModel = None, data: mujoco.MjData = None, *args, **kwargs):
		"""
		Initialize base controller.
		
		Args:
			env: MuJoCo environment instance
			subtree_type: String identifying the kinematic subtree (e.g., 'left_arm', 'torso')
			actuators_prefix: Optional prefix to filter actuators by name
		"""
		super().__init__(env=env, model=model, data=data)


		self.actuators = get_actuators(env, actuators_prefix)

		self.subtree_type = subtree_type
		self._indexes = IndexGetter(env)(subtree_type=self.subtree_type)

	def step(self, qact = None):
		"""Compute control signals for one time step. Must be implemented by subclasses."""
		raise NotImplementedError
	
class NoiseGenerator:
	def __init__(self, ranges: NDArray,  noise_var: float, noise_lambda: float = 1.0) -> None:
		self.nv = noise_var

		self.stds = (ranges[:,1] - ranges[:,0]) / 6 # assume gaussian with limits = 3*std
		self.mus = (ranges[:,1] + ranges[:,0]) / 2
		self.prev_noise = np.zeros_like(self.mus)
		self._lambda = noise_lambda
	def __call__(self):
		noise = self._lambda * (self.nv * self.stds * np.random.randn(len(self.mus)) + self.mus) + (1-self._lambda)*self.prev_noise
		self.prev_noise = noise
		return noise

class NoiseController(Controller):

	def __init__(self, env: MujocoEnv, subtree_type: str, noise_var: float, noise_lambda: float = 1.0, *args, **kwargs):

		super().__init__(env, subtree_type)
		self.noise_gen = NoiseGenerator(self.model.actuator_ctrlrange[self._indexes['actuator_ids']], noise_var, noise_lambda)

	def step(self, qact = None):
		ctrl = np.zeros((self.model.nu), dtype=np.float64)
		ctrl[self._indexes['actuator_ids']] += self.noise_gen()
		return ctrl
	