__all__ = ['IndexGetter', 'quat_wlast', 'quat_wlast', 'get_geoms_for_body_tree']

import numpy as np
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
import gymnasium as gym


from typing import TypedDict, Optional
from numpy.typing import NDArray
from typing import List, Tuple

from functools import partial

### Math ###

def quat_wlast(quat: NDArray) -> NDArray:
	"""Transforms a quaternion from [x, y, z, w] to [w, x, y, z] format."""
	return np.array([quat[1], quat[2], quat[3], quat[0]])

def quat_wfirst(quat: NDArray) -> NDArray:
	"""Transforms a quaternion from [w, x, y, z] to [x, y, z, w] format."""
	return np.array([quat[3], quat[0], quat[1], quat[2]])


def argmedian(x):
  # WARN: only returns item the true median if x has odd len
  return np.argpartition(x, len(x) // 2)[len(x) // 2]

def median_distanced(x: NDArray, point: NDArray) -> float:
  """
  Return the "median-distanced" element of a list of point form a target point.

  The "median-distanced" element is the element whose distance to the target point 
  is the median of the distances of all elements to the target point.
  """

  dists = np.linalg.norm(x - point, axis=1)
  return x[argmedian(dists)]

##############

### Gymnasium ###

def get_model_data(env: Optional[gym.Env] = None, 
				   model: Optional[mujoco.MjModel] = None, data: Optional[mujoco.MjData] = None) -> Tuple[mujoco.MjModel, mujoco.MjData]:
	
	"""Compatibility function to ease passing from gymnsium 0.28 to 1.0
	"""

	if env is not None:
		return env.get_wrapper_attr('model'), env.get_wrapper_attr('data')
	else:
		assert model is not None and data is not None, \
			"If an environment is not specified, both a model and a data structure must be passed."
		
		return model, data

##############

### MuJoCo ###

def get_geoms_for_body(sim_model, body_id):
	""" 
	Original code from [MIMo/mimoEnv/utils.py](https://github.com/trieschlab/MIMo/blob/main/mimoEnv/utils.py) 
	Returns all geom ids belonging to a given body.

	Args:
		sim_model (mujoco.MjModel): The MuJoCo model object.
		body_id (int): The id of the body.

	Returns:
		List[int]: A list of the ids of the geoms belonging to the given body.
	"""
	geom_start = sim_model.body_geomadr[body_id]
	geom_end = geom_start + sim_model.body_geomnum[body_id]
	return range(geom_start, geom_end)

def get_geoms_for_body_tree(sim_model, body_id):
	"""
	Get the range of geometries for a given body tree in the simulation model.
	Args:
		sim_model (mujoco.MjModel): The Mujoco model object.
		body_id (int): The id of the root body.
	Returns:
		range: The range of geometries IDs for the given body tree, or None if the geometries are not consecutive.
	"""	

	bodies = get_child_bodies(sim_model, body_id)
	geoms = []
	for body in sorted(bodies):
		geoms.append(get_geoms_for_body(sim_model, body))
		
	are_geoms_consecutive = True

	for i in range(len(geoms)-1):
		are_geoms_consecutive = are_geoms_consecutive and geoms[i].stop == geoms[i+1].start

		if not are_geoms_consecutive:
			return None

	return range(geoms[0].start, geoms[-1].stop)

def get_child_bodies(sim_model, body_id):
	""" 
	Original code from [MIMo/mimoEnv/utils.py](https://github.com/trieschlab/MIMo/blob/main/mimoEnv/utils.py) 
	
	Returns the subtree of the body structure that has the provided body as its root.

	The body structure is defined in the MuJoCo XMLs. This function returns a list containing the ids of all descendant
	bodies of a given body, including the given body.

	Args:
		sim_model (mujoco.MjModel): The MuJoCo model object.
		body_id (int): The id of the root body.

	Returns:
		List[int]: The ids of the bodies in the subtree.
	"""
	children_dict = {}
	# Built a dictionary listing the children for each node
	for i in range(sim_model.nbody):
		parent = sim_model.body_parentid[i]
		if parent in children_dict:
			children_dict[parent].append(i)
		else:
			children_dict[parent] = [i]
	# Collect all the children in the subtree that has body_id as its root.
	children = []
	to_process = [body_id]
	while len(to_process) > 0:
		child = to_process.pop()
		children.append(child)
		# If this node has children: add them as well
		if child in children_dict:
			to_process.extend(children_dict[child])
	return children


def world_pos_to_body(mujoco_data, position, body_id):
	""" 
	Original code from [MIMo/mimoEnv/utils.py](https://github.com/trieschlab/MIMo/blob/main/mimoEnv/utils.py) 
	Converts position from the world coordinate frame to a body specific frame.

	Position can be a vector or an array of vectors such that the last dimension has size 3.

	Args:
		mujoco_data (mujoco.MjData): The MuJoCo data object.
		position (numpy.ndarray): Array containing position(s). Its shape should be either (3,) or (.., 3).
		body_id (int): The id of the geom.

	Returns:
		numpy.ndarray: Array of the same shape as the input array with the converted coordinates.
	"""
	rel_pos = position - mujoco_data.xpos[body_id]
	rel_pos = np.transpose(
		np.transpose(np.reshape(mujoco_data.xmat[body_id], (3, 3)))).dot(np.transpose(rel_pos))
	return rel_pos

#############

### Model ###

class JointGroup(TypedDict):
	joint_names: 	NDArray[np.str_]
	joint_ids: 		NDArray[np.int_]
	actuator_names: NDArray[np.str_]
	actuator_ids: 	NDArray[np.int_]
	dof_ids: 		NDArray[np.int_]
	q_ids: 			NDArray[np.int_]

class IndexGetter:
	#SUBTREE_TYPES = [
	#	"arm", "arms", "arm_left", "left_arm", "arm_right", "right_arm", 
	#	"eyes", "eye", "eye_left", "left_eye", "eye_right", "right_eye", 
	#	"legs",
	#	"head", 
	#	"anything", "nothing", 
	#]
	
	def __init__(self, env: Optional[gym.Env] = None, subtree_types: List[str] = [], model: mujoco.MjModel = None, data: mujoco.MjData = None):
		
		if env is not None:
			self.model = env.get_wrapper_attr('model')
			self.data = env.get_wrapper_attr('data')
		else:
			assert model is not None and data is not None, \
				"If an environment is not specified, both a model and a data structure must be passed."
			
			self.model = model
			self.data = data

		self.SUBTREE_TYPES = subtree_types

	def __call__(self, 
			  subtree_type=None, root_body_id=None, joint_list=None, 
			  check_constraints: bool = True, include_root: bool = True) -> JointGroup | None:

		assert (subtree_type is not None) + (joint_list is not None) + (root_body_id is not None) == 1, \
			"Only one of subtree_type, joint_list or root_body_id must be provided"


		if subtree_type is not None:
			#assert subtree_type in self.SUBTREE_TYPES, f"Invalid subtree type: {subtree_type}, expected one of {self.SUBTREE_TYPES}"
			if len(self.SUBTREE_TYPES) > 0 and not (subtree_type in self.SUBTREE_TYPES):
				# Silently ignore the wrong subtree type
				return None
			
			_filter = self._get_predef_filter(subtree_type)

		elif root_body_id is not None:
			_filter = self._get_subtree_filter(root_body_id, include_root)
		
		elif joint_list is not None:
			_filter = lambda j: j in joint_list

		joint_names = []
		joint_ids = []
		actuator_names = []
		actuator_ids = []
		dof_ids = []
		q_ids = []

		for i in range(self.model.nu):
			actuator_name = self.model.actuator(i).name
			actuator_id = self.model.actuator(i).id
			actuator_name = actuator_name.split('act:')[1]
			joint_id = self.model.actuator(i).trnid[0]
			joint_name = self.model.joint(joint_id).name

			constraint_id = self.model.equality(joint_name).id # To check if the joint is constrained (e. fixing the initial position)
			
			if _filter(joint_name) and not (check_constraints and self.data.eq_active[constraint_id]):
				joint_names.append(joint_name)
				joint_ids.append(joint_id)

				actuator_names.append(actuator_name)
				actuator_ids.append(actuator_id)

				dof_ids.append(self.model.joint(joint_id).dofadr[0])
				q_ids.append(self.model.joint(joint_id).qposadr[0])
		
		joint_group: JointGroup = {
			'joint_names': np.array(joint_names),
			'joint_ids': np.array(joint_ids, dtype=np.int32),
			'actuator_names': np.array(actuator_names),
			'actuator_ids': np.array(actuator_ids, dtype=np.int32),
			'dof_ids': np.array(dof_ids, dtype=np.int32),
			'q_ids': np.array(q_ids, dtype=np.int32),
		}

		return joint_group
	

	def _get_subtree_filter(self, root_body_id: str | int, include_root: bool) -> callable:
		root_body_id = self.model.body(root_body_id).id
		body_subtree = get_child_bodies(self.model, root_body_id)
	
		if not include_root:
			body_subtree = body_subtree[1:]

		jnts = []
		for body in body_subtree:
			num_jnts = self.model.body(body).jntnum.item()
			frst_jnt = self.model.body(body).jntadr.item()
			jnts += [self.model.joint(frst_jnt + i).name for i in range(num_jnts)]
		
		def _is_in_subtree(joint_name: str) -> bool:
			return joint_name in jnts
	
		return _is_in_subtree
	
	@classmethod
	def _get_predef_filter(cls, subtree_type: str) -> callable:
		if subtree_type in ["arm", "arms"]:
			return cls._joint_is_arm
		elif subtree_type in ["arm_left", "left_arm"] :
			return lambda j: cls._joint_is_arm(j, "left")
		elif subtree_type in ["arm_right", "right_arm"]:
			return lambda j: cls._joint_is_arm(j, "right")
		elif subtree_type in ["eye", "eyes"]:
			return cls._joint_is_eye
		elif subtree_type in ["eye_left", "left_eye"]:
			return partial(cls._joint_is_eye, side='left')
		elif subtree_type in ["eye_right", "right_eye"]:
			return partial(cls._joint_is_eye, side='right')
		elif subtree_type == "head":
			return partial(cls._joint_is_head, eyes=False)
		elif subtree_type == "anything":
			return cls._joint_is_anything
		elif subtree_type == "legs":
			return cls._joint_is_legs
		elif subtree_type == "nothing":
			return cls._joint_is_nothing
		else:
			raise ValueError(f"Invalid subtree type: {subtree_type}, expected one of {cls.SUBTREE_TYPES}")

	@classmethod
	def _joint_is_arm(cls, joint_name: str, side: str = '') -> bool:
		return (joint_name.startswith("robot:") and ( \
				('shoulder' in joint_name) or \
				('elbow' in joint_name) or \
				('hand' in joint_name))) and (side in joint_name)

	@classmethod
	def _joint_is_anything(cls, joint_name: str) -> bool:
		return joint_name.startswith("robot:")

	@classmethod
	def _joint_is_nothing(cls, joint_name: str) -> bool:
		return False

	@classmethod
	def _joint_is_eye(cls, joint_name: str, side: str = '') -> bool:
		return joint_name.startswith("robot:") and \
			('eye' in joint_name) and \
			(side in joint_name)

	@classmethod
	def _joint_is_head(cls, joint_name: str, eyes: bool = True) -> bool:
		return joint_name.startswith("robot:") and \
			('head' in joint_name or \
			(eyes and cls._joint_is_eye(joint_name)))  # material implication, F iff eyes and not joint_is_eye

	@classmethod
	def _joint_is_legs(cls, joint_name: str) -> bool:
		return joint_name.startswith("robot:") and \
			('hip' 	in joint_name or \
			#'left_hip' 		in joint_name or \
			'knee' 			in joint_name or \
			'foot' 			in joint_name or \
			'toes' 			in joint_name	)