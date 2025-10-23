__all__ = ['Frame', 'ControllableFrame', 'OffsetFrame', 'MinOffsetFrame']

import numpy as np
import mujoco

from numpy.typing import NDArray

from gymnasium.envs.mujoco import MujocoEnv
import gymnasium as gym

import lucignolo.core.utils as utils

from functools import partial
from scipy.spatial.transform import Rotation as R

from abc import ABC
from typing import Optional

class iFrame(ABC):
	
	@property
	def xpos(self) -> NDArray: return None

	@property
	def xmat(self) -> NDArray: return None

	@property
	def xquat(self) -> NDArray: return None

	@property
	def jac(self) -> NDArray: return None
	
	@xpos.setter
	def xpos(self, value: NDArray): pass

	@xmat.setter
	def xmat(self, value: NDArray): pass

	@xquat.setter
	def xquat(self, value: NDArray): pass

	@jac.setter
	def jac(self, value: NDArray): pass



class StaticFrame(iFrame):
	
	def __init__(self, xpos = None, xmat = None, xquat = None, jac = None, frame: iFrame = None):
		"""
		Initializes the object with optional parameters for position, matrix, quaternion, and Jacobian.

		Parameters:
		- xpos (NDArray | None): The position array. Defaults to None.
		- xmat (NDArray | None): The matrix array. Defaults to None.
		- xquat (NDArray | None): The quaternion array. Defaults to None.
		- jac (NDArray | None): The Jacobian array. Defaults to None.
		- frame (iFrame): An instance of the iFrame class. If provided, the values of xpos, xmat, xquat, and jac will be set based on the values of the frame object.

		"""

		if frame is not None:
			xpos 	= frame.xpos.copy()
			xmat 	= frame.xmat.copy()
			xquat 	= frame.xquat.copy()
			jac 	= frame.jac.copy()

		self.xpos: NDArray | None = xpos.copy() if xpos is not None else None
		if xmat is not None:
			self.xmat: NDArray | None = xmat.copy()
		if xquat is not None:
			self.xquat: NDArray | None = xquat.copy()
		self.jac: NDArray | None = jac.copy() if jac is not None else None
	
	@property
	def xpos(self) -> NDArray:
		return self._xpos
	@property
	def xmat(self) -> NDArray:
		return self._xmat
	@property
	def xquat(self) -> NDArray:
		return self._xquat
	@property
	def jac(self) -> NDArray:
		return self._jac
	
	@xpos.setter
	def xpos(self, value: NDArray):
		assert value.shape == (3,), f"Expected shape (3,), got {value.shape}"
		setattr(self, '_xpos', value)

	@xmat.setter
	def xmat(self, value: NDArray):
		assert value.shape == (3,3) or value.shape == (9,), f"Expected shape (3,3) or (9,), got {value.shape}"
		setattr(self, '_xmat', value.reshape(3,3))
		setattr(self, '_xquat', utils.quat_wfirst(R.from_matrix(value.reshape(3,3)).as_quat()))

	@xquat.setter
	def xquat(self, value: NDArray):
		assert value.shape == (4,), f"Expected shape (4,), got {value.shape}"
		setattr(self, '_xquat', value)
		setattr(self, '_xmat', R.from_quat(utils.quat_wlast(value)).as_matrix())

	@jac.setter
	def jac(self, value: NDArray):
		setattr(self, '_jac', value)

class Frame(iFrame):
	def __init__(self, env: Optional[gym.Env] = None, name: str = "", ftype: str = "body", heading: NDArray = np.asarray([0, 0, 1]), 
			  model: mujoco.MjModel = None, data: mujoco.MjData = None):

		self.model, self.data = utils.get_model_data(env, model, data)
		
		self.name = name
		self.ftype = ftype
		self.k = heading
		self._get_xpos, self._get_xmat, self._get_xquat, self._get_jac = self.frame_getter()
		self._jac = np.empty((6,self.model.nv), dtype=np.float64)

		self.is_static = True
	
	@property
	def xpos(self) -> NDArray:
		return self._get_xpos(self.name)
	@property
	def xmat(self) -> NDArray:
		return self._get_xmat(self.name).reshape(3,3)
	@property
	def xquat(self) -> NDArray:
		return self._get_xquat(self.name)
	@property
	def jac(self) -> NDArray:
		self._get_jac(jacp=self._jac[:3], jacr=self._jac[3:])
		return self._jac
	
	@xpos.setter
	def xpos(self, value: NDArray):
		raise AttributeError("xpos is read-only")
	@xmat.setter
	def xmat(self, value: NDArray):
		raise AttributeError("xmat is read-only")
	@xquat.setter
	def xquat(self, value: NDArray):
		raise AttributeError("xquat is read-only")
	@jac.setter
	def jac(self, value: NDArray):
		raise AttributeError("jac is read-only")
	
	def frame_getter(self):
		"""Returns a function that retrieves the frame of the given type and name from the environment.
		
		The function can be used to get the xpos, xmat, ... of the frame.
		"""

		ftype = self.ftype

		if ftype == 'site':
			_get_xpos  = lambda name : self.data.site(name).xpos
			_get_xmat  = lambda name : self.data.site(name).xmat
			_get_xquat = lambda name : utils.quat_wfirst(R.from_matrix(self.data.site(name).xmat.reshape(3,3)).as_quat())
			_get_jac   = partial(mujoco.mj_jacSite,  m=self.model, d=self.data, site=int(self.model.site(self.name).id))
		elif ftype == 'body':
			_get_xpos  = lambda name : self.data.body(name).xpos
			_get_xmat  = lambda name : self.data.body(name).xmat
			_get_xquat = lambda name : self.data.body(name).xquat
			_get_jac   = partial(mujoco.mj_jacBody, m=self.model, d=self.data, body=int(self.model.body(self.name).id))
		elif ftype == 'ibody':
			_get_xpos  = lambda name : self.data.body(name).xipos
			_get_xmat  = lambda name : self.data.body(name).ximat
			_get_xquat = lambda name : utils.quat_wfirst(R.from_matrix(self.data.body(name).ximat.reshape(3,3)).as_quat())
			_get_jac   = partial(mujoco.mj_jacBodyCom, m=self.model, d=self.data, body=int(self.model.body(self.name).id))
		elif ftype == 'geom':
			_get_xpos  = lambda name : self.data.geom(name).xpos
			_get_xmat  = lambda name : self.data.geom(name).xmat
			_get_xquat = lambda name : utils.quat_wfirst(R.from_matrix(self.data.geom(name).xmat.reshape(3,3)).as_quat())
			_get_jac   = partial(mujoco.mj_jacGeom, m=self.model, d=self.data, geom=int(self.model.geom(self.name).id))
		elif ftype == 'worldbody':
			_get_xpos  = lambda name : self.data.body(name).xpos
			_get_xmat  = lambda name : self.data.body(name).xmat
			_get_xquat = lambda name : self.data.body(name).xquat
			_get_jac   = lambda *args, **kwargs: None # TODO: check if this works
		

		return _get_xpos, _get_xmat, _get_xquat, _get_jac

	def __copy__(self):
		return self.__class__(model=self.model, data=self.data, name=self.name, ftype=self.ftype, heading=self.k)
	
	def static_copy(self):
		return StaticFrame(xpos=self.xpos, xmat=self.xmat, xquat=self.xquat, jac=self.jac)

class ControllableFrame(Frame):
	def __init__(self, env: MujocoEnv, name: str, heading: NDArray = np.asarray([0, 0, 1])):
		super().__init__(env, name, "body", heading)

		self.body_id = self.model.body(name).id
		self.mocap_id = self.model.body_mocapid[self.body_id]

		assert self.mocap_id >= 0, f"Body <{name}> is not a mocap body"

		self.is_static = False

	@property
	def xpos(self) -> NDArray:
		return self.data.xpos[self.body_id]

	@xpos.setter
	def xpos(self, xpos: NDArray):
		self.data.mocap_pos[self.mocap_id] = xpos

	@property
	def xmat(self) -> NDArray:
		return self.data.xmat[self.body_id].reshape(3,3)
	
	@xmat.setter
	def xmat(self, xmat: NDArray):
		self.data.mocap_quat[self.mocap_id] = utils.quat_wfirst(R.from_matrix(xmat.reshape(3,3)).as_quat())
	
	@property
	def xquat(self) -> NDArray:
		return self.data.mocap_quat[self.mocap_id]
	
	@xquat.setter
	def xquat(self, xquat: NDArray):
		self.data.mocap_quat[self.mocap_id] = xquat

class OffsetFrame(Frame):
	def __init__(self, env: MujocoEnv, name: str, ftype: str, offset: NDArray, 
			  offset_in_world_coords: bool = False, keep_world_z: bool = False,
			  heading: NDArray = np.asarray([0, 0, 1])):
		super().__init__(env, name, ftype, heading)
		self.offset = offset
		self.owc = offset_in_world_coords

		self._get_offset = None

		if self.owc:
			self._get_offset = self._world_offset
		else:
			if keep_world_z:
				self._get_offset = self._local2world_keepz_offset
			else:
				self._get_offset = self._local2world_offset
	@property
	def xpos(self) -> NDArray:
		return self._get_xpos(self.name) + self._get_offset()
	
	def _local2world_keepz_offset(self):
		r = self._get_xmat(self.name).reshape(3,3).copy()

		r[2,:] = np.asarray([0, 0, 1])  # keep world z-axis
		r[:,2] = np.asarray([0, 0, 1])  # keep world z-axis
		r /= np.linalg.norm(r[:2,:2])  # normalize the rotation matrix
		return r.dot(self.offset)

	def _local2world_offset(self):
		return self._get_xmat(self.name).reshape(3,3).dot(self.offset)
	
	def _world_offset(self):
		return self.offset

class MinOffsetFrame(OffsetFrame):
	def __init__(self, env: MujocoEnv, name: str, ftype: str, offset: NDArray, 
			  heading: NDArray = np.asarray([1, 0, 0]),
			  min_z: float = 0.0,
			  ):
		super().__init__(env, name, ftype, offset, offset_in_world_coords=False, keep_world_z=False, heading=heading)

		self.min_z = min_z

	@property
	def xpos(self) -> NDArray:
		xpos = super().xpos
		xpos[2] = max(xpos[2], self.min_z)
		return xpos