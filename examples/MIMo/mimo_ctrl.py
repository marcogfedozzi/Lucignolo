"""
Create a MIMo environment, then add a point in front of it and instruct the inverse dynamics controller to reach it.

What we need:
- MIMo environment
- Inverse Dynamics controller
- point to reach (for simplicity, a sequence with len = 1)
- a translation task that connect one of MIMo's hands' End Effector with the point
"""

import lucignolo as lc
import mimoEnv.utils as me_utils
import gymnasium as gym
import time
import numpy as np
import os
import logging
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import Wrapper

import cv2 as cv


def set_logging(loglevel: str = "INFO"):
	log_c = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]

	if isinstance(loglevel, str):
		ll = loglevel.upper()
		assert ll in log_c, "Invalid log level"

	elif isinstance(loglevel, int):
		assert loglevel >= 0 and loglevel < len(log_c), "Invalid log level"
		ll = log_c[loglevel]

	else:
		raise ValueError("Invalid log level")
	

	loglevel = getattr(logging, ll, None)

	logging.basicConfig(level=loglevel, force=True)

	logging.getLogger().handlers[0].setFormatter(logging.Formatter('%(levelname)s:%(message)s'))

def get_geoms_for_body_tree(sim_model, body_id):
	"""
	Get the range of geometries for a given body tree in the simulation model.
	Args:
		sim_model (mujoco.MjModel): The Mujoco model object.
		body_id (int): The id of the root body.
	Returns:
		range: The range of geometries IDs for the given body tree, or None if the geometries are not consecutive.
	"""	

	bodies = me_utils.get_child_bodies(sim_model, body_id)
	geoms = []
	for body in sorted(bodies):
		geoms.append(me_utils.get_geoms_for_body(sim_model, body))
		
	are_geoms_consecutive = True

	for i in range(len(geoms)-1):
		are_geoms_consecutive = are_geoms_consecutive and geoms[i].stop == geoms[i+1].start

		if not are_geoms_consecutive:
			return None

	return range(geoms[0].start, geoms[-1].stop)


def main():
	set_logging("INFO")
	
	env_name = "MIMoBench-v0"

	"""
	We load the local version of the MIMo XML, that contains the "EEF" sites that 
	we will use as end effectors for the inverse controller
	"""
	file_path = os.path.dirname(os.path.abspath(__file__))
	model_path = os.path.join(file_path, "assets", "control.xml") 

	 #_render_mode = "rgb_array"
	_render_mode = "human"
	
	# Note: the env is wrapped by default
	env: Wrapper = gym.make(env_name, model_path=model_path, render_mode=_render_mode) #, show_sensors=False, print_space_sizes=True)

	model, data = lc.core.utils.get_model_data(env)

	logging.info("Environment created")

	_init_steps = 0 #100
	env.reset()
	for _ in range(_init_steps):
		env.step(np.zeros(env.action_space.shape))
	
	logging.info(f"Simulation initialized with {_init_steps} steps")

	"""
	Available EEF (see XML)
	- head
	- right_eye
	- left_eye
	- right_hand
	- left_hand
	"""
	controlled_body = "left_hand" 

	"""Let's deactivate the contact for the chosen EEF, to simplify the control task"""

	geoms = get_geoms_for_body_tree(model, model.body(controlled_body).id)

	orig_contype = model.geom_contype[geoms]
	orig_conaff = model.geom_conaffinity[geoms]
	model.geom_contype[geoms] = 0
	model.geom_conaffinity[geoms] = 0


	fourcc = cv.VideoWriter_fourcc(*'mp4v')

	_render_video = not (_render_mode == "human")

	if _render_video:
		out = cv.VideoWriter(os.path.join(file_path, f"reachtest_{controlled_body}.mp4"), fourcc, int(1/env.get_wrapper_attr('dt')), (500, 500))
		def render(env):
			img = env.render()
			img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
			out.write(img_bgr)

	else:
		def render(env):
			env.render()

	max_steps = 10000

	"""
	In order to use the Inverse Dynamics Controller to move MIMo we need 3 objects:
	- a [target], the point in space we are intersted in
	- an [end effector], the point on MIMo we want to control
	- one or more [field(s)], the mathematical relation between the [target] and the [end effector]
	"""

	## Target ##

	"""
	A [target] is a mocap body defined in the scene XML. In the 'control.xml' scene we are using here
	we defined 3 targets, one for the head and one for each hand. Note that this is a simple convention,
	as any target could be associated to any end effector, in a many-to-many configuration!

	We can choose among different types of frames, but if we want to be able to move the
	target around in real time we can default to 'ControllableFrame'
	"""

	target = lc.core.frames.ControllableFrame(env, "target:"+controlled_body)

	logging.info("Target created")

	## End Effector ##

	"""
	An [end effector] (EEF) is a 'site' object defined in MIMo's xml. 
	By inspecting that file ('assets/mimo/MIMo_model(v2).xml') you can see we defined an EEF
	for the head, one for each eye, and one for each hand.
	The location in the body tree determines where the site is located on MIMo's body.
	Every site here defined is located in the center of the respective body part, and
	oriented with the 'z' azis exiting perpendicular to its surface 
	(head -> pointing where the nose points, eye -> optical axis, hand -> exiting from the palm)
	"""

	eef_frame = lc.core.frames.Frame(env, "eef:"+controlled_body, "site", heading=np.array([0,0,1]))

	eef = lc.core.eef_point.EEFPoint(eef_frame)

	logging.info("EEF created")


	## Fields ##

	"""
	[field(s)] are nothinig but the mathematical formulation that describes the force that acts
	on the [EEF] given its relative 6D pose (translation, orientation) w.r.t. the [target].
	While any mathematical formulation is possible here, the most frequent one are readily available
	as specific functions.

	Let's add an attractive field that brings the hand towards the target (simple reach).
	"""

	attractive_field = lc.fields.get_field(
		center=target,
		field_type="translation",
		params={
			"k": 800.0,
			"pow": 1.0,
			"max": 0.1,
		}
	)

	alignment_field = lc.fields.get_field(
		center=target,
		field_type="misalignment",
		params={
			"k": 100.0,
			"pow": 2.0,
			"s": 0.5,
			"radii": [0.1, 0.3] 
		}
	)

	viscous_field = lc.fields.get_field(
		center=target,
		field_type="viscosity",
		params={
			"k": [20.0, 10],
			"pow": 1.0,
		}
	)

	"""This is a proportional field, so the force applied at the [EEF] equals 5 times the [EEF]-[target] distance"""

	"""We now specify that the [EEF] will be under the influence of this [field]"""
	eef.add_field(alignment_field)
	eef.add_field(attractive_field)
	eef.add_field(viscous_field)


	logging.info("Field created and added")

	## Controller ##
	"""
	It's finally time to define our controller. For this we will pick the Inverse Dynamic controller, 
	as we want to drive the [EEF] and we do not really care for what the joint do to move it.

	We will further specify that we do not want to move the whole MIMo body, but we only want to control
	the limb that the [EEF] is attached to. To achieve that we specify the subtree type.
	"""

	subtree_type = controlled_body if "head" in controlled_body or "eye" in controlled_body else controlled_body.replace("hand", "arm")

	controller = lc.controllers.IDController(env, eef, subtree_type)

	logging.info("Controller created")


	## Setup ##

	"""Let's now position the target somewhere in front of MIMo"""

	"""This is the relative position w.r.t. MIMo"""
	#rel_target_position = np.asarray([0.4, 0, -0.1])

	"We pass from relative to world coordinates (mimo_location is the root of MIMo's model)"
	#w_target_position = me_utils.body_pos_to_world(data, rel_target_position, model.body('head').id)
	
	"""We clip to avoid falling below the floor"""
	#w_target_position[2] = np.clip(w_target_position[2], 0.1, 0.5)

	#logging.info(f"Setting target at relative position {rel_target_position}, world position {w_target_position}")

	#target.xpos = w_target_position

	target.xpos = np.array([0.2, 0.0, 0.3])


	logging.info(f"Target positioned at {target.xpos}")

	# TODO: simplify the target definition process by creating a Target class with the appropriate methods to set
	# the pose w.r.t. a target body in one go.
	start = time.time()
	for step in range(max_steps):
		action = controller.step()
		obs, reward, done, trunc, info = env.step(action)
		render(env)

		if done or trunc:
				env.reset()

	if _render_video:
		out.release()


	print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.get_wrapper_attr('dt'))
	env.close()

if __name__ == "__main__":

	main()