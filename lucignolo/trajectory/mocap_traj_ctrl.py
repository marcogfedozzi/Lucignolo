# Control a mocap body

import numpy as np
from lucignolo.core.frames import iFrame, ControllableFrame, StaticFrame
from lucignolo.core.timers import GeneratorTimer, DeltaTimer, cTimer, flagTimer
import lucignolo.core.frames as bf
from lucignolo.trajectory.toys_path import MidPoint, Oscillation, ToysPath
from gymnasium.envs.mujoco import MujocoEnv
from lucignolo.core import utils


from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from typing import Tuple
from numpy.typing import NDArray
from collections import deque

from typing import Callable

import logging

from functools import partial

import mujoco


class MocapControl:
    class StackEntry:
        def __init__(self, sim_data: mujoco.MjData, target: iFrame, time: float, func: Callable):
            self.target = target
            self.timer  = GeneratorTimer()(sim_data,time)
            self.func   = partial(func, self.target, self.timer)
        
        def init(self, parent: "MocapControl"):
            pass
    
    class MotionStackEntry(StackEntry):
        def __init__(self, sim_data, target, time, func):
            super().__init__(sim_data, target, time, func)
            self.v = None # velocity
            self.slerp = None
            
        def init(self, parent: "MocapControl"):
            """Compute the velocity of the motion
            
            v is given as the difference between the target position and the current position
            divided by the desired time to reach the target.
            """

            self.v = np.linalg.norm(self.target.xpos - parent.mocap.xpos) / self.timer.T
            self.slerp = Slerp([0, self.timer.T], R.from_matrix([parent.mocap.xmat, self.target.xmat]))

            self.func = partial(self.func, v=self.v, slerp=self.slerp)

    def __init__(self, env: MujocoEnv, mocap_name: str, equality_name: str,
                 MIMo_body_collision: str = "hip", return_to_init_time: float = -1,
                 wait_on_contact: float = 5.0):
        """
        Initialize the MocapControl class.

        Args:
            env (MujocoEnv): The MIMoEnv environment object.
            mocap_name (str): The name of the mocap body.
        """

        self.env = env
        self.mocap = ControllableFrame(env, mocap_name)
        self.ctrl_obj_id = None
        self.ctrl_obj_geoms_idrange = None
        self.timer = None
        self.wait_timer = None

        self.wait_before_eq_toggle = 0.5
        self.wait_on_contact = wait_on_contact

        self.init_frame = None
        self.return_to_init_time = return_to_init_time

        self.dt_timer = DeltaTimer(self.env.data, maxlen=100)

        self.v = 0

        self.eq_id = self.env.model.eq(equality_name).id
        
        self.env.model.eq(equality_name).active = 0

        self._move_foo = self._no_motion

        self.stack = deque()
        self.current_op: MocapControl.StackEntry = None

        # Collision
        if MIMo_body_collision is not None:
            self.body_collision_geoms_idrange = utils.get_geoms_for_body_tree(self.env.model, self.env.model.body(MIMo_body_collision).id)
            if self.body_collision_geoms_idrange is None:
                logging.error(f"Could not find body {MIMo_body_collision} for collision detection")
        else:
            self.body_collision_geoms_idrange = None

    def set_controlled_object(self, name: str):

        self.ctrl_obj_id = self.env.model.body(name).id
        self.ctrl_obj_geoms_idrange = utils.get_geoms_for_body_tree(self.env.model, self.ctrl_obj_id)

        logging.debug(f"Setting controlled object to {name} [{self.ctrl_obj_id}]")

        # Modify equality constraint setting this body as the controlled one
        self.env.model.eq_obj2id[self.eq_id] = self.ctrl_obj_id


    def add_target(self, stack: deque, target: iFrame, time: float, wait: float = 0, check_collision: bool = False):
        """
        Set the target frame and time for the motion.

        Args:
            target (Frame): The target frame.
            time (float): The time for the motion.
        """

        func = self.check_collision if check_collision else lambda x: x

        stack.append(self.MotionStackEntry(self.env.data, target, time, func(self._reach_point)))
        if wait > 0:
            stack.append(self.StackEntry(self.env.data, target, wait, func(self._wait)))

    def return_to_init(self, time: float):
        self.return_to_init_time = time

    def init_motion(self):
        """Initialize the motion of the mocap body."""

        # Move the mocap to the controlled body

        self.stack.append(self.StackEntry(self.env.data, None, 0, self._move_to_target))
        self.stack.append(self.StackEntry(self.env.data, None, 0, self._activate_constraint))
        self.stack.append(self.StackEntry(self.env.data, None, self.wait_before_eq_toggle, self._wait))

        self.stack.append(self.StackEntry(self.env.data, None, 0, self._post_init))

        if self.return_to_init_time > 0:
            #self.stack.append(self.MotionStackEntry(self.env.data, self.init_frame, self.return_to_init_time, self._reach_point))
            self.stack.append(self.StackEntry(self.env.data, None, 0, self._return_to_init))

            #self.return_to_init_time = -1 # reset for future calls


        self._close_stack()


        self.dt_timer.reset()

        self._next_target()

    
    def _return_to_init(self, *args, **kwargs):
        """This adds the return as a separate operation in order to use the init_frame that will
        be defined at the beginning of teh stack"""

        self.stack.appendleft(self.MotionStackEntry(self.env.data, self.init_frame, self.return_to_init_time, self._reach_point))
        return True

    def _post_init(self, *args, **kwargs):
        return True

    def _move_to_target(self, *args, **kwargs):

        self.mocap.xpos = self.env.data.xpos[self.ctrl_obj_id]
        self.mocap.xmat = self.env.data.xmat[self.ctrl_obj_id]

        self.init_frame = StaticFrame(self.env.data.xpos[self.ctrl_obj_id], self.env.data.xmat[self.ctrl_obj_id])


        return True

    def _close_stack(self):
        self.stack.append(self.StackEntry(self.env.data, None, 0, self._deactivate_constraint))
        self.stack.append(self.StackEntry(self.env.data, None, 0, self._no_motion))

    def move(self):
        """Move the mocap body."""
        self.dt_timer.step()

        # WARN: if the object passes too close to the body we switch to the next target
        # this is a very naive way to detect collisions and mostly results in skipping all steps until
        # the object is dropped. This obj collision does not check whether the object is moving towards or
        # away from the body. To be improved.

        done = self.current_op.func()

        if done:
            self._next_target()

    def _activate_constraint(self, *args, **kwargs):
        self.env.data.eq_active[self.eq_id] = 1
        return True

    def _deactivate_constraint(self, *args, **kwargs):
        self.env.data.eq_active[self.eq_id] = 0
        self.ctrl_obj_id = None
        self.ctrl_obj_geoms_idrange = None

        self.init_frame = None

        return True
    
    def _no_motion(self, *args, **kwargs):
        """
        Do nothing.
        
        By returning False it prevents the stack from moving to the next (inexistent) target.
        """
        
        return False

    def _wait(self, target: iFrame, timer: cTimer):
        return timer()

    def _next_target(self):
        
        self.current_op = self.stack.popleft() # remove the reached target form the queue
        self.current_op.timer.reset()
        self.current_op.init(self)

        logging.debug(f"Target reached, {len(self.stack)-2} targets left")
    
    def _reach_point(self, f: iFrame, t: cTimer, v: float = 0, slerp: Slerp = None):

        delta_t = t.time_left()
        time_passed = t.time_passed_lim()

        dt = self.env.dt

        if delta_t <= dt:
            return True
        
        delta_pos = f.xpos - self.mocap.xpos
        norm_delta_pos = np.linalg.norm(delta_pos) # magnitude distance

        if norm_delta_pos > 1e-3:
        
            n_delta_pos = delta_pos / norm_delta_pos # direction distance
            dx = n_delta_pos * v * dt
            self.mocap.xpos += dx

        # Compute the rotation
        
        # use Slerp to interpolate between the current orientation and the target orientation, after time dt from now
        self.mocap.xquat = utils.quat_wfirst(slerp(time_passed).as_quat())
        return False
    
    def add_linear_oscillation(self, amplitude: float, frequency: float, time: float, xz_vel_noise: Tuple[float, float] = (0, 0)):
        self.stack.append(self.MotionStackEntry(self.env.data, self.stack[-1].target, time, self._oscillate_linear(amplitude, frequency, xz_vel_noise)))

    def _oscillate_linear(self, amplitude: float, frequency: float, xz_vel_noise: Tuple[float, float] = (0, 0)):
        def foo(f: iFrame, t: cTimer):
            delta_t = t.time_left()
            time_passed_frac = t.time_passed_frac_lim()

            dt = self.dt_timer.dt

            if delta_t <= dt:
                return True

            self.v = amplitude * np.cos(2 * np.pi * frequency * time_passed_frac) * frequency * 2 * np.pi / t.T

            frame_to_world_ori = f.xmat

            xv, zv = np.random.normal(0, 1, 2) * np.array(xz_vel_noise)

            self.mocap.xpos += frame_to_world_ori.dot(np.array([xv, self.v, zv])*dt)

            return False

        return foo
    
    def add_rotation_oscillation(self, amplitude: float, frequency: float, time: float):
        self.stack.append(self.StackEntry(self.env.data, self.stack[-1].target, time, self._oscillate_rotation(amplitude, frequency)))

    def _oscillate_rotation(self, amplitude: float, frequency: float):
        def foo(f: iFrame, t: cTimer):
            delta_t = t.time_left()
            time_passed_frac = t.time_passed_frac_lim()


            dt = self.dt_timer.dt

            if delta_t <= dt:
                return True

            w = amplitude * np.cos(2 * np.pi * frequency * time_passed_frac) * frequency * 2 * np.pi / t.T
            # rotate the mocap arund the local x axis (assumed to be aligned with the target)

            r = R.from_quat(utils.quat_wlast(self.mocap.xquat)) *  R.from_rotvec(np.array([w*dt, 0, 0]))

            self.mocap.xquat = utils.quat_wfirst(r.as_quat())

            return False

        return foo
    
    def add_oscillation(self, stack: deque, osc_center_frame: iFrame,
                        rotation_amp: float, rotation_freq: float, 
                        linear_amp: float, linear_freq: float, 
                        time: float, check_collision: bool = False):
        
        
        rot_func = self._oscillate_rotation(rotation_amp, rotation_freq)
        lin_func = self._oscillate_linear(linear_amp, linear_freq)

        lin_rot_oscillate = lambda f, t: (lin_func(f, t) or rot_func(f, t))

        oscillate = self.check_collision(lin_rot_oscillate) if check_collision else lin_rot_oscillate
        
        stack.append(self.StackEntry(self.env.data, osc_center_frame, time, oscillate))

    def _detect_obj_collision(self):
        """Detect whether there is a collision between the controlled object and the body.
        
        This is far from the most efficient way to do this, but it is the most straightforward.
        """

        if self.ctrl_obj_geoms_idrange is None:
            return False

        for c in self.env.data.contact: # (ncon, 1)

            if self.ctrl_obj_geoms_idrange.start <= c.geom1 < self.ctrl_obj_geoms_idrange.stop and\
                self.body_collision_geoms_idrange.start <= c.geom2 < self.body_collision_geoms_idrange.stop:
                    
                    logging.debug(f"Collision detected between controlled object [{mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)}] and body [{mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)}]")
                    return True
            
            elif self.ctrl_obj_geoms_idrange.start <= c.geom2 < self.ctrl_obj_geoms_idrange.stop and\
                self.body_collision_geoms_idrange.start <= c.geom1 < self.body_collision_geoms_idrange.stop:

                    logging.debug(f"Collision detected between controlled object [{mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)}] and body [{mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)}]")
                    return True
            
    def check_collision(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if self._detect_obj_collision():
                # TODO: insert wait in stack
                self.stack.insert(1, self.StackEntry(self.env.data, None, self.wait_on_contact, self._wait))

                return True
            return result
        return wrapper

    @property
    def toy_xpos(self):
        return self.mocap.xpos
    @property
    def toy_xquat(self):
        return self.mocap.xquat
    
    @toy_xpos.setter
    def toy_xpos(self, value: NDArray):
        raise AttributeError("Cannot set toy_xpos directly")
    
    @toy_xquat.setter
    def toy_xquat(self, value: NDArray):    
        raise AttributeError("Cannot set toy_xquat directly")


class AutonomousMocapControl(MocapControl):
    """
    A meta-controller that randomly selects the target to move around,
    as well as the parameters of its motion and the time between motions.
    """

    def __init__(self, env: MujocoEnv, mocap_name: str, equality_name: str, 
                num_toys: int,
                toys_path: ToysPath,
                sample_toys_randomly = True, toys_sample_limit = None, return_to_init_time: float = -1,
                wait_on_contact: float = 5.0
                ):
        super().__init__(env, mocap_name, equality_name, return_to_init_time=return_to_init_time, wait_on_contact=wait_on_contact)

        self.stack: deque

        self.sched_timer = flagTimer(env.data, toys_path.time_between_motions*0.9)
        self.wait_before_eq_toggle = toys_path.time_between_motions*0.1
        self.num_toys = num_toys
        self.offset_body = toys_path.offset_body
        self.points = toys_path.points
        self.current_toy_num  = -1

        self._sample_toy_num: Callable[[], int]

        self.toys_sample_limit = toys_sample_limit
        self.tot_sampled_toys = 0

        if sample_toys_randomly:
            self._sample_toy_num = self._sample_random_toy_num
        else:
            self._sample_toy_num = self._sample_sequential_toy_num


        self._is_done = False
        #self._should_reset = False

    def  _sample_random_toy_num(self):
        return np.random.randint(0, self.num_toys)

    def _sample_sequential_toy_num(self):
        return (self.current_toy_num + 1) % self.num_toys

    def start(self):
        self.sched_timer.on()

    def step(self) -> bool:
        """
        Returns "done" signal, if True then the sampling should stop.
        """

        if self.sched_timer():
            logging.debug("Picking new item")
            self.sched_timer.off()
            self._reset_stack()

            self.init_motion()

        if self.current_op is not None:
            self.move()

        return self._is_done

    def _close_stack(self, *args, **kwargs):
        """Activate the timer for the next motion"""

        self.stack.append(self.StackEntry(self.env.data, None, 0, self._deactivate_constraint))
        self.stack.append(self.StackEntry(self.env.data, None, 0, self._activate_timer))
        self.stack.append(self.StackEntry(self.env.data, None, 0, self._no_motion))

        return self._no_motion(*args, **kwargs)
    
    def _activate_timer(self, *args, **kwargs):
        logging.debug("Reactivating timer")
        self.sched_timer.on()
        return True
    
    def _reset_stack(self):
        self.stack.clear()
        self.current_op = None

    @property
    def moved_toy(self):
        return "toy" + str(self.current_toy_num)
    

    def _post_init(self, *args, **kwargs):
        # Call sample motion as part of the stack
        return self._sample_motion()
    
    def init_motion(self):
        
        self.stack.appendleft(self.StackEntry(self.env.data, None, 0, self._sample_toy))

        super().init_motion()
    
    def _sample_toy(self, *args, **kwargs):

        self.current_toy_num = self._sample_toy_num()

        self.tot_sampled_toys += 1

        if self.toys_sample_limit is not None and self.tot_sampled_toys > self.toys_sample_limit:
            self._is_done = True
            
        self.set_controlled_object(self.moved_toy)

        return True
    
    def _sample_motion(self):

        # Since this function will fill the queue with operations, and might itself be called from
        # the queue, we first need to unload the "tail" of the stack, append each motion, 
        # and then reload the tail

        motion_stack = deque()

        raised_frame = bf.StaticFrame(
            frame=bf.OffsetFrame(self.env, self.moved_toy, "body", 
                                offset=self.points[0].offset,
                                offset_in_world_coords=True)
        )
        self.add_target(motion_stack, raised_frame, self.points[0].time, self.points[0].wait, self.points[0].check_collision)

        _, oy, _ = utils.world_pos_to_body(self.env.data, raised_frame.xpos, self.env.data.body(self.points[0].offset_body).id)

        _c = np.array([1,np.sign(oy),1])

        motion_frame = None

        for i, point in enumerate(self.points[1:-1], 1):
            if isinstance(point, MidPoint):
                motion_frame = bf.MinOffsetFrame(self.env, point.offset_body, "body", 
                    min_z=0.15,
                    offset=_c * point.offset)
                
                self.add_target(motion_stack, motion_frame,
                    point.time, point.wait, point.check_collision
                )
            elif isinstance(point, Oscillation):
                self.add_oscillation(motion_stack, motion_frame,
                    point.rotation.amplitude, point.rotation.frequency,
                    point.linear.amplitude, point.linear.frequency,
                    point.rotation.time, point.check_collision
                )
        
        self.add_target(motion_stack, raised_frame, self.points[-1].time, self.points[-1].wait, self.points[-1].check_collision)
    
        # Prepend the stack with the new motion
        motion_stack.extend(self.stack)
        self.stack = motion_stack
    
        return True

    @property
    def is_done(self) -> bool:
        return self._is_done

    @is_done.setter
    def is_done(self, value: bool):
        raise AttributeError("Cannot modify 'is_done' attribute directly")