"""
Lucignolo: a simple task-based controller for MuJoCo.

Mostly devised for Hierarchical Inverse Dynamics, with Tasks
moving in the Null-Space of higher level Tasks.

Why not using the default Mocap-body-based control for MuJoCo?

While it work really well for simple kinematic chains, it does not allow for fine grained
control, which we might need for robotic applications.
This library defines all the components needed to move an End-Effector (e.g. a hand or gripper)
towards a target (e.g. an object), but also allows to align the former with the latter,
to (locally) avoid obstacles, to align with world frames, and more.

Limitations: this is not a fully-fledged product, for that look 
"""

from . import controllers, core, fields, trajectory

__all__ = ["controllers", "core", "fields", "trajectory"]
