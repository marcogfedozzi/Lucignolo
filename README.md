# Lucignolo

A simple task-based controller for MuJoCo.

Mostly devised for Hierarchical Inverse Dynamics, with Tasks
moving in the Null-Space of higher level Tasks.

Why not using the default Mocap-body-based control for MuJoCo?

While it work really well for simple kinematic chains, it does not allow for fine grained
control, which we might need for robotic applications.
This library defines all the components needed to move an End-Effector (e.g. a hand or gripper)
towards a target (e.g. an object), but also allows to align the former with the latter,
to (locally) avoid obstacles, to align with world frames, and more.

Limitations: this is not a fully-fledged product, as the control is limited to simple cases. So no, we cannot make a bipedal robot walk in MuJoCo using this library. Yet?
Depending on the level of interest I might look into that in the future, but no promises.

Repo for the development of Inverse Dynamics/ Kinematics controllers for [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html), originally developed to control [MIMo](https://github.com/trieschlab/MIMo) but extended to general control.

# Q&A

- yes, the name is a play on the amazing [Pinocchio](https://github.com/stack-of-tasks/pinocchio/tree/devel) library, which is more complete (and perhaps complex) than this one.
