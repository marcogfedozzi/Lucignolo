# Lucignolo

A simple task-based controller for [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html).

Mostly devised for Hierarchical Inverse Dynamics, with Tasks moving in the Null-Space of higher level Tasks.

Why not using the default Mocap-body-based control for MuJoCo?

While it works really well for simple kinematic chains, it does not allow for fine grained control, which we might need for robotic applications.
This library defines all the components needed to move an End-Effector (e.g. a hand or gripper)
towards a target (e.g. an object), but also allows to align the former with the latter,
to (locally) avoid obstacles, to align with world frames, and more.

Limitations: this is not a fully-fledged product, as the control is limited to simple cases. So no, we cannot make a bipedal robot walk in MuJoCo using this library (yet?).
Depending on the level of interest I might look into that in the future, but no promises.

This library was originally developed to control [MIMo](https://github.com/trieschlab/MIMo) but is currently being extended to control of general MuJoCo models.

## Control Pipeline

> ⚠️ A necessary heads-up: this is an in-development package, and everything is subject to breaking changes in this stage. Even the control pipeline explained below might be simplified, extended, turned upside-down at a moment notice. Until we reach a satisfactory balance between ease-of-use and complexity, that is.

The Inverse Dynamics controller offered in this library allows to 
The controllers developed in this package need 3 ingredients to make the ~~magic~~ _math_ happen:

- an **End Effector ([EEF](./lucignolo/core/eef_point.py))**: the part on the model that will be moved in **Task Space[1]**. This can be a hand, a gripper, a head, an eye, you name it. EEFs are defined in the scene files as _sites_ located in a specific place in the body tree of your model. From the code you only need the name of the site to access it.
	<details>
	<summary>EEF Example</summary>

	An EEF is here placed in the right hand of [MIMo](https://github.com/trieschlab/MIMo), portions of the code omitted for clarity.

	```xml
	<body name="right_upper_arm" ...>
		<joint name="robot:right_shoulder_horizontal" type="hinge" .../>
		<joint name="robot:right_shoulder_ad_ab" type="hinge".../>
		<joint name="robot:right_shoulder_rotation".../>
		<geom name="right_uarm1" type="capsule".../>
		<body name="right_lower_arm" ...>
			<joint name="robot:right_elbow" type="hinge".../>
			<geom name="right_larm" type="capsule" .../>
			<body name="right_hand" ...> <!-- Hand length is 9.3, Palm length is 5.16 -->
				<joint name="robot:right_hand1" type="hinge" .../>
				<joint name="robot:right_hand2" type="hinge" .../>
				<joint name="robot:right_hand3" type="hinge" .../>
				<geom name="geom:right_hand1" type="box" .../>
				<geom name="geom:right_hand2" type="cylinder" .../>
				
				<site name="eef:right_hand" pos="0 -0.01 .02" size="0.01" group="4" zaxis="0 -1 0"/> <!-- EEF in the center of the palm -->
	```

	</details>

- a **[Target](./lucignolo/core/frames.py)**: this is the interesting frame in space, the goal of the task,  defined by both its position and orientation. Different types of frame exist, from *Static Frames* to *Controllable Frames* that can be easily moved around. Similarly to EEF, Targets also need to be specified in the scene file: they are mocap bodies, to allow for them to be position-controlled from the code.
  <details>
	<summary>Target Example</summary>

	A Target is here created as a sphere-shaped mocap body. It is by default hidden but can be visualized with group 4. It has no physical interaction with other bodies (_contype=0_ and _conaffinity=0_).

	```xml
	<body name="target:right_hand" mocap="true" pos=".0 .0 .0">
		<geom type="sphere" size="0.01" contype="0" conaffinity="0" material="blue" group="4"/> 
	</body>
	```

	</details>

- a **[Field](./lucignolo/fields/factory.py)**: the force that connects the EEF to a Target. Imagine the Target as the origin point of a force field, in which the EEF is immersed. A force will generate on the latter that depends on its pose within the field, and on the nature of the field itself. Under the hood, this is a PD controller in which different components can be specified separately.

	There are currently 4 types of field: 3 of them are Proportional:

	- **translational**: push the EEF closer to the Target;
	- **rotational**: align the EEF frame with the Target's frame;
	- **misalignment**: orient the EEF frame towards the Target [2];
  
	while the last one is Derivative:

  - **viscosity**: a resistance to the motion that depends on the velocity of the EEF, used to stabilize the control;
  ---

	Each proportional field is defined by a polynomial function:

	$k * (\frac{d}{s})^{pow}$

	where:
	- $k$ is the proportional coefficient
	- $d$ is the metric of interest (distance, alignment, etc.)
	- $s$ is athe normalization term
	- $pow$ is the power to which the normalized metric is raised

Here is a code snippet that shows all the necessary steps to from 0 to controller-ready:

<details>
<summary>Control Minimal Example</summary>

```python
import lucignolo as lc

## Environment ##

env = gym.make(...) # get your environment

controlled_body = "hand" 

## Target ##

target = lc.core.frames.ControllableFrame(env, "target:"+controlled_body)
target.xpos = np.array([0.2, 0.0, 0.3])

## End Effector ##

eef_frame = lc.core.frames.Frame(env, "eef:"+controlled_body, "site", heading=np.array([0,0,1]))

eef = lc.core.eef_point.EEFPoint(eef_frame)

## Fields ##

attractive_field = lc.fields.get_field(
	center=target,
	field_type="translation",
	params={
		"k": 800.0,
		"pow": 1.0,
		"max": 0.1,
	}
)

eef.add_field(alignment_field)

## Controller ##
subtree_type = controlled_body if "head" in controlled_body or "eye" in controlled_body else controlled_body.replace("hand", "arm")

controller = lc.controllers.IDController(env, eef, subtree_type)

## Control Loop ##

for step in range(max_steps):
		action = controller.step()
		obs, reward, done, trunc, info = env.step(action)
		render(env)

		if done or trunc:
				env.reset()

```
</details>

Find more complete scripts in the [example](./examples/) folder. 


### Notes and Definitions
<details>
<summary>[1] Task Space</summary>
The Task Space is the (often 3D) physical space of the simulation, where the task is executed (<em>yes this is an extreme simplification, bear with me</em>).Its dual is the Joint Space, where each dimension is one of the Degrees-of-Freedom of the model (i.e. the joints).
An example to understand the difference between the two of them: you want to grab an apple in front of you with your dominant hand.
Controlling your arm in Joint Space would mean thinking about how each muscle should contract in coordination to move your hand towards the target.
Task Space control is instead simply thinking about moving the hand forward, while your brain and body are performing all the complex computation that coordinates the muscles in the background. Quite the taxing feat, isn't it?
</details>

<details>
<summary>[2] Misalignment</summary>
Going in a bit more detail, a vector is defined in the EEF's frame (the <em>heading</em>) and it is pushed to align with the <em>distance vector</em>, the one that goes from the EEF to the Target (geometrically speaking, <var>Target - EEF</var>). The effect is that the EEF "looks" at the target.
</details>

## An in-depth look at the Inverse Kinematics

> TODO

## Installation

This package uses [uv](https://docs.astral.sh/uv/) for managing dependencies. It should, ideally, make the install process buttery smooth and fast.

You can refer to [uv](https://docs.astral.sh/uv/) documentation for an in depth look at all its functionalities, below are reported two use cases for it. Both assume you have already installed uv, and you can find the instructions for there [here](https://docs.astral.sh/uv/getting-started/installation/).

The important thing to remember about uv is that you do not need to activate the virtual environment it creates 

### Test Install

If you want to play around with the package and check that things are working.

<details>
<summary>Test Install</summary>

For this you do not need anything beyond this repo.

1. Clone this repo
	```bash
	git clone https://github.com/marcogfedozzi/Lucignolo.git && cd Lucignolo
	```
2. Run one of the example scripts with uv
	```bash
	uv run --extra mimo examples/MIMo/mimo_ctrl.py
	```

	At this point uv should be creating a virtual environment, adding the _optional_ MIMo dependencies (needed only for this example),
	and start running the script.
	You should see a MuJoCo window with MIMo sitting on the floor and reaching for a point in front of itself.


</details>

### Package Install

If you plan to use this as a package in your existing code.

<details>
<summary>Package Install</summary>

The plan is to make the wheels of this package available on PyPi or similar. For now, however, we are stuck with using the source code of this repo.

1. Clone this repo
	```bash
	git clone https://github.com/marcogfedozzi/Lucignolo.git
	```
2. Go to your project and install Lucignolo with either
   1. pip
		```bash
		cd <your/path>
		pip install <path/to/Lucignolo>
		```
	2. uv
		```bash
		cd <your/path>
		uv add <path/to/Lucignolo>
		```

	Test the correct install with
	
	```bash
	python -c "import lucignolo; print(lucignolo.__version__)"
	```
	or use
	
	```bash
	uv run python -c "import lucignolo; print(lucignolo.__version__)"
	```
	if you are also using a uv environment for your project.

</details>


## Q&A

**Is the name...**  
Yes, the name is a play on the amazing [Pinocchio](https://github.com/stack-of-tasks/pinocchio/tree/devel) library, which is more complete (and perhaps complex) than this one.
