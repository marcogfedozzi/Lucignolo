"""
Trajectory generation and motion planning.

This module provides tools for generating and following trajectories:
- Path generation utilities and primitives
- Motion capture trajectory control
- Reference generation for control systems
- High-level motion planning interfaces

These tools are used for:
- Generating smooth reference trajectories
- Motion capture data processing and replay
- High-level task planning and execution
- Integration with motion planning libraries
"""

from .toys_path import *
from .mocap_traj_ctrl import *
