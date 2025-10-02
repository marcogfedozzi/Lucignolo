"""
Velocity-based field implementations.

This module contains field implementations that depend on velocity
rather than just position. These include viscosity and damping fields.
"""
_all_ = ['TranslationalField', 'MisalignmentField', 'OrientationField', 'TMField', 'ViscosityField', 'get_field']

import numpy as np
from typing import List

from numpy.typing import NDArray

from core.frames import Frame
from abc import ABC, abstractmethod 
from fields.base import XField


class VField(XField):
    """This calss represents a vectorial field that is proportional to the velocity of the end effector.

    It simulates viscosity or friction in the end effector's movement, and acts similarly to the Derivative
    part in a PID controller.
    """

    def __init__(self, k: NDArray | List | float = 1.0, *args, **kwargs):
        """
        k: 
        - float: same coefficient for all axes
        - sizeof(k) == 2: [translational, rotational]
        - sizeof(k) == 6: [vx, vy, vz, wx, wy, wx]
        """

        if isinstance(k, (int, float)):
            self.k = np.ones(6) * k
        elif len(k) == 2:
            self.k = np.array([k[0], k[0], k[0], k[1], k[1], k[1]])
        elif len(k) != 6:
            raise ValueError("k must be a float, a list of 2 elements or a list of 6 elements.")
        
    def __call__(self, point: Frame, qvel: NDArray) -> NDArray:
        """Compute the effect of the field at the specified SE(3) position."""

        v = point.jac @ qvel # [6,] velocity in Cartesian space

        return - self.k * v
    
ViscosityField = VField
