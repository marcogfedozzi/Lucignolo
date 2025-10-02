"""
Abstract base classes for field definitions.

This module contains only the core abstractions that all field types must implement.
No concrete implementations are included here to keep the interface clean.
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from core.frames import Frame


class XField(ABC):
    """
    Abstract base class for all field types.
    
    A field represents a desired behavior or force that acts on a point in space.
    Fields can be force-based (attractors, repulsors) or velocity-based (viscosity).
    
    The field's effect is computed by calling the field object with a Frame and
    joint velocities, returning a 6D spatial vector (3D force + 3D torque).
    """
    
    @abstractmethod
    def __call__(self, point: Frame, qvel: NDArray) -> NDArray:
        """
        Compute the field effect at a given point.
        
        Args:
            point: Frame representing the spatial location to evaluate the field
            qvel: Joint velocities (used for velocity-dependent fields)
            
        Returns:
            NDArray: 6D spatial vector [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        """
        pass
