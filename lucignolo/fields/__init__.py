"""
Force fields and task specifications for robotics control.

This module defines task spaces and objectives through force fields:
- Various types of force fields (attractors, repulsors, alignment)
- End-effector points that can have multiple simultaneous fields
- Task specification layer for operational space control

Force fields provide an intuitive way to specify desired robot behaviors
in task space (e.g., "attract to this position", "align with this orientation").

Main user interface:
    from fields import get_field
    
    # Create fields using the factory function
    attractor = get_field("translation", {"center": target, "k": 5.0})
    damper = get_field("viscosity", {"k": 1.5})

Available field types:
    - "translation": Translational attractor/repulsor  
    - "orientation": Orientation alignment
    - "misalignment": Misalignment correction
    - "tm": Combined translation + misalignment
    - "viscosity": Velocity damping
    - "repulsive_*": Repulsive version of any field type
"""

# Main user-facing interface
from .factory import get_field

# Abstract base class (for advanced users extending the library)
from .base import XField

# Re-export commonly used classes
# Note: Users should prefer get_field() over direct instantiation
from .implementations.force_fields import TranslationalField, OrientationField, MisalignmentField, TMField, FField
from .implementations.velocity_fields import ViscosityField, VField
