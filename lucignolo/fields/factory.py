"""
Field factory and user-facing interface.

This module provides the main entry point for creating fields. Users should
import `get_field` from this module to create field instances without needing
to know about the internal implementation details.

Example usage:
    from fields.factory import get_field
    
    # Create an attractive translational field
    attractor = get_field("translation", {
        "center": target_frame,
        "k": 10.0,
        "pow": 2.0
    })
    
    # Create a viscosity field for damping
    damper = get_field("viscosity", {"k": 1.5})
"""

import numpy as np
from typing import Dict

from lucignolo.core.frames import iFrame
from lucignolo.fields.base import XField
from lucignolo.fields.utils import get_proportional_func
from lucignolo.fields.implementations.force_fields import TranslationalField, MisalignmentField, OrientationField, TMField
from lucignolo.fields.implementations.velocity_fields import ViscosityField
from typing import Optional

def get_field(center: Optional[iFrame], field_type: str, params: Dict) -> XField:
    """
    Factory function to create field instances based on type and parameters.
    
    This is the main user-facing interface for creating fields. It handles
    parameter processing and returns configured field instances.
    
    Args:
        center: the origin of the field (e.g. the target point)

        field_type: String identifying the field type. Supported types:
            - "translation": Translational attractor/repulsor field
            - "misalignment": Orientation alignment field  
            - "orientation": Orientation field (vector alignment)
            - "tm": Combined translation + misalignment field
            - "viscosity": Velocity-dependent damping field
            - "repulsive_translation": Repulsive translational field
            - "repulsive_orientation": Repulsive orientation field
            - etc. (prepending "repulsive_" makes any field repulsive)
            
        params: Dictionary of field parameters. Common parameters:
            - k: Field strength/gain
            - pow: Power law exponent
            - s: Distance scaling factor  
            - min, max: Output clipping bounds
            - thresh: Dead zone threshold
            - radii: Array of radii for multi-zone fields
            - align_same_k: Whether to use same gain for alignment
            
    Returns:
        XField: Configured field instance ready for use
        
    Raises:
        ValueError: If field_type is not recognized
        
    Example:
        # Attractive field toward target
        field = get_field(
            target_frame,
            "translation", 
            {
                "k": 5.0,
                "pow": 1.0,
                "s": 0.1
            }
        )
        
        # Repulsive field avoiding obstacle  
        field = get_field(
            obstacle_frame,
            "repulsive_translation", 
            {
                "k": 10.0,
                "pow": 2.0
            }
        )
        
        # Viscosity damping
        field = get_field(None, "viscosity", {"k": 2.0})
    """
    
    # Handle viscosity field separately (velocity-based, different interface)
    if field_type == "viscosity":
        return ViscosityField(k=params.get("k", 1.0))
    
    # Process common parameters for force fields
    params = params.copy()  # Don't modify original
    is_repulsive = params.pop("is_repulsive", False) or 'repulsive' in field_type
    radii = params.pop("radii", np.array([0.0]))
    align_same_k = params.pop("align_same_k", True)
    
    # Create proportional function from remaining parameters
    proportional_func = get_proportional_func(**params)
    
    # Determine field type and create appropriate instance
    tr_func = None
    al_func = None
    
    if "translation" in field_type:
        field_class = TranslationalField
        tr_func = proportional_func
        
    elif "misalignment" in field_type:
        field_class = MisalignmentField  
        al_func = proportional_func
        
    elif "orientation" in field_type:
        field_class = OrientationField
        al_func = proportional_func
        
    elif "tm" in field_type:
        field_class = TMField
        tr_func = proportional_func
        al_func = proportional_func
        
    else:
        raise ValueError(f"Unknown field type: {field_type}. "
                        f"Supported types: translation, misalignment, orientation, tm, viscosity")
    
    # Create and return field instance
    return field_class(
        center, 
        tr_func=tr_func, al_func=al_func, 
        radii=radii, is_repulsive=is_repulsive, 
        align_same_k=align_same_k
    )
