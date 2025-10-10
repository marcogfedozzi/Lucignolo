"""
Utility functions for field implementations.

This module contains mathematical utilities and helper functions used
by field implementations, separated from the main field logic for clarity.
"""

import numpy as np
from numpy.typing import NDArray
from functools import partial

def get_proportional_func(k: float = 1.0, pow: float = 1.0, s: float = 1.0, 
                         min: float = None, max: float = None, thresh: float = 1e-3):
    """
    Create a proportional function for field strength computation.
    
    This factory function creates a function that computes field strength based on
    distance with customizable scaling, power laws, and clipping.
    
    Args:
        k: Overall gain/strength coefficient
        pow: Power law exponent (e.g., 2.0 for quadratic, 1.0 for linear)
        s: Scaling factor for distance normalization
        min: Minimum output value (None for no minimum)
        max: Maximum output value (None for no maximum)  
        thresh: Threshold below which the field returns zero (dead zone)
        
    Returns:
        Function that takes a vector and returns scaled magnitude with direction
    """
    
    # Only add clipping if min or max are specified, otherwise save computation
    if min is None and max is None:
        magnitude = lambda d: (d/s)**pow
    else:
        if max is None: max = np.inf
        if min is None: min = 0.0
        magnitude = lambda d: np.clip((d/s)**pow, min, max)

    def func(x: NDArray) -> NDArray:
        """Apply proportional scaling to input vector."""
        d = np.linalg.norm(x)
        
        # Dead zone: return zero for very small inputs
        if d < thresh:
            return np.zeros_like(x)
        
        # Compute unit vector and scaled magnitude
        versor = x / d
        return k * magnitude(d) * versor

    return func


get_linear_func         = partial(get_proportional_func, pow=1.0)
get_quadratic_func      = partial(get_proportional_func, pow=2.0)
get_cubic_func          = partial(get_proportional_func, pow=3.0)
get_inverse_func        = partial(get_proportional_func, pow=-1.0)
get_inverse_square_func = partial(get_proportional_func, pow=-2.0)


def get_cosine_func(radius_in: float = 0.2, radius_out: float = 0.3) -> NDArray:
    """Generate a decreasing cosine function that goes from 1 to 0 in the interval [radius1, radius2]."""

    def func(x: NDArray) -> NDArray:
        d = np.linalg.norm(x)
        if d > radius_out:
            return np.zeros_like(x)
    
        if d < radius_in:
            return 1
        
        n = x/d

        return 1/2 * (1 + np.cos(np.pi/(radius_out-radius_in) * (d-radius_in)))

    return func
