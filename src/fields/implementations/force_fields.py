"""
Force-based field implementations.

This module contains field implementations that depend on spatial position
and orientation rather than velocity. These include attractors, repulsors,
and orientation alignment fields.
"""
_all_ = ['TranslationalField', 'MisalignmentField', 'OrientationField', 'TMField', 'ViscosityField', 'get_field']

import numpy as np

from numpy.typing import NDArray
from typing import List, Dict

from functools import partial

from core.frames import Frame
from abc import ABC, abstractmethod 

from fields.base import XField

class FField(XField):
    """This class represents a vectorial field generated around a central point in SE(3).
    
    The effect of the field can be sampled at a specific point by calling the class (__call__).
    """
    def __init__(self, center: Frame,
                is_translational: bool = False, 
                is_alignment: bool = False, # align to target
                is_orientation: bool = False, # orient as vector
                is_repulsive: bool = False,
                tr_func = None, al_func = None,
                radii: NDArray = np.array([0.0]),
                align_same_k: bool = True,
                *args, **kwargs):
        
        self.center = center

        self.is_translational = is_translational
        self.is_alignment = is_alignment
        self.is_orientation = is_orientation

        self.align_same_k = align_same_k # if True and is_alignment, ignore the k vector of the point and align the projected k of this center

        assert is_translational or is_alignment or is_orientation, "At least one of the fields must be active."
        assert not (is_alignment and is_orientation), "Alignment and orientation fields cannot be active at the same time."

        self.is_repulsive = is_repulsive
        self.c = 1 if not self.is_repulsive else -1

        self.a = 0.0

        self.radii = np.asarray(radii, dtype=np.float64)

        if is_translational:
            self.get_trans_err = self._get_translational_error 
            self.tr_func = tr_func if tr_func is not None else get_linear_func()
        else: 
            self.get_trans_err = lambda x: 0
            self.tr_func = lambda x: 0

        if is_alignment:
            self.get_ori_err = self._get_target_misalignment_error
            self.al_func = al_func if al_func is not None else get_linear_func()
        elif is_orientation:
            self.get_ori_err = self._get_vector_misalignment_error
            self.al_func = al_func if al_func is not None else get_linear_func()
        else:
            self.get_ori_err = lambda x: 0
            self.al_func = lambda x: 0

        # Radii
        # if len == 1, cosine goes from radii to center
        # if len == 2, cosine goes from radii[0] to radii[1], 1 before, 0 after

        if np.all(self.radii <= 0): # Alway Active
            self.activation_func = lambda x: 1.0
        else:
            if len(self.radii) == 1:
                self.radii = np.array([0.0, self.radii[0]])
            else:
                assert self.radii[1] > self.radii[0], "The outer radius must be greater than the inner one."

            self.activation_func = get_cosine_func(self.radii[0], self.radii[1])

    def _get_vector_misalignment_error(self, point: Frame):
        """Get the orientation erro bewteen the end effector and a vector expressed in world coordinates."""

        k   = self.center.k # NOTE: assume the same axes is to align, NOT the k of the point
        p_k = k if self.align_same_k else point.k  

        w_k_eef = point.xmat @ p_k

        w_k_center = self.center.xmat @ k

        cp = np.cross(w_k_eef, w_k_center)
        cp_norm = np.clip(np.linalg.norm(cp), -1.0, 1.0) # sanitize rounding errors

        theta = np.arcsin(cp_norm)
        n = cp / cp_norm
        rho = theta * n # [3,]

        return rho

    def _get_target_misalignment_error(self, point: Frame):
        """Get the orientation error between the end effector and the target.
        
        This is used to align the end effector (better, its eef_k vector) with the segment of
        distance to this point.
        """

        xpos    = point.xpos
        xmat    = point.xmat
        k       = point.k

        w_k_eef = xmat @ k
        d = self.center.xpos - xpos
        d_versor = d / np.linalg.norm(d)
        
        cp = np.cross(w_k_eef, d_versor)
        cp_norm = np.linalg.norm(cp)

        theta = np.arcsin(cp_norm)
        n = cp / cp_norm
        rho = theta * n # [3,]

        return rho
    
    def _get_translational_error(self, point: Frame):
        """Get the translational error between the end effector and the target.
        
        This is used to align the end effector with the segment of distance to this point.
        """

        return self.center.xpos - point.xpos
    
    def _get_error(self, point: Frame):
        err = np.zeros(6)

        err[:3] = self.tr_func(self.get_trans_err(point))
        err[3:] = self.al_func(self.get_ori_err(point))


        return err
    
    def __call__(self, point: Frame, qvel: NDArray | None = None):
        """Get the field effect at the specified SE(3) position."""

        self.a = self.activation_func(np.linalg.norm(self.center.xpos - point.xpos))
        
        return self.c * self.a * self._get_error(point)
    
    def __repr__(self):
        """Return a string representation of the field with its parameters."""

        s = f"FField(pos:{self.center.xpos} quat:{self.center.xquat}"

        if self.is_repulsive:
            s += " rep"
        if self.is_translational:
            s += " trn"
        if self.is_alignment:
            s += " alg"
        if self.is_orientation:
            s += " ori"
        
        s += f" radii:{self.radii} a:{self.a}"
        s += ")"

        return s


TranslationalField  = partial(FField, is_translational=True, is_alignment=False)

MisalignmentField   = partial(FField, is_translational=False, is_alignment=True)

OrientationField    = partial(FField, is_translational=False, is_alignment=False, is_orientation=True)

TMField             = partial(FField, is_translational=True, is_alignment=True)

def get_proportional_func(k: float = 1.0, pow: float = 1.0, s: float = 1.0, min: float = None, max: float = None, thresh: float = 1e-3) -> NDArray:

    # Only add clipping if min or max are specified, otherwise save computation
    # Note that I have to add clipping to the magnitude, before multiplying by the versor
    if min is None and max is None:
        magnitude = lambda d: (d/s)**pow
    else:
        if max is None: max = np.inf
        if min is None: min = 0.0

        magnitude = lambda d: np.clip((d/s)**pow, min, max)

    def func(x: NDArray) -> NDArray:
        d = np.linalg.norm(x)

        if d < thresh:
            return np.zeros_like(x)
        n = x/d

        return k * n * magnitude(d)
    
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

# Pipeline:
# generate the fields
# create an EEF point
# assign the fields to the EEF point
# in the loop: for each EEF point, get the fields' effects and sum them
# in the loop: compute the velocity values in joint space

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

def get_field(func_name: str, params: Dict) -> XField:
    """Get a field object based on the name and the parameters."""
    field = None

    if func_name == "viscosity": return partial(ViscosityField, k=params["k"])

    is_repulsive = params.pop("is_repulsive", False) or 'repulsive' in func_name # in case it's already expressed

    radii = params.pop("radii", np.array([0.0]))
    align_same_k = params.pop("align_same_k", True)
    foo = get_proportional_func(**params)

    tr_func = None
    al_func = None

    if "translation"  in func_name: 
        field = TranslationalField
        tr_func = foo
    elif "misalignment" in func_name: 
        field = MisalignmentField
        al_func = foo
    elif "orientation"  in func_name:
        field = OrientationField
        al_func = foo
    elif "tm" in func_name: 
        field = TMField
        tr_func = foo
        al_func = foo
    else:
        raise ValueError("The field function name is not valid.")



    field = partial(field, tr_func=tr_func, al_func=al_func, radii=radii, is_repulsive=is_repulsive, align_same_k=align_same_k)

    return field