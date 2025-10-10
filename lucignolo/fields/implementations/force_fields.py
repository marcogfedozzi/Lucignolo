"""
Force-based field implementations.

This module contains field implementations that depend on spatial position
and orientation rather than velocity. These include attractors, repulsors,
and orientation alignment fields.
"""
_all_ = ['TranslationalField', 'MisalignmentField', 'OrientationField', 'TMField', 'ViscosityField', 'get_field']

import numpy as np

from numpy.typing import NDArray

from functools import partial

from lucignolo.core.frames import Frame

from lucignolo.fields.base import XField
from lucignolo.fields.utils import get_cosine_func, get_linear_func

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