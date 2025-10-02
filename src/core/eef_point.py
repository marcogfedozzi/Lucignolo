import numpy as np

from numpy.typing import NDArray
from typing import List

from fields.forcefield import XField
from core.frames import Frame


class EEFPoint():
    def __init__(self, point: Frame):
        self.point = point
        self.fields = []
    
    def add_field(self, field: XField):
        assert isinstance(field, XField), f"object {field} shoult be a (V,F)Field, found instead {field.__class__.__name__} {field}"
        self.fields.append(field)
    
    def add_fields(self, fields: List):
        for field in fields:
            self.add_field(field)

    def compute_effect(self, qvel: NDArray):
        """Sum the current effect of every field in this point."""

        effect = np.zeros(6)

        for field in self.fields:
            effect += field(self.point, qvel)
        return effect
    
    def __repr__(self):
        return f"EEFPoint({self.point}) || fields: {self.fields}"