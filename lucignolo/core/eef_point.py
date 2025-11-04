import numpy as np

from numpy.typing import NDArray
from typing import List, TYPE_CHECKING

from lucignolo.core.frames import Frame

if TYPE_CHECKING:
    # For type checkers only; avoids runtime circular imports
    from lucignolo.fields.base import XField


class EEFPoint():
    def __init__(self, point: Frame):
        self.point = point
        self.fields = []
        self.current_effect = np.zeros(6)
    
    def add_field(self, field: 'XField'):
        # Avoid importing XField at module import time to prevent circular imports
        try:
            from lucignolo.fields.base import XField
        except Exception:
            XField = None

        if XField is not None:
            assert isinstance(field, XField), f"object {field} should be an XField, found instead {field.__class__.__name__} {field}"
        self.fields.append(field)
    
    def add_fields(self, fields: List):
        for field in fields:
            self.add_field(field)

    def compute_effect(self, qvel: NDArray):
        """Sum the current effect of every field in this point."""

        effect = np.zeros(6)

        for field in self.fields:
            effect += field(self.point, qvel)

        self.current_effect = effect
        return effect
    
    def __repr__(self):
        return f"EEFPoint({self.point}) || fields: {self.fields} || current_effect: {self.current_effect}"