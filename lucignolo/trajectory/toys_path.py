__all__ = ["ToysPath"]

import numpy as np
from typing import Dict, Tuple, List
from random import uniform
from dataclasses import dataclass

@dataclass
class MidPointR:
    @dataclass
    class _Offset:
        min: Tuple[float, float, float]
        max: Tuple[float, float, float]

    time: Tuple[float, float]
    wait: Tuple[float, float]
    offset: _Offset
    check_collision: bool
    offset_body: str | None = None

@dataclass
class OscillationR:
    @dataclass
    class _Rotation:
        amplitude: Tuple[float, float]
        frequency: Tuple[float, float]
        time: Tuple[float, float]

    @dataclass
    class _Linear:
        amplitude: Tuple[float, float]
        frequency: Tuple[float, float]
        time: Tuple[float, float]
        xz_vel_noise: Tuple[float, float]

    rotation: _Rotation
    linear: _Linear
    check_collision: bool
    # offset_body: str


@dataclass
class MidPoint:
    time: float
    wait: float
    offset: Tuple[float, float, float]
    check_collision: bool
    offset_body: str


@dataclass
class Oscillation:
    @dataclass
    class _Rotation:
        amplitude: float
        frequency: float
        time: float
    @dataclass
    class _Linear:
        amplitude: float
        frequency: float
        time: float
        xz_vel_noise: Tuple[float, float]

    rotation: _Rotation
    linear: _Linear
    check_collision: bool

class ToysPath:
    def __init__(self, points: Dict[str, MidPointR | OscillationR], time_between_motions: float, offset_body: str, name: str = ""):
        self.name = name
        self._time_between_motions = time_between_motions
        self._offset_body = offset_body

        self._points = self._sample_points_from_ranges(points)

    def _sample_points_from_ranges(self, points: Dict[str, MidPointR | OscillationR]) -> List[MidPoint | Oscillation]:
        sampled_points = []
        for k, point in points.items():
            if isinstance(point, MidPointR):
                sampled_points.append(
                    MidPoint(
                        time = uniform(*point.time),
                        wait = uniform(*point.wait),
                        offset = np.random.uniform(point.offset.min, point.offset.max),
                        check_collision = point.check_collision,
                        offset_body= point.offset_body if point.offset_body else self._offset_body
                    )
                )
            elif isinstance(point, OscillationR):
                sampled_points.append(
                    Oscillation(
                        rotation = Oscillation._Rotation(
                            amplitude = uniform(*point.rotation.amplitude),
                            frequency = uniform(*point.rotation.frequency),
                            time = uniform(*point.rotation.time)
                        ),
                        linear = Oscillation._Linear(
                            amplitude = uniform(*point.linear.amplitude),
                            frequency = uniform(*point.linear.frequency),
                            time = uniform(*point.linear.time),
                            xz_vel_noise = point.linear.xz_vel_noise
                        ),
                        check_collision = point.check_collision
                    )
                )
        return sampled_points
    
    @property
    def points(self) -> List[MidPoint | Oscillation]:
        return self._points
    @points.setter
    def points(self, value):
        raise AttributeError("Cannot set points directly.")
    
    @property
    def time_between_motions(self) -> float:
        return self._time_between_motions

    @time_between_motions.setter
    def time_between_motions(self, value: float):
        raise AttributeError("Cannot set time_between_motions directly.")

    @property
    def num_toys(self) -> int:
        return self._num_toys

    @num_toys.setter
    def num_toys(self, value: int):
        raise AttributeError("Cannot set num_toys directly.")

    @property
    def offset_body(self) -> str:
        return self._offset_body

    @offset_body.setter
    def offset_body(self, value: str):
        raise AttributeError("Cannot set offset_body directly.")