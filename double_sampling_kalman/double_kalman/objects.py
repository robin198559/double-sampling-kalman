from attr.validators import instance_of
from attrs import define, field
import numpy as np


@define
class DoubleKalmanOutput:
    forward: np.ndarray = field(validator=instance_of(np.ndarray))
    backward: np.ndarray = field(validator=instance_of(np.ndarray))

    @property
    def last_estimate(self) -> np.ndarray:
        return self.x[-1, :, 0]

    @property
    def x(self) -> np.ndarray:
        x_combined = (self.backward + self.forward) / 2
        return x_combined[:, :, 0]
