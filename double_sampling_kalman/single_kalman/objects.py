from attr.validators import instance_of
from attrs import define, field
import numpy as np


@define
class SingleKalmanOutput:
    estimation: np.ndarray = field(validator=instance_of(np.ndarray))
    latest_error_matrix: np.ndarray = field(validator=instance_of(np.ndarray))

    @property
    def last_estimate(self) -> np.ndarray:
        return self.estimation[-1, :, :]

    @property
    def x(self) -> np.ndarray:
        return self.estimation
