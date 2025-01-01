from typing import List

import attrs
from attr.validators import instance_of, deep_iterable
from attrs import define, field
import numpy as np

from double_sampling_kalman.single_kalman.validation import validate_input_dimension


@define
class SingleKalmanOutput:
    estimation: np.ndarray = field(validator=instance_of(np.ndarray))
    latest_error_matrix: np.ndarray = field(validator=instance_of(np.ndarray))

    @property
    def last_estimate(self) -> np.ndarray:
        return self.estimation[-1, :, :]

    @property
    def estimated(self) -> np.ndarray:
        return self.estimation


@attrs.define
class KalmanFilterInput:
    response_name: str = attrs.field(validator=instance_of(str))
    dependent_names: List[str] = attrs.field(
        validator=deep_iterable(
            member_validator=instance_of(str), iterable_validator=instance_of(list)
        )
    )
    observations: np.ndarray = attrs.field(validator=instance_of(np.ndarray))
    system_matrices: np.ndarray = attrs.field(validator=instance_of(np.ndarray))
    measurement_matrices: np.ndarray = attrs.field(validator=instance_of(np.ndarray))
    model_error_covariance_matrix: np.ndarray = attrs.field(
        validator=instance_of(np.ndarray)
    )
    observation_error_covariance: np.ndarray = attrs.field(
        validator=instance_of(np.ndarray)
    )
    initial_x0: np.ndarray = attrs.field(validator=instance_of(np.ndarray))
    initial_p0: np.ndarray = attrs.field(validator=instance_of(np.ndarray))
    control_vectors: np.ndarray = attrs.field(validator=instance_of(np.ndarray))

    def __attrs_post_init__(self):
        validate_input_dimension(
            observations=self.observations,
            system_matrices=self.system_matrices,
            measurement_matrices=self.measurement_matrices,
            model_error_covariance_matrix=self.model_error_covariance_matrix,
            observation_error_covariance=self.observation_error_covariance,
            initial_x0=self.initial_x0,
            initial_p0=self.initial_p0,
            control_vectors=self.control_vectors,
        )

    @staticmethod
    def from_dict(data) -> "KalmanFilterInput":
        return KalmanFilterInput(**data)

    def to_dict(self):
        return attrs.asdict(self)

    def duplicate(self):
        return KalmanFilterInput(**{i: v.copy() for i, v in self.to_dict().items()})
