from copy import copy
from typing import List

from attrs.validators import instance_of, deep_iterable, optional
from attrs import define, field, asdict
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


@define
class DSKFInputCollection:
    response_name: str = field(validator=instance_of(str))
    dependent_names: List[str] = field(
        validator=deep_iterable(member_validator=instance_of(str), iterable_validator=instance_of(list))
    )
    observations: np.ndarray = field(validator=instance_of(np.ndarray))
    system_matrices: np.ndarray = field(validator=instance_of(np.ndarray))
    measurement_matrices: np.ndarray = field(validator=instance_of(np.ndarray))
    model_error_covariance_matrix: np.ndarray = field(validator=instance_of(np.ndarray))
    observation_error_covariance: np.ndarray = field(validator=instance_of(np.ndarray))
    initial_x0: np.ndarray = field(validator=instance_of(np.ndarray))
    initial_p0: np.ndarray = field(validator=instance_of(np.ndarray))
    control_vectors: np.ndarray = field(validator=instance_of(np.ndarray))
    update_ratio_log10_list_log: List[float] = field(
        validator=optional(deep_iterable(member_validator=instance_of(float), iterable_validator=instance_of(list))),
        default=None,
    )

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
    def from_dict(data) -> "DSKFInputCollection":
        return DSKFInputCollection(**data)

    def to_dict(self):
        return asdict(self)

    def duplicate(self):
        return DSKFInputCollection.from_dict({i: copy(v) for i, v in self.to_dict().items()})
