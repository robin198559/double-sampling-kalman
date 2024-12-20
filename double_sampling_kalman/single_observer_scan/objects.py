from typing import Optional

import attrs

import numpy as np
from attr.validators import instance_of, optional

from double_sampling_kalman.single_kalman.validation import validate_input_dimension


@attrs.define
class DoubleSamplingFilter:
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
    control_vectors: Optional[np.ndarray] = attrs.field(
        validator=optional(instance_of(np.ndarray)), default=None
    )

    def __attrs_post_init__(self):
        return validate_input_dimension(
            observations=self.observations,
            system_matrices=self.system_matrices,
            measurement_matrices=self.measurement_matrices,
            model_error_covariance_matrix=self.model_error_covariance_matrix,
            observation_error_covariance=self.observation_error_covariance,
            initial_x0=self.initial_x0,
            initial_p0=self.initial_p0,
            control_vectors=self.control_vectors,
        )
