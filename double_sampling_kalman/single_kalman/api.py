from typing import Optional

import numpy as np

from double_sampling_kalman.single_kalman.methods import _discrete_kalman_filter_core
from double_sampling_kalman.single_kalman.objects import (
    SingleKalmanOutput,
    KalmanFilterInput,
)
from double_sampling_kalman.single_kalman.validation import validate_input_dimension
from double_sampling_kalman.utility.info import log_function


@log_function
def discrete_kalman_filter_numpy_runner(
    filter_input: KalmanFilterInput,
) -> SingleKalmanOutput:
    """
    calculate Kalman filter run

    :param filter_input:
    """

    output, pt = _discrete_kalman_filter_core(
        system_matrices=filter_input.system_matrices,
        measurement_matrices=filter_input.measurement_matrices,
        observations=filter_input.observations,
        model_error_covariance_matrix=filter_input.model_error_covariance_matrix,
        observation_error_covariance=filter_input.observation_error_covariance,
        initial_x0=filter_input.initial_x0,
        initial_p0=filter_input.initial_p0,
        control_vectors=filter_input.control_vectors,
    )

    return SingleKalmanOutput(
        estimation=output,
        latest_error_matrix=pt,
    )
