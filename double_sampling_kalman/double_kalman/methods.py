from typing import Optional

import numpy as np

from double_sampling_kalman.double_kalman.objects import DoubleKalmanOutput
from double_sampling_kalman.single_kalman.methods import _discrete_kalman_filter_core
from double_sampling_kalman.utility.info import log_function


@log_function
def _double_kalman_filter_core(
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: Optional[np.ndarray] = None,
) -> DoubleKalmanOutput:

    # run forward filter
    forward_kalman = _discrete_kalman_filter_core(
        observations=observations,
        system_matrices=system_matrices,
        measurement_matrices=measurement_matrices,
        model_error_covariance_matrix=model_error_covariance_matrix,
        observation_error_covariance=observation_error_covariance,
        initial_x0=initial_x0,
        initial_p0=initial_p0,
        control_vectors=control_vectors,
    )

    # reverse observations
    observations_reversed = observations[::-1, :]
    system_matrices_reversed = system_matrices[::-1, :, :]
    measurement_matrices_reversed = measurement_matrices[::-1, :, :]
    control_vectors_reversed = control_vectors if control_vectors else None

    # run backward filter
    backward_kalman = _discrete_kalman_filter_core(
        observations=observations_reversed,
        system_matrices=system_matrices_reversed,
        measurement_matrices=measurement_matrices_reversed,
        model_error_covariance_matrix=model_error_covariance_matrix,
        observation_error_covariance=observation_error_covariance,
        initial_x0=forward_kalman.last_estimate,
        initial_p0=forward_kalman.latest_error_matrix,
        control_vectors=control_vectors_reversed,
    )

    return DoubleKalmanOutput(forward=forward_kalman, backward=backward_kalman)
