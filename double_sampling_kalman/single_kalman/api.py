from typing import Optional

import numpy as np

from double_sampling_kalman.single_kalman.methods import _discrete_kalman_filter
from double_sampling_kalman.single_kalman.validation import validate_input_dimension
from double_sampling_kalman.utility.info import log_function


@log_function
def discrete_kalman_filter_numpy_runner(
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate one run of kalman filter. Inputs are all numpy.ndarray
    Returns N x M output
    Assuming (N, I) observations, (M, 1) system components

    :param observations:
    :param system_matrices:
    :param measurement_matrices:
    :param model_error_covariance_matrix:
    :param observation_error_covariance:
    :param initial_x0:
    :param initial_p0:
    :param control_vectors:
    :return: N x M
    """

    validate_input_dimension(
        observations=observations,
        system_matrices=system_matrices,
        measurement_matrices=measurement_matrices,
        model_error_covariance_matrix=model_error_covariance_matrix,
        observation_error_covariance=observation_error_covariance,
        initial_x0=initial_x0,
        initial_p0=initial_p0,
        control_vectors=control_vectors,
    )

    result = _discrete_kalman_filter(
        system_matrices=system_matrices,
        measurement_matrices=measurement_matrices,
        observations=observations,
        model_error_covariance_matrix=model_error_covariance_matrix,
        observation_error_covariance=observation_error_covariance,
        initial_x0=initial_x0,
        initial_p0=initial_p0,
        control_vectors=control_vectors,
    )

    return result