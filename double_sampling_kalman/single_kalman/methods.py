from typing import Optional, Tuple

import numpy as np

from double_sampling_kalman.utility.info import log_function


@log_function
def _discrete_kalman_filter_core(
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    core function of kalman filter. It has to be for loop because it is iterative. No way to optimize

    :param measurement_matrices:
    :param system_matrices:
    :param observations:
    :param model_error_covariance_matrix:
    :param observation_error_covariance:
    :param initial_x0:
    :param initial_p0:
    :param control_vectors:
    :return: solution: N x M, xt, pt
    """
    number_of_observations = observations.shape[0]
    number_of_system_components = initial_x0.shape[0]

    # initialize variables
    x_solution = []
    xt, pt = initial_x0, initial_p0
    identity = np.identity(number_of_system_components)
    for t in range(number_of_observations):  # main kalman loop
        measurement_matrix = measurement_matrices[t]
        system_matrix = system_matrices[t]
        # predict
        xt_hat_ = np.matmul(system_matrix, xt) + control_vectors[t]
        pt_hat = (
            np.matmul(system_matrix, np.matmul(pt, system_matrix))
            + model_error_covariance_matrix
        )
        # correct
        pt_ht = np.matmul(pt_hat, measurement_matrix.T)

        kalman_gain_t = pt_ht / (
            np.matmul(measurement_matrix, pt_ht) + observation_error_covariance
        )
        estimated_error = np.matmul(
            kalman_gain_t, (observations[t] - np.matmul(measurement_matrix, xt_hat_))
        )
        xt_hat = xt_hat_ + estimated_error
        pt_hat = np.matmul(
            (identity - np.outer(kalman_gain_t, measurement_matrix)), pt_hat
        )
        # save the data
        x_solution.append(xt_hat)
        # for next iteration
        xt = np.array(xt_hat)
        pt = np.array(pt_hat)
    # Output
    output = np.array(x_solution)
    assert output.shape == (number_of_observations, number_of_system_components, 1)
    return output, pt


def initialize_control_vectors(
    number_of_observations: int, number_of_system_components: int
) -> np.ndarray:
    assert isinstance(number_of_observations, int) and number_of_observations > 0
    assert (
        isinstance(number_of_system_components, int) and number_of_system_components > 0
    )
    control_vectors = np.zeros(
        number_of_system_components * number_of_observations
    ).reshape(number_of_observations, number_of_system_components, 1)
    return control_vectors
