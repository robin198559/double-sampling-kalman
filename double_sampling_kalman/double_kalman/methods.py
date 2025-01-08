from typing import Tuple

import numpy as np

from double_sampling_kalman.single_kalman.methods import (
    discrete_kalman_filter_core,
)
from double_sampling_kalman.utility.info import log_function


def shift_observation(arr, shift, fill_value=np.nan):
    assert shift > 0, "only positive shift"
    arr = np.roll(arr, shift=shift, axis=0)
    arr[:shift, :, :] = fill_value
    return arr


def reverse_filter_parameters(
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    control_vectors: np.ndarray,
):
    observations_reversed = observations[::-1, :]
    system_matrices_reversed = system_matrices[::-1, :, :]
    measurement_matrices_reversed = measurement_matrices[::-1, :, :]
    control_vectors_reversed = shift_observation(
        -control_vectors[::-1, :, :], shift=1, fill_value=0.0
    )  # ctrl needs to shift 1
    return observations_reversed, system_matrices_reversed, measurement_matrices_reversed, control_vectors_reversed


@log_function
def double_kalman_filter_core(
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations_reversed, system_matrices_reversed, measurement_matrices_reversed, control_vectors_reversed = (
        reverse_filter_parameters(
            observations=observations,
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            control_vectors=control_vectors,
        )
    )

    # construct reverse inputs
    observations_concat = np.concatenate((observations, observations_reversed[1:, :]), axis=0)
    system_matrices_concat = np.concatenate((system_matrices, system_matrices_reversed[1:, :, :]), axis=0)
    measurement_matrices_concat = np.concatenate(
        (measurement_matrices, measurement_matrices_reversed[1:, :, :]), axis=0
    )
    control_vectors_concat = np.concatenate((control_vectors, control_vectors_reversed[1:, :, :]), axis=0)

    # run forward filter
    kalman_output, error_matrix = discrete_kalman_filter_core(
        observations=observations_concat,
        system_matrices=system_matrices_concat,
        measurement_matrices=measurement_matrices_concat,
        model_error_covariance_matrix=model_error_covariance_matrix,
        observation_error_covariance=observation_error_covariance,
        initial_x0=initial_x0,
        initial_p0=initial_p0,
        control_vectors=control_vectors_concat,
    )

    forward_kalman = kalman_output[: observations.shape[0], :, :]
    backward_kalman = kalman_output[(observations.shape[0] - 1) :, :, :][::-1, :, :]

    return forward_kalman, backward_kalman, error_matrix


@log_function
def double_kalman_filter_core_open_ends(
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # construct reverse inputs
    observations_reversed, system_matrices_reversed, measurement_matrices_reversed, control_vectors_reversed = (
        reverse_filter_parameters(
            observations=observations,
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            control_vectors=control_vectors,
        )
    )

    # run forward filter
    forward_kalman, initial_p0 = discrete_kalman_filter_core(
        observations=observations,
        system_matrices=system_matrices,
        measurement_matrices=measurement_matrices,
        model_error_covariance_matrix=model_error_covariance_matrix,
        observation_error_covariance=observation_error_covariance,
        initial_x0=initial_x0,
        initial_p0=initial_p0,
        control_vectors=control_vectors,
    )
    # run backward filter
    backward_kalman, error_matrix = discrete_kalman_filter_core(
        observations=observations_reversed,
        system_matrices=system_matrices_reversed,
        measurement_matrices=measurement_matrices_reversed,
        model_error_covariance_matrix=model_error_covariance_matrix,
        observation_error_covariance=observation_error_covariance,
        initial_x0=forward_kalman[-1, :, :],
        initial_p0=initial_p0,
        control_vectors=control_vectors_reversed,
    )

    return forward_kalman, backward_kalman[::-1, :, :], error_matrix


@log_function
def double_kalman_filter_core_w_constraint_factor_model(
    constraint_observation_variance: float,
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # calculate filter using the best multiplier and add constraint
    measurement_matrices_w_constraint = np.concatenate(
        (measurement_matrices, np.ones(measurement_matrices.shape)), axis=1
    )
    observations_with_constraint = np.concatenate((observations, np.ones(observations.shape)), axis=1)
    observation_error_covariance_with_constraint = concat_diagonal(
        observation_error_covariance,
        np.array([constraint_observation_variance]).reshape((1, 1)),
    )

    forward, backward, error_matrix = double_kalman_filter_core_open_ends(
        system_matrices=system_matrices,
        measurement_matrices=measurement_matrices_w_constraint,
        observations=observations_with_constraint,
        model_error_covariance_matrix=model_error_covariance_matrix,
        observation_error_covariance=observation_error_covariance_with_constraint,
        initial_x0=initial_x0,
        initial_p0=initial_p0,
        control_vectors=control_vectors,
    )

    return forward, backward, error_matrix


def concat_diagonal(a, b):
    """Concatenates matrices a and b diagonally."""
    max_dim = max(a.shape[0], b.shape[0])
    a_padded = np.pad(a, ((0, max_dim - a.shape[0]), (0, max_dim - a.shape[1])), mode="constant")
    b_padded = np.pad(b, ((0, max_dim - b.shape[0]), (0, max_dim - b.shape[1])), mode="constant")
    return np.block([[a_padded, np.zeros_like(b_padded)], [np.zeros_like(a_padded), b_padded]])


def calculate_observation_error_cov(
    data: np.ndarray,
    observations: np.ndarray,
    measurement_matrices: np.ndarray,
) -> np.ndarray:
    ave = data
    estimated_error = observations - np.matmul(measurement_matrices, ave)[:, :, 0]
    return np.cov(estimated_error.T).reshape(observations.shape[1], observations.shape[1])


def calculate_model_error_cov(
    data: np.ndarray,
    system_matrices: np.ndarray,
    control_vectors: np.ndarray,
) -> np.ndarray:
    filter_ave = data
    x_hat = np.matmul(system_matrices, filter_ave) + control_vectors
    return np.cov((filter_ave[1:, :, 0] - x_hat[:-1, :, 0]).T)
