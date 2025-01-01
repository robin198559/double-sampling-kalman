from typing import Tuple, List

import numpy as np
from parfun import parfun
from parfun.combine.collection import list_concat
from parfun.partition.api import per_argument
from parfun.partition.collection import list_by_chunk

from double_sampling_kalman.single_kalman.methods import (
    _discrete_kalman_filter_core,
    initialize_control_vectors,
)
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
    control_vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # construct reverse inputs
    observations_reversed = observations[::-1, :][1:, :]
    system_matrices_reversed = system_matrices[::-1, :, :][1:, :, :]
    measurement_matrices_reversed = measurement_matrices[::-1, :, :][1:, :, :]
    observations_concat = np.concatenate((observations, observations_reversed), axis=0)
    system_matrices_concat = np.concatenate(
        (system_matrices, system_matrices_reversed), axis=0
    )
    measurement_matrices_concat = np.concatenate(
        (measurement_matrices, measurement_matrices_reversed), axis=0
    )

    control_vectors_reversed = control_vectors[::-1, :, :][1:, :, :]
    control_vectors_concat = np.concatenate(
        (control_vectors, control_vectors_reversed), axis=0
    )

    # run forward filter
    kalman_output, error_matrix = _discrete_kalman_filter_core(
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
    observations_reversed = observations[::-1, :]
    system_matrices_reversed = system_matrices[::-1, :, :]
    measurement_matrices_reversed = measurement_matrices[::-1, :, :]
    control_vectors_reversed = control_vectors[::-1, :, :]

    # run forward filter
    forward_kalman, initial_p0 = _discrete_kalman_filter_core(
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
    backward_kalman, error_matrix = _discrete_kalman_filter_core(
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
    # calculate filter using best multiplier and add constraint
    measurement_matrices_w_constraint = np.concatenate(
        (measurement_matrices, np.ones(measurement_matrices.shape)), axis=1
    )
    observations_with_constraint = np.concatenate(
        (observations, np.ones(observations.shape)), axis=1
    )
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
    a_padded = np.pad(
        a, ((0, max_dim - a.shape[0]), (0, max_dim - a.shape[1])), mode="constant"
    )
    b_padded = np.pad(
        b, ((0, max_dim - b.shape[0]), (0, max_dim - b.shape[1])), mode="constant"
    )
    return np.block(
        [[a_padded, np.zeros_like(b_padded)], [np.zeros_like(a_padded), b_padded]]
    )


@log_function
@parfun(
    split=per_argument(filter_multipliers=list_by_chunk),
    combine_with=list_concat,
)
def get_double_kalman_filter_signal_in_parallel(
    filter_multipliers: List[float],
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
) -> List[float]:
    signal_list = []
    for k in filter_multipliers:
        forward, backward, error_matrix = _double_kalman_filter_core(
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix / k,
            observation_error_covariance=observation_error_covariance * k,
            initial_x0=initial_x0,
            initial_p0=initial_p0,
            control_vectors=control_vectors,
        )
        signal_list.append(
            calculate_filter_signal(
                forward=forward,
                backward=backward,
            )
        )
    return signal_list


@log_function
@parfun(
    split=per_argument(filter_multipliers=list_by_chunk),
    combine_with=list_concat,
)
def find_double_kalman_filter_control_signal_in_parallel(
    control_location_and_magnitude: List[Tuple[int, float]],
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
) -> List[float]:
    signal_list = []
    for k, c in control_location_and_magnitude:
        control_vectors = control_vectors
        forward, backward, error_matrix = _double_kalman_filter_core(
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix,
            observation_error_covariance=observation_error_covariance,
            initial_x0=initial_x0,
            initial_p0=initial_p0,
            control_vectors=control_vectors,
        )
        signal_list.append(
            calculate_filter_signal(
                forward=forward,
                backward=backward,
            )
        )
    return signal_list


def calculate_filter_signal(
    forward: np.ndarray,
    backward: np.ndarray,
) -> float:
    filter_displacement = np.mean((forward - backward)[:, :, 0], axis=0)
    model_displacement = np.std((forward - backward)[:, :, 0], axis=0)

    np.seterr(invalid="raise")
    try:
        sig = np.sqrt(np.sum(np.pow(filter_displacement / model_displacement, 2)))
    except FloatingPointError as err:
        raise FloatingPointError(
            f"{err}. Last filter values: Most likely the filter exploded."
        )
    return float(np.log(sig))


def calculate_observation_error_cov(
    forward: np.ndarray,
    observations: np.ndarray,
    measurement_matrices: np.ndarray,
) -> np.ndarray:
    ave = forward
    estimated_error = observations - np.matmul(measurement_matrices, ave)[:, :, 0]
    return np.cov(estimated_error.T).reshape(
        observations.shape[1], observations.shape[1]
    )


def calculate_model_error_cov(
    forward: np.ndarray,
    system_matrices: np.ndarray,
    control_vectors: np.ndarray,
) -> np.ndarray:
    filter_ave = forward
    x_hat = np.matmul(system_matrices, filter_ave) + control_vectors
    return np.cov((filter_ave[1:, :, 0] - x_hat[:-1, :, 0]).T)
