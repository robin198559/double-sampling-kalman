from typing import Optional, Tuple, List

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
    control_vectors: Optional[np.ndarray] = None,
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
    if control_vectors:
        control_vectors_reversed = (
            control_vectors[::-1, :, :][1:, :, :] if control_vectors else None
        )
        control_vectors_concat = np.concatenate(
            (control_vectors, control_vectors_reversed), axis=0
        )
    else:
        control_vectors_concat = None

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
@parfun(
    split=per_argument(filter_multipliers=list_by_chunk),
    combine_with=list_concat,
)
def _double_kalman_filter_core_calculate_signal(
    filter_multipliers: List[float],
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: Optional[np.ndarray] = None,
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
                system_matrices=system_matrices,
                control_vectors=control_vectors,
            )
        )
    return signal_list


def calculate_filter_signal(
    forward: np.ndarray,
    backward: np.ndarray,
    system_matrices: np.ndarray,
    control_vectors: Optional[np.ndarray] = None,
) -> float:
    if control_vectors is None:
        control_vectors = initialize_control_vectors(forward.shape[0], forward.shape[1])

    filter_ave = backward + forward / 2
    filter_displacement = np.sum(np.pow((forward - backward)[:, :, 0], 2), axis=0)
    x_hat = np.matmul(system_matrices, filter_ave) + control_vectors
    model_displacement = np.std(filter_ave[1:, :, 0] - x_hat[:-1, :, 0], axis=0)

    sig = np.sqrt(np.sum(np.pow(filter_displacement / model_displacement, 2)))
    return float(np.log(sig))


def calculate_observation_error_cov(
    forward: np.ndarray,
    observations: np.ndarray,
    measurement_matrices: np.ndarray,
) -> np.ndarray:
    ave = forward
    estimated_error = observations - np.matmul(measurement_matrices, ave)[:, :, 0]
    return np.cov(estimated_error.T).reshape(observations.shape[1], 1)


def calculate_model_error_cov(
    forward: np.ndarray,
    system_matrices: np.ndarray,
    control_vectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    if control_vectors is None:
        control_vectors = initialize_control_vectors(forward.shape[0], forward.shape[1])

    filter_ave = forward
    x_hat = np.matmul(system_matrices, filter_ave) + control_vectors
    return np.cov((filter_ave[1:, :, 0] - x_hat[:-1, :, 0]).T)
