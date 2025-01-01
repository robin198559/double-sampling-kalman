import logging
from typing import List, Optional, Tuple, Dict, Union

import numpy as np

from double_sampling_kalman.double_kalman.methods import (
    get_double_kalman_filter_signal_in_parallel,
    calculate_model_error_cov,
    calculate_observation_error_cov,
    double_kalman_filter_core_open_ends,
)
from double_sampling_kalman.single_kalman.objects import KalmanFilterInput
from double_sampling_kalman.utility.info import log_function


def moving_std(a, window_size=3) -> np.array:
    assert window_size > 1
    return np.array(
        [np.std(a[i - window_size : i]) for i in range(window_size, len(a))]
    )


def calculate_filter_convergence(
    error_cov_list: List[float], window_size: int
) -> List[float]:
    if len(error_cov_list) > window_size:
        signal_log = np.log10(error_cov_list)
        return moving_std(signal_log, window_size=window_size).tolist()
    else:
        return []


def stop_filter_scan(
    convergence_series: List[float], window_size: int, convergence_tol: float
):
    if len(convergence_series) > 0:
        within_range = (np.array(convergence_series) < convergence_tol).tolist()
        if np.sum(within_range[-window_size:]) >= window_size:
            return True
    return False


def construct_multiplier_list(
    multiplier_granularity: int, multiplier_log_width: float
) -> List[float]:
    assert multiplier_granularity > 1
    assert multiplier_log_width > 0
    log_filter_multipliers = np.arange(
        -multiplier_log_width,
        multiplier_log_width,
        2 * multiplier_log_width / int(multiplier_granularity),
    ).tolist() + [multiplier_log_width]
    return np.exp(log_filter_multipliers)


@log_function
def _double_kalman_filter_find_parameters(
    max_iter: int,
    window_size_iter: int,
    convergence_tol: float,
    filter_tuning_multiplier_granularity: int,
    initial_filter_tuning_multiplier_log_width: float,
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
) -> Dict[str, Union[str, np.ndarray]]:
    """
    Calculate iteration runs of double sampling kalman filter.
    Returns N x M output
    Assuming (N, I) observations, (M, 1) system components

    :param max_iter:
    :param window_size_iter:
    :param observations:
    :param system_matrices:
    :param measurement_matrices:
    :param model_error_covariance_matrix:
    :param observation_error_covariance:
    :param initial_x0:
    :param initial_p0:
    :param control_vectors:
    :return: SingleKalmanOutput.solution: size N x M
    """
    assert (
        max_iter > window_size_iter
    ), f"max iteration must be bigger than {window_size_iter}. Got {max_iter}"
    error_cov_list = []
    convergence_series = []
    filter_multipliers = construct_multiplier_list(
        multiplier_granularity=filter_tuning_multiplier_granularity,
        multiplier_log_width=initial_filter_tuning_multiplier_log_width,
    )

    for number_of_iterations in range(max_iter):
        # find best multiplier
        signal = get_double_kalman_filter_signal_in_parallel(
            filter_multipliers=filter_multipliers,
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix,
            observation_error_covariance=observation_error_covariance,
            initial_x0=initial_x0,
            initial_p0=initial_p0,
            control_vectors=control_vectors,
        )

        optimal_multiplier = filter_multipliers[np.argmax(np.diff(np.diff(signal))) + 1]

        # calculate filter using best multiplier
        forward, backward, initial_p0 = double_kalman_filter_core_open_ends(
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix
            / optimal_multiplier,
            observation_error_covariance=observation_error_covariance
            * optimal_multiplier,
            initial_x0=initial_x0,
            initial_p0=initial_p0,
            control_vectors=control_vectors,
        )

        # update filter inputs
        model_error_covariance_matrix = calculate_model_error_cov(
            forward=forward,
            system_matrices=system_matrices,
            control_vectors=control_vectors,
        )
        observation_error_covariance = calculate_observation_error_cov(
            forward=forward,
            observations=observations,
            measurement_matrices=measurement_matrices,
        )
        filter_multipliers = zoom_in_signal_multipliers(
            current_multipliers=filter_multipliers,
            current_signal=signal,
            filter_tuning_multiplier_granularity=filter_tuning_multiplier_granularity,
        )
        initial_x0 = backward[0, :, 0].reshape(len(backward[0, :, 0]), 1)
        # determine if the best result has been achieved
        error_cov_list.append(
            float((observation_error_covariance * optimal_multiplier)[0, 0])
        )
        convergence_series = calculate_filter_convergence(
            error_cov_list=error_cov_list,
            window_size=window_size_iter,
        )
        if stop_filter_scan(
            convergence_series=convergence_series,
            window_size=window_size_iter,
            convergence_tol=convergence_tol,
        ):
            logging.info(
                f"successively obtained the filter approximation using {number_of_iterations=}"
            )
            return {
                "observations": observations,
                "system_matrices": system_matrices,
                "measurement_matrices": measurement_matrices,
                "model_error_covariance_matrix": model_error_covariance_matrix,
                "observation_error_covariance": observation_error_covariance,
                "initial_x0": initial_x0,
                "initial_p0": initial_p0,
                "control_vectors": control_vectors,
            }

    raise ValueError(
        f"{convergence_series=}. Failed to obtain converging filter results after {max_iter=} "
        f"with {convergence_tol=}. You may try to increase the ScanConfig.max_iter or ScanConfig.convergence_tol"
    )


def zoom_in_signal_multipliers(
    current_multipliers: List[float],
    current_signal: List[float],
    filter_tuning_multiplier_granularity: int,
) -> List[float]:
    convex = np.diff(np.diff(current_signal))
    negative_convex_index_list = [
        i for i, v in enumerate(convex) if v < 0 and v < min(convex) * 0.8
    ]
    positive_convex_index_list = [
        i for i, v in enumerate(convex) if v > 0 and v > max(convex) * 0.8
    ]
    large_signal_convexity_index = (
        negative_convex_index_list + positive_convex_index_list
    )
    left_wing = current_multipliers[min(large_signal_convexity_index)]
    right_wing = current_multipliers[max(large_signal_convexity_index)]
    multiplier_log_width = max(
        abs(np.log(left_wing)),
        abs(np.log(right_wing)),
    )
    multiplier_log_width = multiplier_log_width + 1
    current_multipliers = construct_multiplier_list(
        multiplier_granularity=filter_tuning_multiplier_granularity,
        multiplier_log_width=multiplier_log_width,
    )
    return current_multipliers
