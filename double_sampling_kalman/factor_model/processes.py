import logging
from typing import Optional, List, Dict, Union, Tuple

import numpy as np
from parfun import parfun
from parfun.combine.collection import list_concat
from parfun.partition.api import per_argument
from parfun.partition.collection import list_by_chunk

from double_sampling_kalman.double_kalman.methods import (
    double_kalman_filter_core_open_ends,
    calculate_model_error_cov,
    calculate_observation_error_cov,
    double_kalman_filter_core,
)
from double_sampling_kalman.factor_model.methods import (
    calculate_filter_diff_squared_log10,
    stop_filter_scan,
    construct_multiplier_list,
    zoom_in_signal_multipliers,
)
from double_sampling_kalman.factor_model.objects import ScanConfig
from double_sampling_kalman.single_kalman.objects import DSKFInputCollection
from double_sampling_kalman.utility.info import log_function


@log_function
def _double_kalman_filter_find_parameters(
    max_iter: int,
    window_size_iter: int,
    convergence_log10_tol: float,
    filter_tuning_multiplier_granularity: int,
    initial_filter_tuning_multiplier_log10_width: float,
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
    update_ratio_log10_list_log: Optional[List[float]] = None,
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
    assert max_iter > window_size_iter, f"max iteration must be bigger than {window_size_iter}. Got {max_iter}"
    n_obs = observations.shape[0]
    n_comp = initial_x0.shape[0]
    if update_ratio_log10_list_log is None:
        update_ratio_log10_list = []
    else:
        update_ratio_log10_list = update_ratio_log10_list_log.copy()

    forward, backward = np.zeros((n_obs, n_comp, 1)), np.zeros((n_obs, n_comp, 1))
    for number_of_iterations in range(max_iter):
        # find best multiplier
        optimal_multiplier, filter_tuning_multiplier_granularity, initial_filter_tuning_multiplier_log10_width = (
            get_optimal_filter_multiplier_brutal_force(
                filter_tuning_multiplier_granularity=filter_tuning_multiplier_granularity,
                initial_filter_tuning_multiplier_log10_width=initial_filter_tuning_multiplier_log10_width,
                system_matrices=system_matrices,
                measurement_matrices=measurement_matrices,
                observations=observations,
                model_error_covariance_matrix=model_error_covariance_matrix,
                observation_error_covariance=observation_error_covariance,
                initial_x0=initial_x0,
                initial_p0=initial_p0,
                control_vectors=control_vectors,
            )
        )

        # calculate filter using best multiplier
        forward_new, backward_new, initial_p0 = double_kalman_filter_core_open_ends(
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix / optimal_multiplier,
            observation_error_covariance=observation_error_covariance * optimal_multiplier,
            initial_x0=initial_x0,
            initial_p0=initial_p0,
            control_vectors=control_vectors,
        )

        update_ratio_log10 = calculate_filter_diff_squared_log10(
            x1=backward_new, x2=backward
        ) - calculate_filter_diff_squared_log10(x1=backward_new[1:], x2=backward_new[:-1])

        # update filter inputs
        forward, backward = forward_new, backward_new
        model_error_covariance_matrix = calculate_model_error_cov(
            data=backward,
            system_matrices=system_matrices,
            control_vectors=control_vectors,
        )
        observation_error_covariance = calculate_observation_error_cov(
            data=backward,
            observations=observations,
            measurement_matrices=measurement_matrices,
        )

        initial_x0 = backward[0, :, 0].reshape(len(backward[0, :, 0]), 1)
        # determine if the best result has been achieved
        update_ratio_log10_list.append(update_ratio_log10)

        if stop_filter_scan(
            convergence_series=update_ratio_log10_list,
            window_size=window_size_iter,
            convergence_log10_tol=convergence_log10_tol,
        ):
            logging.info(f"successively obtained the filter approximation using {number_of_iterations=}")
            filter_parameters = {
                "observations": observations,
                "system_matrices": system_matrices,
                "measurement_matrices": measurement_matrices,
                "model_error_covariance_matrix": model_error_covariance_matrix,
                "observation_error_covariance": observation_error_covariance,
                "initial_x0": initial_x0,
                "initial_p0": initial_p0,
                "control_vectors": control_vectors,
                "update_ratio_log10_list_log": update_ratio_log10_list,
            }

            return filter_parameters

    raise ValueError(
        f"{update_ratio_log10_list=}. Failed to obtain converging filter results after {max_iter=} "
        f"with {convergence_log10_tol=}. You may try to increase the ScanConfig.max_iter or ScanConfig.convergence_tol"
    )


@log_function
def _tune_filter_control(
    scan_config: ScanConfig,
    filter_input: DSKFInputCollection,
) -> DSKFInputCollection:
    n_obs = filter_input.observations.shape[0]
    control_multipliers = construct_multiplier_list(
        multiplier_granularity=scan_config.control_tuning_multiplier_granularity,
        multiplier_log10_width=scan_config.control_tuning_multiplier_log10_width,
    )
    control_vectors = filter_input.control_vectors.copy()
    # find max diff and find max index
    forward, backward, _ = double_kalman_filter_core_open_ends(
        system_matrices=filter_input.system_matrices,
        measurement_matrices=filter_input.measurement_matrices,
        observations=filter_input.observations,
        model_error_covariance_matrix=filter_input.model_error_covariance_matrix,
        observation_error_covariance=filter_input.observation_error_covariance,
        initial_x0=filter_input.initial_x0,
        initial_p0=filter_input.initial_p0,
        control_vectors=control_vectors,
    )
    while True:
        current_signal = calculate_filter_diff_squared_log10(
            x1=forward,
            x2=backward,
        )
        diff = backward - forward
        max_diff_index = np.unravel_index(np.argmax(np.pow(diff, 2), axis=None), diff.shape)
        index_candidates = (
            [
                (max_diff_index[0] + (i + 1), max_diff_index[1], max_diff_index[2])
                for i in range(scan_config.control_index_offset_width)
                if (max_diff_index[0] + (i + 1)) < (n_obs - 1)
            ]
            + [
                (max_diff_index[0] - (i + 1), max_diff_index[1], max_diff_index[2])
                for i in range(scan_config.control_index_offset_width)
                if max_diff_index[0] - (i + 1) > 0
            ]
            + [max_diff_index]
        )
        # create vector and locations
        control_candidates = [(i, k * diff[max_diff_index] / 2) for k in control_multipliers for i in index_candidates]
        # create candidate filter input
        signal = find_double_kalman_filter_control_signal_in_parallel(
            control_candidates=control_candidates,
            system_matrices=filter_input.system_matrices,
            measurement_matrices=filter_input.measurement_matrices,
            observations=filter_input.observations,
            model_error_covariance_matrix=filter_input.model_error_covariance_matrix,
            observation_error_covariance=filter_input.observation_error_covariance,
            initial_x0=filter_input.initial_x0,
            initial_p0=filter_input.initial_p0,
            control_vectors=control_vectors,
        )

        if np.min(signal) < (current_signal - scan_config.control_log10_tol):
            best_candidate_index, best_candidate_value = control_candidates[np.argmin(signal)]
            logging.info(f"updating control at {best_candidate_index} with magnitude {best_candidate_value}")
            control_vectors[best_candidate_index] += best_candidate_value

            # find next
            forward, backward, _ = double_kalman_filter_core_open_ends(
                system_matrices=filter_input.system_matrices,
                measurement_matrices=filter_input.measurement_matrices,
                observations=filter_input.observations,
                model_error_covariance_matrix=filter_input.model_error_covariance_matrix,
                observation_error_covariance=filter_input.observation_error_covariance,
                initial_x0=filter_input.initial_x0,
                initial_p0=filter_input.initial_p0,
                control_vectors=control_vectors,
            )
        else:
            logging.info("no more controls found")
            break

    filter_input_copy = filter_input.duplicate()
    filter_input_copy.control_vectors = control_vectors
    return filter_input_copy


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
    signal_list = [
        get_double_kalman_filter_signal(
            filter_multiplier=k,
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix,
            observation_error_covariance=observation_error_covariance,
            initial_x0=initial_x0,
            initial_p0=initial_p0,
            control_vectors=control_vectors,
        )
        for k in filter_multipliers
    ]
    return signal_list


@log_function
def get_double_kalman_filter_signal(
    filter_multiplier: float,
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
) -> float:
    forward, backward, _ = double_kalman_filter_core(
        system_matrices=system_matrices,
        measurement_matrices=measurement_matrices,
        observations=observations,
        model_error_covariance_matrix=model_error_covariance_matrix / filter_multiplier,
        observation_error_covariance=observation_error_covariance * filter_multiplier,
        initial_x0=initial_x0,
        initial_p0=initial_p0,
        control_vectors=control_vectors,
    )

    return calculate_filter_diff_squared_log10(
        x1=forward,
        x2=backward,
    )


@log_function
def get_optimal_filter_multiplier_brutal_force(
    filter_tuning_multiplier_granularity: int,
    initial_filter_tuning_multiplier_log10_width: float,
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
):
    filter_multipliers = construct_multiplier_list(
        multiplier_granularity=filter_tuning_multiplier_granularity,
        multiplier_log10_width=initial_filter_tuning_multiplier_log10_width,
    )
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

    convex = np.diff(np.diff(signal))
    dip_index = np.argmax(convex)
    optimal_index = np.argmin(convex[dip_index:]) + dip_index + 1
    optimal_multiplier = filter_multipliers[optimal_index]

    filter_tuning_multiplier_granularity, multiplier_log10_width = zoom_in_signal_multipliers(
        current_multipliers=filter_multipliers,
        current_signal=signal,
        filter_tuning_multiplier_granularity=filter_tuning_multiplier_granularity,
    )
    return optimal_multiplier, filter_tuning_multiplier_granularity, multiplier_log10_width


@log_function
def get_optimal_filter_multiplier_gradiant_descent(
    starting_filter_multiplier_log10: float,
    d_filter_multiplier_log10: float,
    learning_rate_log10: float,
    stopping_filter_multiplier_log10: float,
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
) -> float:
    assert d_filter_multiplier_log10 > 0
    k_log10 = starting_filter_multiplier_log10

    while k_log10 > stopping_filter_multiplier_log10:
        current_signal_1, current_signal_2 = get_double_kalman_filter_signal_in_parallel(
            filter_multipliers=[np.pow(10, k_log10), np.pow(10, k_log10 + d_filter_multiplier_log10)],
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix,
            observation_error_covariance=observation_error_covariance,
            initial_x0=initial_x0,
            initial_p0=initial_p0,
            control_vectors=control_vectors,
        )
        slope = -(current_signal_2 - current_signal_1) / d_filter_multiplier_log10  # inverse the signal for minimal
        k_log10 = k_log10 - slope * learning_rate_log10
        if abs(slope * learning_rate_log10) < d_filter_multiplier_log10:
            return np.pow(10, k_log10)

    raise ValueError("not able to find local minimal using gradient descent")


@log_function
@parfun(
    split=per_argument(control_candidates=list_by_chunk),
    combine_with=list_concat,
)
def find_double_kalman_filter_control_signal_in_parallel(
    control_candidates: List[Tuple[Tuple, float]],
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
    for control_index, ctrl in control_candidates:
        control_vectors_i = control_vectors.copy()
        control_vectors_i[control_index] += ctrl
        forward, backward, _ = double_kalman_filter_core_open_ends(
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix,
            observation_error_covariance=observation_error_covariance,
            initial_x0=initial_x0,
            initial_p0=initial_p0,
            control_vectors=control_vectors_i,
        )
        signal_list.append(
            calculate_filter_diff_squared_log10(
                x1=forward,
                x2=backward,
            )
        )
    return signal_list
