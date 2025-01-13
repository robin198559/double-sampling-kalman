import logging
from typing import Optional, List, Dict, Union

import numpy as np
from parfun import parfun
from parfun.combine.collection import list_concat
from parfun.partition.api import per_argument
from parfun.partition.collection import list_by_chunk
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

from double_sampling_kalman.double_kalman.methods import (
    double_kalman_filter_core_open_ends,
    calculate_model_error_cov,
    calculate_observation_error_cov,
    double_kalman_filter_core,
)
from double_sampling_kalman.factor_model.exception import GradientDescentOutOfBoundError, GradientDescentExhaustedError
from double_sampling_kalman.factor_model.methods import (
    calculate_filter_diff_squared_log10,
    stop_filter_scan,
    construct_multiplier_list,
    get_double_kalman_filter_signal,
)
from double_sampling_kalman.utility.info import log_function


@log_function
def _double_kalman_filter_find_parameters(
    max_iter: int,
    window_size_iter: int,
    convergence_log10_tol: float,
    filter_tuning_multiplier_granularity: int,
    filter_tuning_multiplier_log10_width: float,
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

    use_gradient_descent = True

    filter_multipliers = construct_multiplier_list(
        multiplier_granularity=filter_tuning_multiplier_granularity,
        multiplier_log10_width=filter_tuning_multiplier_log10_width,
    )
    # calculate filter using best multiplier
    forward_base, backward_base, initial_p0 = double_kalman_filter_core_open_ends(
        system_matrices=system_matrices,
        measurement_matrices=measurement_matrices,
        observations=observations,
        model_error_covariance_matrix=model_error_covariance_matrix / np.pow(10, filter_tuning_multiplier_log10_width),
        observation_error_covariance=observation_error_covariance * np.pow(10, filter_tuning_multiplier_log10_width),
        initial_x0=initial_x0,
        initial_p0=initial_p0,
        control_vectors=control_vectors,
    )
    base_diff_squared_log10 = calculate_filter_diff_squared_log10(x1=backward_base[1:], x2=backward_base[:-1])

    forward, backward = np.zeros((n_obs, n_comp, 1)), np.zeros((n_obs, n_comp, 1))
    for number_of_iterations in range(max_iter):
        if use_gradient_descent:
            try:
                optimal_multiplier = get_optimal_filter_multiplier_gradient_descent(
                    max_iter=max_iter,
                    starting_filter_multiplier_log10=filter_tuning_multiplier_log10_width,
                    delta_filter_multiplier_log10=1e-3,
                    learning_rate_log10=1,
                    stopping_filter_multiplier_log10=-filter_tuning_multiplier_log10_width,
                    system_matrices=system_matrices,
                    measurement_matrices=measurement_matrices,
                    observations=observations,
                    model_error_covariance_matrix=model_error_covariance_matrix,
                    observation_error_covariance=observation_error_covariance,
                    initial_x0=initial_x0,
                    initial_p0=initial_p0,
                    control_vectors=control_vectors,
                )
            except (GradientDescentOutOfBoundError, GradientDescentExhaustedError):
                use_gradient_descent = False
                logging.warning("gradient descent failed. Trying brutal force method")
        if not use_gradient_descent:
            optimal_multiplier = get_optimal_filter_multiplier_brutal_force(
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

        update_ratio_log10 = calculate_filter_diff_squared_log10(x1=backward_new, x2=backward) - base_diff_squared_log10

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
def get_optimal_filter_multiplier_brutal_force(
    filter_multipliers: List[float],
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: np.ndarray,
):
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

    slope = np.diff(signal)
    convex = np.diff(np.diff(signal))
    if max(slope) > 1e-4:
        # it means local minimum
        dip_index = np.argmax(convex)
        abs_diff = np.abs(np.diff(signal[dip_index:]))
        optimal_index = np.argmin(abs_diff) + dip_index

        # use cubic spline to find the optimal multiplier
        cs = CubicSpline(
            np.log10(filter_multipliers[optimal_index - 2 : optimal_index + 3]),
            signal[optimal_index - 2 : optimal_index + 3],
            extrapolate=False,
        )
        cs_d = cs.derivative()
        roots = cs_d.roots()
        optimal_multiplier_log10 = list(cs(roots))[0]
    else:
        # it means local minimum
        optimal_index = np.argmin(convex) + 1

        # use cubic spline to find the optimal multiplier
        cs = CubicSpline(
            np.log10(filter_multipliers[optimal_index - 2 : optimal_index + 3]),
            convex[optimal_index - 2 : optimal_index + 3],
            extrapolate=False,
        )
        cs_d = cs.derivative()
        roots = cs_d.roots()
        optimal_multiplier_log10 = list(cs(roots))[0]
        breakpoint()
    return np.pow(10, optimal_multiplier_log10)


@log_function
def get_optimal_filter_multiplier_gradient_descent(
    max_iter: int,
    starting_filter_multiplier_log10: float,
    delta_filter_multiplier_log10: float,
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

    k_log10 = starting_filter_multiplier_log10
    number_of_iterations = 0

    while k_log10 > stopping_filter_multiplier_log10 and number_of_iterations < max_iter:

        forward_1, backward_1, initial_p0_1 = double_kalman_filter_core(
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix / np.pow(10, k_log10),
            observation_error_covariance=observation_error_covariance * np.pow(10, k_log10),
            initial_x0=initial_x0,
            initial_p0=initial_p0,
            control_vectors=control_vectors,
        )

        current_signal_1 = calculate_filter_diff_squared_log10(
            x1=forward_1,
            x2=backward_1,
        )

        forward_2, backward_2, _ = double_kalman_filter_core(
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix
            / np.pow(10, k_log10 + delta_filter_multiplier_log10),
            observation_error_covariance=observation_error_covariance
            * np.pow(10, k_log10 + delta_filter_multiplier_log10),
            initial_x0=initial_x0,
            initial_p0=initial_p0,
            control_vectors=control_vectors,
        )

        current_signal_2 = calculate_filter_diff_squared_log10(
            x1=forward_2,
            x2=backward_2,
        )

        # update gradient descent
        slope = -(current_signal_2 - current_signal_1) / delta_filter_multiplier_log10  # inverse the signal for minimal

        # stop the iteration if criteria is met
        if abs(slope * learning_rate_log10) < delta_filter_multiplier_log10:
            logging.info(
                f"successively obtained the optimal multiplier using gradient descent with {number_of_iterations=} "
                f"with {delta_filter_multiplier_log10=}"
            )
            return np.pow(10, k_log10)
        else:
            # if continue the iteration
            k_log10 = k_log10 - slope * learning_rate_log10
            number_of_iterations += 1

    if k_log10 <= stopping_filter_multiplier_log10:
        raise GradientDescentOutOfBoundError(
            f"no local minimal between {starting_filter_multiplier_log10=} and {stopping_filter_multiplier_log10=}"
        )
    else:
        raise GradientDescentExhaustedError(f"not able to find local minimal using gradient descent in {max_iter=}")
