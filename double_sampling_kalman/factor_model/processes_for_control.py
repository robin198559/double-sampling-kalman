import logging
from typing import List, Tuple

import numpy as np
from parfun import parfun
from parfun.combine.collection import list_concat
from parfun.partition.api import per_argument
from parfun.partition.collection import list_by_chunk

from double_sampling_kalman.double_kalman.methods import double_kalman_filter_core_open_ends
from double_sampling_kalman.factor_model.methods import construct_multiplier_list, calculate_filter_diff_squared_log10
from double_sampling_kalman.factor_model.objects import ScanConfig
from double_sampling_kalman.single_kalman.objects import DSKFInputCollection
from double_sampling_kalman.utility.info import log_function


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
