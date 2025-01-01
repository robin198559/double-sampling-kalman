from typing import List, Optional, Dict

import numpy as np
from pandera import check_types
from pandera.typing import DataFrame

from double_sampling_kalman.double_kalman.api import double_kalman_filter
from double_sampling_kalman.factor_model.methods import (
    _double_kalman_filter_find_parameters,
    construct_multiplier_list,
)
from double_sampling_kalman.factor_model.objects import ScanConfig
from double_sampling_kalman.single_kalman.methods import initialize_control_vectors
from double_sampling_kalman.single_kalman.objects import KalmanFilterInput

from double_sampling_kalman.factor_model.schema import ObservationSchema
from double_sampling_kalman.utility.info import log_function


NUM_OBSERVATION_CHANNELS = 1


@check_types()
def initialize_filter_input(
    data: DataFrame[ObservationSchema],
    response_name: str,
    dependent_names: Optional[List[str]] = None,
    control_vector_adjustments: Optional[Dict[int, List[float]]] = None,
) -> KalmanFilterInput:

    if dependent_names is None:
        dependent_names = list(set(data["label"]).difference([response_name]))

    # prepare input data
    observations_table = data.pivot_table(
        index="index", columns="label", values="value"
    )

    observations_table = observations_table[[response_name] + list(dependent_names)]

    contains_missing = observations_table.isnull().any(axis=0).tolist()
    assert not any(
        contains_missing
    ), f"observation data contains missing values: {contains_missing}"

    n_comp = len(dependent_names)
    n_obs = observations_table.shape[0]
    observations_np = observations_table.values
    observations = observations_np[:, 0].reshape((n_obs, NUM_OBSERVATION_CHANNELS))
    measurement_matrices = observations_np[:, 1:].reshape(
        (n_obs, NUM_OBSERVATION_CHANNELS, n_comp)
    )

    # initialize filter parameters
    system_matrices = np.array([np.identity(n_comp)] * n_obs).reshape(
        (n_obs, n_comp, n_comp)
    )
    model_error_covariance_matrix = np.identity(n_comp).reshape((n_comp, n_comp))
    observation_error_covariance_matrix = np.array([[1e-8]]).reshape(
        (NUM_OBSERVATION_CHANNELS, NUM_OBSERVATION_CHANNELS)
    )
    initial_x0 = np.zeros(n_comp).reshape((n_comp, 1))
    initial_p0 = np.identity(n_comp).reshape((n_comp, n_comp))
    control_vectors = initialize_control_vectors(n_obs, n_comp)
    if control_vector_adjustments:
        for i, v in control_vector_adjustments.items():
            assert (
                len(v) == n_comp
            ), f"control vector adjustments must be list of size {n_comp}. Got {len(v)} at index {i}"
            control_vectors[i, :, 0] = v
    return KalmanFilterInput.from_dict(
        {
            "response_name": response_name,
            "dependent_names": dependent_names,
            "observations": observations,
            "system_matrices": system_matrices,
            "measurement_matrices": measurement_matrices,
            "model_error_covariance_matrix": model_error_covariance_matrix,
            "observation_error_covariance": observation_error_covariance_matrix,
            "initial_x0": initial_x0,
            "initial_p0": initial_p0,
            "control_vectors": control_vectors,
        }
    )


@log_function
def tune_filter(
    scan_config: ScanConfig,
    filter_input: KalmanFilterInput,
) -> KalmanFilterInput:
    """ """

    optimal_filter_parameters = _double_kalman_filter_find_parameters(
        max_iter=scan_config.max_iter,
        window_size_iter=scan_config.window_size_iter,
        convergence_tol=scan_config.convergence_tol,
        filter_tuning_multiplier_granularity=scan_config.filter_tuning_multiplier_granularity,
        initial_filter_tuning_multiplier_log_width=scan_config.initial_filter_tuning_multiplier_log_width,
        observations=filter_input.observations,
        system_matrices=filter_input.system_matrices,
        measurement_matrices=filter_input.measurement_matrices,
        model_error_covariance_matrix=filter_input.model_error_covariance_matrix,
        observation_error_covariance=filter_input.observation_error_covariance,
        initial_x0=filter_input.initial_x0,
        initial_p0=filter_input.initial_p0,
        control_vectors=filter_input.control_vectors,
    )
    optimal_filter_parameters.update(
        {
            "response_name": filter_input.response_name,
            "dependent_names": filter_input.dependent_names,
        }
    )
    return KalmanFilterInput.from_dict(optimal_filter_parameters)


@log_function
def tune_filter_control(
    scan_config: ScanConfig,
    filter_input: KalmanFilterInput,
) -> KalmanFilterInput:
    multipliers = construct_multiplier_list(
        multiplier_granularity=scan_config.control_tuning_multiplier_granularity,
        multiplier_log_width=scan_config.control_tuning_multiplier_log_width,
    )
    # find max diff
    filter_output = double_kalman_filter(filter_input=filter_input)
    diff = filter_output.backward - filter_output.forward
    max_diff_index = np.unravel_index(np.argmax(np.abs(diff), axis=None), diff.shape)
    # create candidate filter input
    filter_input_candidate = filter_input.duplicate()

    for k in multipliers:
        filter_input_candidate.control_vectors[max_diff_index] = (
            diff[max_diff_index] * k
        )
        result = double_kalman_filter(filter_input=filter_input_candidate)
