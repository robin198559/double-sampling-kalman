from double_sampling_kalman.double_kalman.methods import (
    double_kalman_filter_core_open_ends,
)
from double_sampling_kalman.double_kalman.objects import DoubleKalmanOutput
from double_sampling_kalman.single_kalman.objects import DSKFInputCollection
from double_sampling_kalman.utility.info import log_function


@log_function
def double_kalman_filter(
    filter_input: DSKFInputCollection,
) -> DoubleKalmanOutput:
    """
    Calculate 2 runs of kalman filter - forward and backward.

    :param filter_input:
    """
    forward, backward, _ = double_kalman_filter_core_open_ends(
        system_matrices=filter_input.system_matrices,
        measurement_matrices=filter_input.measurement_matrices,
        observations=filter_input.observations,
        model_error_covariance_matrix=filter_input.model_error_covariance_matrix,
        observation_error_covariance=filter_input.observation_error_covariance,
        initial_x0=filter_input.initial_x0,
        initial_p0=filter_input.initial_p0,
        control_vectors=filter_input.control_vectors,
    )

    return DoubleKalmanOutput(dependent_names=filter_input.dependent_names, forward=forward, backward=backward)
