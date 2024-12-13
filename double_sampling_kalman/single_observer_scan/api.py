from operator import index
from typing import List, Optional

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from double_sampling_kalman.single_observer_scan.methods import (
    _double_kalman_filter_numpy_scan,
)
from double_sampling_kalman.double_kalman.objects import DoubleKalmanOutput

from double_sampling_kalman.single_observer_scan.schema import ObservationSchema
from double_sampling_kalman.utility.info import log_function


def _double_sampling_kalman_filter_scan(
    data: DataFrame[ObservationSchema],
    response: str,
    dependents: List[str],
):

    # prepare input data
    observations_table = data.pivot_table(
        index="index", columns="label", values="value"
    )
    observations_table = observations_table[[response] + list(dependents)]

    contains_missing = observations_table.isnull().any(axis=1).tolist()
    assert (
        not contains_missing
    ), f"observation data contains missing values: {contains_missing}"

    n_comp = len(dependents)
    n_obs = observations_table.shape[0]
    observations_np = observations_table.values
    observations = observations_np[:, 0]
    measurement_matrices = observations_np[:, 1:]

    # initialize filter parameters
    system_matrices = np.array([np.identity(n_comp)] * n_obs)
    model_error_covariance_matrix = np.identity(n_comp)
    observation_error_covariance_matrix = np.array([[1e-8]])
    initial_x0 = np.zeros(n_comp).reshape((n_comp, 1))
    initial_p0 = np.identity(n_comp)

    # initial scan

    # control adjustment

    # final scan
    return


@log_function
def double_kalman_filter_numpy_scan(
    max_iter: int,
    observations: np.ndarray,
    system_matrices: np.ndarray,
    measurement_matrices: np.ndarray,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.ndarray,
    initial_x0: np.ndarray,
    initial_p0: np.ndarray,
    control_vectors: Optional[np.ndarray] = None,
) -> DoubleKalmanOutput:
    """
    Calculate iteration runs of double sampling kalman filter.
    Returns N x M output
    Assuming (N, I) observations, (M, 1) system components

    :param max_iter:
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

    forward, backward = _double_kalman_filter_numpy_scan(
        max_iter=max_iter,
        observations=observations,
        system_matrices=system_matrices,
        measurement_matrices=measurement_matrices,
        model_error_covariance_matrix=model_error_covariance_matrix,
        observation_error_covariance=observation_error_covariance,
        initial_x0=initial_x0,
        initial_p0=initial_p0,
        control_vectors=control_vectors,
    )
    return DoubleKalmanOutput(forward=forward, backward=backward)
