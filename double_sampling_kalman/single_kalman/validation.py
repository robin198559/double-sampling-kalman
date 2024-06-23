from typing import Optional

import numpy as np


def validate_input_dimension(
    observations: np.array,
    system_matrices: np.array,
    measurement_matrices: np.array,
    model_error_covariance_matrix: np.ndarray,
    observation_error_covariance: np.array,
    initial_x0: np.array,
    initial_p0: np.array,
    control_vectors: Optional[np.array] = None,
):
    """

    Assume (N, I) observations, (M, 1) system components
        Then
        system_matrices: N x (M, M)
        measurement_matrices: N x (I, M)
        model_error_covariance_matrix: M x M
        observation_error_covariance_matrix: I x I
        initial_x0: (M, 1)
        initial_p0: M x M
        control_vectors: N x (M, 1)

    :param observations:
    :param system_matrices:
    :param measurement_matrices:
    :param model_error_covariance_matrix:
    :param observation_error_covariance:
    :param initial_x0:
    :param initial_p0:
    :param control_vectors:
    :return:
    """
    assert isinstance(observations, np.ndarray)
    assert isinstance(system_matrices, np.ndarray)
    assert isinstance(measurement_matrices, np.ndarray)
    assert isinstance(model_error_covariance_matrix, np.ndarray)
    assert isinstance(observation_error_covariance, np.ndarray)
    assert isinstance(initial_x0, np.ndarray)
    assert isinstance(initial_p0, np.ndarray)

    # get dimension number from key variables
    n = observations.shape[0]
    assert n > 0, "observations can't be empty"
    assert (
        len(observations.shape) == 2
    ), f"observation must be (N, I) shape, where N number of obs and I number of channels. Got {observations.shape}"
    i = observations.shape[1]
    m = initial_x0.shape[0]

    # validating other dimensions
    assert system_matrices.shape == (n, m, m)
    assert measurement_matrices.shape == (n, i, m)
    assert model_error_covariance_matrix.shape == (m, m)
    assert initial_p0.shape == (m, m)
    assert initial_x0.shape == (m, 1)
    assert observation_error_covariance.shape == (i, i)

    if control_vectors is not None:
        assert isinstance(control_vectors, np.ndarray)
        assert control_vectors.shape == (n, m, 1)
