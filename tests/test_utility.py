import numpy as np


def get_simple_test_case(
    n_obs: int, n_components: int, return_std: float, error_std: float
):
    measurement_matrices = return_std * (
        np.random.rand(n_obs * n_components).reshape(n_obs, 1, n_components) * 2 - 1
    )
    solution = np.zeros((n_obs, n_components, 1))
    solution[: int(n_obs / 2), 0, 0] = 0.45
    solution[int(n_obs / 2) :, 0, 0] = 0.5
    solution[:, 1, 0] = 1 - solution[:, 0, 0]

    observations = np.matmul(measurement_matrices, solution).reshape(
        n_obs
    ) + error_std * (np.random.rand(n_obs) * 2 - 1)
    observations = observations.reshape((n_obs, 1))
    return observations, measurement_matrices, solution


def get_complex_test_case(
    n_obs: int, n_components: int, return_std: float, error_std: float
):
    measurement_matrices = return_std * (
        np.random.rand(n_obs * n_components).reshape(n_obs, 1, n_components) * 2 - 1
    )
    solution = np.ones((n_obs, n_components, 1))
    solution[: int(n_obs * 1 / 4), 0, 0] = (
        np.cumsum(solution[: int(n_obs * 1 / 4), 0, 0] * 0.4 / n_obs) + 0.1
    )
    solution[int(n_obs * 1 / 4) : int(n_obs * 2 / 4), 0, 0] = 0.2
    solution[int(n_obs * 2 / 4) : int(n_obs * 3 / 4), 0, 0] = (
        np.cumsum(solution[int(n_obs * 2 / 4) : int(n_obs * 3 / 4), 0, 0] * 0.4 / n_obs)
        + 0.2
    )
    solution[int(n_obs * 3 / 4) :, 0, 0] = 0.7

    solution[:, 1, 0] = 1 - solution[:, 0, 0]

    observations = np.matmul(measurement_matrices, solution).reshape(
        n_obs
    ) + error_std * (np.random.rand(n_obs) * 2 - 1)
    observations = observations.reshape((n_obs, 1))
    return observations, measurement_matrices, solution
