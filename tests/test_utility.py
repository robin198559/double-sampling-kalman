import numpy as np


def get_simple_test_case(
    n_obs: int, n_components: int, return_std: float, error_std: float
):
    measurement_matrices = return_std * (
        np.random.rand(n_obs * n_components).reshape(n_obs, 1, n_components) * 2 - 1
    )
    solution = np.zeros((n_obs, n_components, 1))
    solution[: int(n_obs / 2), 0, 0] = 0.45
    solution[: int(n_obs / 2), 1, 0] = 0.55
    solution[int(n_obs / 2) :, 0, 0] = 0.5
    solution[int(n_obs / 2) :, 1, 0] = 0.5

    observations = np.matmul(measurement_matrices, solution).reshape(
        n_obs
    ) + error_std * (np.random.rand(n_obs) * 2 - 1)
    observations = observations.reshape((n_obs, 1))
    return observations, measurement_matrices, solution
