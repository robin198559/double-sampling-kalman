import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from double_sampling_kalman.factor_model.schema import ObservationSchema


def get_simple_test_case_numpy(
    solution: np.ndarray, return_std: float, error_std: float
):
    n_obs, n_components, _ = solution.shape
    measurement_matrices = return_std * (
        np.random.rand(n_obs * n_components).reshape(n_obs, 1, n_components) * 2 - 1
    )

    observations = np.matmul(measurement_matrices, solution).reshape(
        n_obs
    ) + error_std * (np.random.rand(n_obs) * 2 - 1)
    observations = observations.reshape((n_obs, 1))
    return observations, measurement_matrices


def get_simple_solution_numpy(n_obs: int, n_components: int):
    solution = np.zeros((n_obs, n_components, 1))
    solution[: int(n_obs / 2), 0, 0] = 0.45
    solution[int(n_obs / 2) :, 0, 0] = 0.5
    solution[:, 1, 0] = 1 - solution[:, 0, 0]
    return solution


def get_simple_test_case_df(
    solution: np.ndarray, return_std: float, error_std: float
) -> DataFrame[ObservationSchema]:
    n_obs, n_components, _ = solution.shape
    measurement_matrices = return_std * (np.random.rand(n_obs * n_components) * 2 - 1)

    observations = np.matmul(
        measurement_matrices.reshape(n_obs, 1, n_components), solution
    ).reshape(n_obs) + error_std * (np.random.rand(n_obs) * 2 - 1)
    observations = observations.reshape((n_obs, 1))

    data_df = pd.DataFrame(
        np.concatenate(
            (observations, measurement_matrices.reshape(n_obs, n_components)), axis=1
        ),
        columns=["y"] + [f"x{i+1}" for i in range(n_components)],
    )
    data_df.index.name = "index"
    data_df.columns.name = "label"
    data = data_df.unstack().reset_index().rename(columns={0: "value"})
    return data


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
