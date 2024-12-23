import unittest
import numpy as np

from double_sampling_kalman.factor_model_scan.api import (
    double_kalman_filter_numpy_scan,
)
from tests.test_utility import get_simple_test_case


class TestSingleKalman(unittest.TestCase):
    def test_single_kalman_simple_linear(self):
        np.random.seed(0)
        n_obs = 30
        n_comp = 2
        std = 0.1
        error_std = 0.001

        observations, measurement_matrices, solution = get_simple_test_case(
            n_obs=n_obs,
            n_components=n_comp,
            return_std=std,
            error_std=error_std,
        )

        system_matrices = np.array([np.identity(n_comp)] * n_obs)
        model_error_covariance_matrix = np.array([[0.13, 0], [0, 0.13]])
        observation_error_covariance_matrix = np.array([[0.0001]])
        initial_x0 = np.array([0.3, 0.7]).reshape((2, 1))
        initial_p0 = np.array([[0.01, 0.01], [0.01, 0.01]])

        result = double_kalman_filter_numpy_scan(
            max_iter=50,
            log10std_tol=0.1,
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix,
            observation_error_covariance=observation_error_covariance_matrix,
            initial_x0=initial_x0,
            initial_p0=initial_p0,
        )
        breakpoint()
