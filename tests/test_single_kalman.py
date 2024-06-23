import unittest

import numpy as np

from double_sampling_kalman.single_kalman.api import discrete_kalman_filter_numpy_runner
from tests.test_utility import get_simple_test_case


class TestSingleKalman(unittest.TestCase):
    def test_single_kalman_simple_linear(self):
        np.random.seed(0)
        n_obs = 30
        n_comp = 2
        std = 0.02
        error_std = 0.0001
        expected = np.array(
            [
                0.2965542,
                0.29977733,
                0.31230881,
                0.32148675,
                0.41269284,
                0.42282208,
                0.41978839,
                0.42649139,
                0.44420546,
                0.44135612,
                0.44599717,
                0.44724558,
                0.4464233,
                0.44815663,
                0.44807461,
                0.47044644,
                0.4711174,
                0.49909483,
                0.49611237,
                0.49171277,
                0.49230087,
                0.49696695,
                0.4944412,
                0.49429621,
                0.49471085,
                0.49421624,
                0.49825375,
                0.50093443,
                0.50148784,
                0.50077752,
            ]
        )

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

        result = discrete_kalman_filter_numpy_runner(
            system_matrices=system_matrices,
            measurement_matrices=measurement_matrices,
            observations=observations,
            model_error_covariance_matrix=model_error_covariance_matrix,
            observation_error_covariance=observation_error_covariance_matrix,
            initial_x0=initial_x0,
            initial_p0=initial_p0,
        )

        assert np.sum(result[:, 0, 0] - expected) < 1e-8
