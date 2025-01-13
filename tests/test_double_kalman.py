import logging
import unittest
import numpy as np

from double_sampling_kalman.double_kalman.api import double_kalman_filter
from double_sampling_kalman.factor_model.api import (
    tune_filter,
    initialize_filter,
    tune_filter_control,
)
from double_sampling_kalman.factor_model.objects import ScanConfig, ScanConfigGD
from tests.test_utility import get_simple_solution_numpy, get_simple_test_case_df


class TestSingleKalman(unittest.TestCase):
    def test_single_kalman_simple_linear(self):
        logging.basicConfig(level=logging.INFO)
        np.random.seed(0)
        n_obs = 30
        n_comp = 2
        std = 0.1
        error_std = 0.001

        solution = get_simple_solution_numpy(
            n_obs=n_obs,
            n_components=n_comp,
        )
        data = get_simple_test_case_df(
            solution=solution,
            return_std=std,
            error_std=error_std,
        )
        scan_config = ScanConfig()
        filter_input = initialize_filter(
            data=data,
            response_name="y",
        )

        filter_input = tune_filter(
            scan_config=scan_config,
            filter_input=filter_input,
        )
        result = double_kalman_filter(filter_input)
        breakpoint()
        filter_input = tune_filter_control(
            scan_config=scan_config,
            filter_input=filter_input,
        )
        result = double_kalman_filter(filter_input)
        breakpoint()

    def test_single_kalman_simple_linear_with_control_bias(self):
        logging.basicConfig(level=logging.INFO)
        np.random.seed(0)
        n_obs = 30
        n_comp = 2
        std = 0.1
        error_std = 0.001

        solution = get_simple_solution_numpy(
            n_obs=n_obs,
            n_components=n_comp,
        )
        data = get_simple_test_case_df(
            solution=solution,
            return_std=std,
            error_std=error_std,
        )
        scan_config = ScanConfig()
        filter_input = initialize_filter(data=data, response_name="y", control_vector_adjustments={15: [0.05, -0.05]})

        filter_input = tune_filter(
            scan_config=scan_config,
            filter_input=filter_input,
        )
        result = double_kalman_filter(filter_input)
        breakpoint()
