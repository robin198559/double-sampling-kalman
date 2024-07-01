from attr.validators import instance_of
from attrs import define, field
import numpy as np

from double_sampling_kalman.single_kalman.objects import SingleKalmanOutput


@define
class DoubleKalmanOutput:
    forward: SingleKalmanOutput = field(validator=instance_of(SingleKalmanOutput))
    backward: SingleKalmanOutput = field(validator=instance_of(SingleKalmanOutput))

    @property
    def last_estimate(self) -> np.ndarray:
        return self.x[-1, :, :]

    @property
    def x(self) -> np.ndarray:
        x_combined = (self.backward.x[::-1, :, :] + self.forward.x) / 2
        return x_combined

    @property
    def model_error_cov(self) -> np.ndarray:
        model_error = (self.backward.x[::-1, :, :] - self.forward.x) / 2
        model_error = model_error.reshape(model_error.shape[:2])
        return np.cov(model_error, rowvar=False)

    def calculate_observation_error_covariance(
        self,
        measurement_matrices: np.ndarray,
        observations: np.ndarray,
    ) -> np.ndarray:
        i = observations.shape[1]
        error = observations - np.matmul(measurement_matrices, self.x).reshape(
            observations.shape
        )
        new_observation_error_covariance = np.cov(error, rowvar=False).reshape((i, i))
        return new_observation_error_covariance
