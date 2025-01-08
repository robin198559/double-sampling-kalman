from typing import Optional

import attrs
import numpy as np
from attrs.validators import instance_of, ge, gt


@attrs.define
class FactorModel:
    observations: np.ndarray = attrs.field(validator=instance_of(np.ndarray))
    measurement_matrices: np.ndarray = attrs.field(validator=instance_of(np.ndarray))

    @staticmethod
    def from_dict(data) -> "FactorModel":
        return FactorModel(**data)

    def to_dict(self):
        return attrs.asdict(self)


@attrs.define
class ScanConfig:
    max_iter: Optional[int] = attrs.field(validator=[instance_of(int), ge(3)], default=50)
    window_size_iter: Optional[int] = attrs.field(validator=[instance_of(int), ge(2)], default=5)
    convergence_log10_tol: Optional[float] = attrs.field(
        validator=instance_of(float),
        default=-2.0,
    )
    filter_tuning_multiplier_granularity: Optional[int] = attrs.field(validator=[instance_of(int), gt(0)], default=200)
    initial_filter_tuning_multiplier_log10_width: Optional[float] = attrs.field(
        validator=[instance_of(float), gt(0)], default=10.0
    )
    control_index_offset_width: Optional[int] = attrs.field(validator=[instance_of(int), gt(0)], default=3)
    control_tuning_multiplier_log10_width: Optional[float] = attrs.field(
        validator=[instance_of(float), gt(0)], default=0.5
    )
    control_tuning_multiplier_granularity: Optional[int] = attrs.field(validator=[instance_of(int), gt(0)], default=10)
    control_log10_tol: Optional[float] = attrs.field(
        validator=[instance_of(float), ge(0)],
        default=0.02,
    )

    def __attrs_post_init__(self):
        assert (
            self.max_iter > self.window_size_iter
        ), f"max_iter {self.max_iter} must be larger than window_size_iter {self.window_size_iter}"
