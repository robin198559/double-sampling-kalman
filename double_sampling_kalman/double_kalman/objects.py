from typing import List, Dict
import pandas as pd

from attr.validators import instance_of, deep_iterable
from attrs import define, field
import numpy as np


@define
class DoubleKalmanOutput:
    dependent_names: List[str] = field(
        validator=deep_iterable(
            member_validator=instance_of(str), iterable_validator=instance_of(list)
        )
    )
    forward: np.ndarray = field(validator=instance_of(np.ndarray))
    backward: np.ndarray = field(validator=instance_of(np.ndarray))

    @property
    def last_estimate(self) -> Dict[str, float]:
        return {
            name: value
            for name, value in zip(self.dependent_names, self.estimated[-1, :])
        }

    @property
    def estimated(self) -> pd.DataFrame:
        x_combined = (self.backward + self.forward) / 2
        return pd.DataFrame(x_combined[:, :, 0], columns=self.dependent_names)
