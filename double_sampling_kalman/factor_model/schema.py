from pandera import DataFrameModel, Field
from pandera.typing import Series


class ObservationSchema(DataFrameModel):
    label: Series[str] = Field(nullable=False)
    index: Series[int] = Field(nullable=False)
    value: Series[float] = Field(nullable=False)

    class Config:
        strict = "filter"
