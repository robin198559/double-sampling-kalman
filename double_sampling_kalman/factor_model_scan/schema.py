from pandera import DataFrameSchema, Column

ObservationSchema = DataFrameSchema(
    {
        "label": Column(str, nullable=False),
        "index": Column(int, nullable=False),
        "value": Column(float, nullable=False),
    },
    strict="filter",
)