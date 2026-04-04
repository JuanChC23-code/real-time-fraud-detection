import pandera as pa
from pandera import Column, DataFrameSchema


transaction_schema = DataFrameSchema({
    "request_id": Column(str),
    "timestamp": Column(str),
    "label": Column(int),
})


prediction_schema = DataFrameSchema({
    "request_id": Column(str),
    "timestamp": Column(str),
    "model_version": Column(str),
    "fraud_probability": Column(float),
    "prediction": Column(int),
    "label": Column(int),
})