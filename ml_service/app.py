from typing import Any
from contextlib import asynccontextmanager
from time import perf_counter

import asyncio
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from ml_service import config
from ml_service.features import to_dataframe
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)
from ml_service.evidently_monitoring import DRIFT_MONITOR, evidently_worker


MODEL = Model()

LATENCY_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

REQUEST_COUNT = Counter(
    "ml_service_requests_total",
    "Total number of HTTP requests",
    ["endpoint", "method", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "ml_service_request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint", "method"],
    buckets=LATENCY_BUCKETS,
)

PREDICT_REQUESTS = Counter(
    "ml_service_predict_requests_total",
    "Total number of predict requests",
    ["status"],
)

PREDICT_PREPROCESSING_TIME = Histogram(
    "ml_service_preprocessing_seconds",
    "Time spent on preprocessing input data",
    buckets=LATENCY_BUCKETS,
)

MODEL_INFERENCE_TIME = Histogram(
    "ml_service_inference_seconds",
    "Time spent on model inference",
    buckets=LATENCY_BUCKETS,
)

MODEL_PREDICTION_COUNT = Counter(
    "ml_service_predictions_total",
    "Total number of predictions by class",
    ["prediction"],
)

MODEL_PROBABILITY = Histogram(
    "ml_service_prediction_probability",
    "Distribution of predicted probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL_UPDATES = Counter(
    "ml_service_model_updates_total",
    "Total number of model update attempts",
    ["status"],
)

CURRENT_MODEL_INFO = Gauge(
    "ml_service_model_info",
    "Current loaded model information",
    ["run_id", "model_type"],
)

CURRENT_MODEL_FEATURE_COUNT = Gauge(
    "ml_service_model_feature_count",
    "Number of features required by current model",
)

CPU_USAGE = Gauge(
    "ml_service_cpu_percent",
    "Current CPU usage percent",
)

MEMORY_USAGE = Gauge(
    "ml_service_memory_percent",
    "Current memory usage percent",
)


def update_resource_metrics() -> None:
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)


def set_model_metrics() -> None:
    model_state = MODEL.get()
    model = model_state.model
    run_id = model_state.run_id or "none"
    model_type = type(model).__name__ if model is not None else "none"

    CURRENT_MODEL_INFO.labels(run_id=run_id, model_type=model_type).set(1)

    feature_count = len(MODEL.features) if model is not None else 0
    CURRENT_MODEL_FEATURE_COUNT.set(feature_count)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_mlflow()
    run_id = config.default_run_id()
    MODEL.set(run_id=run_id)
    set_model_metrics()
    update_resource_metrics()

    numeric_defaults = {
        "age": 39,
        "fnlwgt": 77516,
        "education.num": 13,
        "capital.gain": 2174,
        "capital.loss": 0,
        "hours.per.week": 40,
    }

    categorical_defaults = {
        "workclass": "Private",
        "education": "Bachelors",
        "marital.status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "native.country": "United-States",
    }

    reference_rows = []
    for _ in range(30):
        row = {}
        for col in MODEL.features:
            if col in numeric_defaults:
                row[col] = numeric_defaults[col]
            else:
                row[col] = categorical_defaults.get(col, "unknown")
        reference_rows.append(row)

    reference_df = pd.DataFrame(reference_rows)
    reference_df["prediction"] = 0
    reference_df["probability"] = 0.5
    DRIFT_MONITOR.set_reference_data(reference_df)

    asyncio.ensure_future(evidently_worker(period_seconds=60, min_records=10))

    yield


def create_app() -> FastAPI:
    app = FastAPI(title="MLflow FastAPI service", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        started = perf_counter()
        status_code = 200
        try:
            model_state = MODEL.get()
            run_id = model_state.run_id
            return {"status": "ok", "run_id": run_id}
        finally:
            REQUEST_COUNT.labels(
                endpoint="/health",
                method="GET",
                status_code=str(status_code),
            ).inc()
            REQUEST_LATENCY.labels(endpoint="/health", method="GET").observe(
                perf_counter() - started
            )
            update_resource_metrics()

    @app.get("/metrics")
    def metrics() -> Response:
        update_resource_metrics()
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        request_started = perf_counter()
        status_code = 200

        try:
            model_state = MODEL.get()
            model = model_state.model

            if model is None:
                status_code = 503
                PREDICT_REQUESTS.labels(status="model_not_loaded").inc()
                raise HTTPException(status_code=503, detail="Model is not loaded yet")

            preprocess_started = perf_counter()
            try:
                df = to_dataframe(request, needed_columns=MODEL.features)
            except Exception as e:
                status_code = 422
                PREDICT_REQUESTS.labels(status="preprocessing_error").inc()
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to prepare input data: {e}",
                )
            finally:
                PREDICT_PREPROCESSING_TIME.observe(perf_counter() - preprocess_started)

            if df.empty:
                status_code = 422
                PREDICT_REQUESTS.labels(status="empty_input").inc()
                raise HTTPException(status_code=422, detail="Input data is empty")

            missing_columns = [col for col in MODEL.features if col not in df.columns]
            if missing_columns:
                status_code = 422
                PREDICT_REQUESTS.labels(status="missing_columns").inc()
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing required features: {missing_columns}",
                )

            null_columns = [col for col in MODEL.features if df[col].isnull().any()]
            if null_columns:
                status_code = 422
                PREDICT_REQUESTS.labels(status="null_values").inc()
                raise HTTPException(
                    status_code=422,
                    detail=f"Required features contain null values: {null_columns}",
                )

            inference_started = perf_counter()
            try:
                probability = float(model.predict_proba(df)[0][1])
                prediction = int(probability >= 0.5)
            except ValueError as e:
                status_code = 422
                PREDICT_REQUESTS.labels(status="invalid_feature_values").inc()
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid feature values for model: {e}",
                )
            except Exception as e:
                status_code = 500
                PREDICT_REQUESTS.labels(status="inference_error").inc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Model inference failed: {e}",
                )
            finally:
                MODEL_INFERENCE_TIME.observe(perf_counter() - inference_started)

            MODEL_PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
            MODEL_PROBABILITY.observe(probability)
            PREDICT_REQUESTS.labels(status="success").inc()

            DRIFT_MONITOR.add_record(df, prediction, probability)
            return PredictResponse(prediction=prediction, probability=probability)

        except HTTPException as e:
            status_code = e.status_code
            raise
        finally:
            REQUEST_COUNT.labels(
                endpoint="/predict",
                method="POST",
                status_code=str(status_code),
            ).inc()
            REQUEST_LATENCY.labels(endpoint="/predict", method="POST").observe(
                perf_counter() - request_started
            )
            update_resource_metrics()

    @app.post("/updateModel", response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        request_started = perf_counter()
        status_code = 200

        try:
            run_id = req.run_id.strip()

            if not run_id:
                status_code = 422
                MODEL_UPDATES.labels(status="empty_run_id").inc()
                raise HTTPException(status_code=422, detail="run_id must not be empty")

            try:
                MODEL.set(run_id=run_id)
                set_model_metrics()
                MODEL_UPDATES.labels(status="success").inc()
            except Exception as e:
                status_code = 404
                MODEL_UPDATES.labels(status="failed").inc()
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to load model for run_id={run_id}: {e}",
                )

            return UpdateModelResponse(run_id=run_id)

        except HTTPException as e:
            status_code = e.status_code
            raise
        finally:
            REQUEST_COUNT.labels(
                endpoint="/updateModel",
                method="POST",
                status_code=str(status_code),
            ).inc()
            REQUEST_LATENCY.labels(endpoint="/updateModel", method="POST").observe(
                perf_counter() - request_started
            )
            update_resource_metrics()

    return app


app = create_app()