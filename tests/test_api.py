import pytest
from fastapi.testclient import TestClient

from ml_service.app import create_app, MODEL
from ml_service.model import ModelData


class DummyModel:
    feature_names_in_ = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education.num",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
        "native.country",
    ]

    def predict_proba(self, df):
        return [[0.2, 0.8]]


@pytest.fixture
def client(monkeypatch):
    dummy_model = DummyModel()
    old_data = MODEL.get()

    def fake_configure_mlflow():
        return None

    def fake_model_set(run_id: str):
        MODEL.data = ModelData(model=dummy_model, run_id=run_id)

    monkeypatch.setattr("ml_service.app.configure_mlflow", fake_configure_mlflow)
    monkeypatch.setattr("ml_service.app.MODEL.set", fake_model_set)
    monkeypatch.setenv("DEFAULT_RUN_ID", "test_run_id")

    app = create_app()

    with TestClient(app) as test_client:
        yield test_client

    MODEL.data = old_data


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["run_id"] == "test_run_id"


def test_predict_ok(client):
    payload = {
        "age": 39,
        "workclass": "Private",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education.num": 13,
        "marital.status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital.gain": 2174,
        "capital.loss": 0,
        "hours.per.week": 40,
        "native.country": "United-States",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] == 1
    assert 0 <= data["probability"] <= 1


def test_predict_invalid_type(client):
    payload = {
        "age": "abc",
        "workclass": "Private",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education.num": 13,
        "marital.status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital.gain": 2174,
        "capital.loss": 0,
        "hours.per.week": 40,
        "native.country": "United-States",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_empty_json(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_update_model_empty_run_id(client):
    response = client.post("/updateModel", json={"run_id": "   "})
    assert response.status_code == 422
    assert response.json()["detail"] == "run_id must not be empty"