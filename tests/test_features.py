from ml_service.features import to_dataframe
from ml_service.schemas import PredictRequest


def test_to_dataframe_all_columns():
    req = PredictRequest(
        age=39,
        workclass="Private",
        fnlwgt=77516,
        education="Bachelors",
        education_num=13,
        marital_status="Never-married",
        occupation="Adm-clerical",
        relationship="Not-in-family",
        race="White",
        sex="Male",
        capital_gain=2174,
        capital_loss=0,
        hours_per_week=40,
        native_country="United-States",
    )

    df = to_dataframe(req)

    assert df.shape == (1, 14)
    assert "education.num" in df.columns
    assert "marital.status" in df.columns
    assert "capital.gain" in df.columns
    assert df.loc[0, "age"] == 39
    assert df.loc[0, "sex"] == "Male"


def test_to_dataframe_only_needed_columns():
    req = PredictRequest(
        age=39,
        sex="Male",
        race="White",
    )

    df = to_dataframe(req, needed_columns=["age", "sex", "race"])

    assert list(df.columns) == ["age", "sex", "race"]
    assert df.shape == (1, 3)
    assert df.loc[0, "age"] == 39
    assert df.loc[0, "sex"] == "Male"
    assert df.loc[0, "race"] == "White"