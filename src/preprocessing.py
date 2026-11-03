import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]

NUMERIC_FEATURES = [
    "distance",
    "weight",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create common freight prediction features."""

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear


    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    # Treat negative shipment weights as missing values.
    data.loc[data["weight"] < 0, "weight"] = float("nan")

    return data[feature_columns]


def build_preprocessor():
    """Build the numerical and categorical preprocessing pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor
