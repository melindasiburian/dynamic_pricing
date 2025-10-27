"""Training pipeline for dynamic pricing model using preprocessed dataset.

This script loads the `dataset_preprocessed.csv` file, performs the necessary
transformations (categorical encoding) and normalization of numeric features,
trains a machine learning model, evaluates it, and stores the trained pipeline
for later use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "dataset_preprocessed.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "dynamic_pricing_model.joblib"


def load_dataset(path: Path = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the preprocessed dataset and split into features and target."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    dataset = pd.read_csv(path)

    feature_columns = [
        column
        for column in dataset.columns
        if column not in {"base_price_idr", "date"}
    ]

    features = dataset[feature_columns]
    target = dataset["base_price_idr"]
    return features, target


def build_pipeline() -> Pipeline:
    """Construct the training pipeline with preprocessing and model."""
    categorical_features = ["category"]
    numeric_features = [
        "competitive_price",
        "competitor_count",
        "category_quantity",
        "total_visitors",
        "monthly_event_days",
        "temperature_celsius",
        "prcp_mm",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_model() -> dict:
    """Run the training pipeline and return evaluation metrics."""
    features, target = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": float(np.sqrt(mse)),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return metrics


def main() -> None:
    metrics = train_model()

    print("Dynamic pricing model trained successfully. Evaluation metrics:")
    print(f"  R^2 Score : {metrics['r2']:.4f}")
    print(f"  MAE       : {metrics['mae']:.2f}")
    print(f"  RMSE      : {metrics['rmse']:.2f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
