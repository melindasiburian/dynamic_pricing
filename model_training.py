"""Training utilities for the dynamic pricing baseline model."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "dataset_preprocessed.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "dynamic_pricing_model.joblib"


def load_dataset(path: Path = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the preprocessed dataset and split into features and target."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    dataset = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    target = dataset["base_price_idr"]
    feature_columns = [
        column
        for column in dataset.columns
        if column not in {"base_price_idr", "date"}
    ]
    features = dataset[feature_columns]
    return features, target


def build_pipeline() -> Pipeline:
    """Construct the preprocessing and modelling pipeline."""
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
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_and_save_model() -> Dict[str, float]:
    """Train the baseline model and persist the fitted pipeline."""
    features, target = load_dataset()
    train_size = int(len(features) * 0.8)
    if train_size == 0 or train_size == len(features):
        raise ValueError("Dataset is too small to create a temporal train/test split.")

    X_train = features.iloc[:train_size]
    X_test = features.iloc[train_size:]
    y_train = target.iloc[:train_size]
    y_test = target.iloc[train_size:]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    metrics: Dict[str, float] = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mse)),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return metrics


def main() -> None:
    metrics = train_and_save_model()
    print("Dynamic pricing model trained successfully. Evaluation metrics:")
    print(f"  R^2 Score : {metrics['r2']:.4f}")
    print(f"  MAE       : {metrics['mae']:.2f}")
    print(f"  RMSE      : {metrics['rmse']:.2f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
