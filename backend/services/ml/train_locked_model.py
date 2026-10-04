from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
    / "train.csv"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
    / "validation.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "locked_xgboost_bundle.joblib"
)


TARGET_THRESHOLD = 0.002
CONFIDENCE_THRESHOLD = 0.55


NON_FEATURE_COLUMNS = [
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
    "future_return_1h",
    "target",
]


def create_target(series):
    return pd.Series(
        np.select(
            [
                series > TARGET_THRESHOLD,
                series < -TARGET_THRESHOLD,
            ],
            [
                "UP",
                "DOWN",
            ],
            default="NEUTRAL",
        ),
        index=series.index,
    )


def main():
    print("Loading training data...")

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    combined = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    feature_columns = [
        column
        for column in combined.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X = combined[
        feature_columns
    ].copy()

    y = create_target(
        combined["future_return_1h"]
    )

    encoder = LabelEncoder()

    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    y_encoded = encoder.transform(y)

    print(
        f"Training rows: {len(X)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        f"Classes: {list(encoder.classes_)}"
    )

    print(
        "\nTraining locked XGBoost model..."
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X,
        y_encoded,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = {
        "model": model,
        "encoder": encoder,
        "feature_columns": feature_columns,
        "target_threshold": TARGET_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_name": "locked_xgboost_paper_v1",
    }

    joblib.dump(
        bundle,
        MODEL_FILE,
    )

    print(
        "\nLocked model saved."
    )

    print(
        f"Model: {MODEL_FILE}"
    )

    print(
        f"Feature count: {len(feature_columns)}"
    )

    print(
        f"Target threshold: "
        f"±{TARGET_THRESHOLD * 100:.2f}%"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )


if __name__ == "__main__":
    main()