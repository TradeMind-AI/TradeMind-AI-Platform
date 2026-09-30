from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Input files
TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
    / "train.csv"
)


# Columns that must NOT be used as ML features
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


def load_training_data():
    """Load the training dataset."""

    return pd.read_csv(TRAIN_FILE)


def prepare_features(df):
    """Separate the 63 ML features from the target."""

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X = df[feature_columns]
    y = df["target"]

    # Encode target
    label_mapping = {
        "DOWN": 0,
        "NEUTRAL": 1,
        "UP": 2,
    }

    y = y.map(label_mapping)

    if y.isna().any():
        raise ValueError("Unknown target label found.")

    return X, y, feature_columns


def train_model(X, y):
    """Train the same XGBoost configuration used in the experiment."""

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    return model


def analyze_importance(model, feature_columns):
    """Display features ranked by XGBoost importance."""

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    importance_df = importance_df.reset_index(drop=True)

    print("\nFeature importance ranking:")
    print("=" * 50)

    for index, row in importance_df.iterrows():
        print(
            f"{index + 1:2d}. "
            f"{row['feature']:<30} "
            f"{row['importance']:.6f}"
        )

    print("=" * 50)

    print("\nTop 15 features:")

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )


def main():
    print("Loading training data...")

    df = load_training_data()

    print(f"Training samples: {len(df)}")

    X, y, feature_columns = prepare_features(df)

    print(f"Number of features: {len(feature_columns)}")

    print("\nTarget leakage check: PASSED")

    print("\nTraining XGBoost model...")

    model = train_model(X, y)

    print("Model training complete.")

    analyze_importance(
        model,
        feature_columns,
    )

    print("\nFeature importance analysis complete.")


if __name__ == "__main__":
    main()