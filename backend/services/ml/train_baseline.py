from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
)


# Columns that must not be used as model features
NON_FEATURE_COLUMNS = {
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
}


def load_split(filename: str) -> pd.DataFrame:
    """Load one chronological dataset split."""

    return pd.read_csv(
        SPLIT_DIR / filename
    )


def separate_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target."""

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X = df[feature_columns].copy()
    y = df["target"].copy()

    return X, y


def main() -> None:
    """Train and evaluate the baseline Logistic Regression model."""

    print("Loading ML splits...")

    train_df = load_split("train.csv")
    validation_df = load_split("validation.csv")

    X_train, y_train = separate_features_and_target(
        train_df
    )

    X_validation, y_validation = (
        separate_features_and_target(validation_df)
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_validation)}"
    )

    print(
        f"Features: {X_train.shape[1]}"
    )

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    print("\nFitting scaler on training data only...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    print(
        "Training-only scaling: OK"
    )

    # --------------------------------------------------
    # Baseline model
    # --------------------------------------------------

    print("\nTraining Logistic Regression baseline...")

    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    print(
        "Model training complete."
    )

    # --------------------------------------------------
    # Validation prediction
    # --------------------------------------------------

    print("\nGenerating validation predictions...")

    y_validation_pred = model.predict(
        X_validation_scaled
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_validation,
        y_validation_pred,
    )

    print(
        f"\nValidation accuracy: "
        f"{accuracy:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_validation,
            y_validation_pred,
            labels=[
                "DOWN",
                "NEUTRAL",
                "UP",
            ],
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    confusion = confusion_matrix(
        y_validation,
        y_validation_pred,
        labels=[
            "DOWN",
            "NEUTRAL",
            "UP",
        ],
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=[
            "Actual DOWN",
            "Actual NEUTRAL",
            "Actual UP",
        ],
        columns=[
            "Predicted DOWN",
            "Predicted NEUTRAL",
            "Predicted UP",
        ],
    )

    print(
        confusion_df.to_string()
    )

    print(
        "\nBaseline model evaluation complete."
    )


if __name__ == "__main__":
    main()