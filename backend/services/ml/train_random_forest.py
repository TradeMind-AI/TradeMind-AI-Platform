from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
    / "validation.csv"
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


def load_data():
    """Load training and validation datasets."""

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)

    return train_df, validation_df


def prepare_features(train_df, validation_df):
    """Separate features (X) and target (y)."""

    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    # Make sure both datasets contain the same features
    validation_feature_columns = [
        column
        for column in validation_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    if feature_columns != validation_feature_columns:
        raise ValueError(
            "Training and validation feature columns do not match."
        )

    # Leakage checks
    if "target" in feature_columns:
        raise ValueError("Target leakage detected: 'target' is in X.")

    if "future_return_1h" in feature_columns:
        raise ValueError(
            "Target leakage detected: 'future_return_1h' is in X."
        )

    X_train = train_df[feature_columns]
    y_train = train_df["target"]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df["target"]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    )


def train_model(X_train, y_train):
    """Train the Random Forest classifier."""

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_validation, y_validation):
    """Evaluate the model on the validation dataset."""

    predictions = model.predict(X_validation)

    accuracy = accuracy_score(y_validation, predictions)

    print("\nValidation accuracy:")
    print(f"{accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            digits=4,
        )
    )

    print("Confusion matrix:")

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=["DOWN", "NEUTRAL", "UP"],
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual DOWN", "Actual NEUTRAL", "Actual UP"],
        columns=["Pred DOWN", "Pred NEUTRAL", "Pred UP"],
    )

    print(matrix_df)


def main():
    print("Loading ML data...")

    train_df, validation_df = load_data()

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    ) = prepare_features(
        train_df,
        validation_df,
    )

    print(f"Number of features: {len(feature_columns)}")

    print("\nTarget leakage check: PASSED")

    print("\nTraining Random Forest model...")

    model = train_model(
        X_train,
        y_train,
    )

    print("Model training complete.")

    evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    print("\nRandom Forest experiment complete.")


if __name__ == "__main__":
    main()