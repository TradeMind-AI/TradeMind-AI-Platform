from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
)


# Columns that must never be used as model features
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

    file_path = SPLIT_DIR / filename

    df = pd.read_csv(file_path)

    return df


def separate_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the 63 model features from the target."""

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X = df[feature_columns].copy()
    y = df["target"].copy()

    return X, y


def main() -> None:
    """Prepare train, validation, and test data."""

    print("Loading ML splits...")

    train_df = load_split("train.csv")
    validation_df = load_split("validation.csv")
    test_df = load_split("test.csv")

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")

    # Separate features and targets
    X_train, y_train = separate_features_and_target(
        train_df
    )

    X_validation, y_validation = (
        separate_features_and_target(validation_df)
    )

    X_test, y_test = separate_features_and_target(
        test_df
    )

    print(
        f"\nNumber of features: {X_train.shape[1]}"
    )

    # Verify all splits contain the same features
    if list(X_train.columns) != list(
        X_validation.columns
    ):
        raise ValueError(
            "Train and validation feature columns do not match."
        )

    if list(X_train.columns) != list(
        X_test.columns
    ):
        raise ValueError(
            "Train and test feature columns do not match."
        )

    print("Feature columns match across all splits.")

    # Verify target is not accidentally included
    forbidden_columns = {
        "future_return_1h",
        "target",
    }

    leaked_columns = (
        forbidden_columns
        & set(X_train.columns)
    )

    if leaked_columns:
        raise ValueError(
            f"Target leakage detected: {leaked_columns}"
        )

    print(
        "Target leakage check: PASSED"
    )

    # --------------------------------------------------
    # Standardization
    # --------------------------------------------------
    #
    # IMPORTANT:
    # The scaler is FIT ONLY on training data.
    #
    # Validation and test are transformed using
    # the statistics learned from training data.
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print(
        "\nPreprocessing:"
    )

    print(
        "Scaler fitted using training data only."
    )

    print(
        "Validation transformed using training scaler."
    )

    print(
        "Test transformed using training scaler."
    )

    # Convert back to DataFrames so feature names
    # remain available.
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index,
    )

    X_validation_scaled = pd.DataFrame(
        X_validation_scaled,
        columns=X_validation.columns,
        index=X_validation.index,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index,
    )

    # --------------------------------------------------
    # Final checks
    # --------------------------------------------------

    print("\nFinal dataset shapes:")

    print(
        f"X_train:       {X_train_scaled.shape}"
    )

    print(
        f"X_validation:  {X_validation_scaled.shape}"
    )

    print(
        f"X_test:        {X_test_scaled.shape}"
    )

    print(
        f"y_train:       {y_train.shape}"
    )

    print(
        f"y_validation:  {y_validation.shape}"
    )

    print(
        f"y_test:        {y_test.shape}"
    )

    print("\nTarget classes:")

    print(
        sorted(y_train.unique())
    )

    print("\nML data preparation complete.")


if __name__ == "__main__":
    main()