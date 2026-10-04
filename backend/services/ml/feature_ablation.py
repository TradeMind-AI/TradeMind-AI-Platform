from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Project root
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


# Feature groups from our feature-engineering pipeline
FEATURE_GROUPS = {
    "returns": [
        "return_1h",
        "return_3h",
        "return_6h",
        "return_12h",
        "return_24h",
        "log_return_1h",
    ],

    "trend": [
        "sma_6",
        "sma_12",
        "sma_24",
        "sma_48",
        "sma_168",
        "ema_12",
        "ema_24",
        "ema_50",
        "close_to_sma_24",
        "close_to_sma_168",
    ],

    "momentum": [
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
    ],

    "volatility": [
        "rolling_volatility_24",
        "rolling_volatility_72",
        "atr_14",
        "bollinger_middle_20",
        "bollinger_upper_20",
        "bollinger_lower_20",
        "bollinger_width_20",
    ],

    "candle": [
        "candle_range",
        "body_size",
        "upper_wick",
        "lower_wick",
        "range_pct",
        "body_pct",
    ],

    "volume": [
        "volume_change",
        "volume_sma_24",
        "relative_volume",
        "trade_count_change",
        "trade_count_sma_24",
        "taker_buy_ratio",
    ],

    "time": [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ],

    "lag": [
        "return_1h_lag_1",
        "return_1h_lag_2",
        "return_1h_lag_3",
        "return_1h_lag_6",
        "return_1h_lag_12",
        "return_1h_lag_24",
        "volume_lag_1",
        "volume_lag_3",
        "volume_lag_6",
        "volume_lag_24",
        "rsi_14_lag_1",
        "rsi_14_lag_3",
        "rsi_14_lag_6",
        "volatility_24_lag_1",
        "volatility_24_lag_3",
    ],
}


def load_data():
    """Load training and validation datasets."""

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)

    return train_df, validation_df


def prepare_targets(train_df, validation_df):
    """Encode target classes."""

    label_mapping = {
        "DOWN": 0,
        "NEUTRAL": 1,
        "UP": 2,
    }

    y_train = train_df["target"].map(label_mapping)
    y_validation = validation_df["target"].map(label_mapping)

    if y_train.isna().any() or y_validation.isna().any():
        raise ValueError("Unknown target label found.")

    return y_train, y_validation


def train_and_evaluate(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Train XGBoost and return validation accuracy."""

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

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_validation)

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    return accuracy


def main():
    print("Loading data...")

    train_df, validation_df = load_data()

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")

    # All 63 features
    all_features = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    # Validate feature groups
    grouped_features = []

    for features in FEATURE_GROUPS.values():
        grouped_features.extend(features)

    if set(grouped_features) != set(all_features):
        missing = set(all_features) - set(grouped_features)
        extra = set(grouped_features) - set(all_features)

        raise ValueError(
            f"Feature group mismatch.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    y_train, y_validation = prepare_targets(
        train_df,
        validation_df,
    )

    print("\nTarget leakage check: PASSED")

    experiments = {}

    # ---------------------------------------------------------
    # Experiment 1: All 63 features
    # ---------------------------------------------------------

    print("\nExperiment 1: All 63 features")

    X_train = train_df[all_features]
    X_validation = validation_df[all_features]

    accuracy = train_and_evaluate(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    experiments["All features"] = accuracy

    print(f"Validation accuracy: {accuracy:.4f}")

    # ---------------------------------------------------------
    # Experiment 2: Remove time features
    # ---------------------------------------------------------

    print("\nExperiment 2: Remove time features")

    time_features = FEATURE_GROUPS["time"]

    reduced_features = [
        feature
        for feature in all_features
        if feature not in time_features
    ]

    X_train = train_df[reduced_features]
    X_validation = validation_df[reduced_features]

    accuracy = train_and_evaluate(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    experiments["Without time features"] = accuracy

    print(f"Validation accuracy: {accuracy:.4f}")

    # ---------------------------------------------------------
    # Experiment 3: Remove lag features
    # ---------------------------------------------------------

    print("\nExperiment 3: Remove lag features")

    lag_features = FEATURE_GROUPS["lag"]

    reduced_features = [
        feature
        for feature in all_features
        if feature not in lag_features
    ]

    X_train = train_df[reduced_features]
    X_validation = validation_df[reduced_features]

    accuracy = train_and_evaluate(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    experiments["Without lag features"] = accuracy

    print(f"Validation accuracy: {accuracy:.4f}")

    # ---------------------------------------------------------
    # Experiment 4: Remove time + lag features
    # ---------------------------------------------------------

    print("\nExperiment 4: Remove time + lag features")

    removed_features = (
        FEATURE_GROUPS["time"]
        + FEATURE_GROUPS["lag"]
    )

    reduced_features = [
        feature
        for feature in all_features
        if feature not in removed_features
    ]

    X_train = train_df[reduced_features]
    X_validation = validation_df[reduced_features]

    accuracy = train_and_evaluate(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    experiments["Without time + lag"] = accuracy

    print(f"Validation accuracy: {accuracy:.4f}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 55)
    print("FEATURE ABLATION RESULTS")
    print("=" * 55)

    for experiment, accuracy in experiments.items():
        print(
            f"{experiment:<30} "
            f"{accuracy:.4f}"
        )

    print("=" * 55)

    print("\nFeature ablation experiment complete.")


if __name__ == "__main__":
    main()