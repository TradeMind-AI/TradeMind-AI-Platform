from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
)
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


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

TARGET_THRESHOLDS = [
    0.0020,  # 0.20%
    0.0030,  # 0.30%
    0.0040,  # 0.40%
    0.0050,  # 0.50%
]

CONFIDENCE_THRESHOLD = 0.55

# Estimated round-trip/trade cost used for comparison.
TRANSACTION_COST_PCT = 0.05


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


def create_target(
    future_return: pd.Series,
    threshold: float,
) -> pd.Series:
    conditions = [
        future_return > threshold,
        future_return < -threshold,
    ]

    choices = [
        "UP",
        "DOWN",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="NEUTRAL",
        ),
        index=future_return.index,
    )


def prepare_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X_train = train_df[
        feature_columns
    ].copy()

    X_validation = validation_df[
        feature_columns
    ].copy()

    return (
        X_train,
        X_validation,
        feature_columns,
    )


def evaluate_trading(
    validation_df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    encoder: LabelEncoder,
):
    result = validation_df[
        [
            "future_return_1h"
        ]
    ].copy()

    result["prediction"] = predictions

    class_names = list(
        encoder.classes_
    )

    class_index = {
        name: index
        for index, name in enumerate(
            class_names
        )
    }

    result["probability_up"] = (
        probabilities[
            :,
            class_index["UP"]
        ]
    )

    result["probability_down"] = (
        probabilities[
            :,
            class_index["DOWN"]
        ]
    )

    result["confidence"] = probabilities.max(
        axis=1
    )

    result["signal"] = "HOLD"

    buy_mask = (
        (result["prediction"] == "UP")
        & (
            result["confidence"]
            >= CONFIDENCE_THRESHOLD
        )
    )

    sell_mask = (
        (result["prediction"] == "DOWN")
        & (
            result["confidence"]
            >= CONFIDENCE_THRESHOLD
        )
    )

    result.loc[
        buy_mask,
        "signal"
    ] = "BUY"

    result.loc[
        sell_mask,
        "signal"
    ] = "SELL"

    trades = result[
        result["signal"].isin(
            ["BUY", "SELL"]
        )
    ].copy()

    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "gross_return": 0.0,
            "cost": 0.0,
            "net_return": 0.0,
        }

    trades["trade_return"] = 0.0

    trades.loc[
        trades["signal"] == "BUY",
        "trade_return"
    ] = trades.loc[
        trades["signal"] == "BUY",
        "future_return_1h"
    ]

    trades.loc[
        trades["signal"] == "SELL",
        "trade_return"
    ] = -trades.loc[
        trades["signal"] == "SELL",
        "future_return_1h"
    ]

    cost_decimal = (
        TRANSACTION_COST_PCT / 100
    )

    trades[
        "net_trade_return"
    ] = (
        trades["trade_return"]
        - cost_decimal
    )

    win_rate = (
        (
            trades[
                "net_trade_return"
            ] > 0
        ).mean()
        * 100
    )

    gross_return = (
        trades[
            "trade_return"
        ].sum()
        * 100
    )

    cost = (
        len(trades)
        * TRANSACTION_COST_PCT
    )

    net_return = (
        trades[
            "net_trade_return"
        ].sum()
        * 100
    )

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "gross_return": gross_return,
        "cost": cost,
        "net_return": net_return,
    }


def run_experiment(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    threshold: float,
):
    y_train = create_target(
        train_df["future_return_1h"],
        threshold,
    )

    y_validation = create_target(
        validation_df[
            "future_return_1h"
        ],
        threshold,
    )

    encoder = LabelEncoder()

    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    y_train_encoded = (
        encoder.transform(
            y_train
        )
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
        X_train,
        y_train_encoded,
    )

    prediction_encoded = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )

    predictions = encoder.inverse_transform(
        prediction_encoded
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    up_precision = precision_score(
        y_validation == "UP",
        predictions == "UP",
        zero_division=0,
    )

    down_precision = precision_score(
        y_validation == "DOWN",
        predictions == "DOWN",
        zero_division=0,
    )

    trading = evaluate_trading(
        validation_df,
        predictions,
        probabilities,
        encoder,
    )

    return {
        "threshold_pct": (
            threshold * 100
        ),
        "accuracy_pct": (
            accuracy * 100
        ),
        "macro_f1_pct": (
            macro_f1 * 100
        ),
        "up_precision_pct": (
            up_precision * 100
        ),
        "down_precision_pct": (
            down_precision * 100
        ),
        "trades": trading["trades"],
        "win_rate_pct": trading[
            "win_rate"
        ],
        "gross_return_pct": trading[
            "gross_return"
        ],
        "transaction_cost_pct": trading[
            "cost"
        ],
        "net_return_pct": trading[
            "net_return"
        ],
        "train_up": (
            y_train == "UP"
        ).sum(),
        "train_down": (
            y_train == "DOWN"
        ).sum(),
        "train_neutral": (
            y_train == "NEUTRAL"
        ).sum(),
    }


def main():
    print("Loading ML datasets...")

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Training rows: "
        f"{len(train_df)}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df)}"
    )

    (
        X_train,
        X_validation,
        feature_columns,
    ) = prepare_features(
        train_df,
        validation_df,
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    results = []

    for threshold in TARGET_THRESHOLDS:
        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Testing target threshold: "
            f"±{threshold * 100:.2f}%"
        )

        print(
            "=" * 80
        )

        result = run_experiment(
            train_df,
            validation_df,
            X_train,
            X_validation,
            threshold,
        )

        results.append(
            result
        )

        print(
            f"Accuracy: "
            f"{result['accuracy_pct']:.2f}%"
        )

        print(
            f"Macro F1: "
            f"{result['macro_f1_pct']:.2f}%"
        )

        print(
            f"UP precision: "
            f"{result['up_precision_pct']:.2f}%"
        )

        print(
            f"DOWN precision: "
            f"{result['down_precision_pct']:.2f}%"
        )

        print(
            f"Trades: "
            f"{result['trades']}"
        )

        print(
            f"Trade win rate: "
            f"{result['win_rate_pct']:.2f}%"
        )

        print(
            f"Net return: "
            f"{result['net_return_pct']:.2f}%"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by="net_return_pct",
            ascending=False,
        )
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "TARGET THRESHOLD COMPARISON"
    )

    print(
        "=" * 120
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold_pct":
                    lambda x: f"{x:.2f}%",
                "accuracy_pct":
                    lambda x: f"{x:.2f}%",
                "macro_f1_pct":
                    lambda x: f"{x:.2f}%",
                "up_precision_pct":
                    lambda x: f"{x:.2f}%",
                "down_precision_pct":
                    lambda x: f"{x:.2f}%",
                "win_rate_pct":
                    lambda x: f"{x:.2f}%",
                "gross_return_pct":
                    lambda x: f"{x:.2f}%",
                "transaction_cost_pct":
                    lambda x: f"{x:.2f}%",
                "net_return_pct":
                    lambda x: f"{x:.2f}%",
            },
        )
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()