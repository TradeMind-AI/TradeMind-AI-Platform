from pathlib import Path

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

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
    / "test.csv"
)


TARGET_THRESHOLD = 0.002
CONFIDENCE_THRESHOLD = 0.55
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
    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_validation_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    feature_columns = [
        column
        for column in train_validation_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X_train = train_validation_df[
        feature_columns
    ]

    X_test = test_df[
        feature_columns
    ]

    y_train = create_target(
        train_validation_df["future_return_1h"]
    )

    y_test = create_target(
        test_df["future_return_1h"]
    )

    encoder = LabelEncoder()
    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    y_train_encoded = encoder.transform(
        y_train
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Test rows: {len(X_test)}"
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    print("\nTraining locked XGBoost model...")

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

    probabilities = model.predict_proba(
        X_test
    )

    prediction_encoded = model.predict(
        X_test
    )

    predictions = encoder.inverse_transform(
        prediction_encoded
    )

    result = test_df[
        [
            "open_time",
            "close",
            "future_return_1h",
        ]
    ].copy()

    result["actual_target"] = y_test
    result["prediction"] = predictions

    class_index = {
        class_name: index
        for index, class_name in enumerate(
            encoder.classes_
        )
    }

    result["probability_down"] = (
        probabilities[
            :,
            class_index["DOWN"]
        ]
    )

    result["probability_neutral"] = (
        probabilities[
            :,
            class_index["NEUTRAL"]
        ]
    )

    result["probability_up"] = (
        probabilities[
            :,
            class_index["UP"]
        ]
    )

    result["confidence"] = probabilities.max(
        axis=1
    )

    result["signal"] = "HOLD"

    result.loc[
        (
            (result["prediction"] == "UP")
            & (
                result["confidence"]
                >= CONFIDENCE_THRESHOLD
            )
        ),
        "signal",
    ] = "BUY"

    result.loc[
        (
            (result["prediction"] == "DOWN")
            & (
                result["confidence"]
                >= CONFIDENCE_THRESHOLD
            )
        ),
        "signal",
    ] = "SELL"

    trades = result[
        result["signal"].isin(
            [
                "BUY",
                "SELL",
            ]
        )
    ].copy()

    trades["trade_return"] = 0.0

    buy_mask = (
        trades["signal"] == "BUY"
    )

    sell_mask = (
        trades["signal"] == "SELL"
    )

    trades.loc[
        buy_mask,
        "trade_return",
    ] = trades.loc[
        buy_mask,
        "future_return_1h",
    ]

    trades.loc[
        sell_mask,
        "trade_return",
    ] = -trades.loc[
        sell_mask,
        "future_return_1h",
    ]

    cost_decimal = (
        TRANSACTION_COST_PCT / 100
    )

    trades["net_trade_return"] = (
        trades["trade_return"]
        - cost_decimal
    )

    total_trades = len(trades)

    winning_trades = (
        trades["net_trade_return"] > 0
    ).sum()

    win_rate = (
        winning_trades
        / total_trades
        * 100
        if total_trades > 0
        else 0
    )

    gross_return = (
        trades["trade_return"].sum()
        * 100
    )

    transaction_cost = (
        total_trades
        * TRANSACTION_COST_PCT
    )

    net_return = (
        trades[
            "net_trade_return"
        ].sum()
        * 100
    )

    print("\n" + "=" * 80)
    print("LOCKED XGBOOST FINAL TEST")
    print("=" * 80)

    print(
        f"Target threshold:       "
        f"±{TARGET_THRESHOLD * 100:.2f}%"
    )

    print(
        f"Confidence threshold:   "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"Total test rows:        "
        f"{len(result)}"
    )

    print(
        f"BUY signals:            "
        f"{(result['signal'] == 'BUY').sum()}"
    )

    print(
        f"SELL signals:           "
        f"{(result['signal'] == 'SELL').sum()}"
    )

    print(
        f"HOLD signals:           "
        f"{(result['signal'] == 'HOLD').sum()}"
    )

    print(
        f"\nTotal trades:           "
        f"{total_trades}"
    )

    print(
        f"Winning trades:         "
        f"{winning_trades}"
    )

    print(
        f"Win rate:               "
        f"{win_rate:.2f}%"
    )

    print(
        f"\nGross return:           "
        f"{gross_return:.2f}%"
    )

    print(
        f"Transaction cost:       "
        f"{transaction_cost:.2f}%"
    )

    print(
        f"Net return:             "
        f"{net_return:.2f}%"
    )

    print("=" * 80)

    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "locked_xgboost_test_results.csv"
    )

    result.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nSaved test results to:"
        f"\n{output_file}"
    )


if __name__ == "__main__":
    main()