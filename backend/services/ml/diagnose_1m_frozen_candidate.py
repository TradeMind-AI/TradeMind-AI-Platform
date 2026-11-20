from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_1m"
)

TRAIN_FILE = SPLIT_DIR / "train.csv"
VALIDATION_FILE = SPLIT_DIR / "validation.csv"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "1m_frozen_candidate_validation_trades.csv"
)


# Frozen candidate. Do not tune these values in this diagnostic.
FORECAST_HORIZON = 3
TARGET_THRESHOLD = 0.0010
CONFIDENCE_THRESHOLD = 0.45
ROUND_TRIP_COST = 0.0010


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
    "future_return_1m",
}


def create_future_return(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        df["close"].shift(
            -FORECAST_HORIZON
        )
        / df["close"]
        - 1
    )


def create_target(
    future_return: pd.Series,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                future_return
                < -TARGET_THRESHOLD,
                future_return
                > TARGET_THRESHOLD,
            ],
            [
                "DOWN",
                "UP",
            ],
            default="NEUTRAL",
        ),
        index=future_return.index,
    )


def remove_overlapping_trades(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    accepted = []
    next_available_index = -1

    for _, row in (
        candidates
        .sort_values("row_index")
        .iterrows()
    ):
        row_index = int(
            row["row_index"]
        )

        if (
            row_index
            < next_available_index
        ):
            continue

        accepted.append(row)

        next_available_index = (
            row_index
            + FORECAST_HORIZON
        )

    if not accepted:
        return (
            candidates
            .iloc[0:0]
            .copy()
        )

    return (
        pd.DataFrame(
            accepted
        )
        .reset_index(drop=True)
    )


def summarize_period(
    name: str,
    trades: pd.DataFrame,
) -> dict:
    if trades.empty:
        return {
            "period": name,
            "trades": 0,
            "wins": 0,
            "win_rate_pct": 0.0,
            "gross_return_pct": 0.0,
            "cost_pct": 0.0,
            "net_return_pct": 0.0,
            "avg_net_trade_pct": 0.0,
        }

    wins = int(
        (
            trades["net_return"]
            > 0
        ).sum()
    )

    return {
        "period": name,
        "trades": len(trades),
        "wins": wins,
        "win_rate_pct": (
            wins
            / len(trades)
            * 100
        ),
        "gross_return_pct": (
            trades[
                "gross_return"
            ].sum()
            * 100
        ),
        "cost_pct": (
            len(trades)
            * ROUND_TRIP_COST
            * 100
        ),
        "net_return_pct": (
            trades[
                "net_return"
            ].sum()
            * 100
        ),
        "avg_net_trade_pct": (
            trades[
                "net_return"
            ].mean()
            * 100
        ),
    }


def main() -> None:
    print(
        "Loading frozen 1m candidate "
        "validation diagnostic..."
    )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    train_df["open_time"] = (
        pd.to_datetime(
            train_df["open_time"],
            utc=True,
        )
    )

    validation_df["open_time"] = (
        pd.to_datetime(
            validation_df["open_time"],
            utc=True,
        )
    )

    feature_columns = [
        column
        for column in train_df.columns
        if column
        not in NON_FEATURE_COLUMNS
    ]

    train_future = (
        create_future_return(
            train_df
        )
    )

    validation_future = (
        create_future_return(
            validation_df
        )
    )

    train_mask = (
        train_future.notna()
    )

    validation_mask = (
        validation_future.notna()
    )

    X_train = (
        train_df.loc[
            train_mask,
            feature_columns,
        ]
    )

    X_validation = (
        validation_df.loc[
            validation_mask,
            feature_columns,
        ]
    )

    y_train = create_target(
        train_future.loc[
            train_mask
        ]
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

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_validation):,}"
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    print(
        "\nFrozen configuration:"
    )

    print(
        f"Forecast horizon: "
        f"{FORECAST_HORIZON}m"
    )

    print(
        f"Target threshold: "
        f"±{TARGET_THRESHOLD * 100:.2f}%"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"Round-trip cost: "
        f"{ROUND_TRIP_COST * 100:.2f}%"
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    print(
        "\nTraining frozen candidate..."
    )

    model.fit(
        X_train,
        y_train_encoded,
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )
    )

    prediction_encoded = (
        model.predict(
            X_validation
        )
    )

    predictions = (
        encoder.inverse_transform(
            prediction_encoded
        )
    )

    validation_rows = (
        validation_df.loc[
            validation_mask,
            [
                "open_time",
                "close",
            ],
        ]
        .copy()
        .reset_index(drop=True)
    )

    validation_rows[
        "future_return"
    ] = (
        validation_future.loc[
            validation_mask
        ]
        .to_numpy()
    )

    validation_rows[
        "prediction"
    ] = predictions

    class_index = {
        name: index
        for index, name
        in enumerate(
            encoder.classes_
        )
    }

    validation_rows[
        "probability_down"
    ] = probabilities[
        :,
        class_index["DOWN"],
    ]

    validation_rows[
        "probability_neutral"
    ] = probabilities[
        :,
        class_index["NEUTRAL"],
    ]

    validation_rows[
        "probability_up"
    ] = probabilities[
        :,
        class_index["UP"],
    ]

    validation_rows[
        "confidence"
    ] = probabilities.max(
        axis=1
    )

    validation_rows[
        "row_index"
    ] = np.arange(
        len(validation_rows)
    )

    candidates = validation_rows[
        validation_rows[
            "prediction"
        ].isin(
            [
                "UP",
                "DOWN",
            ]
        )
        & (
            validation_rows[
                "confidence"
            ]
            >= CONFIDENCE_THRESHOLD
        )
    ].copy()

    candidates[
        "gross_return"
    ] = np.where(
        candidates[
            "prediction"
        ] == "UP",
        candidates[
            "future_return"
        ],
        -candidates[
            "future_return"
        ],
    )

    trades = (
        remove_overlapping_trades(
            candidates
        )
    )

    trades[
        "net_return"
    ] = (
        trades[
            "gross_return"
        ]
        - ROUND_TRIP_COST
    )

    trades[
        "result"
    ] = np.where(
        trades[
            "net_return"
        ] > 0,
        "WIN",
        "LOSS",
    )

    trades[
        "equity"
    ] = (
        1
        + trades[
            "net_return"
        ]
    ).cumprod()

    trades[
        "equity_peak"
    ] = (
        trades[
            "equity"
        ]
        .cummax()
    )

    trades[
        "drawdown"
    ] = (
        trades[
            "equity"
        ]
        / trades[
            "equity_peak"
        ]
        - 1
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    total_summary = (
        summarize_period(
            "ALL",
            trades,
        )
    )

    boundaries = np.linspace(
        0,
        len(trades),
        4,
        dtype=int,
    )

    period_summaries = [
        total_summary
    ]

    for index in range(3):
        part = trades.iloc[
            boundaries[index]:
            boundaries[index + 1]
        ]

        period_summaries.append(
            summarize_period(
                f"PERIOD_{index + 1}",
                part,
            )
        )

    summary_df = pd.DataFrame(
        period_summaries
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FROZEN 1M CANDIDATE "
        "VALIDATION DIAGNOSTIC"
    )

    print(
        "=" * 100
    )

    print(
        summary_df.to_string(
            index=False,
            formatters={
                "win_rate_pct":
                    lambda value:
                        f"{value:.2f}%",

                "gross_return_pct":
                    lambda value:
                        f"{value:.2f}%",

                "cost_pct":
                    lambda value:
                        f"{value:.2f}%",

                "net_return_pct":
                    lambda value:
                        f"{value:.2f}%",

                "avg_net_trade_pct":
                    lambda value:
                        f"{value:.4f}%",
            },
        )
    )

    buy_trades = trades[
        trades[
            "prediction"
        ] == "UP"
    ]

    sell_trades = trades[
        trades[
            "prediction"
        ] == "DOWN"
    ]

    print(
        "\nDirection breakdown:"
    )

    for name, frame in [
        ("BUY", buy_trades),
        ("SELL", sell_trades),
    ]:
        if frame.empty:
            print(
                f"{name}: 0 trades"
            )
            continue

        print(
            f"{name}: "
            f"{len(frame)} trades, "
            f"net "
            f"{frame['net_return'].sum() * 100:.2f}%, "
            f"win rate "
            f"{(frame['net_return'] > 0).mean() * 100:.2f}%"
        )

    if not trades.empty:
        max_drawdown = (
            trades[
                "drawdown"
            ].min()
            * 100
        )

        print(
            f"\nMaximum trade-sequence "
            f"drawdown: "
            f"{max_drawdown:.2f}%"
        )

        loss_flags = (
            trades["result"]
            == "LOSS"
        )

        max_consecutive_losses = 0
        current_losses = 0

        for is_loss in loss_flags:
            if is_loss:
                current_losses += 1

                max_consecutive_losses = max(
                    max_consecutive_losses,
                    current_losses,
                )
            else:
                current_losses = 0

        print(
            "Maximum consecutive losses: "
            f"{max_consecutive_losses}"
        )

        print(
            "\nValidation trade dates:"
        )

        print(
            f"First: "
            f"{trades['open_time'].min()}"
        )

        print(
            f"Last:  "
            f"{trades['open_time'].max()}"
        )

    print(
        "\nTEST SPLIT WAS NOT USED."
    )

    print(
        "\nDo not change the frozen "
        "configuration based on this "
        "diagnostic."
    )

    print(
        f"\nDetailed trades saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
