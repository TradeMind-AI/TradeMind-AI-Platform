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
    / "1m_frozen_walk_forward_results.csv"
)

TRADE_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "1m_frozen_walk_forward_trades.csv"
)


# Frozen configuration. This script is diagnostic only.
FORECAST_HORIZON = 3
TARGET_THRESHOLD = 0.0010
CONFIDENCE_THRESHOLD = 0.45
ROUND_TRIP_COST = 0.0010
VALIDATION_FOLDS = 4


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


def future_return(
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
    returns: pd.Series,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                returns < -TARGET_THRESHOLD,
                returns > TARGET_THRESHOLD,
            ],
            [
                "DOWN",
                "UP",
            ],
            default="NEUTRAL",
        ),
        index=returns.index,
    )


def remove_overlapping_trades(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    accepted = []
    next_available = -1

    for _, row in (
        trades
        .sort_values("row_index")
        .iterrows()
    ):
        row_index = int(
            row["row_index"]
        )

        if row_index < next_available:
            continue

        accepted.append(row)

        next_available = (
            row_index
            + FORECAST_HORIZON
        )

    if not accepted:
        return (
            trades
            .iloc[0:0]
            .copy()
        )

    return (
        pd.DataFrame(accepted)
        .reset_index(drop=True)
    )


def build_model() -> XGBClassifier:
    return XGBClassifier(
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


def evaluate_fold(
    fold_number: int,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    encoder: LabelEncoder,
) -> tuple[dict, pd.DataFrame]:
    train_returns = future_return(
        train_df
    )

    eval_returns = future_return(
        eval_df
    )

    train_mask = (
        train_returns.notna()
    )

    eval_mask = (
        eval_returns.notna()
    )

    X_train = (
        train_df.loc[
            train_mask,
            feature_columns,
        ]
    )

    y_train = create_target(
        train_returns.loc[
            train_mask
        ]
    )

    X_eval = (
        eval_df.loc[
            eval_mask,
            feature_columns,
        ]
    )

    y_train_encoded = (
        encoder.transform(
            y_train
        )
    )

    model = build_model()

    print(
        f"Training fold {fold_number}: "
        f"{len(X_train):,} train rows, "
        f"{len(X_eval):,} evaluation rows..."
    )

    model.fit(
        X_train,
        y_train_encoded,
    )

    probabilities = (
        model.predict_proba(
            X_eval
        )
    )

    prediction_encoded = (
        model.predict(
            X_eval
        )
    )

    predictions = (
        encoder.inverse_transform(
            prediction_encoded
        )
    )

    rows = (
        eval_df.loc[
            eval_mask,
            [
                "open_time",
                "close",
            ],
        ]
        .copy()
        .reset_index(drop=True)
    )

    rows["future_return"] = (
        eval_returns.loc[
            eval_mask
        ]
        .to_numpy()
    )

    rows["prediction"] = (
        predictions
    )

    rows["confidence"] = (
        probabilities.max(axis=1)
    )

    rows["row_index"] = np.arange(
        len(rows)
    )

    directional = (
        rows["prediction"]
        .isin(
            [
                "UP",
                "DOWN",
            ]
        )
    )

    confident = (
        rows["confidence"]
        >= CONFIDENCE_THRESHOLD
    )

    trades = rows[
        directional
        & confident
    ].copy()

    trades["gross_return"] = np.where(
        trades["prediction"] == "UP",
        trades["future_return"],
        -trades["future_return"],
    )

    trades = remove_overlapping_trades(
        trades
    )

    trades["net_return"] = (
        trades["gross_return"]
        - ROUND_TRIP_COST
    )

    trades["fold"] = fold_number

    trades["result"] = np.where(
        trades["net_return"] > 0,
        "WIN",
        "LOSS",
    )

    if trades.empty:
        summary = {
            "fold": fold_number,
            "train_rows": len(X_train),
            "evaluation_rows": len(X_eval),
            "start": eval_df["open_time"].iloc[0],
            "end": eval_df["open_time"].iloc[-1],
            "trades": 0,
            "wins": 0,
            "win_rate_pct": 0.0,
            "gross_return_pct": 0.0,
            "cost_pct": 0.0,
            "net_return_pct": 0.0,
            "avg_net_trade_pct": 0.0,
        }
    else:
        wins = int(
            (
                trades["net_return"] > 0
            ).sum()
        )

        summary = {
            "fold": fold_number,
            "train_rows": len(X_train),
            "evaluation_rows": len(X_eval),
            "start": eval_df["open_time"].iloc[0],
            "end": eval_df["open_time"].iloc[-1],
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

    return summary, trades


def main() -> None:
    print(
        "Running frozen 1m "
        "walk-forward validation..."
    )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    for frame in [
        train_df,
        validation_df,
    ]:
        frame["open_time"] = (
            pd.to_datetime(
                frame["open_time"],
                utc=True,
            )
        )

    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    encoder = LabelEncoder()

    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    print()
    print(
        "Frozen configuration:"
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
    print(
        f"Features: "
        f"{len(feature_columns)}"
    )
    print(
        f"Validation folds: "
        f"{VALIDATION_FOLDS}"
    )

    fold_boundaries = np.linspace(
        0,
        len(validation_df),
        VALIDATION_FOLDS + 1,
        dtype=int,
    )

    summaries = []
    all_trades = []

    for fold_index in range(
        VALIDATION_FOLDS
    ):
        start = (
            fold_boundaries[
                fold_index
            ]
        )

        end = (
            fold_boundaries[
                fold_index + 1
            ]
        )

        eval_df = (
            validation_df.iloc[
                start:end
            ]
            .copy()
            .reset_index(drop=True)
        )

        if fold_index == 0:
            expanding_train = (
                train_df
                .copy()
                .reset_index(drop=True)
            )
        else:
            previous_validation = (
                validation_df.iloc[
                    :start
                ]
            )

            expanding_train = (
                pd.concat(
                    [
                        train_df,
                        previous_validation,
                    ],
                    ignore_index=True,
                )
            )

        summary, trades = (
            evaluate_fold(
                fold_index + 1,
                expanding_train,
                eval_df,
                feature_columns,
                encoder,
            )
        )

        summaries.append(
            summary
        )

        if not trades.empty:
            all_trades.append(
                trades
            )

    summary_df = pd.DataFrame(
        summaries
    )

    if all_trades:
        trades_df = pd.concat(
            all_trades,
            ignore_index=True,
        )
    else:
        trades_df = pd.DataFrame()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    trades_df.to_csv(
        TRADE_OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "=" * 125
    )
    print(
        "FROZEN 1M WALK-FORWARD RESULTS"
    )
    print(
        "=" * 125
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

    total_trades = int(
        summary_df[
            "trades"
        ].sum()
    )

    total_wins = int(
        summary_df[
            "wins"
        ].sum()
    )

    total_net = (
        summary_df[
            "net_return_pct"
        ].sum()
    )

    positive_folds = int(
        (
            summary_df[
                "net_return_pct"
            ] > 0
        ).sum()
    )

    print()
    print(
        "Combined walk-forward:"
    )
    print(
        f"Trades: {total_trades}"
    )

    if total_trades > 0:
        print(
            f"Win rate: "
            f"{total_wins / total_trades * 100:.2f}%"
        )
    else:
        print(
            "Win rate: 0.00%"
        )

    print(
        f"Net return: "
        f"{total_net:.2f}%"
    )

    print(
        f"Positive folds: "
        f"{positive_folds}"
        f"/{VALIDATION_FOLDS}"
    )

    print()
    print(
        "TEST SPLIT WAS NOT USED."
    )

    print()
    print(
        "Do not change the frozen "
        "configuration based on "
        "individual fold outcomes."
    )

    print()
    print(
        f"Summary saved to:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        f"Trades saved to:"
        f"\n{TRADE_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
