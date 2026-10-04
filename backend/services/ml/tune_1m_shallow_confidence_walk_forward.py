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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "1m_shallow_confidence_walk_forward.csv"
)


FORECAST_HORIZON = 3
TARGET_THRESHOLD = 0.0010
ROUND_TRIP_COST = 0.0010
VALIDATION_FOLDS = 4

CONFIDENCE_THRESHOLDS = [
    0.300,
    0.325,
    0.350,
    0.375,
    0.400,
    0.425,
    0.450,
]


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


def build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
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


def remove_overlaps(
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


def main() -> None:
    print(
        "Running shallow 1m confidence "
        "walk-forward sweep..."
    )

    train_df = pd.read_csv(
        SPLIT_DIR / "train.csv"
    )

    validation_df = pd.read_csv(
        SPLIT_DIR / "validation.csv"
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
        "Frozen model:"
    )
    print(
        "depth=3, trees=200, "
        "learning_rate=0.05"
    )
    print(
        "81 features, 3m horizon, "
        "±0.10% target, "
        "0.10% round-trip cost"
    )

    boundaries = np.linspace(
        0,
        len(validation_df),
        VALIDATION_FOLDS + 1,
        dtype=int,
    )

    fold_payloads = []

    for fold in range(
        VALIDATION_FOLDS
    ):
        start = boundaries[fold]
        end = boundaries[fold + 1]

        eval_df = (
            validation_df.iloc[
                start:end
            ]
            .copy()
            .reset_index(drop=True)
        )

        if fold == 0:
            expanding_train = (
                train_df
                .copy()
                .reset_index(drop=True)
            )
        else:
            expanding_train = (
                pd.concat(
                    [
                        train_df,
                        validation_df.iloc[
                            :start
                        ],
                    ],
                    ignore_index=True,
                )
            )

        train_returns = future_return(
            expanding_train
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
            expanding_train.loc[
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

        print(
            f"Training fold {fold + 1}: "
            f"{len(X_train):,} train rows, "
            f"{len(X_eval):,} eval rows..."
        )

        model = build_model()

        model.fit(
            X_train,
            encoder.transform(
                y_train
            ),
        )

        probabilities = (
            model.predict_proba(
                X_eval
            )
        )

        predictions = (
            encoder.inverse_transform(
                model.predict(
                    X_eval
                )
            )
        )

        fold_payloads.append(
            {
                "fold": fold + 1,
                "returns":
                    eval_returns.loc[
                        eval_mask
                    ].to_numpy(),
                "predictions":
                    predictions,
                "confidence":
                    probabilities.max(
                        axis=1
                    ),
            }
        )

    results = []

    for confidence_threshold in (
        CONFIDENCE_THRESHOLDS
    ):
        fold_returns = []
        total_trades = 0
        total_wins = 0
        all_net_returns = []

        for payload in fold_payloads:
            rows = pd.DataFrame(
                {
                    "future_return":
                        payload["returns"],
                    "prediction":
                        payload["predictions"],
                    "confidence":
                        payload["confidence"],
                    "row_index":
                        np.arange(
                            len(
                                payload[
                                    "returns"
                                ]
                            )
                        ),
                }
            )

            trades = rows[
                rows[
                    "prediction"
                ].isin(
                    [
                        "UP",
                        "DOWN",
                    ]
                )
                & (
                    rows[
                        "confidence"
                    ]
                    >= confidence_threshold
                )
            ].copy()

            trades[
                "gross_return"
            ] = np.where(
                trades[
                    "prediction"
                ] == "UP",
                trades[
                    "future_return"
                ],
                -trades[
                    "future_return"
                ],
            )

            trades = remove_overlaps(
                trades
            )

            trades[
                "net_return"
            ] = (
                trades[
                    "gross_return"
                ]
                - ROUND_TRIP_COST
            )

            fold_net = (
                trades[
                    "net_return"
                ].sum()
                * 100
            )

            fold_returns.append(
                fold_net
            )

            total_trades += len(
                trades
            )

            total_wins += int(
                (
                    trades[
                        "net_return"
                    ] > 0
                ).sum()
            )

            all_net_returns.extend(
                trades[
                    "net_return"
                ].tolist()
            )

        if total_trades > 0:
            win_rate = (
                total_wins
                / total_trades
                * 100
            )

            avg_net = (
                np.mean(
                    all_net_returns
                )
                * 100
            )

            median_net = (
                np.median(
                    all_net_returns
                )
                * 100
            )
        else:
            win_rate = 0.0
            avg_net = 0.0
            median_net = 0.0

        results.append(
            {
                "confidence_threshold":
                    confidence_threshold,
                "trades":
                    total_trades,
                "win_rate_pct":
                    win_rate,
                "net_return_pct":
                    sum(
                        fold_returns
                    ),
                "avg_net_trade_pct":
                    avg_net,
                "median_net_trade_pct":
                    median_net,
                "positive_folds":
                    sum(
                        value > 0
                        for value in fold_returns
                    ),
                "fold_1_net_pct":
                    fold_returns[0],
                "fold_2_net_pct":
                    fold_returns[1],
                "fold_3_net_pct":
                    fold_returns[2],
                "fold_4_net_pct":
                    fold_returns[3],
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by=[
                "positive_folds",
                "net_return_pct",
                "avg_net_trade_pct",
                "trades",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "=" * 145
    )
    print(
        "1M SHALLOW MODEL CONFIDENCE "
        "WALK-FORWARD RESULTS"
    )
    print(
        "=" * 145
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "confidence_threshold":
                    lambda value:
                        f"{value:.3f}",
                "win_rate_pct":
                    lambda value:
                        f"{value:.2f}%",
                "net_return_pct":
                    lambda value:
                        f"{value:.2f}%",
                "avg_net_trade_pct":
                    lambda value:
                        f"{value:.4f}%",
                "median_net_trade_pct":
                    lambda value:
                        f"{value:.4f}%",
                "fold_1_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "fold_2_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "fold_3_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "fold_4_net_pct":
                    lambda value:
                        f"{value:.2f}%",
            },
        )
    )

    print()
    print(
        "TEST SPLIT WAS NOT USED."
    )

    print()
    print(
        "Selection rule: require enough "
        "trades, positive after-cost "
        "expectancy, and better fold "
        "consistency before considering "
        "the untouched test."
    )

    print()
    print(
        f"Saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
