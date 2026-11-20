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
    / "ml_splits_10m"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "10m_multi_horizon_validation.csv"
)


FORECAST_HORIZONS_BARS = [
    1,
    2,
    3,
    6,
]

TARGET_THRESHOLDS = [
    0.0015,
    0.0020,
    0.0030,
    0.0040,
]

CONFIDENCE_THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

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
    "future_return_10m",
}


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


def make_return(
    df: pd.DataFrame,
    horizon_bars: int,
) -> pd.Series:
    return (
        df["close"].shift(
            -horizon_bars
        )
        / df["close"]
        - 1
    )


def make_target(
    returns: pd.Series,
    threshold: float,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                returns < -threshold,
                returns > threshold,
            ],
            [
                "DOWN",
                "UP",
            ],
            default="NEUTRAL",
        ),
        index=returns.index,
    )


def remove_overlaps(
    trades: pd.DataFrame,
    horizon_bars: int,
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
            + horizon_bars
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
        "Running 10m multi-horizon "
        "walk-forward validation..."
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

    boundaries = np.linspace(
        0,
        len(validation_df),
        VALIDATION_FOLDS + 1,
        dtype=int,
    )

    results = []

    for horizon_bars in (
        FORECAST_HORIZONS_BARS
    ):
        horizon_minutes = (
            horizon_bars * 10
        )

        print()
        print(
            f"Forecast horizon: "
            f"{horizon_minutes} minutes"
        )

        for target_threshold in (
            TARGET_THRESHOLDS
        ):
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

                train_returns = (
                    make_return(
                        expanding_train,
                        horizon_bars,
                    )
                )

                eval_returns = (
                    make_return(
                        eval_df,
                        horizon_bars,
                    )
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

                y_train = (
                    make_target(
                        train_returns.loc[
                            train_mask
                        ],
                        target_threshold,
                    )
                )

                X_eval = (
                    eval_df.loc[
                        eval_mask,
                        feature_columns,
                    ]
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
                                payload[
                                    "predictions"
                                ],
                            "confidence":
                                payload[
                                    "confidence"
                                ],
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
                        trades,
                        horizon_bars,
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
                else:
                    win_rate = 0.0
                    avg_net = 0.0

                results.append(
                    {
                        "horizon_minutes":
                            horizon_minutes,
                        "target_threshold_pct":
                            target_threshold
                            * 100,
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
        "=" * 150
    )

    print(
        "TOP 10M MULTI-HORIZON "
        "WALK-FORWARD RESULTS"
    )

    print(
        "=" * 150
    )

    print(
        results_df
        .head(30)
        .to_string(
            index=False,
            formatters={
                "target_threshold_pct":
                    lambda value:
                        f"{value:.3f}%",
                "confidence_threshold":
                    lambda value:
                        f"{value:.2f}",
                "win_rate_pct":
                    lambda value:
                        f"{value:.2f}%",
                "net_return_pct":
                    lambda value:
                        f"{value:.2f}%",
                "avg_net_trade_pct":
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
        f"Saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
