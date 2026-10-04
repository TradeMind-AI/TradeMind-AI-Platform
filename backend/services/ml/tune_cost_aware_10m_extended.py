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
    / "extended_research_splits_10m"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cost_aware_10m_validation.csv"
)

VALIDATION_FOLDS = 4

# Broad, deliberately limited research grid.
# We are looking for larger moves and lower turnover,
# not trying to rescue the rejected 30-minute-horizon model.
HORIZON_BARS = [
    3,   # 30m
    6,   # 60m
    12,  # 120m
    24,  # 240m
]

TARGET_THRESHOLDS = [
    0.0030,
    0.0040,
    0.0060,
    0.0080,
    0.0100,
]

CONFIDENCE_THRESHOLDS = [
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
]

EXECUTION_SCENARIOS = {
    "LOW": 0.0006,
    "MID": 0.0014,
    "HIGH": 0.0027,
}

BASE_NON_FEATURE_COLUMNS = {
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
    "target",
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


def horizon_return(
    df: pd.DataFrame,
    horizon_bars: int,
) -> pd.Series:
    return (
        df["close"].shift(-horizon_bars)
        / df["close"]
        - 1.0
    )


def remove_overlaps(
    trades: pd.DataFrame,
    horizon_bars: int,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    accepted = []
    next_available = -1

    for _, row in trades.sort_values(
        "row_index"
    ).iterrows():
        row_index = int(
            row["row_index"]
        )

        if row_index < next_available:
            continue

        accepted.append(row)
        next_available = (
            row_index + horizon_bars
        )

    if not accepted:
        return trades.iloc[0:0].copy()

    return (
        pd.DataFrame(accepted)
        .reset_index(drop=True)
    )


def realized_return(
    trades: pd.DataFrame,
) -> pd.Series:
    long_return = (
        trades["exit_price"]
        / trades["entry_price"]
        - 1.0
    )

    short_return = (
        trades["entry_price"]
        / trades["exit_price"]
        - 1.0
    )

    return pd.Series(
        np.where(
            trades["prediction"] == "UP",
            long_return,
            short_return,
        ),
        index=trades.index,
    )


def main() -> None:
    print(
        "Running cost-aware 10m extended-history research..."
    )

    print()
    print(
        "Goal: find lower-turnover setups whose gross edge "
        "comfortably exceeds execution costs."
    )

    print(
        "No KNOWN_RECENT data is loaded."
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
        if column not in BASE_NON_FEATURE_COLUMNS
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

    for horizon_bars in HORIZON_BARS:
        horizon_minutes = (
            horizon_bars * 10
        )

        print()
        print(
            f"Horizon: {horizon_minutes} minutes"
        )

        for target_threshold in TARGET_THRESHOLDS:
            fold_predictions = []

            for fold in range(
                VALIDATION_FOLDS
            ):
                start = boundaries[fold]
                end = boundaries[fold + 1]

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

                eval_df = (
                    validation_df.iloc[
                        start:end
                    ]
                    .copy()
                    .reset_index(drop=True)
                )

                train_returns = (
                    horizon_return(
                        expanding_train,
                        horizon_bars,
                    )
                )

                train_mask = (
                    train_returns.notna()
                )

                model = build_model()

                model.fit(
                    expanding_train.loc[
                        train_mask,
                        feature_columns,
                    ],
                    encoder.transform(
                        make_target(
                            train_returns.loc[
                                train_mask
                            ],
                            target_threshold,
                        )
                    ),
                )

                probabilities = (
                    model.predict_proba(
                        eval_df[
                            feature_columns
                        ]
                    )
                )

                predictions = (
                    encoder.inverse_transform(
                        model.predict(
                            eval_df[
                                feature_columns
                            ]
                        )
                    )
                )

                rows = eval_df[
                    [
                        "open_time",
                        "open",
                        "close",
                    ]
                ].copy()

                rows["prediction"] = (
                    predictions
                )

                rows["confidence"] = (
                    probabilities.max(axis=1)
                )

                rows["entry_price"] = (
                    eval_df["open"].shift(-1)
                )

                rows["exit_price"] = (
                    eval_df["close"].shift(
                        -horizon_bars
                    )
                )

                rows["row_index"] = np.arange(
                    len(rows)
                )

                rows = rows.dropna(
                    subset=[
                        "entry_price",
                        "exit_price",
                    ]
                ).copy()

                fold_predictions.append(
                    rows
                )

            for confidence_threshold in (
                CONFIDENCE_THRESHOLDS
            ):
                fold_metrics = []

                for rows in fold_predictions:
                    trades = rows[
                        rows["prediction"].isin(
                            [
                                "UP",
                                "DOWN",
                            ]
                        )
                        & (
                            rows["confidence"]
                            >= confidence_threshold
                        )
                    ].copy()

                    trades = remove_overlaps(
                        trades,
                        horizon_bars,
                    )

                    trades["gross_return"] = (
                        realized_return(
                            trades
                        )
                    )

                    fold_result = {
                        "trades": len(trades),
                        "gross_return_pct":
                            trades[
                                "gross_return"
                            ].sum()
                            * 100,
                        "gross_returns":
                            trades[
                                "gross_return"
                            ].tolist(),
                    }

                    for scenario_name, cost in (
                        EXECUTION_SCENARIOS.items()
                    ):
                        net = (
                            trades[
                                "gross_return"
                            ]
                            - cost
                        )

                        fold_result[
                            f"{scenario_name}_net_pct"
                        ] = (
                            net.sum() * 100
                        )

                    fold_metrics.append(
                        fold_result
                    )

                total_trades = sum(
                    fold["trades"]
                    for fold in fold_metrics
                )

                all_gross = pd.Series(
                    [
                        value
                        for fold in fold_metrics
                        for value in fold[
                            "gross_returns"
                        ]
                    ],
                    dtype=float,
                )

                gross_total_pct = (
                    all_gross.sum() * 100
                )

                avg_gross_trade_pct = (
                    all_gross.mean() * 100
                    if total_trades
                    else 0.0
                )

                row = {
                    "horizon_minutes":
                        horizon_minutes,
                    "target_threshold_pct":
                        target_threshold * 100,
                    "confidence_threshold":
                        confidence_threshold,
                    "trades":
                        total_trades,
                    "gross_return_pct":
                        gross_total_pct,
                    "avg_gross_trade_pct":
                        avg_gross_trade_pct,
                }

                for scenario_name, cost in (
                    EXECUTION_SCENARIOS.items()
                ):
                    scenario_key = (
                        scenario_name.lower()
                    )

                    net_returns = (
                        all_gross - cost
                    )

                    row[
                        f"{scenario_key}_cost_pct"
                    ] = cost * 100

                    row[
                        f"{scenario_key}_net_pct"
                    ] = (
                        net_returns.sum()
                        * 100
                    )

                    row[
                        f"{scenario_key}_avg_net_trade_pct"
                    ] = (
                        net_returns.mean()
                        * 100
                        if total_trades
                        else 0.0
                    )

                    row[
                        f"{scenario_key}_positive_folds"
                    ] = sum(
                        fold[
                            f"{scenario_name}_net_pct"
                        ] > 0
                        for fold in fold_metrics
                    )

                row["mid_fold_1_net_pct"] = (
                    fold_metrics[0][
                        "MID_net_pct"
                    ]
                )

                row["mid_fold_2_net_pct"] = (
                    fold_metrics[1][
                        "MID_net_pct"
                    ]
                )

                row["mid_fold_3_net_pct"] = (
                    fold_metrics[2][
                        "MID_net_pct"
                    ]
                )

                row["mid_fold_4_net_pct"] = (
                    fold_metrics[3][
                        "MID_net_pct"
                    ]
                )

                # A simple research flag, not a deployment rule:
                # enough trades, positive at MID cost, at least
                # 3/4 positive MID folds, and gross edge per trade
                # at least 1.5x the MID all-in cost.
                row["robust_candidate"] = bool(
                    total_trades >= 40
                    and row[
                        "mid_net_pct"
                    ] > 0
                    and row[
                        "mid_positive_folds"
                    ] >= 3
                    and avg_gross_trade_pct
                    >= (
                        EXECUTION_SCENARIOS[
                            "MID"
                        ]
                        * 100
                        * 1.5
                    )
                )

                results.append(row)

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df.sort_values(
            by=[
                "robust_candidate",
                "mid_positive_folds",
                "mid_net_pct",
            ],
            ascending=[
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
        "=" * 220
    )

    print(
        "TOP COST-AWARE 10M EXTENDED-HISTORY RESULTS"
    )

    print(
        "=" * 220
    )

    display = results_df.head(
        30
    ).copy()

    columns = [
        "horizon_minutes",
        "target_threshold_pct",
        "confidence_threshold",
        "trades",
        "gross_return_pct",
        "avg_gross_trade_pct",
        "low_net_pct",
        "mid_net_pct",
        "high_net_pct",
        "mid_positive_folds",
        "mid_fold_1_net_pct",
        "mid_fold_2_net_pct",
        "mid_fold_3_net_pct",
        "mid_fold_4_net_pct",
        "robust_candidate",
    ]

    print(
        display[
            columns
        ].to_string(
            index=False,
            formatters={
                "target_threshold_pct":
                    lambda value:
                        f"{value:.2f}%",
                "confidence_threshold":
                    lambda value:
                        f"{value:.2f}",
                "gross_return_pct":
                    lambda value:
                        f"{value:.2f}%",
                "avg_gross_trade_pct":
                    lambda value:
                        f"{value:.4f}%",
                "low_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "mid_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "high_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "mid_fold_1_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "mid_fold_2_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "mid_fold_3_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "mid_fold_4_net_pct":
                    lambda value:
                        f"{value:.2f}%",
            },
        )
    )

    print()
    print(
        f"Robust candidates found: "
        f"{int(results_df['robust_candidate'].sum())}"
    )

    print()
    print(
        "KNOWN_RECENT was NOT loaded."
    )

    print()
    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
