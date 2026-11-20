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

VALIDATION_FOLDS = 4

HORIZON_BARS = 3
TARGET_THRESHOLD = 0.0020
CONFIDENCE_THRESHOLD = 0.45

EXECUTION_SCENARIOS = {
    "LOW_COST": {
        "fee_per_side": 0.0002,
        "spread_round_trip": 0.0001,
        "slippage_per_side": 0.00005,
    },
    "MID_COST": {
        "fee_per_side": 0.0005,
        "spread_round_trip": 0.0002,
        "slippage_per_side": 0.0001,
    },
    "HIGH_COST": {
        "fee_per_side": 0.0010,
        "spread_round_trip": 0.0003,
        "slippage_per_side": 0.0002,
    },
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


def horizon_return_from_close(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        df["close"].shift(-HORIZON_BARS)
        / df["close"]
        - 1.0
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


def remove_overlaps(
    trades: pd.DataFrame,
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
            row_index
            + HORIZON_BARS
        )

    if not accepted:
        return trades.iloc[0:0].copy()

    return (
        pd.DataFrame(accepted)
        .reset_index(drop=True)
    )


def scenario_cost(
    scenario: dict,
) -> float:
    return (
        2 * scenario["fee_per_side"]
        + scenario["spread_round_trip"]
        + 2 * scenario["slippage_per_side"]
    )


def max_drawdown_pct(
    net_returns: pd.Series,
) -> float:
    if net_returns.empty:
        return 0.0

    equity = (
        1.0 + net_returns
    ).cumprod()

    running_max = equity.cummax()

    drawdown = (
        equity / running_max - 1.0
    )

    return float(
        drawdown.min() * 100
    )


def main() -> None:
    print(
        "Running frozen 10m extended-history "
        "replication validation..."
    )

    print()
    print(
        "Frozen configuration:"
    )

    print(
        "  10m candles"
    )

    print(
        "  30m forecast horizon"
    )

    print(
        "  target ±0.20%"
    )

    print(
        "  confidence >= 0.45"
    )

    print(
        "  4 expanding validation folds"
    )

    print(
        "  signal at candle close"
    )

    print(
        "  entry at next candle open"
    )

    print(
        "  exit at frozen horizon close"
    )

    print()

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

    scenario_folds = {
        name: []
        for name in EXECUTION_SCENARIOS
    }

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
            expanding_train = pd.concat(
                [
                    train_df,
                    validation_df.iloc[
                        :start
                    ],
                ],
                ignore_index=True,
            )

        eval_df = (
            validation_df.iloc[
                start:end
            ]
            .copy()
            .reset_index(drop=True)
        )

        train_returns = (
            horizon_return_from_close(
                expanding_train
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
                    ]
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
                "close_time",
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
                -HORIZON_BARS
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

        rows["signal"] = (
            rows["prediction"].isin(
                [
                    "UP",
                    "DOWN",
                ]
            )
            & (
                rows["confidence"]
                >= CONFIDENCE_THRESHOLD
            )
        )

        trades = rows[
            rows["signal"]
        ].copy()

        trades = remove_overlaps(
            trades
        )

        trades["gross_return"] = (
            realized_return(trades)
        )

        print(
            f"Fold {fold + 1}: "
            f"{len(trades)} trades, "
            f"gross "
            f"{trades['gross_return'].sum() * 100:.2f}%"
        )

        for scenario_name, scenario in (
            EXECUTION_SCENARIOS.items()
        ):
            cost = scenario_cost(
                scenario
            )

            net = (
                trades["gross_return"]
                - cost
            )

            scenario_folds[
                scenario_name
            ].append(
                {
                    "trades":
                        len(trades),
                    "wins":
                        int(
                            (net > 0).sum()
                        ),
                    "gross_return_pct":
                        trades[
                            "gross_return"
                        ].sum()
                        * 100,
                    "net_return_pct":
                        net.sum() * 100,
                    "net_returns":
                        net.tolist(),
                }
            )

    output_rows = []

    for scenario_name, folds in (
        scenario_folds.items()
    ):
        total_trades = sum(
            fold["trades"]
            for fold in folds
        )

        total_wins = sum(
            fold["wins"]
            for fold in folds
        )

        all_net = pd.Series(
            [
                value
                for fold in folds
                for value in fold[
                    "net_returns"
                ]
            ],
            dtype=float,
        )

        output_rows.append(
            {
                "scenario":
                    scenario_name,
                "all_in_round_trip_cost_pct":
                    scenario_cost(
                        EXECUTION_SCENARIOS[
                            scenario_name
                        ]
                    )
                    * 100,
                "trades":
                    total_trades,
                "win_rate_pct":
                    (
                        total_wins
                        / total_trades
                        * 100
                        if total_trades
                        else 0.0
                    ),
                "gross_return_pct":
                    sum(
                        fold[
                            "gross_return_pct"
                        ]
                        for fold in folds
                    ),
                "net_return_pct":
                    sum(
                        fold[
                            "net_return_pct"
                        ]
                        for fold in folds
                    ),
                "avg_net_trade_pct":
                    (
                        all_net.mean() * 100
                        if total_trades
                        else 0.0
                    ),
                "median_net_trade_pct":
                    (
                        all_net.median() * 100
                        if total_trades
                        else 0.0
                    ),
                "compounded_net_pct":
                    (
                        (
                            np.prod(
                                1.0
                                + all_net.to_numpy()
                            )
                            - 1.0
                        )
                        * 100
                        if total_trades
                        else 0.0
                    ),
                "max_drawdown_pct":
                    max_drawdown_pct(
                        all_net
                    ),
                "positive_folds":
                    sum(
                        fold[
                            "net_return_pct"
                        ] > 0
                        for fold in folds
                    ),
                "fold_1_net_pct":
                    folds[0][
                        "net_return_pct"
                    ],
                "fold_2_net_pct":
                    folds[1][
                        "net_return_pct"
                    ],
                "fold_3_net_pct":
                    folds[2][
                        "net_return_pct"
                    ],
                "fold_4_net_pct":
                    folds[3][
                        "net_return_pct"
                    ],
            }
        )

    results = pd.DataFrame(
        output_rows
    )

    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "extended_10m_replication_validation.csv"
    )

    results.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "=" * 185
    )

    print(
        "FROZEN 10M EXTENDED-HISTORY "
        "REPLICATION RESULTS"
    )

    print(
        "=" * 185
    )

    print(
        results.to_string(
            index=False,
            formatters={
                "all_in_round_trip_cost_pct":
                    lambda value:
                        f"{value:.3f}%",
                "win_rate_pct":
                    lambda value:
                        f"{value:.2f}%",
                "gross_return_pct":
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
                "compounded_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "max_drawdown_pct":
                    lambda value:
                        f"{value:.2f}%",
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
        "KNOWN_RECENT was NOT loaded."
    )

    print(
        "This is a replication check on "
        "2025-01-01 through 2025-09-30 only."
    )

    print()
    print("Saved to:")
    print(output_file)


if __name__ == "__main__":
    main()
