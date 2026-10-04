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
    / "cost_aware_10m_known_recent_evaluation.csv"
)

HORIZON_BARS = 6
HORIZON_MINUTES = 60
TARGET_THRESHOLD = 0.0040
CONFIDENCE_THRESHOLD = 0.55

EXECUTION_SCENARIOS = {
    "LOW_COST": 0.0006,
    "MID_COST": 0.0014,
    "HIGH_COST": 0.0027,
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


def horizon_return(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        df["close"].shift(-HORIZON_BARS)
        / df["close"]
        - 1.0
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
        "Running frozen COST_AWARE_10M_V1 "
        "known-recent evaluation..."
    )

    print()
    print(
        "Frozen configuration:"
    )

    print(
        "  candles: 10m"
    )

    print(
        "  forecast horizon: 60m"
    )

    print(
        "  target: ±0.40%"
    )

    print(
        "  confidence >= 0.55"
    )

    print(
        "  entry: next bar open"
    )

    print(
        "  exit: 60m horizon close"
    )

    print()
    print(
        "IMPORTANT: KNOWN_RECENT is useful "
        "secondary out-of-sample evidence, "
        "but it is NOT a pristine final test "
        "because this market period was examined "
        "during earlier research."
    )

    train_df = pd.read_csv(
        SPLIT_DIR / "train.csv"
    )

    validation_df = pd.read_csv(
        SPLIT_DIR / "validation.csv"
    )

    known_recent_df = pd.read_csv(
        SPLIT_DIR / "known_recent.csv"
    )

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    feature_columns = [
        column
        for column in development_df.columns
        if column not in BASE_NON_FEATURE_COLUMNS
    ]

    train_returns = horizon_return(
        development_df
    )

    train_mask = (
        train_returns.notna()
    )

    encoder = LabelEncoder()

    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    model = build_model()

    model.fit(
        development_df.loc[
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
            known_recent_df[
                feature_columns
            ]
        )
    )

    predictions = (
        encoder.inverse_transform(
            model.predict(
                known_recent_df[
                    feature_columns
                ]
            )
        )
    )

    rows = known_recent_df[
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
        known_recent_df["open"]
        .shift(-1)
    )

    rows["exit_price"] = (
        known_recent_df["close"]
        .shift(-HORIZON_BARS)
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
        realized_return(
            trades
        )
    )

    output_rows = []

    for scenario_name, cost in (
        EXECUTION_SCENARIOS.items()
    ):
        net = (
            trades["gross_return"]
            - cost
        )

        trade_count = len(trades)

        wins = int(
            (net > 0).sum()
        )

        output_rows.append(
            {
                "scenario":
                    scenario_name,
                "all_in_round_trip_cost_pct":
                    cost * 100,
                "trades":
                    trade_count,
                "win_rate_pct":
                    (
                        wins
                        / trade_count
                        * 100
                        if trade_count
                        else 0.0
                    ),
                "gross_return_pct":
                    trades[
                        "gross_return"
                    ].sum()
                    * 100,
                "avg_gross_trade_pct":
                    (
                        trades[
                            "gross_return"
                        ].mean()
                        * 100
                        if trade_count
                        else 0.0
                    ),
                "net_return_pct":
                    net.sum() * 100,
                "avg_net_trade_pct":
                    (
                        net.mean() * 100
                        if trade_count
                        else 0.0
                    ),
                "median_net_trade_pct":
                    (
                        net.median() * 100
                        if trade_count
                        else 0.0
                    ),
                "compounded_net_pct":
                    (
                        (
                            np.prod(
                                1.0
                                + net.to_numpy()
                            )
                            - 1.0
                        )
                        * 100
                        if trade_count
                        else 0.0
                    ),
                "max_drawdown_pct":
                    max_drawdown_pct(
                        net
                    ),
            }
        )

    results = pd.DataFrame(
        output_rows
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "=" * 180
    )

    print(
        "COST_AWARE_10M_V1 "
        "KNOWN-RECENT RESULTS"
    )

    print(
        "=" * 180
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
                "avg_gross_trade_pct":
                    lambda value:
                        f"{value:.4f}%",
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
            },
        )
    )

    print()
    print(
        "Do not tune COST_AWARE_10M_V1 "
        "against these results."
    )

    print()
    print(
        "A genuinely clean final test still "
        "requires future unseen candles."
    )

    print()
    print("Saved to:")
    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
