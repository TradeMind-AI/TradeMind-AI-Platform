from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Validation-only execution stress test.
# These are scenario assumptions, not claims about a specific exchange fee tier.
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


CONFIGS = {
    "2m": {
        "split_dir": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ml_splits_2m"
        ),
        "future_return_column": "future_return_2m",
        "horizon_bars": 3,
        "target_threshold": 0.0010,
        "confidence_threshold": 0.50,
    },
    "10m": {
        "split_dir": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ml_splits_10m"
        ),
        "future_return_column": "future_return_10m",
        "horizon_bars": 3,
        "target_threshold": 0.0020,
        "confidence_threshold": 0.45,
    },
    "30m": {
        "split_dir": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ml_splits_30m"
        ),
        "future_return_column": "future_return_30m",
        "horizon_bars": 4,
        "target_threshold": 0.0040,
        "confidence_threshold": 0.45,
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


def horizon_return_from_close(
    df: pd.DataFrame,
    horizon_bars: int,
) -> pd.Series:
    return (
        df["close"].shift(-horizon_bars)
        / df["close"]
        - 1
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
        row_index = int(row["row_index"])

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


def prepare_validation_predictions(
    config: dict,
) -> pd.DataFrame:
    split_dir = config["split_dir"]

    train_df = pd.read_csv(
        split_dir / "train.csv"
    )

    validation_df = pd.read_csv(
        split_dir / "validation.csv"
    )

    future_column = config[
        "future_return_column"
    ]

    non_feature_columns = (
        BASE_NON_FEATURE_COLUMNS
        | {
            future_column,
            "target",
        }
    )

    feature_columns = [
        column
        for column in train_df.columns
        if column not in non_feature_columns
    ]

    train_returns = (
        horizon_return_from_close(
            train_df,
            config["horizon_bars"],
        )
    )

    train_mask = train_returns.notna()

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
        train_df.loc[
            train_mask,
            feature_columns,
        ],
        encoder.transform(
            make_target(
                train_returns.loc[
                    train_mask
                ],
                config[
                    "target_threshold"
                ],
            )
        ),
    )

    probabilities = model.predict_proba(
        validation_df[
            feature_columns
        ]
    )

    predictions = (
        encoder.inverse_transform(
            model.predict(
                validation_df[
                    feature_columns
                ]
            )
        )
    )

    rows = validation_df[
        [
            "open_time",
            "close_time",
            "open",
            "close",
        ]
    ].copy()

    rows["prediction"] = predictions
    rows["confidence"] = (
        probabilities.max(axis=1)
    )

    # Signal is generated at the close of row t.
    # More realistic execution enters at the next bar open.
    rows["entry_price"] = (
        validation_df["open"]
        .shift(-1)
    )

    # Exit at the close corresponding to the frozen forecast horizon.
    rows["exit_price"] = (
        validation_df["close"]
        .shift(
            -config["horizon_bars"]
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
            >= config[
                "confidence_threshold"
            ]
        )
    )

    return rows


def gross_return(
    trades: pd.DataFrame,
) -> pd.Series:
    long_return = (
        trades["exit_price"]
        / trades["entry_price"]
        - 1
    )

    short_return = (
        trades["entry_price"]
        / trades["exit_price"]
        - 1
    )

    return pd.Series(
        np.where(
            trades["prediction"] == "UP",
            long_return,
            short_return,
        ),
        index=trades.index,
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
        equity / running_max - 1
    )

    return float(
        drawdown.min() * 100
    )


def main() -> None:
    print(
        "Running validation-only "
        "execution realism stress test..."
    )

    print()
    print(
        "No test split is loaded."
    )

    print(
        "Entry model: next bar open "
        "after signal candle closes."
    )

    print(
        "Exit model: frozen horizon "
        "close."
    )

    print()
    print(
        "Execution scenarios "
        "(assumptions, not exchange quotes):"
    )

    for name, scenario in (
        EXECUTION_SCENARIOS.items()
    ):
        total = scenario_cost(
            scenario
        )

        print(
            f"  {name}: "
            f"fee/side "
            f"{scenario['fee_per_side'] * 100:.3f}%, "
            f"spread RT "
            f"{scenario['spread_round_trip'] * 100:.3f}%, "
            f"slippage/side "
            f"{scenario['slippage_per_side'] * 100:.3f}%, "
            f"all-in RT "
            f"{total * 100:.3f}%"
        )

    output_rows = []

    for timeframe, config in (
        CONFIGS.items()
    ):
        print()
        print(
            f"Evaluating frozen "
            f"{timeframe} validation setup..."
        )

        rows = (
            prepare_validation_predictions(
                config
            )
        )

        trades = rows[
            rows["signal"]
        ].copy()

        trades = remove_overlaps(
            trades,
            config["horizon_bars"],
        )

        trades["gross_return"] = (
            gross_return(trades)
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

            trade_count = len(trades)

            wins = int(
                (net > 0).sum()
            )

            output_rows.append(
                {
                    "timeframe":
                        timeframe,
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
                        (
                            trades[
                                "gross_return"
                            ].sum()
                            * 100
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

    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "execution_realism_validation.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "=" * 165
    )

    print(
        "EXECUTION REALISM "
        "VALIDATION RESULTS"
    )

    print(
        "=" * 165
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
            },
        )
    )

    print()
    print(
        "TEST SPLITS WERE NOT USED."
    )

    print()
    print(
        "Do not select a model from one "
        "cost scenario alone; look for "
        "robustness across plausible "
        "execution assumptions."
    )

    print()
    print("Saved to:")
    print(output_file)


if __name__ == "__main__":
    main()
