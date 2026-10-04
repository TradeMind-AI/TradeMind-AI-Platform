from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ROUND_TRIP_COST = 0.0010


CONFIGS = {
    "10m": {
        "split_dir": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ml_splits_10m"
        ),
        "future_return_column":
            "future_return_10m",
        "horizon_bars": 3,
        "horizon_minutes": 30,
        "target_threshold": 0.0020,
        "confidence_threshold": 0.45,
        "model_params": {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    },
    "30m": {
        "split_dir": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ml_splits_30m"
        ),
        "future_return_column":
            "future_return_30m",
        "horizon_bars": 4,
        "horizon_minutes": 120,
        "target_threshold": 0.0040,
        "confidence_threshold": 0.45,
        "model_params": {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
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


def build_model(
    params: dict,
) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def future_return(
    df: pd.DataFrame,
    horizon_bars: int,
) -> pd.Series:
    return (
        df["close"].shift(-horizon_bars)
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
        return trades.iloc[0:0].copy()

    return (
        pd.DataFrame(accepted)
        .reset_index(drop=True)
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


def evaluate_timeframe(
    timeframe: str,
    config: dict,
    encoder: LabelEncoder,
) -> tuple[dict, pd.DataFrame]:
    split_dir = config["split_dir"]

    train_df = pd.read_csv(
        split_dir / "train.csv"
    )

    validation_df = pd.read_csv(
        split_dir / "validation.csv"
    )

    test_df = pd.read_csv(
        split_dir / "test.csv"
    )

    train_validation = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    future_column = (
        config[
            "future_return_column"
        ]
    )

    non_feature_columns = (
        BASE_NON_FEATURE_COLUMNS
        | {
            future_column,
            "target",
        }
    )

    feature_columns = [
        column
        for column
        in train_validation.columns
        if column
        not in non_feature_columns
    ]

    horizon_bars = (
        config["horizon_bars"]
    )

    target_threshold = (
        config["target_threshold"]
    )

    confidence_threshold = (
        config[
            "confidence_threshold"
        ]
    )

    train_returns = future_return(
        train_validation,
        horizon_bars,
    )

    train_mask = (
        train_returns.notna()
    )

    X_train = (
        train_validation.loc[
            train_mask,
            feature_columns,
        ]
    )

    y_train = make_target(
        train_returns.loc[
            train_mask
        ],
        target_threshold,
    )

    test_returns = future_return(
        test_df,
        horizon_bars,
    )

    test_mask = (
        test_returns.notna()
    )

    test_scored = (
        test_df.loc[
            test_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_returns = (
        test_returns.loc[
            test_mask
        ]
        .reset_index(drop=True)
    )

    model = build_model(
        config["model_params"]
    )

    model.fit(
        X_train,
        encoder.transform(
            y_train
        ),
    )

    probabilities = (
        model.predict_proba(
            test_scored[
                feature_columns
            ]
        )
    )

    predictions = (
        encoder.inverse_transform(
            model.predict(
                test_scored[
                    feature_columns
                ]
            )
        )
    )

    result = test_scored[
        [
            "open_time",
            "close_time",
            "close",
        ]
    ].copy()

    result["prediction"] = (
        predictions
    )

    result["confidence"] = (
        probabilities.max(axis=1)
    )

    result["future_return"] = (
        test_returns.to_numpy()
    )

    result["row_index"] = np.arange(
        len(result)
    )

    result["signal"] = (
        result["prediction"].isin(
            [
                "UP",
                "DOWN",
            ]
        )
        & (
            result["confidence"]
            >= confidence_threshold
        )
    )

    trades = result[
        result["signal"]
    ].copy()

    trades["gross_return"] = np.where(
        trades["prediction"] == "UP",
        trades["future_return"],
        -trades["future_return"],
    )

    trades = remove_overlaps(
        trades,
        horizon_bars,
    )

    trades["net_return"] = (
        trades["gross_return"]
        - ROUND_TRIP_COST
    )

    trade_count = len(trades)

    wins = int(
        (
            trades["net_return"] > 0
        ).sum()
    )

    losses = int(
        (
            trades["net_return"] <= 0
        ).sum()
    )

    win_rate = (
        wins / trade_count * 100
        if trade_count
        else 0.0
    )

    gross_return_pct = (
        trades["gross_return"].sum()
        * 100
    )

    total_cost_pct = (
        trade_count
        * ROUND_TRIP_COST
        * 100
    )

    net_return_pct = (
        trades["net_return"].sum()
        * 100
    )

    avg_net_trade_pct = (
        trades["net_return"].mean()
        * 100
        if trade_count
        else 0.0
    )

    median_net_trade_pct = (
        trades["net_return"].median()
        * 100
        if trade_count
        else 0.0
    )

    compounded_net_pct = (
        (
            np.prod(
                1.0
                + trades["net_return"]
                .to_numpy()
            )
            - 1.0
        )
        * 100
        if trade_count
        else 0.0
    )

    max_dd_pct = max_drawdown_pct(
        trades["net_return"]
    )

    trades["timeframe"] = timeframe

    summary = {
        "timeframe": timeframe,
        "horizon_minutes":
            config[
                "horizon_minutes"
            ],
        "target_threshold_pct":
            target_threshold * 100,
        "confidence_threshold":
            confidence_threshold,
        "feature_count":
            len(feature_columns),
        "train_validation_rows":
            len(train_validation),
        "test_rows_scored":
            len(test_scored),
        "trades": trade_count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "gross_return_pct":
            gross_return_pct,
        "cost_pct":
            total_cost_pct,
        "net_return_pct":
            net_return_pct,
        "avg_net_trade_pct":
            avg_net_trade_pct,
        "median_net_trade_pct":
            median_net_trade_pct,
        "compounded_net_pct":
            compounded_net_pct,
        "max_drawdown_pct":
            max_dd_pct,
    }

    return summary, trades


def main() -> None:
    print(
        "Running ONE-SHOT untouched "
        "test evaluation..."
    )

    print()
    print(
        "Frozen candidates:"
    )

    print(
        "10m: 30m horizon, ±0.20% target, "
        "confidence >= 0.45"
    )

    print(
        "30m: 120m horizon, ±0.40% target, "
        "confidence >= 0.45"
    )

    print(
        "Round-trip cost: 0.10%"
    )

    print()
    print(
        "IMPORTANT: these test results "
        "must not be used for further "
        "hyperparameter tuning."
    )

    encoder = LabelEncoder()

    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    summaries = []
    all_trades = []

    for timeframe, config in (
        CONFIGS.items()
    ):
        print()
        print(
            f"Evaluating frozen "
            f"{timeframe} candidate..."
        )

        summary, trades = (
            evaluate_timeframe(
                timeframe,
                config,
                encoder,
            )
        )

        summaries.append(summary)
        all_trades.append(trades)

    summary_df = pd.DataFrame(
        summaries
    )

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = (
        output_dir
        / "frozen_short_tf_final_test_summary.csv"
    )

    trades_file = (
        output_dir
        / "frozen_short_tf_final_test_trades.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    if all_trades:
        pd.concat(
            all_trades,
            ignore_index=True,
        ).to_csv(
            trades_file,
            index=False,
        )

    print()
    print(
        "=" * 170
    )

    print(
        "FROZEN SHORT-TIMEFRAME "
        "ONE-SHOT FINAL TEST RESULTS"
    )

    print(
        "=" * 170
    )

    print(
        summary_df.to_string(
            index=False,
            formatters={
                "target_threshold_pct":
                    lambda value:
                        f"{value:.2f}%",
                "confidence_threshold":
                    lambda value:
                        f"{value:.2f}",
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
        "FINAL-TEST RULE: "
        "do not tune these candidates "
        "against these results."
    )

    print()
    print("Saved summary:")
    print(summary_file)

    print()
    print("Saved trades:")
    print(trades_file)


if __name__ == "__main__":
    main()
