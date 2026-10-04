from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

V1_SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_1m"
)

V2_SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_1m_v2"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "1m_v1_vs_v2_validation_comparison.csv"
)


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
        pd.DataFrame(
            accepted
        )
        .reset_index(drop=True)
    )


def evaluate_version(
    version: str,
    split_dir: Path,
) -> dict:
    train_df = pd.read_csv(
        split_dir / "train.csv"
    )

    validation_df = pd.read_csv(
        split_dir / "validation.csv"
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
        if column not in NON_FEATURE_COLUMNS
    ]

    train_returns = future_return(
        train_df
    )

    validation_returns = future_return(
        validation_df
    )

    train_mask = (
        train_returns.notna()
    )

    validation_mask = (
        validation_returns.notna()
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
        train_returns.loc[
            train_mask
        ]
    )

    y_validation = create_target(
        validation_returns.loc[
            validation_mask
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
        f"Training {version} "
        f"with {len(feature_columns)} features..."
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

    accuracy = (
        accuracy_score(
            y_validation,
            predictions,
        )
    )

    macro_f1 = (
        f1_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    up_precision = (
        precision_score(
            y_validation == "UP",
            predictions == "UP",
            zero_division=0,
        )
    )

    down_precision = (
        precision_score(
            y_validation == "DOWN",
            predictions == "DOWN",
            zero_division=0,
        )
    )

    rows = (
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

    rows["future_return"] = (
        validation_returns.loc[
            validation_mask
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

    if trades.empty:
        win_rate = 0.0
        gross_return = 0.0
        net_return = 0.0
        avg_net_trade = 0.0
        median_net_trade = 0.0
        max_drawdown = 0.0
        positive_periods = 0
    else:
        win_rate = (
            (
                trades["net_return"]
                > 0
            ).mean()
            * 100
        )

        gross_return = (
            trades[
                "gross_return"
            ].sum()
            * 100
        )

        net_return = (
            trades[
                "net_return"
            ].sum()
            * 100
        )

        avg_net_trade = (
            trades[
                "net_return"
            ].mean()
            * 100
        )

        median_net_trade = (
            trades[
                "net_return"
            ].median()
            * 100
        )

        equity = (
            1
            + trades[
                "net_return"
            ]
        ).cumprod()

        drawdown = (
            equity
            / equity.cummax()
            - 1
        )

        max_drawdown = (
            drawdown.min()
            * 100
        )

        boundaries = np.linspace(
            0,
            len(trades),
            4,
            dtype=int,
        )

        period_returns = []

        for index in range(3):
            part = trades.iloc[
                boundaries[index]:
                boundaries[index + 1]
            ]

            period_returns.append(
                part[
                    "net_return"
                ].sum()
                * 100
            )

        positive_periods = sum(
            value > 0
            for value in period_returns
        )

    return {
        "version": version,
        "feature_count":
            len(feature_columns),
        "accuracy_pct":
            accuracy * 100,
        "macro_f1_pct":
            macro_f1 * 100,
        "up_precision_pct":
            up_precision * 100,
        "down_precision_pct":
            down_precision * 100,
        "trades":
            len(trades),
        "win_rate_pct":
            win_rate,
        "gross_return_pct":
            gross_return,
        "cost_pct":
            len(trades)
            * ROUND_TRIP_COST
            * 100,
        "net_return_pct":
            net_return,
        "average_net_trade_pct":
            avg_net_trade,
        "median_net_trade_pct":
            median_net_trade,
        "positive_periods":
            positive_periods,
        "max_drawdown_pct":
            max_drawdown,
    }


def main() -> None:
    print(
        "Comparing frozen 1m V1 "
        "against chart-reading V2..."
    )

    print(
        "\nFrozen comparison settings:"
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

    results = [
        evaluate_version(
            "V1_BASELINE",
            V1_SPLIT_DIR,
        ),
        evaluate_version(
            "V2_CHART_READING",
            V2_SPLIT_DIR,
        ),
    ]

    results_df = pd.DataFrame(
        results
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "1M V1 VS V2 VALIDATION COMPARISON"
    )

    print(
        "=" * 120
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "accuracy_pct":
                    lambda value:
                        f"{value:.2f}%",
                "macro_f1_pct":
                    lambda value:
                        f"{value:.2f}%",
                "up_precision_pct":
                    lambda value:
                        f"{value:.2f}%",
                "down_precision_pct":
                    lambda value:
                        f"{value:.2f}%",
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
                "average_net_trade_pct":
                    lambda value:
                        f"{value:.4f}%",
                "median_net_trade_pct":
                    lambda value:
                        f"{value:.4f}%",
                "max_drawdown_pct":
                    lambda value:
                        f"{value:.2f}%",
            },
        )
    )

    v1 = results_df.iloc[0]
    v2 = results_df.iloc[1]

    print(
        "\nV2 change vs V1:"
    )

    print(
        f"Net return: "
        f"{v2['net_return_pct'] - v1['net_return_pct']:+.2f} percentage points"
    )

    print(
        f"Average net trade: "
        f"{v2['average_net_trade_pct'] - v1['average_net_trade_pct']:+.4f} percentage points"
    )

    print(
        f"Trade count: "
        f"{int(v2['trades'] - v1['trades']):+d}"
    )

    print(
        f"Positive periods: "
        f"{int(v2['positive_periods'] - v1['positive_periods']):+d}"
    )

    print(
        "\nTEST SPLITS WERE NOT USED."
    )

    print(
        f"\nFull comparison saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
