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
    / "1m_multi_horizon_tuning_results.csv"
)


FORECAST_HORIZONS = [
    3,
    5,
    10,
]

TARGET_THRESHOLDS = [
    0.00050,   # 0.05%
    0.00075,   # 0.075%
    0.00100,   # 0.10%
    0.00150,   # 0.15%
    0.00200,   # 0.20%
]

CONFIDENCE_THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

ROUND_TRIP_COSTS = [
    0.00050,   # 0.05%
    0.00075,   # 0.075%
    0.00100,   # 0.10%
    0.00125,   # 0.125%
    0.00150,   # 0.15%
]

MIN_TRADES_FOR_ROBUST = 50


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
    horizon_minutes: int,
) -> pd.Series:
    return (
        df["close"].shift(
            -horizon_minutes
        )
        / df["close"]
        - 1
    )


def create_target(
    future_return: pd.Series,
    threshold: float,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                future_return < -threshold,
                future_return > threshold,
            ],
            [
                "DOWN",
                "UP",
            ],
            default="NEUTRAL",
        ),
        index=future_return.index,
    )


def build_trade_candidates(
    validation_df: pd.DataFrame,
    future_return: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    encoder: LabelEncoder,
    confidence_threshold: float,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "open_time":
                validation_df[
                    "open_time"
                ].to_numpy(),

            "future_return":
                future_return
                .to_numpy(),

            "prediction":
                predictions,
        }
    )

    class_index = {
        name: index
        for index, name
        in enumerate(
            encoder.classes_
        )
    }

    result["probability_down"] = (
        probabilities[
            :,
            class_index["DOWN"]
        ]
    )

    result["probability_neutral"] = (
        probabilities[
            :,
            class_index["NEUTRAL"]
        ]
    )

    result["probability_up"] = (
        probabilities[
            :,
            class_index["UP"]
        ]
    )

    result["confidence"] = (
        probabilities.max(axis=1)
    )

    directional = (
        result["prediction"]
        .isin(
            [
                "UP",
                "DOWN",
            ]
        )
    )

    confident = (
        result["confidence"]
        >= confidence_threshold
    )

    candidates = result[
        directional
        & confident
    ].copy()

    candidates["row_index"] = (
        candidates.index
    )

    candidates = candidates.dropna(
        subset=[
            "future_return",
        ]
    )

    candidates["gross_return"] = np.where(
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

    return candidates


def remove_overlapping_trades(
    candidates: pd.DataFrame,
    horizon_minutes: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    accepted_rows = []

    next_available_index = -1

    for _, row in (
        candidates
        .sort_values("row_index")
        .iterrows()
    ):
        row_index = int(
            row["row_index"]
        )

        if row_index < next_available_index:
            continue

        accepted_rows.append(
            row
        )

        next_available_index = (
            row_index
            + horizon_minutes
        )

    if not accepted_rows:
        return (
            candidates
            .iloc[0:0]
            .copy()
        )

    return (
        pd.DataFrame(
            accepted_rows
        )
        .reset_index(drop=True)
    )


def evaluate_cost(
    trades: pd.DataFrame,
    round_trip_cost: float,
) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "gross_return_pct": 0.0,
            "cost_pct": 0.0,
            "net_return_pct": 0.0,
            "average_net_trade_pct": 0.0,
            "median_net_trade_pct": 0.0,
        }

    net_return = (
        trades["gross_return"]
        - round_trip_cost
    )

    return {
        "trades":
            len(trades),

        "win_rate_pct":
            (
                net_return > 0
            ).mean()
            * 100,

        "gross_return_pct":
            trades[
                "gross_return"
            ].sum()
            * 100,

        "cost_pct":
            len(trades)
            * round_trip_cost
            * 100,

        "net_return_pct":
            net_return.sum()
            * 100,

        "average_net_trade_pct":
            net_return.mean()
            * 100,

        "median_net_trade_pct":
            net_return.median()
            * 100,
    }


def period_robustness(
    trades: pd.DataFrame,
    round_trip_cost: float,
) -> dict:
    if trades.empty:
        return {
            "positive_periods": 0,
            "period_1_net_pct": 0.0,
            "period_2_net_pct": 0.0,
            "period_3_net_pct": 0.0,
        }

    temp = trades.copy()

    temp["open_time"] = pd.to_datetime(
        temp["open_time"],
        utc=True,
    )

    temp = (
        temp
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    boundaries = np.linspace(
        0,
        len(temp),
        4,
        dtype=int,
    )

    period_returns = []

    for index in range(3):
        start = boundaries[index]
        end = boundaries[index + 1]

        part = temp.iloc[
            start:end
        ]

        if part.empty:
            period_returns.append(
                0.0
            )
            continue

        net = (
            part["gross_return"]
            - round_trip_cost
        )

        period_returns.append(
            net.sum() * 100
        )

    return {
        "positive_periods":
            sum(
                value > 0
                for value in period_returns
            ),

        "period_1_net_pct":
            period_returns[0],

        "period_2_net_pct":
            period_returns[1],

        "period_3_net_pct":
            period_returns[2],
    }


def main() -> None:
    print(
        "Loading 1-minute train "
        "and validation data..."
    )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X_train_full = (
        train_df[
            feature_columns
        ]
    )

    X_validation_full = (
        validation_df[
            feature_columns
        ]
    )

    print(
        f"Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    encoder = LabelEncoder()

    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    results = []

    for horizon in FORECAST_HORIZONS:
        print(
            "\n"
            + "#" * 90
        )

        print(
            f"FORECAST HORIZON: "
            f"{horizon} MINUTES"
        )

        print(
            "#" * 90
        )

        train_future_return = (
            create_future_return(
                train_df,
                horizon,
            )
        )

        validation_future_return = (
            create_future_return(
                validation_df,
                horizon,
            )
        )

        train_valid_mask = (
            train_future_return.notna()
        )

        validation_valid_mask = (
            validation_future_return.notna()
        )

        X_train = (
            X_train_full.loc[
                train_valid_mask
            ]
        )

        X_validation = (
            X_validation_full.loc[
                validation_valid_mask
            ]
        )

        train_returns = (
            train_future_return.loc[
                train_valid_mask
            ]
        )

        validation_returns = (
            validation_future_return.loc[
                validation_valid_mask
            ]
        )

        validation_for_results = (
            validation_df.loc[
                validation_valid_mask
            ]
        )

        for target_threshold in (
            TARGET_THRESHOLDS
        ):
            print(
                "\n"
                + "=" * 80
            )

            print(
                "Training target threshold "
                f"±{target_threshold * 100:.3f}%"
            )

            y_train = create_target(
                train_returns,
                target_threshold,
            )

            y_validation = create_target(
                validation_returns,
                target_threshold,
            )

            print(
                "Train target distribution:"
            )

            print(
                (
                    y_train
                    .value_counts(
                        normalize=True
                    )
                    .sort_index()
                    * 100
                )
                .round(2)
                .to_string()
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

            model.fit(
                X_train,
                y_train_encoded,
            )

            prediction_encoded = (
                model.predict(
                    X_validation
                )
            )

            probabilities = (
                model.predict_proba(
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

            for confidence_threshold in (
                CONFIDENCE_THRESHOLDS
            ):
                candidates = (
                    build_trade_candidates(
                        validation_for_results,
                        validation_returns,
                        predictions,
                        probabilities,
                        encoder,
                        confidence_threshold,
                    )
                )

                trades = (
                    remove_overlapping_trades(
                        candidates,
                        horizon,
                    )
                )

                for round_trip_cost in (
                    ROUND_TRIP_COSTS
                ):
                    trading = (
                        evaluate_cost(
                            trades,
                            round_trip_cost,
                        )
                    )

                    robustness = (
                        period_robustness(
                            trades,
                            round_trip_cost,
                        )
                    )

                    robust_candidate = (
                        trading["trades"]
                        >= MIN_TRADES_FOR_ROBUST
                        and trading[
                            "net_return_pct"
                        ] > 0
                        and trading[
                            "average_net_trade_pct"
                        ] > 0
                        and robustness[
                            "positive_periods"
                        ] >= 2
                    )

                    results.append(
                        {
                            "horizon_minutes":
                                horizon,

                            "target_threshold_pct":
                                target_threshold
                                * 100,

                            "confidence_threshold":
                                confidence_threshold,

                            "round_trip_cost_pct":
                                round_trip_cost
                                * 100,

                            "accuracy_pct":
                                accuracy
                                * 100,

                            "macro_f1_pct":
                                macro_f1
                                * 100,

                            "up_precision_pct":
                                up_precision
                                * 100,

                            "down_precision_pct":
                                down_precision
                                * 100,

                            **trading,
                            **robustness,

                            "robust_candidate":
                                robust_candidate,
                        }
                    )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by=[
                "robust_candidate",
                "net_return_pct",
                "average_net_trade_pct",
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

    print(
        "\n"
        + "=" * 150
    )

    print(
        "TOP MULTI-HORIZON "
        "1-MINUTE VALIDATION RESULTS"
    )

    print(
        "=" * 150
    )

    print(
        results_df
        .head(40)
        .to_string(
            index=False,
            formatters={
                "target_threshold_pct":
                    lambda value:
                        f"{value:.3f}%",

                "confidence_threshold":
                    lambda value:
                        f"{value:.2f}",

                "round_trip_cost_pct":
                    lambda value:
                        f"{value:.3f}%",

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

                "period_1_net_pct":
                    lambda value:
                        f"{value:.2f}%",

                "period_2_net_pct":
                    lambda value:
                        f"{value:.2f}%",

                "period_3_net_pct":
                    lambda value:
                        f"{value:.2f}%",
            },
        )
    )

    robust = results_df[
        results_df[
            "robust_candidate"
        ]
    ]

    print(
        f"\nRobust candidates: "
        f"{len(robust)}"
    )

    if not robust.empty:
        print(
            "\nBest robust candidates:"
        )

        print(
            robust
            .head(20)
            .to_string(
                index=False
            )
        )

    print(
        "\nTEST SPLIT WAS NOT USED."
    )

    print(
        f"\nFull results saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
