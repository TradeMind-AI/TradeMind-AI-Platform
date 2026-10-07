from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PIPELINE_DIR = (
    PROJECT_ROOT
    / "backend"
    / "services"
    / "data_pipeline"
)

if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(DATA_PIPELINE_DIR),
    )

from feature_engineering_1m import (
    create_candle_features,
    create_momentum_features,
    create_return_features,
    create_time_features,
    create_trend_features,
    create_volatility_features,
    create_volume_features,
)


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "btcusdt_1m.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "microstructure_short_horizon_1m_validation.csv"
)

RESEARCH_START = pd.Timestamp(
    "2024-01-01",
    tz="UTC",
)

VALIDATION_START = pd.Timestamp(
    "2026-01-01",
    tz="UTC",
)

KNOWN_RECENT_START = pd.Timestamp(
    "2026-07-01",
    tz="UTC",
)

HORIZONS = [
    3,
    5,
]

MOVE_THRESHOLDS = [
    0.0005,
    0.0010,
]

CONFIDENCE_THRESHOLDS = [
    0.45,
    0.50,
    0.55,
]

BASE_FEATURE_COLUMNS = [
    "return_1m",
    "return_3m",
    "return_5m",
    "return_10m",
    "return_20m",
    "return_30m",
    "return_60m",
    "log_return_1m",
    "close_to_sma_20",
    "close_to_sma_60",
    "close_to_sma_240",
    "rsi_14",
    "macd",
    "macd_histogram",
    "rolling_volatility_10",
    "rolling_volatility_30",
    "rolling_volatility_60",
    "atr_14",
    "bollinger_width_20",
    "range_pct",
    "body_pct",
    "close_location",
    "volume_change",
    "relative_volume_30",
    "relative_volume_60",
    "trade_count_change",
    "taker_buy_ratio",
    "taker_buy_imbalance",
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

MICROSTRUCTURE_FEATURE_COLUMNS = [
    "return_accel_1m",
    "return_accel_3m",
    "abs_return_1m",
    "signed_volume",
    "buy_volume",
    "sell_volume",
    "buy_sell_ratio",
    "imbalance_change_1m",
    "imbalance_change_3m",
    "imbalance_mean_3m",
    "imbalance_mean_5m",
    "imbalance_mean_10m",
    "imbalance_slope_3m",
    "imbalance_slope_5m",
    "volume_zscore_20m",
    "trade_count_zscore_20m",
    "volume_burst",
    "trade_burst",
    "range_expansion_5m",
    "body_direction",
    "body_signed_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "upper_wick_pressure_3m",
    "lower_wick_pressure_3m",
    "distance_recent_high_5m",
    "distance_recent_low_5m",
    "distance_recent_high_15m",
    "distance_recent_low_15m",
    "higher_close_count_5m",
    "lower_close_count_5m",
    "up_streak_3m",
    "down_streak_3m",
    "volatility_ratio_10_60",
    "price_volume_pressure_3m",
    "price_volume_pressure_5m",
    "buy_pressure_change_3m",
    "buy_pressure_change_5m",
]

FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + MICROSTRUCTURE_FEATURE_COLUMNS
)


def build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def rolling_zscore(
    series: pd.Series,
    window: int,
) -> pd.Series:
    mean = (
        series.rolling(
            window
        ).mean()
    )

    std = (
        series.rolling(
            window
        ).std()
    )

    return (
        (series - mean)
        / std.replace(
            0,
            np.nan,
        )
    )


def create_microstructure_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df["return_accel_1m"] = (
        df["return_1m"]
        - df["return_1m"].shift(1)
    )

    df["return_accel_3m"] = (
        df["return_1m"]
        - df["return_1m"].shift(3)
    )

    df["abs_return_1m"] = (
        df["return_1m"].abs()
    )

    df["buy_volume"] = (
        df["taker_buy_base_volume"]
    )

    df["sell_volume"] = (
        df["volume"]
        - df["taker_buy_base_volume"]
    ).clip(
        lower=0
    )

    df["signed_volume"] = (
        df["buy_volume"]
        - df["sell_volume"]
    )

    df["buy_sell_ratio"] = (
        df["buy_volume"]
        / df["sell_volume"].replace(
            0,
            np.nan,
        )
    )

    df["imbalance_change_1m"] = (
        df["taker_buy_imbalance"]
        .diff(1)
    )

    df["imbalance_change_3m"] = (
        df["taker_buy_imbalance"]
        .diff(3)
    )

    for window in [
        3,
        5,
        10,
    ]:
        df[
            f"imbalance_mean_{window}m"
        ] = (
            df["taker_buy_imbalance"]
            .rolling(window)
            .mean()
        )

    df["imbalance_slope_3m"] = (
        df["imbalance_mean_3m"]
        - df["imbalance_mean_3m"]
        .shift(3)
    )

    df["imbalance_slope_5m"] = (
        df["imbalance_mean_5m"]
        - df["imbalance_mean_5m"]
        .shift(5)
    )

    df["volume_zscore_20m"] = (
        rolling_zscore(
            df["volume"],
            20,
        )
    )

    df["trade_count_zscore_20m"] = (
        rolling_zscore(
            df["number_of_trades"],
            20,
        )
    )

    df["volume_burst"] = (
        df["volume"]
        / df["volume"]
        .rolling(20)
        .median()
        .replace(
            0,
            np.nan,
        )
    )

    df["trade_burst"] = (
        df["number_of_trades"]
        / df["number_of_trades"]
        .rolling(20)
        .median()
        .replace(
            0,
            np.nan,
        )
    )

    df["range_expansion_5m"] = (
        df["range_pct"]
        / df["range_pct"]
        .rolling(5)
        .mean()
        .replace(
            0,
            np.nan,
        )
    )

    df["body_direction"] = np.sign(
        df["close"]
        - df["open"]
    )

    df["body_signed_pct"] = (
        (
            df["close"]
            - df["open"]
        )
        / df["open"]
    )

    df["upper_wick_pct"] = (
        df["upper_wick"]
        / df["open"]
    )

    df["lower_wick_pct"] = (
        df["lower_wick"]
        / df["open"]
    )

    df["upper_wick_pressure_3m"] = (
        df["upper_wick_pct"]
        .rolling(3)
        .mean()
    )

    df["lower_wick_pressure_3m"] = (
        df["lower_wick_pct"]
        .rolling(3)
        .mean()
    )

    for window in [
        5,
        15,
    ]:
        recent_high = (
            df["high"]
            .rolling(window)
            .max()
        )

        recent_low = (
            df["low"]
            .rolling(window)
            .min()
        )

        df[
            f"distance_recent_high_{window}m"
        ] = (
            df["close"]
            / recent_high
            - 1.0
        )

        df[
            f"distance_recent_low_{window}m"
        ] = (
            df["close"]
            / recent_low
            - 1.0
        )

    higher_close = (
        df["close"]
        > df["close"].shift(1)
    ).astype(float)

    lower_close = (
        df["close"]
        < df["close"].shift(1)
    ).astype(float)

    df["higher_close_count_5m"] = (
        higher_close
        .rolling(5)
        .sum()
    )

    df["lower_close_count_5m"] = (
        lower_close
        .rolling(5)
        .sum()
    )

    df["up_streak_3m"] = (
        higher_close
        .rolling(3)
        .sum()
    )

    df["down_streak_3m"] = (
        lower_close
        .rolling(3)
        .sum()
    )

    df["volatility_ratio_10_60"] = (
        df["rolling_volatility_10"]
        / df["rolling_volatility_60"]
        .replace(
            0,
            np.nan,
        )
    )

    signed_return = (
        df["return_1m"]
        * df["volume"]
    )

    df["price_volume_pressure_3m"] = (
        signed_return
        .rolling(3)
        .sum()
    )

    df["price_volume_pressure_5m"] = (
        signed_return
        .rolling(5)
        .sum()
    )

    df["buy_pressure_change_3m"] = (
        df["taker_buy_ratio"]
        - df["taker_buy_ratio"]
        .shift(3)
    )

    df["buy_pressure_change_5m"] = (
        df["taker_buy_ratio"]
        - df["taker_buy_ratio"]
        .shift(5)
    )

    return df


def build_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    df = raw.copy()

    df = create_return_features(df)
    df = create_trend_features(df)
    df = create_momentum_features(df)
    df = create_volatility_features(df)
    df = create_candle_features(df)
    df = create_volume_features(df)
    df = create_time_features(df)
    df = create_microstructure_features(df)

    return (
        df.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=FEATURE_COLUMNS
        )
        .reset_index(drop=True)
    )


def make_target(
    future_return: pd.Series,
    threshold: float,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                future_return
                < -threshold,
                future_return
                > threshold,
            ],
            [
                "DOWN",
                "UP",
            ],
            default="NEUTRAL",
        ),
        index=future_return.index,
    )


def class_metric(
    actual: np.ndarray,
    predicted: np.ndarray,
    label: str,
) -> tuple[float, float, float]:
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=[label],
            zero_division=0,
        )
    )

    return (
        float(precision[0]),
        float(recall[0]),
        float(f1[0]),
    )


def main() -> None:
    print(
        "TRADEMIND - MICROSTRUCTURE "
        "SHORT-HORIZON 1M RESEARCH"
    )

    print()
    print(
        "Goal: predict meaningful BTC moves "
        "over the next 3-5 minutes instead "
        "of guessing every 1m candle color."
    )

    print(
        "The frozen 10m paper model is "
        "NOT changed."
    )

    print(
        "2026-07-01 onward remains excluded "
        "for a later one-shot check."
    )

    raw = pd.read_csv(
        INPUT_FILE
    )

    raw["open_time"] = pd.to_datetime(
        raw["open_time"],
        utc=True,
        errors="coerce",
    )

    raw["close_time"] = pd.to_datetime(
        raw["close_time"],
        utc=True,
        errors="coerce",
    )

    raw = (
        raw.dropna(
            subset=["open_time"]
        )
        .sort_values(
            "open_time"
        )
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    warmup_start = (
        RESEARCH_START
        - pd.Timedelta(
            minutes=300
        )
    )

    raw = raw[
        (
            raw["open_time"]
            >= warmup_start
        )
        & (
            raw["open_time"]
            < KNOWN_RECENT_START
        )
    ].copy()

    print()
    print(
        f"Raw rows: "
        f"{len(raw):,}"
    )

    df = build_features(
        raw
    )

    df = df[
        df["open_time"]
        >= RESEARCH_START
    ].reset_index(drop=True)

    print(
        f"ML-ready rows: "
        f"{len(df):,}"
    )

    print(
        f"Feature count: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Period: "
        f"{df['open_time'].min()} "
        f"-> "
        f"{df['open_time'].max()}"
    )

    train_mask = (
        df["open_time"]
        < VALIDATION_START
    )

    validation_mask = (
        (
            df["open_time"]
            >= VALIDATION_START
        )
        & (
            df["open_time"]
            < KNOWN_RECENT_START
        )
    )

    print(
        f"Train rows: "
        f"{int(train_mask.sum()):,}"
    )

    print(
        f"Validation rows: "
        f"{int(validation_mask.sum()):,}"
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

    for horizon in HORIZONS:
        future_return = (
            df["close"]
            .shift(-horizon)
            / df["close"]
            - 1.0
        )

        usable = (
            future_return.notna()
        )

        horizon_train = (
            train_mask
            & usable
        )

        horizon_validation = (
            validation_mask
            & usable
        )

        for move_threshold in (
            MOVE_THRESHOLDS
        ):
            print()
            print(
                f"Training horizon={horizon}m "
                f"threshold="
                f"{move_threshold * 100:.2f}%"
            )

            target = make_target(
                future_return,
                move_threshold,
            )

            train_target = (
                target.loc[
                    horizon_train
                ]
            )

            validation_target = (
                target.loc[
                    horizon_validation
                ]
            )

            distribution = (
                validation_target
                .value_counts(
                    normalize=True
                )
                .to_dict()
            )

            model = build_model()

            model.fit(
                df.loc[
                    horizon_train,
                    FEATURE_COLUMNS,
                ],
                encoder.transform(
                    train_target
                ),
            )

            probabilities = (
                model.predict_proba(
                    df.loc[
                        horizon_validation,
                        FEATURE_COLUMNS,
                    ]
                )
            )

            raw_prediction = (
                encoder.inverse_transform(
                    probabilities.argmax(
                        axis=1
                    )
                )
            )

            confidence = (
                probabilities.max(
                    axis=1
                )
            )

            actual = (
                validation_target
                .to_numpy()
            )

            for confidence_threshold in (
                CONFIDENCE_THRESHOLDS
            ):
                active = (
                    confidence
                    >= confidence_threshold
                )

                if not active.any():
                    continue

                predicted_active = (
                    raw_prediction[
                        active
                    ]
                )

                actual_active = (
                    actual[
                        active
                    ]
                )

                directional = np.isin(
                    predicted_active,
                    [
                        "UP",
                        "DOWN",
                    ],
                )

                directional_pred = (
                    predicted_active[
                        directional
                    ]
                )

                directional_actual = (
                    actual_active[
                        directional
                    ]
                )

                directional_accuracy = (
                    float(
                        (
                            directional_pred
                            == directional_actual
                        ).mean()
                    )
                    if len(
                        directional_pred
                    )
                    else 0.0
                )

                up_precision, up_recall, up_f1 = (
                    class_metric(
                        directional_actual,
                        directional_pred,
                        "UP",
                    )
                    if len(
                        directional_pred
                    )
                    else (
                        0.0,
                        0.0,
                        0.0,
                    )
                )

                down_precision, down_recall, down_f1 = (
                    class_metric(
                        directional_actual,
                        directional_pred,
                        "DOWN",
                    )
                    if len(
                        directional_pred
                    )
                    else (
                        0.0,
                        0.0,
                        0.0,
                    )
                )

                results.append(
                    {
                        "horizon_minutes":
                            horizon,
                        "move_threshold_pct":
                            move_threshold
                            * 100,
                        "confidence_threshold":
                            confidence_threshold,
                        "coverage_pct":
                            active.mean()
                            * 100,
                        "predictions":
                            int(
                                active.sum()
                            ),
                        "directional_predictions":
                            int(
                                directional.sum()
                            ),
                        "directional_coverage_pct":
                            directional.mean()
                            * active.mean()
                            * 100,
                        "accuracy_pct":
                            accuracy_score(
                                actual_active,
                                predicted_active,
                            )
                            * 100,
                        "balanced_accuracy_pct":
                            balanced_accuracy_score(
                                actual_active,
                                predicted_active,
                            )
                            * 100,
                        "directional_accuracy_pct":
                            directional_accuracy
                            * 100,
                        "up_precision_pct":
                            up_precision
                            * 100,
                        "up_recall_pct":
                            up_recall
                            * 100,
                        "up_f1_pct":
                            up_f1
                            * 100,
                        "down_precision_pct":
                            down_precision
                            * 100,
                        "down_recall_pct":
                            down_recall
                            * 100,
                        "down_f1_pct":
                            down_f1
                            * 100,
                        "validation_up_pct":
                            distribution.get(
                                "UP",
                                0.0,
                            )
                            * 100,
                        "validation_down_pct":
                            distribution.get(
                                "DOWN",
                                0.0,
                            )
                            * 100,
                        "validation_neutral_pct":
                            distribution.get(
                                "NEUTRAL",
                                0.0,
                            )
                            * 100,
                    }
                )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df.sort_values(
            by=[
                "directional_accuracy_pct",
                "directional_coverage_pct",
            ],
            ascending=[
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
    print("=" * 165)
    print(
        "TOP MICROSTRUCTURE SHORT-HORIZON "
        "VALIDATION RESULTS"
    )
    print("=" * 165)

    display_columns = [
        "horizon_minutes",
        "move_threshold_pct",
        "confidence_threshold",
        "coverage_pct",
        "directional_coverage_pct",
        "directional_predictions",
        "directional_accuracy_pct",
        "balanced_accuracy_pct",
        "up_precision_pct",
        "down_precision_pct",
        "validation_neutral_pct",
    ]

    print(
        results_df.head(
            20
        )[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "move_threshold_pct":
                    lambda value:
                        f"{value:.2f}%",
                "confidence_threshold":
                    lambda value:
                        f"{value:.2f}",
                "coverage_pct":
                    lambda value:
                        f"{value:.2f}%",
                "directional_coverage_pct":
                    lambda value:
                        f"{value:.2f}%",
                "directional_accuracy_pct":
                    lambda value:
                        f"{value:.2f}%",
                "balanced_accuracy_pct":
                    lambda value:
                        f"{value:.2f}%",
                "up_precision_pct":
                    lambda value:
                        f"{value:.2f}%",
                "down_precision_pct":
                    lambda value:
                        f"{value:.2f}%",
                "validation_neutral_pct":
                    lambda value:
                        f"{value:.2f}%",
            },
        )
    )

    print()
    print(
        "What to look for:"
    )

    print(
        "- Directional accuracy meaningfully "
        "above 50%."
    )

    print(
        "- Useful directional coverage, not "
        "a model that almost always says "
        "NEUTRAL."
    )

    print(
        "- Similar UP and DOWN precision."
    )

    print(
        "- Balanced accuracy that confirms "
        "the result is not class imbalance."
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print(
        "2026-07-01 onward was NOT loaded "
        "into validation."
    )


if __name__ == "__main__":
    main()
