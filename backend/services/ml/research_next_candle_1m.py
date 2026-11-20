from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
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
    create_lag_features,
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
    / "processed"
    / "btcusdt_1m.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "next_candle_1m_validation.csv"
)

RESEARCH_START = pd.Timestamp(
    "2025-01-01",
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

TARGET_THRESHOLDS = [
    0.0000,
    0.0001,
    0.0002,
    0.0003,
]

CONFIDENCE_THRESHOLDS = [
    0.45,
    0.50,
    0.55,
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
    df = create_lag_features(df)

    df["future_return_1m"] = (
        df["close"].shift(-1)
        / df["close"]
        - 1.0
    )

    return (
        df.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .reset_index(drop=True)
    )


def directional_precision(
    actual: np.ndarray,
    predicted: np.ndarray,
    label: str,
) -> float:
    mask = predicted == label

    if not mask.any():
        return 0.0

    return float(
        (
            actual[mask]
            == label
        ).mean()
    )


def main() -> None:
    print(
        "TRADEMIND - 1M NEXT-CANDLE "
        "PREDICTOR RESEARCH"
    )

    print()
    print(
        "Purpose: estimate the direction of "
        "the NEXT completed 1-minute candle."
    )

    print(
        "This is research only. It does NOT "
        "modify the frozen 10m trading model."
    )

    print(
        "KNOWN_RECENT from 2026-07-01 onward "
        "is excluded from this research run."
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
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Keep enough warm-up history for the
    # longest 240-minute rolling features.
    warmup_start = (
        RESEARCH_START
        - pd.Timedelta(
            minutes=300
        )
    )

    raw = raw[
        (raw["open_time"] >= warmup_start)
        & (
            raw["open_time"]
            < KNOWN_RECENT_START
        )
    ].copy()

    print()
    print(
        f"Raw research rows: "
        f"{len(raw):,}"
    )

    df = build_features(raw)

    df = df[
        df["open_time"]
        >= RESEARCH_START
    ].copy()

    train_df = df[
        df["open_time"]
        < VALIDATION_START
    ].copy()

    validation_df = df[
        (
            df["open_time"]
            >= VALIDATION_START
        )
        & (
            df["open_time"]
            < KNOWN_RECENT_START
        )
    ].copy()

    print(
        f"Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Train: "
        f"{train_df['open_time'].min()} "
        f"-> "
        f"{train_df['open_time'].max()}"
    )

    print(
        f"Validation: "
        f"{validation_df['open_time'].min()} "
        f"-> "
        f"{validation_df['open_time'].max()}"
    )

    feature_columns = [
        column
        for column in train_df.columns
        if column
        not in NON_FEATURE_COLUMNS
    ]

    print(
        f"Feature count: "
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

    for threshold in TARGET_THRESHOLDS:
        train_target = make_target(
            train_df[
                "future_return_1m"
            ],
            threshold,
        )

        validation_target = make_target(
            validation_df[
                "future_return_1m"
            ],
            threshold,
        )

        model = build_model()

        model.fit(
            train_df[
                feature_columns
            ],
            encoder.transform(
                train_target
            ),
        )

        probabilities = (
            model.predict_proba(
                validation_df[
                    feature_columns
                ]
            )
        )

        raw_prediction = (
            encoder.inverse_transform(
                model.predict(
                    validation_df[
                        feature_columns
                    ]
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

            predicted = (
                raw_prediction[
                    active
                ]
            )

            actual_active = (
                actual[
                    active
                ]
            )

            if len(predicted) == 0:
                continue

            directional = np.isin(
                predicted,
                [
                    "UP",
                    "DOWN",
                ],
            )

            directional_pred = (
                predicted[
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

            row = {
                "target_threshold_pct":
                    threshold * 100,
                "confidence_threshold":
                    confidence_threshold,
                "coverage_pct":
                    active.mean() * 100,
                "predictions":
                    int(active.sum()),
                "directional_predictions":
                    int(
                        directional.sum()
                    ),
                "accuracy_pct":
                    accuracy_score(
                        actual_active,
                        predicted,
                    )
                    * 100,
                "balanced_accuracy_pct":
                    balanced_accuracy_score(
                        actual_active,
                        predicted,
                    )
                    * 100,
                "directional_accuracy_pct":
                    directional_accuracy
                    * 100,
                "up_precision_pct":
                    directional_precision(
                        directional_actual,
                        directional_pred,
                        "UP",
                    )
                    * 100,
                "down_precision_pct":
                    directional_precision(
                        directional_actual,
                        directional_pred,
                        "DOWN",
                    )
                    * 100,
            }

            results.append(row)

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df.sort_values(
            by=[
                "directional_accuracy_pct",
                "directional_predictions",
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
    print("=" * 110)
    print(
        "TOP NEXT-CANDLE VALIDATION RESULTS"
    )
    print("=" * 110)

    print(
        results_df.head(
            20
        ).to_string(
            index=False,
            formatters={
                "target_threshold_pct":
                    lambda value:
                        f"{value:.3f}%",
                "confidence_threshold":
                    lambda value:
                        f"{value:.2f}",
                "coverage_pct":
                    lambda value:
                        f"{value:.2f}%",
                "accuracy_pct":
                    lambda value:
                        f"{value:.2f}%",
                "balanced_accuracy_pct":
                    lambda value:
                        f"{value:.2f}%",
                "directional_accuracy_pct":
                    lambda value:
                        f"{value:.2f}%",
                "up_precision_pct":
                    lambda value:
                        f"{value:.2f}%",
                "down_precision_pct":
                    lambda value:
                        f"{value:.2f}%",
            },
        )
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "- directional_accuracy_pct measures "
        "whether predicted UP/DOWN direction "
        "matched the next candle class."
    )

    print(
        "- coverage_pct shows how often the "
        "model was confident enough to speak."
    )

    print(
        "- KNOWN_RECENT is still untouched by "
        "this script."
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
