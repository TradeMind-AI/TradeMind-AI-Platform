from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ai_rsi_threshold_sweep"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "rsi_threshold_results.csv"
)

TRADE_FILE = (
    OUTPUT_DIR
    / "rsi_threshold_trades.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "summary.json"
)

RESEARCH_START = pd.Timestamp(
    "2023-03-26",
    tz="UTC",
)

VALIDATION_START = pd.Timestamp(
    "2025-01-01",
    tz="UTC",
)

TEST_START = pd.Timestamp(
    "2025-10-01",
    tz="UTC",
)

HORIZON_BARS = 6
CONFIDENCE_THRESHOLD = 0.55
TARGET_THRESHOLD = 0.0040

COSTS = {
    "low": 0.0006,
    "mid": 0.0014,
    "high": 0.0027,
}

LONG_THRESHOLDS = [
    25.0,
    30.0,
    35.0,
    40.0,
]

SHORT_THRESHOLDS = [
    75.0,
    70.0,
    65.0,
    60.0,
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
    "future_return_10m",
    "target",
}


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


def add_rsi(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    delta = (
        result["close"]
        .diff()
    )

    gain = delta.clip(
        lower=0.0,
    )

    loss = (
        -delta.clip(
            upper=0.0,
        )
    )

    average_gain = gain.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=14,
    ).mean()

    average_loss = loss.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = (
        average_gain
        / average_loss.replace(
            0.0,
            np.nan,
        )
    )

    result["rsi14_filter"] = (
        100.0
        - 100.0
        / (
            1.0
            + rs
        )
    )

    return result


def train_model(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    XGBClassifier,
    LabelEncoder,
]:
    train = df[
        df["open_time"]
        < VALIDATION_START
    ].copy()

    future_return = (
        train["close"].shift(
            -HORIZON_BARS
        )
        / train["close"]
        - 1.0
    )

    mask = future_return.notna()

    X = train.loc[
        mask,
        feature_columns,
    ]

    y_text = make_target(
        future_return.loc[
            mask
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

    y = encoder.transform(
        y_text
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
        X,
        y,
    )

    return (
        model,
        encoder,
    )


def add_predictions(
    df: pd.DataFrame,
    feature_columns: list[str],
    model: XGBClassifier,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    result = df.copy()

    mask = (
        result["open_time"]
        >= VALIDATION_START
    )

    X = result.loc[
        mask,
        feature_columns,
    ]

    probabilities = (
        model.predict_proba(
            X
        )
    )

    encoded = model.predict(
        X
    )

    labels = (
        encoder.inverse_transform(
            encoded
        )
    )

    result["ai_prediction"] = pd.NA
    result["ai_confidence"] = np.nan

    result.loc[
        mask,
        "ai_prediction",
    ] = labels

    result.loc[
        mask,
        "ai_confidence",
    ] = probabilities.max(
        axis=1
    )

    return result


def filter_allows(
    rsi_value: float,
    direction: int,
    long_threshold: float,
    short_threshold: float,
) -> bool:
    if pd.isna(
        rsi_value
    ):
        return False

    if direction == 1:
        return (
            float(rsi_value)
            <= long_threshold
        )

    return (
        float(rsi_value)
        >= short_threshold
    )


def simulate(
    df: pd.DataFrame,
    period_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    long_threshold: float | None,
    short_threshold: float | None,
    label: str,
) -> list[dict]:
    trades = []

    index = 0

    while (
        index
        < len(df)
        - HORIZON_BARS
        - 1
    ):
        row = df.iloc[index]

        signal_time = row[
            "open_time"
        ]

        if signal_time < start:
            index += 1
            continue

        if (
            end is not None
            and signal_time >= end
        ):
            break

        prediction = row[
            "ai_prediction"
        ]

        confidence = row[
            "ai_confidence"
        ]

        if (
            pd.isna(prediction)
            or pd.isna(confidence)
            or float(confidence)
            < CONFIDENCE_THRESHOLD
            or prediction
            not in {
                "UP",
                "DOWN",
            }
        ):
            index += 1
            continue

        direction = (
            1
            if prediction == "UP"
            else -1
        )

        if (
            long_threshold
            is not None
            and short_threshold
            is not None
            and not filter_allows(
                row[
                    "rsi14_filter"
                ],
                direction,
                long_threshold,
                short_threshold,
            )
        ):
            index += 1
            continue

        entry_index = (
            index + 1
        )

        exit_index = (
            index
            + HORIZON_BARS
        )

        if exit_index >= len(df):
            break

        entry_row = df.iloc[
            entry_index
        ]

        exit_row = df.iloc[
            exit_index
        ]

        if (
            end is not None
            and entry_row[
                "open_time"
            ] >= end
        ):
            break

        entry_price = float(
            entry_row["open"]
        )

        exit_price = float(
            exit_row["close"]
        )

        gross_return = (
            (
                exit_price
                / entry_price
                - 1.0
            )
            * direction
        )

        trade = {
            "rule":
                label,
            "period":
                period_name,
            "signal_time":
                signal_time,
            "entry_time":
                entry_row[
                    "open_time"
                ],
            "exit_time":
                exit_row[
                    "open_time"
                ],
            "prediction":
                prediction,
            "confidence":
                float(
                    confidence
                ),
            "direction":
                (
                    "LONG"
                    if direction == 1
                    else "SHORT"
                ),
            "rsi14":
                float(
                    row[
                        "rsi14_filter"
                    ]
                ),
            "entry_price":
                entry_price,
            "exit_price":
                exit_price,
            "gross_return_pct":
                gross_return
                * 100.0,
        }

        for (
            cost_name,
            cost,
        ) in COSTS.items():
            trade[
                f"net_{cost_name}_pct"
            ] = (
                gross_return
                - cost
            ) * 100.0

        trades.append(
            trade
        )

        index = exit_index + 1

    return trades


def summarize(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if trades.empty:
        return pd.DataFrame()

    for (
        rule,
        period,
    ), subset in trades.groupby(
        [
            "rule",
            "period",
        ],
        sort=False,
    ):
        row = {
            "rule":
                rule,
            "period":
                period,
            "trades":
                int(
                    len(
                        subset
                    )
                ),
            "win_rate_mid_pct":
                float(
                    (
                        subset[
                            "net_mid_pct"
                        ]
                        > 0.0
                    ).mean()
                    * 100.0
                ),
            "avg_net_mid_pct":
                float(
                    subset[
                        "net_mid_pct"
                    ].mean()
                ),
            "net_mid_sum_pct":
                float(
                    subset[
                        "net_mid_pct"
                    ].sum()
                ),
            "avg_net_high_pct":
                float(
                    subset[
                        "net_high_pct"
                    ].mean()
                ),
            "net_high_sum_pct":
                float(
                    subset[
                        "net_high_pct"
                    ].sum()
                ),
        }

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    for period in [
        "validation",
        "test",
    ]:
        baseline = result[
            (
                result["rule"]
                == "AI baseline"
            )
            & (
                result["period"]
                == period
            )
        ]

        if baseline.empty:
            continue

        base_avg = float(
            baseline.iloc[0][
                "avg_net_mid_pct"
            ]
        )

        base_win = float(
            baseline.iloc[0][
                "win_rate_mid_pct"
            ]
        )

        mask = (
            result["period"]
            == period
        )

        result.loc[
            mask,
            "avg_mid_improvement_vs_baseline_pct_points",
        ] = (
            result.loc[
                mask,
                "avg_net_mid_pct",
            ]
            - base_avg
        )

        result.loc[
            mask,
            "win_rate_improvement_vs_baseline_pct_points",
        ] = (
            result.loc[
                mask,
                "win_rate_mid_pct",
            ]
            - base_win
        )

    return result


def threshold_name(
    long_threshold: float,
    short_threshold: float,
) -> str:
    return (
        "AI + RSI "
        f"{int(long_threshold)}/"
        f"{int(short_threshold)}"
    )


def main() -> None:
    print(
        "TRADEMIND - AI RSI THRESHOLD SWEEP"
    )
    print()

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing feature file: {FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
        errors="coerce",
    )

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "open_time",
                *required,
            ]
        )
        .sort_values(
            "open_time"
        )
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    df = df[
        df["open_time"]
        >= RESEARCH_START
    ].copy().reset_index(
        drop=True
    )

    df = add_rsi(
        df
    )

    feature_columns = [
        column
        for column in df.columns
        if (
            column
            not in NON_FEATURE_COLUMNS
            and column
            != "rsi14_filter"
            and pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]

    model_rows = df[
        [
            "open_time",
            *feature_columns,
        ]
    ].dropna()

    valid_times = set(
        model_rows[
            "open_time"
        ]
    )

    df = df[
        df["open_time"]
        .isin(
            valid_times
        )
    ].copy().reset_index(
        drop=True
    )

    model, encoder = train_model(
        df,
        feature_columns,
    )

    df = add_predictions(
        df,
        feature_columns,
        model,
        encoder,
    )

    all_trades = []

    for period_name, start, end in [
        (
            "validation",
            VALIDATION_START,
            TEST_START,
        ),
        (
            "test",
            TEST_START,
            None,
        ),
    ]:
        all_trades.extend(
            simulate(
                df,
                period_name,
                start,
                end,
                None,
                None,
                "AI baseline",
            )
        )

    for (
        long_threshold,
        short_threshold,
    ) in zip(
        LONG_THRESHOLDS,
        SHORT_THRESHOLDS,
    ):
        label = threshold_name(
            long_threshold,
            short_threshold,
        )

        for period_name, start, end in [
            (
                "validation",
                VALIDATION_START,
                TEST_START,
            ),
            (
                "test",
                TEST_START,
                None,
            ),
        ]:
            all_trades.extend(
                simulate(
                    df,
                    period_name,
                    start,
                    end,
                    long_threshold,
                    short_threshold,
                    label,
                )
            )

    trades = pd.DataFrame(
        all_trades
    )

    results = summarize(
        trades
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        TRADE_FILE,
        index=False,
    )

    results.to_csv(
        RESULT_FILE,
        index=False,
    )

    validation = results[
        results["period"]
        == "validation"
    ].copy()

    candidates = validation[
        validation["rule"]
        != "AI baseline"
    ].copy()

    candidates = candidates[
        candidates["trades"]
        >= 10
    ].copy()

    if not candidates.empty:
        candidates = (
            candidates.sort_values(
                [
                    "avg_net_mid_pct",
                    "win_rate_mid_pct",
                ],
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        validation_winner = (
            candidates.iloc[0][
                "rule"
            ]
        )
    else:
        validation_winner = None

    summary = {
        "purpose":
            "Check whether the promising RSI post-model filter is stable around nearby thresholds.",
        "timeframe":
            "10m",
        "horizon_minutes":
            60,
        "confidence_threshold":
            CONFIDENCE_THRESHOLD,
        "target_threshold_pct":
            TARGET_THRESHOLD
            * 100.0,
        "training_end":
            str(
                VALIDATION_START
            ),
        "validation_start":
            str(
                VALIDATION_START
            ),
        "test_start":
            str(
                TEST_START
            ),
        "thresholds": [
            {
                "long_rsi_max":
                    long_value,
                "short_rsi_min":
                    short_value,
            }
            for (
                long_value,
                short_value,
            )
            in zip(
                LONG_THRESHOLDS,
                SHORT_THRESHOLDS,
            )
        ],
        "selection_rule":
            "Choose on validation only, require at least 10 validation trades, then inspect the corresponding test result.",
        "validation_winner":
            validation_winner,
        "important_note":
            "Research only. This does not modify the existing frozen forward candidate.",
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    columns = [
        "rule",
        "period",
        "trades",
        "win_rate_mid_pct",
        "avg_net_mid_pct",
        "net_mid_sum_pct",
        "avg_mid_improvement_vs_baseline_pct_points",
        "win_rate_improvement_vs_baseline_pct_points",
        "net_high_sum_pct",
    ]

    print(
        results[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Validation-selected candidate:"
    )
    print(
        validation_winner
        if validation_winner
        else "None"
    )

    print()
    print(
        "Do not choose a threshold from the test set."
    )

    print(
        "This script does NOT modify the frozen forward model."
    )

    print()
    print(
        "Saved:"
    )
    print(
        RESULT_FILE
    )
    print(
        TRADE_FILE
    )
    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()
