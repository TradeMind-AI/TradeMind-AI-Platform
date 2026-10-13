from __future__ import annotations

from dataclasses import dataclass
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
    / "ai_technical_filter_benchmark"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "filter_results.csv"
)

TRADE_FILE = (
    OUTPUT_DIR
    / "filter_trades.csv"
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


@dataclass(frozen=True)
class FilterRule:
    key: str
    name: str


FILTERS = [
    FilterRule(
        key="baseline",
        name="AI baseline",
    ),
    FilterRule(
        key="ema_alignment",
        name="AI + EMA alignment",
    ),
    FilterRule(
        key="macd_alignment",
        name="AI + MACD alignment",
    ),
    FilterRule(
        key="rsi_40_60",
        name="AI + RSI 40/60",
    ),
    FilterRule(
        key="rsi_30_70",
        name="AI + RSI 30/70",
    ),
    FilterRule(
        key="bollinger_reversion",
        name="AI + Bollinger reversion",
    ),
    FilterRule(
        key="ema_macd",
        name="AI + EMA + MACD",
    ),
    FilterRule(
        key="ema_rsi",
        name="AI + EMA + RSI 40/60",
    ),
]


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


def ema(
    series: pd.Series,
    span: int,
) -> pd.Series:
    return series.ewm(
        span=span,
        adjust=False,
        min_periods=span,
    ).mean()


def add_technical_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    close = result["close"]

    result["ema20_filter"] = ema(
        close,
        20,
    )

    delta = close.diff()

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

    sma20 = close.rolling(
        20,
        min_periods=20,
    ).mean()

    std20 = close.rolling(
        20,
        min_periods=20,
    ).std(
        ddof=0,
    )

    result["bb_upper_filter"] = (
        sma20
        + 2.0
        * std20
    )

    result["bb_lower_filter"] = (
        sma20
        - 2.0
        * std20
    )

    ema12 = ema(
        close,
        12,
    )

    ema26 = ema(
        close,
        26,
    )

    macd = (
        ema12
        - ema26
    )

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    result["macd_hist_filter"] = (
        macd
        - macd_signal
    )

    return result


def filter_allows(
    row: pd.Series,
    direction: int,
    key: str,
) -> bool:
    if key == "baseline":
        return True

    close = float(
        row["close"]
    )

    ema20 = row[
        "ema20_filter"
    ]

    rsi14 = row[
        "rsi14_filter"
    ]

    bb_upper = row[
        "bb_upper_filter"
    ]

    bb_lower = row[
        "bb_lower_filter"
    ]

    macd_hist = row[
        "macd_hist_filter"
    ]

    if key == "ema_alignment":
        if pd.isna(ema20):
            return False

        return (
            close > ema20
            if direction == 1
            else close < ema20
        )

    if key == "macd_alignment":
        if pd.isna(macd_hist):
            return False

        return (
            macd_hist > 0.0
            if direction == 1
            else macd_hist < 0.0
        )

    if key == "rsi_40_60":
        if pd.isna(rsi14):
            return False

        return (
            rsi14 <= 40.0
            if direction == 1
            else rsi14 >= 60.0
        )

    if key == "rsi_30_70":
        if pd.isna(rsi14):
            return False

        return (
            rsi14 <= 30.0
            if direction == 1
            else rsi14 >= 70.0
        )

    if key == "bollinger_reversion":
        if (
            pd.isna(bb_upper)
            or pd.isna(bb_lower)
        ):
            return False

        return (
            close <= bb_lower
            if direction == 1
            else close >= bb_upper
        )

    if key == "ema_macd":
        return (
            filter_allows(
                row,
                direction,
                "ema_alignment",
            )
            and filter_allows(
                row,
                direction,
                "macd_alignment",
            )
        )

    if key == "ema_rsi":
        return (
            filter_allows(
                row,
                direction,
                "ema_alignment",
            )
            and filter_allows(
                row,
                direction,
                "rsi_40_60",
            )
        )

    raise ValueError(
        f"Unknown filter: {key}"
    )


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


def add_model_predictions(
    df: pd.DataFrame,
    feature_columns: list[str],
    model: XGBClassifier,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    result = df.copy()

    prediction_mask = (
        result["open_time"]
        >= VALIDATION_START
    )

    X = result.loc[
        prediction_mask,
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
        prediction_mask,
        "ai_prediction",
    ] = labels

    result.loc[
        prediction_mask,
        "ai_confidence",
    ] = probabilities.max(
        axis=1
    )

    return result


def simulate_filter(
    df: pd.DataFrame,
    rule: FilterRule,
    period_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
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

        if not filter_allows(
            row,
            direction,
            rule.key,
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
            entry_row[
                "open"
            ]
        )

        exit_price = float(
            exit_row[
                "close"
            ]
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
            "filter":
                rule.name,
            "filter_key":
                rule.key,
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
            "entry_price":
                entry_price,
            "exit_price":
                exit_price,
            "gross_return_pct":
                gross_return
                * 100.0,
            "rsi14":
                row[
                    "rsi14_filter"
                ],
            "ema20":
                row[
                    "ema20_filter"
                ],
            "macd_hist":
                row[
                    "macd_hist_filter"
                ],
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

        # Keep execution comparable to the
        # forward collector: no overlapping
        # 60-minute positions.
        index = exit_index + 1

    return trades


def summarize(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for rule in FILTERS:
        for period in [
            "validation",
            "test",
        ]:
            subset = trades[
                (
                    trades[
                        "filter_key"
                    ]
                    == rule.key
                )
                & (
                    trades[
                        "period"
                    ]
                    == period
                )
            ]

            row = {
                "filter":
                    rule.name,
                "filter_key":
                    rule.key,
                "period":
                    period,
                "trades":
                    int(
                        len(
                            subset
                        )
                    ),
            }

            if subset.empty:
                row.update(
                    {
                        "win_rate_mid_pct":
                            0.0,
                        "net_mid_sum_pct":
                            0.0,
                        "avg_net_mid_pct":
                            0.0,
                        "net_high_sum_pct":
                            0.0,
                        "avg_net_high_pct":
                            0.0,
                    }
                )

                rows.append(
                    row
                )
                continue

            row[
                "win_rate_mid_pct"
            ] = float(
                (
                    subset[
                        "net_mid_pct"
                    ]
                    > 0.0
                ).mean()
                * 100.0
            )

            row[
                "net_mid_sum_pct"
            ] = float(
                subset[
                    "net_mid_pct"
                ].sum()
            )

            row[
                "avg_net_mid_pct"
            ] = float(
                subset[
                    "net_mid_pct"
                ].mean()
            )

            row[
                "net_high_sum_pct"
            ] = float(
                subset[
                    "net_high_pct"
                ].sum()
            )

            row[
                "avg_net_high_pct"
            ] = float(
                subset[
                    "net_high_pct"
                ].mean()
            )

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
                result[
                    "filter_key"
                ]
                == "baseline"
            )
            & (
                result[
                    "period"
                ]
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
            result[
                "period"
            ]
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


def main() -> None:
    print(
        "TRADEMIND - AI + TECHNICAL FILTER BENCHMARK"
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

    numeric_columns = [
        column
        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        if column in df.columns
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
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

    df = add_technical_indicators(
        df
    )

    feature_columns = [
        column
        for column in df.columns
        if (
            column
            not in NON_FEATURE_COLUMNS
            and not column.endswith(
                "_filter"
            )
            and pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]

    # Match the model input universe and
    # avoid technical filter columns leaking
    # into the AI model itself.
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
        df[
            "open_time"
        ].isin(
            valid_times
        )
    ].copy().reset_index(
        drop=True
    )

    model, encoder = train_model(
        df,
        feature_columns,
    )

    df = add_model_predictions(
        df,
        feature_columns,
        model,
        encoder,
    )

    all_trades = []

    for rule in FILTERS:
        all_trades.extend(
            simulate_filter(
                df,
                rule,
                "validation",
                VALIDATION_START,
                TEST_START,
            )
        )

        all_trades.extend(
            simulate_filter(
                df,
                rule,
                "test",
                TEST_START,
                None,
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

    summary = {
        "purpose":
            "Test whether common technical indicators improve the 10m TradeMind AI candidate as post-model filters.",
        "model":
            "XGBoost using the same core 10m configuration, trained only before validation.",
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
        "entry_model":
            "Signal candle close; enter next 10m open",
        "exit_model":
            "Fixed 60m horizon close",
        "overlap_policy":
            "No overlapping positions within each tested filter",
        "costs_pct": {
            name:
                value * 100.0
            for name, value
            in COSTS.items()
        },
        "important_note":
            "This is a research reconstruction of the 10m model family, not a retuning of the currently frozen forward candidate.",
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    columns = [
        "filter",
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
        "Decision rule:"
    )
    print(
        "A filter is interesting only if it improves "
        "the AI baseline in BOTH validation and test, "
        "while retaining a meaningful number of trades."
    )

    print()
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
