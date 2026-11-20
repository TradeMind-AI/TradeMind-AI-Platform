from __future__ import annotations

from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_5m_features.csv"
)

CONTEXT_10M_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m_features.csv"
)

CONTEXT_1H_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v3_barrier_opportunity"
)

RESULT_FILE = OUTPUT_DIR / "candidate_results.csv"
TOP_FILE = OUTPUT_DIR / "top_candidates.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"

RESEARCH_START = pd.Timestamp(
    "2024-01-01",
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

BASE_MINUTES = 5

HORIZON_MINUTES = [
    15,
    30,
    60,
]

TAKE_PROFIT_LEVELS = [
    0.0035,
    0.0050,
    0.0075,
]

STOP_LOSS_LEVELS = [
    0.0020,
    0.0030,
    0.0040,
]

CONFIDENCE_THRESHOLDS = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

MID_COST = 0.0014
HIGH_COST = 0.0027

TARGET_CONFIDENCE = 0.80
TARGET_WIN_RATE_PCT = 80.0
TARGET_TRADES_PER_DAY = 4.0
STRETCH_TRADES_PER_DAY = 8.0
TARGET_MEDIAN_GAP_HOURS = 2.0
MIN_TARGET_TRADES = 50

CONTEXT_10M_COLUMNS = [
    "return_60m",
    "return_120m",
    "close_to_sma_60m",
    "close_to_sma_240m",
    "rsi_14",
    "macd_histogram",
    "rolling_volatility_60m",
    "relative_volume_60m",
    "taker_buy_imbalance",
]

CONTEXT_1H_COLUMNS = [
    "return_1h",
    "return_3h",
    "return_6h",
    "close_to_sma_24",
    "close_to_sma_168",
    "rsi_14",
    "macd_histogram",
    "rolling_volatility_24",
    "relative_volume",
]

NON_FEATURE_COLUMNS = {
    "open_time",
    "close_time",
    "decision_time",
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
    "future_return_5m",
    "target",
}


def load_context(
    path: Path,
    columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    usecols = [
        "close_time",
        *columns,
    ]

    frame = pd.read_csv(
        path,
        usecols=usecols,
    )

    frame[
        "decision_time"
    ] = pd.to_datetime(
        frame["close_time"],
        utc=True,
        errors="coerce",
    )

    frame = frame.drop(
        columns=[
            "close_time",
        ]
    )

    frame = frame.rename(
        columns={
            column:
                f"{prefix}{column}"
            for column in columns
        }
    )

    return (
        frame.dropna(
            subset=["decision_time"]
        )
        .sort_values(
            "decision_time"
        )
        .drop_duplicates(
            subset=["decision_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def load_dataset() -> tuple[
    pd.DataFrame,
    list[str],
]:
    print(
        "Loading 5m base features..."
    )

    frame = pd.read_csv(
        BASE_FILE
    )

    frame["open_time"] = pd.to_datetime(
        frame["open_time"],
        utc=True,
        errors="coerce",
    )

    frame["close_time"] = pd.to_datetime(
        frame["close_time"],
        utc=True,
        errors="coerce",
    )

    frame[
        "decision_time"
    ] = frame["close_time"]

    frame = (
        frame.dropna(
            subset=[
                "open_time",
                "decision_time",
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .sort_values(
            "decision_time"
        )
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    frame = frame[
        frame["open_time"]
        >= RESEARCH_START
    ].reset_index(drop=True)

    print(
        "Loading closed 10m context..."
    )

    context_10m = load_context(
        CONTEXT_10M_FILE,
        CONTEXT_10M_COLUMNS,
        "ctx10_",
    )

    print(
        "Loading closed 1h context..."
    )

    context_1h = load_context(
        CONTEXT_1H_FILE,
        CONTEXT_1H_COLUMNS,
        "ctx1h_",
    )

    frame = pd.merge_asof(
        frame.sort_values(
            "decision_time"
        ),
        context_10m,
        on="decision_time",
        direction="backward",
    )

    frame = pd.merge_asof(
        frame.sort_values(
            "decision_time"
        ),
        context_1h,
        on="decision_time",
        direction="backward",
    )

    feature_columns = [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and not column.startswith(
            "future_return_"
        )
    ]

    numeric = frame[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    finite = np.isfinite(
        numeric.to_numpy(
            dtype=float
        )
    ).all(axis=1)

    frame = frame.loc[
        finite
    ].reset_index(drop=True)

    print(
        f"ML rows: {len(frame):,}"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        "Period: "
        f"{frame['open_time'].min()} "
        "-> "
        f"{frame['open_time'].max()}"
    )

    return (
        frame,
        feature_columns,
    )


def barrier_outcomes(
    frame: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
    horizon_bars: int,
) -> dict[str, np.ndarray]:
    length = len(frame)

    entry = (
        pd.to_numeric(
            frame["open"],
            errors="coerce",
        )
        .shift(-1)
        .to_numpy(
            dtype=float
        )
    )

    long_gross = np.full(
        length,
        np.nan,
        dtype=float,
    )

    short_gross = np.full(
        length,
        np.nan,
        dtype=float,
    )

    long_exit = np.full(
        length,
        horizon_bars,
        dtype=np.int16,
    )

    short_exit = np.full(
        length,
        horizon_bars,
        dtype=np.int16,
    )

    long_resolved = np.zeros(
        length,
        dtype=bool,
    )

    short_resolved = np.zeros(
        length,
        dtype=bool,
    )

    valid = np.isfinite(
        entry
    )

    for offset in range(
        1,
        horizon_bars + 1,
    ):
        high = (
            pd.to_numeric(
                frame["high"],
                errors="coerce",
            )
            .shift(-offset)
            .to_numpy(
                dtype=float
            )
        )

        low = (
            pd.to_numeric(
                frame["low"],
                errors="coerce",
            )
            .shift(-offset)
            .to_numpy(
                dtype=float
            )
        )

        bar_valid = (
            valid
            & np.isfinite(high)
            & np.isfinite(low)
        )

        unresolved_long = (
            bar_valid
            & (~long_resolved)
        )

        long_tp = (
            unresolved_long
            & (
                high
                >= entry
                * (
                    1.0
                    + take_profit
                )
            )
        )

        long_sl = (
            unresolved_long
            & (
                low
                <= entry
                * (
                    1.0
                    - stop_loss
                )
            )
        )

        # Conservative same-bar handling:
        # if both TP and SL are touched,
        # count the stop as first.
        long_loss = long_sl
        long_win = (
            long_tp
            & (~long_sl)
        )

        long_gross[
            long_loss
        ] = -stop_loss

        long_gross[
            long_win
        ] = take_profit

        long_hit = (
            long_loss
            | long_win
        )

        long_exit[
            long_hit
        ] = offset

        long_resolved |= long_hit

        unresolved_short = (
            bar_valid
            & (~short_resolved)
        )

        short_tp = (
            unresolved_short
            & (
                low
                <= entry
                * (
                    1.0
                    - take_profit
                )
            )
        )

        short_sl = (
            unresolved_short
            & (
                high
                >= entry
                * (
                    1.0
                    + stop_loss
                )
            )
        )

        short_loss = short_sl
        short_win = (
            short_tp
            & (~short_sl)
        )

        short_gross[
            short_loss
        ] = -stop_loss

        short_gross[
            short_win
        ] = take_profit

        short_hit = (
            short_loss
            | short_win
        )

        short_exit[
            short_hit
        ] = offset

        short_resolved |= short_hit

    horizon_close = (
        pd.to_numeric(
            frame["close"],
            errors="coerce",
        )
        .shift(-horizon_bars)
        .to_numpy(
            dtype=float
        )
    )

    final_valid = (
        valid
        & np.isfinite(
            horizon_close
        )
    )

    unresolved_long = (
        final_valid
        & (~long_resolved)
    )

    long_gross[
        unresolved_long
    ] = (
        horizon_close[
            unresolved_long
        ]
        / entry[
            unresolved_long
        ]
        - 1.0
    )

    unresolved_short = (
        final_valid
        & (~short_resolved)
    )

    short_gross[
        unresolved_short
    ] = (
        entry[
            unresolved_short
        ]
        / horizon_close[
            unresolved_short
        ]
        - 1.0
    )

    valid = (
        final_valid
        & np.isfinite(
            long_gross
        )
        & np.isfinite(
            short_gross
        )
    )

    return {
        "long_gross":
            long_gross,
        "short_gross":
            short_gross,
        "long_exit":
            long_exit,
        "short_exit":
            short_exit,
        "valid":
            valid,
    }


def build_classifier(
    y: np.ndarray,
) -> XGBClassifier:
    positives = float(
        y.sum()
    )

    negatives = float(
        len(y)
        - positives
    )

    scale_pos_weight = (
        negatives
        / max(
            positives,
            1.0,
        )
    )

    return XGBClassifier(
        n_estimators=170,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=6,
        reg_alpha=0.1,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        device=os.getenv(
            "TRADEMIND_XGB_DEVICE",
            "cpu",
        ),
        scale_pos_weight=
            scale_pos_weight,
    )


def fit_side_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_rows: np.ndarray,
    calibration_rows: np.ndarray,
) -> tuple[
    XGBClassifier,
    IsotonicRegression | None,
    float,
]:
    model = build_classifier(
        y[
            model_rows
        ]
    )

    model.fit(
        X.iloc[
            model_rows
        ],
        y[
            model_rows
        ],
    )

    raw_calibration = (
        model.predict_proba(
            X.iloc[
                calibration_rows
            ]
        )[:, 1]
    )

    calibration_target = y[
        calibration_rows
    ]

    calibrator = None

    if (
        len(
            np.unique(
                calibration_target
            )
        )
        >= 2
        and len(
            np.unique(
                raw_calibration
            )
        )
        >= 2
    ):
        calibrator = (
            IsotonicRegression(
                out_of_bounds="clip",
            )
        )

        calibrator.fit(
            raw_calibration,
            calibration_target,
        )

        calibrated = (
            calibrator.transform(
                raw_calibration
            )
        )
    else:
        calibrated = (
            raw_calibration
        )

    brier = float(
        brier_score_loss(
            calibration_target,
            calibrated,
        )
    )

    return (
        model,
        calibrator,
        brier,
    )


def predict_probability(
    model: XGBClassifier,
    calibrator:
        IsotonicRegression | None,
    X: pd.DataFrame,
    rows: np.ndarray,
) -> np.ndarray:
    raw = model.predict_proba(
        X.iloc[
            rows
        ]
    )[:, 1]

    if calibrator is None:
        return raw

    return calibrator.transform(
        raw
    )


def max_drawdown_pct(
    net_returns: np.ndarray,
) -> float:
    if len(
        net_returns
    ) == 0:
        return 0.0

    equity = np.cumprod(
        1.0
        + net_returns
    )

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity / peak
        - 1.0
    )

    return float(
        drawdown.min()
        * 100.0
    )


def frequency_metrics(
    times: pd.Series,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> tuple[
    float,
    float,
    float,
]:
    days = max(
        (
            period_end
            - period_start
        ).total_seconds()
        / 86400.0,
        1e-9,
    )

    per_day = (
        len(times)
        / days
    )

    if len(times) < 2:
        return (
            per_day,
            math.inf,
            math.inf,
        )

    ordered = pd.to_datetime(
        times,
        utc=True,
    ).sort_values()

    gaps = (
        ordered.diff()
        .dropna()
        .dt.total_seconds()
        / 3600.0
    )

    return (
        per_day,
        float(
            gaps.median()
        ),
        float(
            gaps.quantile(
                0.90
            )
        ),
    )


def evaluate(
    frame: pd.DataFrame,
    rows: np.ndarray,
    long_probability: np.ndarray,
    short_probability: np.ndarray,
    outcomes: dict[str, np.ndarray],
    confidence_threshold: float,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> dict:
    choose_long = (
        long_probability
        >= short_probability
    )

    probability = np.maximum(
        long_probability,
        short_probability,
    )

    selected_positions = (
        np.flatnonzero(
            probability
            >= confidence_threshold
        )
    )

    accepted_positions = []

    next_available = -1

    for position in (
        selected_positions
    ):
        row_index = int(
            rows[
                position
            ]
        )

        if (
            row_index
            < next_available
        ):
            continue

        if choose_long[
            position
        ]:
            exit_offset = int(
                outcomes[
                    "long_exit"
                ][
                    row_index
                ]
            )
        else:
            exit_offset = int(
                outcomes[
                    "short_exit"
                ][
                    row_index
                ]
            )

        accepted_positions.append(
            int(position)
        )

        next_available = (
            row_index
            + max(
                exit_offset,
                1,
            )
        )

    accepted_positions = np.asarray(
        accepted_positions,
        dtype=np.int64,
    )

    if len(
        accepted_positions
    ) == 0:
        return {
            "trades": 0,
            "trades_per_day": 0.0,
            "median_gap_hours":
                math.inf,
            "p90_gap_hours":
                math.inf,
            "mean_confidence_pct":
                0.0,
            "win_rate_mid_pct":
                0.0,
            "avg_net_mid_pct":
                0.0,
            "net_mid_sum_pct":
                0.0,
            "net_high_sum_pct":
                0.0,
            "max_drawdown_mid_pct":
                0.0,
        }

    accepted_rows = rows[
        accepted_positions
    ]

    accepted_long = choose_long[
        accepted_positions
    ]

    gross = np.where(
        accepted_long,
        outcomes[
            "long_gross"
        ][
            accepted_rows
        ],
        outcomes[
            "short_gross"
        ][
            accepted_rows
        ],
    )

    mid = (
        gross
        - MID_COST
    )

    high = (
        gross
        - HIGH_COST
    )

    (
        trades_per_day,
        median_gap,
        p90_gap,
    ) = frequency_metrics(
        frame.iloc[
            accepted_rows
        ][
            "open_time"
        ],
        period_start,
        period_end,
    )

    return {
        "trades":
            int(
                len(
                    accepted_positions
                )
            ),
        "trades_per_day":
            trades_per_day,
        "median_gap_hours":
            median_gap,
        "p90_gap_hours":
            p90_gap,
        "mean_confidence_pct":
            float(
                probability[
                    accepted_positions
                ].mean()
                * 100.0
            ),
        "win_rate_mid_pct":
            float(
                (mid > 0.0).mean()
                * 100.0
            ),
        "avg_net_mid_pct":
            float(
                mid.mean()
                * 100.0
            ),
        "net_mid_sum_pct":
            float(
                mid.sum()
                * 100.0
            ),
        "net_high_sum_pct":
            float(
                high.sum()
                * 100.0
            ),
        "max_drawdown_mid_pct":
            max_drawdown_pct(
                mid
            ),
    }


def research_score(
    row: pd.Series,
) -> float:
    win_shortfall = max(
        0.0,
        TARGET_WIN_RATE_PCT
        - float(
            row[
                "win_rate_mid_pct"
            ]
        ),
    )

    frequency_shortfall = max(
        0.0,
        TARGET_TRADES_PER_DAY
        - float(
            row[
                "trades_per_day"
            ]
        ),
    )

    gap = float(
        row[
            "median_gap_hours"
        ]
    )

    gap_penalty = (
        max(
            0.0,
            gap
            - TARGET_MEDIAN_GAP_HOURS,
        )
        if np.isfinite(gap)
        else 100.0
    )

    count_shortfall = max(
        0,
        MIN_TARGET_TRADES
        - int(
            row[
                "trades"
            ]
        ),
    )

    return float(
        row[
            "win_rate_mid_pct"
        ]
        + row[
            "trades_per_day"
        ]
        * 6.0
        + row[
            "avg_net_mid_pct"
        ]
        * 25.0
        + min(
            int(
                row[
                    "trades"
                ]
            ),
            300,
        )
        * 0.03
        - win_shortfall
        * 0.8
        - frequency_shortfall
        * 7.0
        - gap_penalty
        * 2.0
        - count_shortfall
        * 0.08
        + (
            12.0
            if row[
                "net_high_sum_pct"
            ] > 0.0
            else -12.0
        )
    )


def main() -> None:
    print(
        "TRADEMIND V3 - MULTI-TIMEFRAME "
        "BARRIER OPPORTUNITY RESEARCH"
    )

    print()
    print(
        "Goal: predict whether a real LONG or "
        "SHORT trade will finish profitable "
        "after costs, using 5m execution plus "
        "closed 10m and 1h context."
    )

    print()
    print(
        "Execution is path-aware:"
    )

    print(
        "  entry: next 5m bar open"
    )

    print(
        "  exit: TP, SL, or max-hold close"
    )

    print(
        "  if TP and SL touch in the same bar, "
        "the stop is assumed first"
    )

    print()
    print(
        "Strict research target (not forced):"
    )

    print(
        "  calibrated confidence >= 80%"
    )

    print(
        "  win rate >= 80%"
    )

    print(
        f"  trades/day >= "
        f"{TARGET_TRADES_PER_DAY:.1f} "
        f"(stretch "
        f"{STRETCH_TRADES_PER_DAY:.1f})"
    )

    print(
        "  median gap <= 2h"
    )

    print(
        f"  minimum trades >= "
        f"{MIN_TARGET_TRADES}"
    )

    print(
        "  positive mid and high-cost net"
    )

    (
        frame,
        feature_columns,
    ) = load_dataset()

    X = frame[
        feature_columns
    ]

    train_mask = (
        frame[
            "open_time"
        ]
        < VALIDATION_START
    ).to_numpy()

    validation_mask = (
        (
            frame[
                "open_time"
            ]
            >= VALIDATION_START
        )
        & (
            frame[
                "open_time"
            ]
            < TEST_START
        )
    ).to_numpy()

    test_mask = (
        frame[
            "open_time"
        ]
        >= TEST_START
    ).to_numpy()

    all_train_rows = np.flatnonzero(
        train_mask
    )

    split = int(
        len(
            all_train_rows
        )
        * 0.75
    )

    model_rows = (
        all_train_rows[
            :split
        ]
    )

    calibration_rows = (
        all_train_rows[
            split:
        ]
    )

    validation_rows = (
        np.flatnonzero(
            validation_mask
        )
    )

    test_rows = (
        np.flatnonzero(
            test_mask
        )
    )

    result_rows = []

    config_count = 0

    for horizon_minutes in (
        HORIZON_MINUTES
    ):
        horizon_bars = (
            horizon_minutes
            // BASE_MINUTES
        )

        for take_profit in (
            TAKE_PROFIT_LEVELS
        ):
            for stop_loss in (
                STOP_LOSS_LEVELS
            ):
                config_count += 1

                print()
                print(
                    "=" * 105
                )

                print(
                    "Config "
                    f"{config_count}: "
                    f"hold<={horizon_minutes}m "
                    f"TP={take_profit * 100:.2f}% "
                    f"SL={stop_loss * 100:.2f}%"
                )

                print(
                    "=" * 105
                )

                outcomes = (
                    barrier_outcomes(
                        frame,
                        take_profit,
                        stop_loss,
                        horizon_bars,
                    )
                )

                valid = outcomes[
                    "valid"
                ]

                config_model_rows = (
                    model_rows[
                        valid[
                            model_rows
                        ]
                    ]
                )

                config_calibration_rows = (
                    calibration_rows[
                        valid[
                            calibration_rows
                        ]
                    ]
                )

                config_validation_rows = (
                    validation_rows[
                        valid[
                            validation_rows
                        ]
                    ]
                )

                config_test_rows = (
                    test_rows[
                        valid[
                            test_rows
                        ]
                    ]
                )

                long_target = (
                    (
                        outcomes[
                            "long_gross"
                        ]
                        - MID_COST
                    )
                    > 0.0
                ).astype(
                    np.int8
                )

                short_target = (
                    (
                        outcomes[
                            "short_gross"
                        ]
                        - MID_COST
                    )
                    > 0.0
                ).astype(
                    np.int8
                )

                (
                    long_model,
                    long_calibrator,
                    long_brier,
                ) = fit_side_model(
                    X,
                    long_target,
                    config_model_rows,
                    config_calibration_rows,
                )

                (
                    short_model,
                    short_calibrator,
                    short_brier,
                ) = fit_side_model(
                    X,
                    short_target,
                    config_model_rows,
                    config_calibration_rows,
                )

                print(
                    "Calibration Brier: "
                    f"long={long_brier:.4f} "
                    f"short={short_brier:.4f}"
                )

                for (
                    period,
                    rows,
                    period_start,
                    period_end,
                ) in [
                    (
                        "validation",
                        config_validation_rows,
                        VALIDATION_START,
                        TEST_START,
                    ),
                    (
                        "test",
                        config_test_rows,
                        TEST_START,
                        frame[
                            "open_time"
                        ].iloc[-1]
                        + pd.Timedelta(
                            minutes=BASE_MINUTES
                        ),
                    ),
                ]:
                    long_probability = (
                        predict_probability(
                            long_model,
                            long_calibrator,
                            X,
                            rows,
                        )
                    )

                    short_probability = (
                        predict_probability(
                            short_model,
                            short_calibrator,
                            X,
                            rows,
                        )
                    )

                    for confidence in (
                        CONFIDENCE_THRESHOLDS
                    ):
                        metrics = evaluate(
                            frame,
                            rows,
                            long_probability,
                            short_probability,
                            outcomes,
                            confidence,
                            period_start,
                            period_end,
                        )

                        strict_target = bool(
                            confidence
                            >= TARGET_CONFIDENCE
                            and metrics[
                                "trades"
                            ]
                            >= MIN_TARGET_TRADES
                            and metrics[
                                "win_rate_mid_pct"
                            ]
                            >= TARGET_WIN_RATE_PCT
                            and metrics[
                                "trades_per_day"
                            ]
                            >= TARGET_TRADES_PER_DAY
                            and metrics[
                                "median_gap_hours"
                            ]
                            <= TARGET_MEDIAN_GAP_HOURS
                            and metrics[
                                "net_mid_sum_pct"
                            ]
                            > 0.0
                            and metrics[
                                "net_high_sum_pct"
                            ]
                            > 0.0
                        )

                        result_rows.append(
                            {
                                "period":
                                    period,
                                "horizon_minutes":
                                    horizon_minutes,
                                "take_profit_pct":
                                    take_profit
                                    * 100.0,
                                "stop_loss_pct":
                                    stop_loss
                                    * 100.0,
                                "confidence_threshold":
                                    confidence,
                                "long_calibration_brier":
                                    long_brier,
                                "short_calibration_brier":
                                    short_brier,
                                **metrics,
                                "strict_target_met":
                                    strict_target,
                            }
                        )

    results = pd.DataFrame(
        result_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULT_FILE,
        index=False,
    )

    validation = results[
        results[
            "period"
        ] == "validation"
    ].copy()

    validation[
        "research_score"
    ] = validation.apply(
        research_score,
        axis=1,
    )

    validation = (
        validation.sort_values(
            [
                "strict_target_met",
                "research_score",
                "win_rate_mid_pct",
                "trades_per_day",
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

    top = validation.head(
        20
    ).copy()

    test = (
        results[
            results[
                "period"
            ] == "test"
        ]
        .set_index(
            [
                "horizon_minutes",
                "take_profit_pct",
                "stop_loss_pct",
                "confidence_threshold",
            ]
        )
    )

    test_columns = [
        "trades",
        "trades_per_day",
        "median_gap_hours",
        "p90_gap_hours",
        "mean_confidence_pct",
        "win_rate_mid_pct",
        "avg_net_mid_pct",
        "net_mid_sum_pct",
        "net_high_sum_pct",
        "max_drawdown_mid_pct",
        "strict_target_met",
    ]

    for column in test_columns:
        values = []

        for row in top.itertuples(
            index=False
        ):
            key = (
                row.horizon_minutes,
                row.take_profit_pct,
                row.stop_loss_pct,
                row.confidence_threshold,
            )

            if key in test.index:
                values.append(
                    test.loc[
                        key,
                        column,
                    ]
                )
            else:
                values.append(
                    np.nan
                )

        top[
            f"test_{column}"
        ] = values

    top.to_csv(
        TOP_FILE,
        index=False,
    )

    best = top.iloc[0]

    summary = {
        "research_only": True,
        "architecture": (
            "5m dual-side profitable-trade "
            "classifiers + closed 10m/1h context "
            "+ TP/SL/max-hold path simulation"
        ),
        "configurations": config_count,
        "targets": {
            "confidence":
                TARGET_CONFIDENCE,
            "win_rate_pct":
                TARGET_WIN_RATE_PCT,
            "trades_per_day":
                TARGET_TRADES_PER_DAY,
            "stretch_trades_per_day":
                STRETCH_TRADES_PER_DAY,
            "median_gap_hours":
                TARGET_MEDIAN_GAP_HOURS,
            "minimum_trades":
                MIN_TARGET_TRADES,
            "mid_cost_pct":
                MID_COST * 100.0,
            "high_cost_pct":
                HIGH_COST * 100.0,
        },
        "best_validation_candidate": {
            "horizon_minutes":
                int(
                    best[
                        "horizon_minutes"
                    ]
                ),
            "take_profit_pct":
                float(
                    best[
                        "take_profit_pct"
                    ]
                ),
            "stop_loss_pct":
                float(
                    best[
                        "stop_loss_pct"
                    ]
                ),
            "confidence_threshold":
                float(
                    best[
                        "confidence_threshold"
                    ]
                ),
            "validation_trades":
                int(
                    best[
                        "trades"
                    ]
                ),
            "validation_trades_per_day":
                float(
                    best[
                        "trades_per_day"
                    ]
                ),
            "validation_win_rate_mid_pct":
                float(
                    best[
                        "win_rate_mid_pct"
                    ]
                ),
            "validation_net_mid_sum_pct":
                float(
                    best[
                        "net_mid_sum_pct"
                    ]
                ),
            "validation_net_high_sum_pct":
                float(
                    best[
                        "net_high_sum_pct"
                    ]
                ),
            "test_trades":
                int(
                    best[
                        "test_trades"
                    ]
                ),
            "test_trades_per_day":
                float(
                    best[
                        "test_trades_per_day"
                    ]
                ),
            "test_win_rate_mid_pct":
                float(
                    best[
                        "test_win_rate_mid_pct"
                    ]
                ),
            "test_net_mid_sum_pct":
                float(
                    best[
                        "test_net_mid_sum_pct"
                    ]
                ),
            "test_net_high_sum_pct":
                float(
                    best[
                        "test_net_high_sum_pct"
                    ]
                ),
            "test_target_met":
                bool(
                    best[
                        "test_strict_target_met"
                    ]
                ),
        },
        "warning": (
            "Validation selects the candidate. "
            "Later test is evidence only. A promising "
            "barrier model must still be frozen and "
            "forward paper tested before it can replace "
            "an existing candidate."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    display_columns = [
        "horizon_minutes",
        "take_profit_pct",
        "stop_loss_pct",
        "confidence_threshold",
        "trades",
        "trades_per_day",
        "median_gap_hours",
        "mean_confidence_pct",
        "win_rate_mid_pct",
        "avg_net_mid_pct",
        "net_mid_sum_pct",
        "net_high_sum_pct",
        "test_trades",
        "test_trades_per_day",
        "test_win_rate_mid_pct",
        "test_net_mid_sum_pct",
        "test_net_high_sum_pct",
    ]

    print()
    print(
        "=" * 180
    )

    print(
        "TOP BARRIER OPPORTUNITY CANDIDATES"
    )

    print(
        "=" * 180
    )

    print(
        top[
            display_columns
        ].head(15).to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(RESULT_FILE)
    print(TOP_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
