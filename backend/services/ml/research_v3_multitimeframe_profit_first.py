from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v3_multitimeframe_profit_first"
)

RESULT_FILE = OUTPUT_DIR / "candidate_results.csv"
TRADE_FILE = OUTPUT_DIR / "candidate_trades.csv"
TOP_FILE = OUTPUT_DIR / "top_candidates.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"

TIMEFRAMES = {
    "1m": 1,
    "2m": 2,
    "5m": 5,
    "10m": 10,
}

HORIZONS_MINUTES = [
    5,
    10,
    20,
    30,
    60,
]

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

MID_COST = 0.0014
HIGH_COST = 0.0027

PROPOSER_THRESHOLDS = [
    0.50,
    0.52,
    0.55,
    0.58,
    0.60,
    0.65,
]

SELECTOR_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]

TARGET_SELECTOR_CONFIDENCE = 0.80
TARGET_WIN_RATE_PCT = 80.0
TARGET_TRADES_PER_DAY = 4.0
STRETCH_TRADES_PER_DAY = 8.0
TARGET_MEDIAN_GAP_HOURS = 2.0
MIN_TARGET_TRADES = 50

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
    "future_return_2m",
    "future_return_5m",
    "future_return_10m",
    "future_return_20m",
    "future_return_30m",
    "future_return_60m",
    "target",
}

META_FEATURES = [
    "proposer_confidence",
    "proposer_margin",
    "direction_up",
]


def build_xgb(
    objective: str,
) -> XGBClassifier:
    if objective == "direction":
        return XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

    return XGBClassifier(
        n_estimators=260,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=4,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def data_file(
    timeframe: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"btcusdt_{timeframe}_features.csv"
    )


def feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and not column.startswith(
            "future_return_"
        )
    ]


def future_trade_arrays(
    frame: pd.DataFrame,
    horizon_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entry = (
        pd.to_numeric(
            frame["open"],
            errors="coerce",
        )
        .shift(-1)
        .to_numpy(dtype=float)
    )

    exit_price = (
        pd.to_numeric(
            frame["close"],
            errors="coerce",
        )
        .shift(-horizon_bars)
        .to_numpy(dtype=float)
    )

    gross_long = (
        exit_price / entry
        - 1.0
    )

    gross_short = (
        entry / exit_price
        - 1.0
    )

    direction_up = (
        gross_long >= gross_short
    ).astype(np.int8)

    best_gross = np.where(
        direction_up == 1,
        gross_long,
        gross_short,
    )

    valid = (
        np.isfinite(entry)
        & np.isfinite(exit_price)
        & np.isfinite(best_gross)
    )

    return (
        direction_up,
        best_gross,
        valid,
    )


def realized_for_direction(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    directions_up: np.ndarray,
    horizon_bars: int,
) -> np.ndarray:
    entry = (
        frame["open"]
        .shift(-1)
        .iloc[row_indices]
        .to_numpy(dtype=float)
    )

    exit_price = (
        frame["close"]
        .shift(-horizon_bars)
        .iloc[row_indices]
        .to_numpy(dtype=float)
    )

    return np.where(
        directions_up == 1,
        exit_price / entry - 1.0,
        entry / exit_price - 1.0,
    )


def chronological_split(
    train_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    split = int(
        len(train_rows)
        * 0.70
    )

    return (
        train_rows[:split],
        train_rows[split:],
    )


def selector_matrix(
    frame: pd.DataFrame,
    rows: np.ndarray,
    base_features: list[str],
    proposer_probability: np.ndarray,
) -> pd.DataFrame:
    result = frame.iloc[
        rows
    ][base_features].reset_index(
        drop=True
    )

    direction_up = (
        proposer_probability >= 0.5
    ).astype(float)

    confidence = np.maximum(
        proposer_probability,
        1.0 - proposer_probability,
    )

    result[
        "proposer_confidence"
    ] = confidence

    result[
        "proposer_margin"
    ] = np.abs(
        proposer_probability
        - 0.5
    )

    result[
        "direction_up"
    ] = direction_up

    return result


def frequency_metrics(
    signal_times: pd.Series,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> tuple[float, float, float]:
    days = max(
        (
            period_end
            - period_start
        ).total_seconds()
        / 86400.0,
        1e-9,
    )

    trades_per_day = (
        len(signal_times)
        / days
    )

    if len(signal_times) < 2:
        return (
            trades_per_day,
            math.inf,
            math.inf,
        )

    ordered = pd.to_datetime(
        signal_times,
        utc=True,
    ).sort_values()

    gaps = (
        ordered.diff()
        .dropna()
        .dt.total_seconds()
        / 3600.0
    )

    return (
        trades_per_day,
        float(
            gaps.median()
        ),
        float(
            gaps.quantile(0.90)
        ),
    )


def max_drawdown_pct(
    net_return: np.ndarray,
) -> float:
    if len(net_return) == 0:
        return 0.0

    equity = np.cumprod(
        1.0 + net_return
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


def evaluate_thresholds(
    frame: pd.DataFrame,
    timeframe: str,
    horizon_minutes: int,
    horizon_bars: int,
    period: str,
    rows: np.ndarray,
    direction_probability: np.ndarray,
    selector_probability: np.ndarray,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> tuple[list[dict], list[dict]]:
    result_rows = []
    trade_rows = []

    direction_up = (
        direction_probability
        >= 0.5
    ).astype(np.int8)

    proposer_confidence = np.maximum(
        direction_probability,
        1.0 - direction_probability,
    )

    gross = realized_for_direction(
        frame,
        rows,
        direction_up,
        horizon_bars,
    )

    for proposer_threshold in (
        PROPOSER_THRESHOLDS
    ):
        for selector_threshold in (
            SELECTOR_THRESHOLDS
        ):
            candidate_positions = np.flatnonzero(
                (
                    proposer_confidence
                    >= proposer_threshold
                )
                & (
                    selector_probability
                    >= selector_threshold
                )
            )

            accepted_positions = []
            next_available_row = -1

            for position in candidate_positions:
                row_index = int(
                    rows[position]
                )

                if (
                    row_index
                    < next_available_row
                ):
                    continue

                accepted_positions.append(
                    int(position)
                )

                # Do not allow overlapping positions.
                next_available_row = (
                    row_index
                    + horizon_bars
                )

            accepted_positions = np.asarray(
                accepted_positions,
                dtype=np.int64,
            )

            if len(
                accepted_positions
            ) == 0:
                count = 0
                win_rate = 0.0
                avg_mid = 0.0
                net_mid_sum = 0.0
                net_high_sum = 0.0
                drawdown = 0.0
                avg_selector = 0.0
                avg_proposer = 0.0
                trades_per_day = 0.0
                median_gap = math.inf
                p90_gap = math.inf
            else:
                accepted_rows = rows[
                    accepted_positions
                ]

                accepted_gross = gross[
                    accepted_positions
                ]

                mid = (
                    accepted_gross
                    - MID_COST
                )

                high = (
                    accepted_gross
                    - HIGH_COST
                )

                signal_times = frame.iloc[
                    accepted_rows
                ][
                    "open_time"
                ]

                (
                    trades_per_day,
                    median_gap,
                    p90_gap,
                ) = frequency_metrics(
                    signal_times,
                    period_start,
                    period_end,
                )

                count = len(
                    accepted_positions
                )

                win_rate = float(
                    (mid > 0).mean()
                    * 100.0
                )

                avg_mid = float(
                    mid.mean()
                    * 100.0
                )

                net_mid_sum = float(
                    mid.sum()
                    * 100.0
                )

                net_high_sum = float(
                    high.sum()
                    * 100.0
                )

                drawdown = (
                    max_drawdown_pct(
                        mid
                    )
                )

                avg_selector = float(
                    selector_probability[
                        accepted_positions
                    ].mean()
                    * 100.0
                )

                avg_proposer = float(
                    proposer_confidence[
                        accepted_positions
                    ].mean()
                    * 100.0
                )

                for position in (
                    accepted_positions
                ):
                    row_index = int(
                        rows[position]
                    )

                    trade_rows.append(
                        {
                            "timeframe":
                                timeframe,
                            "horizon_minutes":
                                horizon_minutes,
                            "period":
                                period,
                            "proposer_threshold":
                                proposer_threshold,
                            "selector_threshold":
                                selector_threshold,
                            "signal_time":
                                frame.iloc[
                                    row_index
                                ][
                                    "open_time"
                                ],
                            "direction":
                                (
                                    "LONG"
                                    if direction_up[
                                        position
                                    ] == 1
                                    else "SHORT"
                                ),
                            "proposer_confidence":
                                float(
                                    proposer_confidence[
                                        position
                                    ]
                                ),
                            "selector_probability":
                                float(
                                    selector_probability[
                                        position
                                    ]
                                ),
                            "gross_return_pct":
                                float(
                                    gross[
                                        position
                                    ]
                                    * 100.0
                                ),
                            "mid_net_return_pct":
                                float(
                                    (
                                        gross[
                                            position
                                        ]
                                        - MID_COST
                                    )
                                    * 100.0
                                ),
                            "high_net_return_pct":
                                float(
                                    (
                                        gross[
                                            position
                                        ]
                                        - HIGH_COST
                                    )
                                    * 100.0
                                ),
                        }
                    )

            strict_target_met = bool(
                selector_threshold
                >= TARGET_SELECTOR_CONFIDENCE
                and count
                >= MIN_TARGET_TRADES
                and win_rate
                >= TARGET_WIN_RATE_PCT
                and trades_per_day
                >= TARGET_TRADES_PER_DAY
                and median_gap
                <= TARGET_MEDIAN_GAP_HOURS
                and net_mid_sum > 0.0
                and net_high_sum > 0.0
            )

            result_rows.append(
                {
                    "timeframe":
                        timeframe,
                    "horizon_minutes":
                        horizon_minutes,
                    "period":
                        period,
                    "proposer_threshold":
                        proposer_threshold,
                    "selector_threshold":
                        selector_threshold,
                    "trades":
                        count,
                    "trades_per_day":
                        trades_per_day,
                    "median_gap_hours":
                        median_gap,
                    "p90_gap_hours":
                        p90_gap,
                    "mean_proposer_confidence_pct":
                        avg_proposer,
                    "mean_selector_confidence_pct":
                        avg_selector,
                    "win_rate_mid_pct":
                        win_rate,
                    "avg_net_mid_pct":
                        avg_mid,
                    "net_mid_sum_pct":
                        net_mid_sum,
                    "net_high_sum_pct":
                        net_high_sum,
                    "max_drawdown_mid_pct":
                        drawdown,
                    "strict_target_met":
                        strict_target_met,
                }
            )

    return (
        result_rows,
        trade_rows,
    )


def run_candidate(
    timeframe: str,
    timeframe_minutes: int,
    horizon_minutes: int,
) -> tuple[list[dict], list[dict]]:
    horizon_bars = (
        horizon_minutes
        // timeframe_minutes
    )

    path = data_file(
        timeframe
    )

    print()
    print(
        "=" * 95
    )
    print(
        f"V3 candidate: {timeframe} "
        f"/ horizon {horizon_minutes}m"
    )
    print(
        "=" * 95
    )

    frame = pd.read_csv(
        path
    )

    frame["open_time"] = pd.to_datetime(
        frame["open_time"],
        utc=True,
        errors="coerce",
    )

    frame = (
        frame.dropna(
            subset=[
                "open_time",
                "open",
                "close",
            ]
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

    frame = frame[
        frame["open_time"]
        >= RESEARCH_START
    ].reset_index(drop=True)

    base_features = (
        feature_columns(
            frame
        )
    )

    numeric = frame[
        base_features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    finite_mask = (
        np.isfinite(
            numeric.to_numpy(
                dtype=float
            )
        ).all(axis=1)
    )

    frame = frame.loc[
        finite_mask
    ].reset_index(drop=True)

    (
        direction_target,
        _,
        target_valid,
    ) = future_trade_arrays(
        frame,
        horizon_bars,
    )

    train_mask = (
        (
            frame["open_time"]
            < VALIDATION_START
        )
        & target_valid
    ).to_numpy()

    validation_mask = (
        (
            frame["open_time"]
            >= VALIDATION_START
        )
        & (
            frame["open_time"]
            < TEST_START
        )
        & target_valid
    ).to_numpy()

    test_mask = (
        (
            frame["open_time"]
            >= TEST_START
        )
        & target_valid
    ).to_numpy()

    train_rows = np.flatnonzero(
        train_mask
    )

    validation_rows = np.flatnonzero(
        validation_mask
    )

    test_rows = np.flatnonzero(
        test_mask
    )

    proposer_train_rows, selector_train_rows = (
        chronological_split(
            train_rows
        )
    )

    if (
        len(proposer_train_rows) < 1000
        or len(selector_train_rows) < 500
        or len(validation_rows) < 500
        or len(test_rows) < 500
    ):
        raise RuntimeError(
            "Insufficient rows for "
            f"{timeframe}/{horizon_minutes}m"
        )

    proposer = build_xgb(
        "direction"
    )

    proposer.fit(
        frame.iloc[
            proposer_train_rows
        ][base_features],
        direction_target[
            proposer_train_rows
        ],
    )

    selector_train_direction_probability = (
        proposer.predict_proba(
            frame.iloc[
                selector_train_rows
            ][base_features]
        )[:, 1]
    )

    selector_train_direction_up = (
        selector_train_direction_probability
        >= 0.5
    ).astype(np.int8)

    selector_train_gross = (
        realized_for_direction(
            frame,
            selector_train_rows,
            selector_train_direction_up,
            horizon_bars,
        )
    )

    selector_target = (
        selector_train_gross
        - MID_COST
        > 0.0
    ).astype(np.int8)

    selector_columns = (
        base_features
        + META_FEATURES
    )

    selector = build_xgb(
        "selector"
    )

    selector.fit(
        selector_matrix(
            frame,
            selector_train_rows,
            base_features,
            selector_train_direction_probability,
        )[selector_columns],
        selector_target,
    )

    # Final proposer is refit on all pre-validation rows.
    proposer_final = build_xgb(
        "direction"
    )

    proposer_final.fit(
        frame.iloc[
            train_rows
        ][base_features],
        direction_target[
            train_rows
        ],
    )

    all_results = []
    all_trades = []

    for (
        period,
        rows,
        period_start,
        period_end,
    ) in [
        (
            "validation",
            validation_rows,
            VALIDATION_START,
            TEST_START,
        ),
        (
            "test",
            test_rows,
            TEST_START,
            frame[
                "open_time"
            ].iloc[-1]
            + pd.Timedelta(
                minutes=timeframe_minutes
            ),
        ),
    ]:
        direction_probability = (
            proposer_final.predict_proba(
                frame.iloc[
                    rows
                ][base_features]
            )[:, 1]
        )

        selector_probability = (
            selector.predict_proba(
                selector_matrix(
                    frame,
                    rows,
                    base_features,
                    direction_probability,
                )[selector_columns]
            )[:, 1]
        )

        (
            result_rows,
            trade_rows,
        ) = evaluate_thresholds(
            frame,
            timeframe,
            horizon_minutes,
            horizon_bars,
            period,
            rows,
            direction_probability,
            selector_probability,
            period_start,
            period_end,
        )

        all_results.extend(
            result_rows
        )

        all_trades.extend(
            trade_rows
        )

    validation_direction_probability = (
        proposer_final.predict_proba(
            frame.iloc[
                validation_rows
            ][base_features]
        )[:, 1]
    )

    validation_direction_prediction = (
        validation_direction_probability
        >= 0.5
    ).astype(np.int8)

    direction_accuracy = (
        accuracy_score(
            direction_target[
                validation_rows
            ],
            validation_direction_prediction,
        )
        * 100.0
    )

    try:
        direction_auc = (
            roc_auc_score(
                direction_target[
                    validation_rows
                ],
                validation_direction_probability,
            )
        )
    except ValueError:
        direction_auc = 0.5

    print(
        "Direction validation: "
        f"accuracy={direction_accuracy:.2f}% "
        f"auc={direction_auc:.4f}"
    )

    return (
        all_results,
        all_trades,
    )


def score_validation(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result[
        "win_shortfall"
    ] = np.maximum(
        0.0,
        TARGET_WIN_RATE_PCT
        - result[
            "win_rate_mid_pct"
        ],
    )

    result[
        "frequency_shortfall"
    ] = np.maximum(
        0.0,
        TARGET_TRADES_PER_DAY
        - result[
            "trades_per_day"
        ],
    )

    result[
        "gap_penalty"
    ] = np.maximum(
        0.0,
        result[
            "median_gap_hours"
        ].replace(
            np.inf,
            999.0,
        )
        - TARGET_MEDIAN_GAP_HOURS,
    )

    result[
        "research_score"
    ] = (
        result[
            "win_rate_mid_pct"
        ]
        + result[
            "trades_per_day"
        ] * 5.0
        + result[
            "avg_net_mid_pct"
        ] * 20.0
        + np.minimum(
            result[
                "trades"
            ],
            200,
        ) * 0.03
        - result[
            "win_shortfall"
        ] * 0.7
        - result[
            "frequency_shortfall"
        ] * 6.0
        - result[
            "gap_penalty"
        ] * 2.0
        + np.where(
            result[
                "net_high_sum_pct"
            ] > 0.0,
            8.0,
            -8.0,
        )
    )

    return result


def main() -> None:
    print(
        "TRADEMIND V3 - MULTI-TIMEFRAME "
        "PROFIT-FIRST TRADE SEARCH"
    )

    print()
    print(
        "V3 removes NEUTRAL/HOLD from the "
        "direction model."
    )
    print(
        "Stage 1 always proposes LONG or SHORT."
    )
    print(
        "Stage 2 estimates whether the proposed "
        "trade should be profitable after costs."
    )
    print()
    print(
        "Research targets (not forced):"
    )
    print(
        "  win rate >= 80%"
    )
    print(
        "  selector confidence >= 80%"
    )
    print(
        f"  trades/day >= {TARGET_TRADES_PER_DAY:.1f} "
        f"(stretch {STRETCH_TRADES_PER_DAY:.1f})"
    )
    print(
        "  median trade gap <= 2h"
    )
    print(
        f"  minimum trades >= {MIN_TARGET_TRADES}"
    )
    print(
        "  positive mid/high-cost returns"
    )

    all_results = []
    all_trades = []

    candidate_count = 0

    for (
        timeframe,
        timeframe_minutes,
    ) in TIMEFRAMES.items():
        for horizon_minutes in (
            HORIZONS_MINUTES
        ):
            if (
                horizon_minutes
                < timeframe_minutes
                or horizon_minutes
                % timeframe_minutes
                != 0
            ):
                continue

            candidate_count += 1

            (
                results,
                trades,
            ) = run_candidate(
                timeframe,
                timeframe_minutes,
                horizon_minutes,
            )

            all_results.extend(
                results
            )

            all_trades.extend(
                trades
            )

    results = pd.DataFrame(
        all_results
    )

    trades = pd.DataFrame(
        all_trades
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULT_FILE,
        index=False,
    )

    trades.to_csv(
        TRADE_FILE,
        index=False,
    )

    validation = score_validation(
        results[
            results["period"]
            == "validation"
        ].copy()
    )

    validation = validation.sort_values(
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
    ).reset_index(drop=True)

    top = validation.head(
        20
    ).copy()

    test = results[
        results["period"]
        == "test"
    ].copy()

    test_lookup = test.set_index(
        [
            "timeframe",
            "horizon_minutes",
            "proposer_threshold",
            "selector_threshold",
        ]
    )

    test_columns = [
        "trades",
        "trades_per_day",
        "median_gap_hours",
        "p90_gap_hours",
        "mean_selector_confidence_pct",
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
                row.timeframe,
                row.horizon_minutes,
                row.proposer_threshold,
                row.selector_threshold,
            )

            if key in test_lookup.index:
                value = (
                    test_lookup.loc[
                        key,
                        column,
                    ]
                )
            else:
                value = np.nan

            values.append(
                value
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
            "binary LONG/SHORT proposer + "
            "profit-after-cost selector"
        ),
        "candidate_combinations":
            candidate_count,
        "timeframes":
            list(
                TIMEFRAMES.keys()
            ),
        "horizons_minutes":
            HORIZONS_MINUTES,
        "targets": {
            "selector_confidence":
                TARGET_SELECTOR_CONFIDENCE,
            "win_rate_pct":
                TARGET_WIN_RATE_PCT,
            "minimum_trades_per_day":
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
            "timeframe":
                str(
                    best[
                        "timeframe"
                    ]
                ),
            "horizon_minutes":
                int(
                    best[
                        "horizon_minutes"
                    ]
                ),
            "proposer_threshold":
                float(
                    best[
                        "proposer_threshold"
                    ]
                ),
            "selector_threshold":
                float(
                    best[
                        "selector_threshold"
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
            "Validation determines candidate ranking. "
            "Later test is evidence only. Any promising "
            "V3 configuration must be frozen and evaluated "
            "on new forward paper data before deployment."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 180
    )
    print(
        "TOP V3 CANDIDATES"
    )
    print(
        "=" * 180
    )

    display_columns = [
        "timeframe",
        "horizon_minutes",
        "proposer_threshold",
        "selector_threshold",
        "trades",
        "trades_per_day",
        "median_gap_hours",
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
    print(TRADE_FILE)
    print(TOP_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
