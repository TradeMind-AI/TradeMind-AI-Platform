from __future__ import annotations

from pathlib import Path
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

from research_v3_barrier_opportunity import (
    HIGH_COST,
    MID_COST,
    TEST_START,
    VALIDATION_START,
    barrier_outcomes,
    evaluate,
    load_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4_xgb_tuning"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
)

RESULT_FILE = OUTPUT_DIR / "validation_search.csv"
TOP_FILE = OUTPUT_DIR / "top_validation_candidates.csv"
TEST_FILE = OUTPUT_DIR / "selected_test_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
MODEL_FILE = MODEL_DIR / "v4_xgb_candidate.joblib"

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
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

TARGET_CONFIDENCE = 0.80
TARGET_WIN_RATE_PCT = 80.0
TARGET_TRADES_PER_DAY = 4.0
TARGET_MEDIAN_GAP_HOURS = 2.0
MIN_TARGET_TRADES = 50

MODEL_CONFIGS = [
    {
        "name": "balanced_shallow",
        "n_estimators": 220,
        "max_depth": 3,
        "learning_rate": 0.035,
        "min_child_weight": 8,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.20,
        "reg_lambda": 2.00,
        "gamma": 0.00,
        "class_weight_scale": 1.00,
    },
    {
        "name": "balanced_medium",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.030,
        "min_child_weight": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 1.50,
        "gamma": 0.00,
        "class_weight_scale": 1.00,
    },
    {
        "name": "strong_regularization",
        "n_estimators": 260,
        "max_depth": 4,
        "learning_rate": 0.035,
        "min_child_weight": 10,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.50,
        "reg_lambda": 3.00,
        "gamma": 0.05,
        "class_weight_scale": 1.00,
    },
    {
        "name": "high_recall",
        "n_estimators": 260,
        "max_depth": 4,
        "learning_rate": 0.040,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.50,
        "gamma": 0.00,
        "class_weight_scale": 1.35,
    },
    {
        "name": "precision_weighted",
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.025,
        "min_child_weight": 8,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.30,
        "reg_lambda": 2.50,
        "gamma": 0.05,
        "class_weight_scale": 0.70,
    },
    {
        "name": "deeper_slow",
        "n_estimators": 360,
        "max_depth": 6,
        "learning_rate": 0.020,
        "min_child_weight": 10,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.40,
        "reg_lambda": 3.00,
        "gamma": 0.10,
        "class_weight_scale": 0.85,
    },
]


def build_model(
    y: np.ndarray,
    config: dict,
) -> XGBClassifier:
    positives = float(y.sum())
    negatives = float(len(y) - positives)

    base_weight = (
        negatives
        / max(positives, 1.0)
    )

    scale_pos_weight = (
        base_weight
        * float(
            config[
                "class_weight_scale"
            ]
        )
    )

    return XGBClassifier(
        n_estimators=int(
            config["n_estimators"]
        ),
        max_depth=int(
            config["max_depth"]
        ),
        learning_rate=float(
            config["learning_rate"]
        ),
        min_child_weight=float(
            config["min_child_weight"]
        ),
        subsample=float(
            config["subsample"]
        ),
        colsample_bytree=float(
            config["colsample_bytree"]
        ),
        reg_alpha=float(
            config["reg_alpha"]
        ),
        reg_lambda=float(
            config["reg_lambda"]
        ),
        gamma=float(
            config["gamma"]
        ),
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


def fit_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_rows: np.ndarray,
    calibration_rows: np.ndarray,
    config: dict,
) -> tuple[
    XGBClassifier,
    IsotonicRegression | None,
    float,
]:
    model = build_model(
        y[model_rows],
        config,
    )

    model.fit(
        X.iloc[model_rows],
        y[model_rows],
    )

    raw = model.predict_proba(
        X.iloc[calibration_rows]
    )[:, 1]

    calibration_target = y[
        calibration_rows
    ]

    calibrator = None

    if (
        len(np.unique(raw)) >= 2
        and len(
            np.unique(
                calibration_target
            )
        ) >= 2
    ):
        calibrator = IsotonicRegression(
            out_of_bounds="clip",
        )

        calibrator.fit(
            raw,
            calibration_target,
        )

        calibrated = calibrator.transform(
            raw
        )
    else:
        calibrated = raw

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


def predict(
    model: XGBClassifier,
    calibrator:
        IsotonicRegression | None,
    X: pd.DataFrame,
    rows: np.ndarray,
) -> np.ndarray:
    raw = model.predict_proba(
        X.iloc[rows]
    )[:, 1]

    if calibrator is None:
        return raw

    return calibrator.transform(raw)


def strict_target(
    threshold: float,
    metrics: dict,
) -> bool:
    return bool(
        threshold
        >= TARGET_CONFIDENCE
        and metrics["trades"]
        >= MIN_TARGET_TRADES
        and metrics["win_rate_mid_pct"]
        >= TARGET_WIN_RATE_PCT
        and metrics["trades_per_day"]
        >= TARGET_TRADES_PER_DAY
        and metrics["median_gap_hours"]
        <= TARGET_MEDIAN_GAP_HOURS
        and metrics["net_mid_sum_pct"]
        > 0.0
        and metrics["net_high_sum_pct"]
        > 0.0
    )


def score(
    row: dict,
) -> float:
    win_rate = float(
        row["win_rate_mid_pct"]
    )
    frequency = float(
        row["trades_per_day"]
    )
    avg_net = float(
        row["avg_net_mid_pct"]
    )
    high_net = float(
        row["net_high_sum_pct"]
    )
    trades = int(
        row["trades"]
    )
    gap = float(
        row["median_gap_hours"]
    )

    if not np.isfinite(gap):
        gap = 999.0

    return (
        win_rate
        + frequency * 8.0
        + avg_net * 30.0
        + min(trades, 300) * 0.03
        - max(
            0.0,
            TARGET_WIN_RATE_PCT
            - win_rate,
        ) * 0.9
        - max(
            0.0,
            TARGET_TRADES_PER_DAY
            - frequency,
        ) * 8.0
        - max(
            0.0,
            gap
            - TARGET_MEDIAN_GAP_HOURS,
        ) * 2.0
        + (
            15.0
            if high_net > 0.0
            else -15.0
        )
    )


def main() -> None:
    print(
        "TRADEMIND V4 - XGBOOST "
        "TRADE-SUCCESS TUNER"
    )

    print()
    print(
        "Validation chooses the model."
    )
    print(
        "Later test is not used during "
        "hyperparameter selection."
    )

    (
        frame,
        feature_columns,
    ) = load_dataset()

    X = frame[
        feature_columns
    ]

    prevalidation_rows = np.flatnonzero(
        (
            frame["open_time"]
            < VALIDATION_START
        ).to_numpy()
    )

    validation_rows = np.flatnonzero(
        (
            (
                frame["open_time"]
                >= VALIDATION_START
            )
            & (
                frame["open_time"]
                < TEST_START
            )
        ).to_numpy()
    )

    test_rows = np.flatnonzero(
        (
            frame["open_time"]
            >= TEST_START
        ).to_numpy()
    )

    split = int(
        len(prevalidation_rows)
        * 0.80
    )

    model_rows = (
        prevalidation_rows[:split]
    )

    calibration_rows = (
        prevalidation_rows[split:]
    )

    print(
        f"Prevalidation rows: "
        f"{len(prevalidation_rows):,}"
    )
    print(
        f"Validation rows: "
        f"{len(validation_rows):,}"
    )
    print(
        f"Later test rows: "
        f"{len(test_rows):,}"
    )
    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    search_rows = []

    total_configs = (
        len(HORIZON_MINUTES)
        * len(TAKE_PROFIT_LEVELS)
        * len(STOP_LOSS_LEVELS)
        * len(MODEL_CONFIGS)
    )

    completed = 0

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
                outcomes = (
                    barrier_outcomes(
                        frame,
                        take_profit,
                        stop_loss,
                        horizon_bars,
                    )
                )

                valid = outcomes["valid"]

                mr = model_rows[
                    valid[model_rows]
                ]
                cr = calibration_rows[
                    valid[
                        calibration_rows
                    ]
                ]
                vr = validation_rows[
                    valid[
                        validation_rows
                    ]
                ]

                long_target = (
                    (
                        outcomes[
                            "long_gross"
                        ]
                        - MID_COST
                    )
                    > 0.0
                ).astype(np.int8)

                short_target = (
                    (
                        outcomes[
                            "short_gross"
                        ]
                        - MID_COST
                    )
                    > 0.0
                ).astype(np.int8)

                for config in (
                    MODEL_CONFIGS
                ):
                    completed += 1

                    print(
                        f"[{completed}/"
                        f"{total_configs}] "
                        f"{config['name']} "
                        f"hold={horizon_minutes}m "
                        f"TP={take_profit*100:.2f}% "
                        f"SL={stop_loss*100:.2f}%"
                    )

                    (
                        long_model,
                        long_calibrator,
                        long_brier,
                    ) = fit_model(
                        X,
                        long_target,
                        mr,
                        cr,
                        config,
                    )

                    (
                        short_model,
                        short_calibrator,
                        short_brier,
                    ) = fit_model(
                        X,
                        short_target,
                        mr,
                        cr,
                        config,
                    )

                    long_probability = (
                        predict(
                            long_model,
                            long_calibrator,
                            X,
                            vr,
                        )
                    )

                    short_probability = (
                        predict(
                            short_model,
                            short_calibrator,
                            X,
                            vr,
                        )
                    )

                    for threshold in (
                        CONFIDENCE_THRESHOLDS
                    ):
                        metrics = evaluate(
                            frame,
                            vr,
                            long_probability,
                            short_probability,
                            outcomes,
                            threshold,
                            VALIDATION_START,
                            TEST_START,
                        )

                        row = {
                            "model_config":
                                config["name"],
                            "horizon_minutes":
                                horizon_minutes,
                            "take_profit_pct":
                                take_profit
                                * 100.0,
                            "stop_loss_pct":
                                stop_loss
                                * 100.0,
                            "confidence_threshold":
                                threshold,
                            "long_brier":
                                long_brier,
                            "short_brier":
                                short_brier,
                            **metrics,
                        }

                        row[
                            "strict_target_met"
                        ] = strict_target(
                            threshold,
                            metrics,
                        )

                        row[
                            "research_score"
                        ] = score(row)

                        search_rows.append(
                            row
                        )

    search = pd.DataFrame(
        search_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    search.to_csv(
        RESULT_FILE,
        index=False,
    )

    ranked = search.sort_values(
        [
            "strict_target_met",
            "research_score",
            "win_rate_mid_pct",
            "trades_per_day",
            "net_high_sum_pct",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    top = ranked.head(25)
    top.to_csv(
        TOP_FILE,
        index=False,
    )

    selected = ranked.iloc[0]

    selected_config = next(
        config
        for config in MODEL_CONFIGS
        if (
            config["name"]
            == selected[
                "model_config"
            ]
        )
    )

    horizon_minutes = int(
        selected[
            "horizon_minutes"
        ]
    )
    horizon_bars = (
        horizon_minutes
        // BASE_MINUTES
    )
    take_profit = float(
        selected[
            "take_profit_pct"
        ]
    ) / 100.0
    stop_loss = float(
        selected[
            "stop_loss_pct"
        ]
    ) / 100.0
    threshold = float(
        selected[
            "confidence_threshold"
        ]
    )

    print()
    print(
        "Validation-selected candidate:"
    )
    print(
        selected.to_string()
    )

    outcomes = barrier_outcomes(
        frame,
        take_profit,
        stop_loss,
        horizon_bars,
    )

    valid = outcomes["valid"]

    mr = model_rows[
        valid[model_rows]
    ]
    cr = calibration_rows[
        valid[calibration_rows]
    ]
    tr = test_rows[
        valid[test_rows]
    ]

    long_target = (
        (
            outcomes["long_gross"]
            - MID_COST
        )
        > 0.0
    ).astype(np.int8)

    short_target = (
        (
            outcomes["short_gross"]
            - MID_COST
        )
        > 0.0
    ).astype(np.int8)

    (
        long_model,
        long_calibrator,
        long_brier,
    ) = fit_model(
        X,
        long_target,
        mr,
        cr,
        selected_config,
    )

    (
        short_model,
        short_calibrator,
        short_brier,
    ) = fit_model(
        X,
        short_target,
        mr,
        cr,
        selected_config,
    )

    long_probability = predict(
        long_model,
        long_calibrator,
        X,
        tr,
    )

    short_probability = predict(
        short_model,
        short_calibrator,
        X,
        tr,
    )

    test_metrics = evaluate(
        frame,
        tr,
        long_probability,
        short_probability,
        outcomes,
        threshold,
        TEST_START,
        frame[
            "open_time"
        ].iloc[-1]
        + pd.Timedelta(
            minutes=BASE_MINUTES
        ),
    )

    test_row = {
        "model_config":
            selected_config["name"],
        "horizon_minutes":
            horizon_minutes,
        "take_profit_pct":
            take_profit * 100.0,
        "stop_loss_pct":
            stop_loss * 100.0,
        "confidence_threshold":
            threshold,
        "long_brier":
            long_brier,
        "short_brier":
            short_brier,
        **test_metrics,
        "strict_target_met":
            strict_target(
                threshold,
                test_metrics,
            ),
    }

    pd.DataFrame(
        [test_row]
    ).to_csv(
        TEST_FILE,
        index=False,
    )

    joblib.dump(
        {
            "research_only": True,
            "feature_columns":
                feature_columns,
            "long_model":
                long_model,
            "short_model":
                short_model,
            "long_calibrator":
                long_calibrator,
            "short_calibrator":
                short_calibrator,
            "model_config":
                selected_config,
            "horizon_minutes":
                horizon_minutes,
            "take_profit":
                take_profit,
            "stop_loss":
                stop_loss,
            "confidence_threshold":
                threshold,
        },
        MODEL_FILE,
    )

    summary = {
        "research_only": True,
        "selection_policy": (
            "Hyperparameters, barrier settings "
            "and confidence threshold are selected "
            "using validation only. Later test is "
            "evaluated once after selection."
        ),
        "target": {
            "confidence":
                TARGET_CONFIDENCE,
            "win_rate_pct":
                TARGET_WIN_RATE_PCT,
            "trades_per_day":
                TARGET_TRADES_PER_DAY,
            "median_gap_hours":
                TARGET_MEDIAN_GAP_HOURS,
            "minimum_trades":
                MIN_TARGET_TRADES,
            "positive_high_cost_net":
                True,
        },
        "search_space": {
            "model_configs":
                len(MODEL_CONFIGS),
            "barrier_configs":
                (
                    len(
                        HORIZON_MINUTES
                    )
                    * len(
                        TAKE_PROFIT_LEVELS
                    )
                    * len(
                        STOP_LOSS_LEVELS
                    )
                ),
            "thresholds":
                len(
                    CONFIDENCE_THRESHOLDS
                ),
            "total_validation_rows":
                len(search),
        },
        "selected_validation": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (bool, np.bool_),
                )
                else float(value)
                if isinstance(
                    value,
                    (float, np.floating),
                )
                else int(value)
                if isinstance(
                    value,
                    (int, np.integer),
                )
                else value
            )
            for key, value
            in selected.to_dict().items()
        },
        "later_test": test_row,
        "qualified_for_forward_paper":
            bool(
                selected[
                    "strict_target_met"
                ]
                and test_row[
                    "strict_target_met"
                ]
            ),
        "warning": (
            "Meeting the target historically "
            "does not guarantee future performance. "
            "A qualifying candidate must still be "
            "frozen and forward-paper tested."
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
        "Later-test result:"
    )
    print(
        pd.Series(
            test_row
        ).to_string()
    )

    print()
    print(
        "Qualified for forward paper: "
        f"{summary['qualified_for_forward_paper']}"
    )

    print()
    print("Saved:")
    print(RESULT_FILE)
    print(TOP_FILE)
    print(TEST_FILE)
    print(SUMMARY_FILE)
    print(MODEL_FILE)


if __name__ == "__main__":
    main()
