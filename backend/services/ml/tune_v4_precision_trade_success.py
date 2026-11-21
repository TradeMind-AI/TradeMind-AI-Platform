from __future__ import annotations

from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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
    / "v4_precision_tuning"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
)

RESULT_FILE = OUTPUT_DIR / "validation_search.csv"
TOP_FILE = OUTPUT_DIR / "top_validation_candidates.csv"
TEST_FILE = OUTPUT_DIR / "selected_test_result.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
MODEL_FILE = MODEL_DIR / "v4_precision_candidate.joblib"

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
    0.0015,
    0.0020,
    0.0030,
]

CONFIDENCE_THRESHOLDS = [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

TARGET_CONFIDENCE = 0.80
TARGET_WIN_RATE_PCT = 80.0
TARGET_TRADES_PER_DAY = 4.0
TARGET_MEDIAN_GAP_HOURS = 2.0
MIN_TARGET_TRADES = 50

TARGET_MODES = {
    "high_cost_profit": HIGH_COST,
    "high_cost_plus_buffer": HIGH_COST + 0.0010,
}

MODEL_CONFIGS = [
    {
        "name": "precision_shallow",
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.025,
        "min_child_weight": 12,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.50,
        "reg_lambda": 3.00,
        "gamma": 0.10,
        "weight_mode": "none",
    },
    {
        "name": "light_balance",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.025,
        "min_child_weight": 8,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.25,
        "reg_lambda": 2.00,
        "gamma": 0.05,
        "weight_mode": "sqrt_balanced",
    },
]

CALIBRATION_METHODS = [
    "platt",
]


class ConstantProbabilityModel:
    def __init__(
        self,
        probability: float,
    ) -> None:
        self.probability = float(
            np.clip(
                probability,
                0.0,
                1.0,
            )
        )

    def predict_proba(
        self,
        X,
    ) -> np.ndarray:
        count = len(X)
        positive = np.full(
            count,
            self.probability,
            dtype=float,
        )
        return np.column_stack(
            [
                1.0 - positive,
                positive,
            ]
        )


def scale_pos_weight(
    y: np.ndarray,
    mode: str,
) -> float:
    positives = float(y.sum())
    negatives = float(
        len(y) - positives
    )

    ratio = (
        negatives
        / max(positives, 1.0)
    )

    if mode == "none":
        return 1.0

    if mode == "sqrt_balanced":
        return float(
            np.sqrt(ratio)
        )

    return ratio


def build_model(
    y: np.ndarray,
    config: dict,
) -> XGBClassifier:
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
            scale_pos_weight(
                y,
                config["weight_mode"],
            ),
    )


def fit_calibrator(
    raw_probability: np.ndarray,
    y: np.ndarray,
    method: str,
):
    if method == "platt":
        eps = 1e-6
        p = np.clip(
            raw_probability,
            eps,
            1.0 - eps,
        )
        logits = np.log(
            p / (1.0 - p)
        ).reshape(-1, 1)

        model = LogisticRegression(
            C=1.0,
            max_iter=500,
        )
        model.fit(
            logits,
            y,
        )

        calibrated = model.predict_proba(
            logits
        )[:, 1]

        return (
            model,
            calibrated,
        )

    model = IsotonicRegression(
        out_of_bounds="clip",
    )
    model.fit(
        raw_probability,
        y,
    )

    return (
        model,
        model.transform(
            raw_probability
        ),
    )


def apply_calibrator(
    raw_probability: np.ndarray,
    calibrator,
    method: str,
) -> np.ndarray:
    if method == "platt":
        eps = 1e-6
        p = np.clip(
            raw_probability,
            eps,
            1.0 - eps,
        )
        logits = np.log(
            p / (1.0 - p)
        ).reshape(-1, 1)

        return calibrator.predict_proba(
            logits
        )[:, 1]

    return calibrator.transform(
        raw_probability
    )


def fit_side(
    X: pd.DataFrame,
    y: np.ndarray,
    model_rows: np.ndarray,
    calibration_rows: np.ndarray,
    config: dict,
    calibration_method: str,
):
    model_target = y[
        model_rows
    ]

    unique_model_target = np.unique(
        model_target
    )

    if len(
        unique_model_target
    ) < 2:
        probability = float(
            unique_model_target[0]
        )

        model = ConstantProbabilityModel(
            probability
        )
    else:
        model = build_model(
            model_target,
            config,
        )

        model.fit(
            X.iloc[model_rows],
            model_target,
        )

    raw = model.predict_proba(
        X.iloc[calibration_rows]
    )[:, 1]

    calibration_target = y[
        calibration_rows
    ]

    if (
        len(
            np.unique(
                calibration_target
            )
        ) < 2
        or len(
            np.unique(
                raw
            )
        ) < 2
    ):
        brier = float(
            brier_score_loss(
                calibration_target,
                raw,
            )
        )

        return (
            model,
            None,
            calibration_method,
            brier,
        )

    (
        calibrator,
        calibrated,
    ) = fit_calibrator(
        raw,
        calibration_target,
        calibration_method,
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
        calibration_method,
        brier,
    )


def predict(
    model,
    calibrator,
    method: str,
    X: pd.DataFrame,
    rows: np.ndarray,
) -> np.ndarray:
    raw = model.predict_proba(
        X.iloc[rows]
    )[:, 1]

    if calibrator is None:
        return raw

    return apply_calibrator(
        raw,
        calibrator,
        method,
    )


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


def eligibility_tier(
    threshold: float,
    metrics: dict,
) -> int:
    if strict_target(
        threshold,
        metrics,
    ):
        return 4

    if (
        threshold
        >= TARGET_CONFIDENCE
        and metrics["trades"] >= 50
        and metrics["net_high_sum_pct"] > 0.0
    ):
        return 3

    if (
        metrics["trades"] >= 50
        and metrics["net_high_sum_pct"] > 0.0
    ):
        return 2

    if (
        threshold
        >= TARGET_CONFIDENCE
        and metrics["trades"] >= 50
    ):
        return 1

    return 0


def distance_score(
    threshold: float,
    metrics: dict,
) -> float:
    win = float(
        metrics["win_rate_mid_pct"]
    )
    freq = float(
        metrics["trades_per_day"]
    )
    gap = float(
        metrics["median_gap_hours"]
    )
    trades = int(
        metrics["trades"]
    )
    high_net = float(
        metrics["net_high_sum_pct"]
    )

    if not np.isfinite(gap):
        gap = 999.0

    penalty = (
        max(
            0.0,
            TARGET_CONFIDENCE
            - threshold,
        )
        * 100.0
        * 2.0
        + max(
            0.0,
            TARGET_WIN_RATE_PCT
            - win,
        )
        * 1.5
        + max(
            0.0,
            TARGET_TRADES_PER_DAY
            - freq,
        )
        * 12.0
        + max(
            0.0,
            gap
            - TARGET_MEDIAN_GAP_HOURS,
        )
        * 2.0
        + max(
            0,
            MIN_TARGET_TRADES
            - trades,
        )
        * 0.25
        + (
            50.0
            if high_net <= 0.0
            else 0.0
        )
    )

    reward = (
        min(win, 100.0)
        * 0.25
        + min(freq, 12.0)
        * 2.0
        + max(
            min(high_net, 100.0),
            -100.0,
        )
        * 0.10
    )

    return reward - penalty


def main() -> None:
    print(
        "TRADEMIND V4.1 - "
        "PRECISION-FIRST TUNER"
    )
    print()
    print(
        "Changes from V4:"
    )
    print(
        "  labels can require profit "
        "after HIGH costs"
    )
    print(
        "  no-reweight and light-reweight "
        "models are tested"
    )
    print(
        "  Platt and isotonic calibration "
        "are compared"
    )
    print(
        "  ranking cannot prefer frequency "
        "while ignoring the hard criteria"
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

    validation_results = []

    total = (
        len(HORIZON_MINUTES)
        * len(TAKE_PROFIT_LEVELS)
        * len(STOP_LOSS_LEVELS)
        * len(TARGET_MODES)
        * len(MODEL_CONFIGS)
        * len(CALIBRATION_METHODS)
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
                vr = validation_rows[
                    valid[validation_rows]
                ]

                for (
                    target_mode,
                    target_cost,
                ) in TARGET_MODES.items():
                    long_target = (
                        (
                            outcomes[
                                "long_gross"
                            ]
                            - target_cost
                        )
                        > 0.0
                    ).astype(np.int8)

                    short_target = (
                        (
                            outcomes[
                                "short_gross"
                            ]
                            - target_cost
                        )
                        > 0.0
                    ).astype(np.int8)

                    for config in (
                        MODEL_CONFIGS
                    ):
                        for (
                            calibration_method
                        ) in (
                            CALIBRATION_METHODS
                        ):
                            completed += 1

                            if (
                                completed == 1
                                or completed % 20 == 0
                            ):
                                print(
                                    f"[{completed}/{total}] "
                                    f"{config['name']} "
                                    f"{target_mode} "
                                    f"cal={calibration_method} "
                                    f"hold={horizon_minutes}m "
                                    f"TP={take_profit*100:.2f}% "
                                    f"SL={stop_loss*100:.2f}%"
                                )

                            (
                                long_model,
                                long_calibrator,
                                _,
                                long_brier,
                            ) = fit_side(
                                X,
                                long_target,
                                mr,
                                cr,
                                config,
                                calibration_method,
                            )

                            (
                                short_model,
                                short_calibrator,
                                _,
                                short_brier,
                            ) = fit_side(
                                X,
                                short_target,
                                mr,
                                cr,
                                config,
                                calibration_method,
                            )

                            long_probability = predict(
                                long_model,
                                long_calibrator,
                                calibration_method,
                                X,
                                vr,
                            )

                            short_probability = predict(
                                short_model,
                                short_calibrator,
                                calibration_method,
                                X,
                                vr,
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
                                    "target_mode":
                                        target_mode,
                                    "calibration_method":
                                        calibration_method,
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
                                    **metrics,
                                }

                                row[
                                    "strict_target_met"
                                ] = strict_target(
                                    threshold,
                                    metrics,
                                )

                                row[
                                    "eligibility_tier"
                                ] = eligibility_tier(
                                    threshold,
                                    metrics,
                                )

                                row[
                                    "distance_score"
                                ] = distance_score(
                                    threshold,
                                    metrics,
                                )

                                validation_results.append(
                                    row
                                )

    results = pd.DataFrame(
        validation_results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULT_FILE,
        index=False,
    )

    ranked = results.sort_values(
        [
            "eligibility_tier",
            "distance_score",
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

    ranked.head(30).to_csv(
        TOP_FILE,
        index=False,
    )

    selected = ranked.iloc[0]

    config = next(
        item
        for item in MODEL_CONFIGS
        if item["name"]
        == selected["model_config"]
    )

    target_mode = str(
        selected["target_mode"]
    )
    target_cost = TARGET_MODES[
        target_mode
    ]
    calibration_method = str(
        selected[
            "calibration_method"
        ]
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
            - target_cost
        )
        > 0.0
    ).astype(np.int8)

    short_target = (
        (
            outcomes["short_gross"]
            - target_cost
        )
        > 0.0
    ).astype(np.int8)

    (
        long_model,
        long_calibrator,
        _,
        long_brier,
    ) = fit_side(
        X,
        long_target,
        mr,
        cr,
        config,
        calibration_method,
    )

    (
        short_model,
        short_calibrator,
        _,
        short_brier,
    ) = fit_side(
        X,
        short_target,
        mr,
        cr,
        config,
        calibration_method,
    )

    long_probability = predict(
        long_model,
        long_calibrator,
        calibration_method,
        X,
        tr,
    )

    short_probability = predict(
        short_model,
        short_calibrator,
        calibration_method,
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
            config["name"],
        "target_mode":
            target_mode,
        "calibration_method":
            calibration_method,
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
        "eligibility_tier":
            eligibility_tier(
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
            "calibration_method":
                calibration_method,
            "model_config":
                config,
            "target_mode":
                target_mode,
            "target_cost":
                target_cost,
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
            "All tuning uses prevalidation + validation. "
            "The later test is evaluated once after the "
            "validation winner is selected."
        ),
        "validation_candidates":
            len(results),
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
            "A historical pass does not guarantee "
            "future performance. Any qualifying "
            "candidate must be frozen before new "
            "forward-paper evaluation."
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
        "VALIDATION WINNER"
    )
    print(
        selected.to_string()
    )
    print()
    print(
        "LATER TEST"
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
