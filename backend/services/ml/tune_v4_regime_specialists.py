from __future__ import annotations

from pathlib import Path
import json
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "v4_regime_tuning"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
RESULT_FILE = OUTPUT_DIR / "validation_search.csv"
TOP_FILE = OUTPUT_DIR / "top_validation_candidates.csv"
TEST_FILE = OUTPUT_DIR / "selected_test_result.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
MODEL_FILE = MODEL_DIR / "v4_regime_candidate.joblib"

BASE_MINUTES = 5
HORIZONS = [30, 60]
TPS = [0.0035, 0.0050, 0.0075]
SLS = [0.0020, 0.0030]
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

TARGET_CONFIDENCE = 0.80
TARGET_WIN_RATE_PCT = 80.0
TARGET_TRADES_PER_DAY = 4.0
TARGET_MEDIAN_GAP_HOURS = 2.0
MIN_TARGET_TRADES = 50

MODEL_CONFIGS = [
    dict(
        name="precision",
        n_estimators=320,
        max_depth=3,
        learning_rate=0.025,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.4,
        reg_lambda=2.5,
        gamma=0.05,
        weight_mode="none",
    ),
    dict(
        name="light_balance",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.025,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        gamma=0.05,
        weight_mode="sqrt",
    ),
]


def pos_weight(y, mode):
    pos = float(y.sum())
    neg = float(len(y) - pos)
    ratio = neg / max(pos, 1.0)
    if mode == "none":
        return 1.0
    return float(np.sqrt(ratio))


def build_model(y, cfg):
    return XGBClassifier(
        n_estimators=int(cfg["n_estimators"]),
        max_depth=int(cfg["max_depth"]),
        learning_rate=float(cfg["learning_rate"]),
        min_child_weight=float(cfg["min_child_weight"]),
        subsample=float(cfg["subsample"]),
        colsample_bytree=float(cfg["colsample_bytree"]),
        reg_alpha=float(cfg["reg_alpha"]),
        reg_lambda=float(cfg["reg_lambda"]),
        gamma=float(cfg["gamma"]),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        device=os.getenv("TRADEMIND_XGB_DEVICE", "cpu"),
        scale_pos_weight=pos_weight(y, cfg["weight_mode"]),
    )


class ConstantModel:
    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, X):
        p = np.full(len(X), self.p, dtype=float)
        return np.column_stack([1.0 - p, p])


def fit_side(X, y, model_rows, cal_rows, cfg):
    yt = y[model_rows]
    unique = np.unique(yt)

    if len(unique) < 2:
        model = ConstantModel(float(unique[0]))
    else:
        model = build_model(yt, cfg)
        model.fit(X.iloc[model_rows], yt)

    raw = model.predict_proba(X.iloc[cal_rows])[:, 1]
    cy = y[cal_rows]

    if len(np.unique(raw)) < 2 or len(np.unique(cy)) < 2:
        calibrator = None
        calibrated = raw
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw, cy)
        calibrated = calibrator.transform(raw)

    brier = float(brier_score_loss(cy, calibrated))
    return model, calibrator, brier


def predict(model, calibrator, X, rows):
    raw = model.predict_proba(X.iloc[rows])[:, 1]
    if calibrator is None:
        return raw
    return calibrator.transform(raw)


def strict(threshold, m):
    return bool(
        threshold >= TARGET_CONFIDENCE
        and m["trades"] >= MIN_TARGET_TRADES
        and m["win_rate_mid_pct"] >= TARGET_WIN_RATE_PCT
        and m["trades_per_day"] >= TARGET_TRADES_PER_DAY
        and m["median_gap_hours"] <= TARGET_MEDIAN_GAP_HOURS
        and m["net_mid_sum_pct"] > 0
        and m["net_high_sum_pct"] > 0
    )


def rank_tier(threshold, m):
    if strict(threshold, m):
        return 5
    if (
        threshold >= 0.80
        and m["trades"] >= 50
        and m["win_rate_mid_pct"] >= 70
        and m["net_high_sum_pct"] > 0
    ):
        return 4
    if threshold >= 0.80 and m["trades"] >= 50 and m["net_high_sum_pct"] > 0:
        return 3
    if m["trades"] >= 50 and m["net_high_sum_pct"] > 0:
        return 2
    if threshold >= 0.80 and m["trades"] >= 50:
        return 1
    return 0


def distance_score(threshold, m):
    gap = m["median_gap_hours"] if np.isfinite(m["median_gap_hours"]) else 999.0
    penalty = (
        max(0.0, TARGET_CONFIDENCE - threshold) * 200
        + max(0.0, TARGET_WIN_RATE_PCT - m["win_rate_mid_pct"]) * 1.5
        + max(0.0, TARGET_TRADES_PER_DAY - m["trades_per_day"]) * 12
        + max(0.0, gap - TARGET_MEDIAN_GAP_HOURS) * 2
        + max(0, MIN_TARGET_TRADES - m["trades"]) * 0.2
        + (60 if m["net_high_sum_pct"] <= 0 else 0)
    )
    reward = (
        m["win_rate_mid_pct"] * 0.3
        + min(m["trades_per_day"], 12) * 2
        + max(min(m["net_high_sum_pct"], 100), -100) * 0.1
    )
    return reward - penalty


def regime_masks(frame, train_rows):
    train = frame.iloc[train_rows]

    trend_abs_q = float(
        train["close_to_sma_60m"].abs().quantile(0.65)
    )
    vol_hi = float(
        train["rolling_volatility_60m"].quantile(0.70)
    )
    vol_lo = float(
        train["rolling_volatility_60m"].quantile(0.30)
    )
    bb_hi = float(
        train["bollinger_width_30m"].quantile(0.70)
    )

    trend_abs = frame["close_to_sma_60m"].abs()
    vol = frame["rolling_volatility_60m"]
    rsi = frame["rsi_14"]
    relvol = frame["relative_volume_60m"]
    macd_hist = frame["macd_histogram"]
    bb_width = frame["bollinger_width_30m"]

    return {
        "all": np.ones(len(frame), dtype=bool),
        "strong_trend": (
            (trend_abs >= trend_abs_q)
            & (
                (
                    (frame["close_to_sma_60m"] > 0)
                    & (macd_hist > 0)
                )
                | (
                    (frame["close_to_sma_60m"] < 0)
                    & (macd_hist < 0)
                )
            )
        ).to_numpy(),
        "mean_reversion": (
            (rsi <= 30)
            | (rsi >= 70)
        ).to_numpy(),
        "high_volatility": (
            (vol >= vol_hi)
            | (bb_width >= bb_hi)
        ).to_numpy(),
        "low_volatility": (
            vol <= vol_lo
        ).to_numpy(),
        "high_volume": (
            relvol >= 1.25
        ).to_numpy(),
    }


def main():
    print("TRADEMIND V4 REGIME-SPECIALIST TUNER")
    print("Labels require profit after HIGH trading costs.")

    frame, features = load_dataset()
    X = frame[features]

    pre = np.flatnonzero((frame["open_time"] < VALIDATION_START).to_numpy())
    val = np.flatnonzero(
        (
            (frame["open_time"] >= VALIDATION_START)
            & (frame["open_time"] < TEST_START)
        ).to_numpy()
    )
    test = np.flatnonzero((frame["open_time"] >= TEST_START).to_numpy())

    split = int(len(pre) * 0.80)
    model_base = pre[:split]
    cal_base = pre[split:]

    regimes = regime_masks(frame, model_base)
    rows_out = []
    artifacts = {}

    total = (
        len(HORIZONS)
        * len(TPS)
        * len(SLS)
        * len(regimes)
        * len(MODEL_CONFIGS)
    )
    done = 0

    for horizon_minutes in HORIZONS:
        horizon_bars = horizon_minutes // BASE_MINUTES

        for tp in TPS:
            for sl in SLS:
                outcomes = barrier_outcomes(frame, tp, sl, horizon_bars)
                valid = outcomes["valid"]

                long_target = (
                    (outcomes["long_gross"] - HIGH_COST) > 0
                ).astype(np.int8)
                short_target = (
                    (outcomes["short_gross"] - HIGH_COST) > 0
                ).astype(np.int8)

                for regime_name, regime_mask in regimes.items():
                    mr = model_base[valid[model_base] & regime_mask[model_base]]
                    cr = cal_base[valid[cal_base] & regime_mask[cal_base]]
                    vr = val[valid[val] & regime_mask[val]]

                    if len(mr) < 1000 or len(cr) < 200 or len(vr) < 200:
                        continue

                    for cfg in MODEL_CONFIGS:
                        done += 1
                        if done == 1 or done % 20 == 0:
                            print(
                                f"[{done}/{total}] {regime_name} "
                                f"{cfg['name']} hold={horizon_minutes} "
                                f"TP={tp*100:.2f}% SL={sl*100:.2f}%"
                            )

                        lm, lc, lb = fit_side(
                            X, long_target, mr, cr, cfg
                        )
                        sm, sc, sb = fit_side(
                            X, short_target, mr, cr, cfg
                        )

                        lp = predict(lm, lc, X, vr)
                        sp = predict(sm, sc, X, vr)

                        artifact_key = (
                            regime_name,
                            cfg["name"],
                            horizon_minutes,
                            tp,
                            sl,
                        )
                        artifacts[artifact_key] = (
                            lm, lc, sm, sc, cfg
                        )

                        for threshold in THRESHOLDS:
                            m = evaluate(
                                frame,
                                vr,
                                lp,
                                sp,
                                outcomes,
                                threshold,
                                VALIDATION_START,
                                TEST_START,
                            )
                            rows_out.append({
                                "regime": regime_name,
                                "model_config": cfg["name"],
                                "horizon_minutes": horizon_minutes,
                                "take_profit_pct": tp * 100.0,
                                "stop_loss_pct": sl * 100.0,
                                "confidence_threshold": threshold,
                                "long_brier": lb,
                                "short_brier": sb,
                                **m,
                                "strict_target_met": strict(threshold, m),
                                "tier": rank_tier(threshold, m),
                                "score": distance_score(threshold, m),
                            })

    results = pd.DataFrame(rows_out)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_FILE, index=False)

    ranked = results.sort_values(
        ["tier", "score", "win_rate_mid_pct", "trades_per_day"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    ranked.head(30).to_csv(TOP_FILE, index=False)
    selected = ranked.iloc[0]

    regime_name = str(selected["regime"])
    cfg_name = str(selected["model_config"])
    horizon_minutes = int(selected["horizon_minutes"])
    tp = float(selected["take_profit_pct"]) / 100.0
    sl = float(selected["stop_loss_pct"]) / 100.0
    threshold = float(selected["confidence_threshold"])

    outcomes = barrier_outcomes(
        frame, tp, sl, horizon_minutes // BASE_MINUTES
    )
    valid = outcomes["valid"]
    regime_mask = regimes[regime_name]

    tr = test[valid[test] & regime_mask[test]]
    key = (regime_name, cfg_name, horizon_minutes, tp, sl)
    lm, lc, sm, sc, cfg = artifacts[key]

    lp = predict(lm, lc, X, tr)
    sp = predict(sm, sc, X, tr)

    tm = evaluate(
        frame,
        tr,
        lp,
        sp,
        outcomes,
        threshold,
        TEST_START,
        frame["open_time"].iloc[-1] + pd.Timedelta(minutes=5),
    )

    test_row = {
        "regime": regime_name,
        "model_config": cfg_name,
        "horizon_minutes": horizon_minutes,
        "take_profit_pct": tp * 100.0,
        "stop_loss_pct": sl * 100.0,
        "confidence_threshold": threshold,
        **tm,
        "strict_target_met": strict(threshold, tm),
        "tier": rank_tier(threshold, tm),
    }

    pd.DataFrame([test_row]).to_csv(TEST_FILE, index=False)

    joblib.dump({
        "research_only": True,
        "feature_columns": features,
        "regime": regime_name,
        "regime_thresholds_from_training": True,
        "model_config": cfg,
        "long_model": lm,
        "long_calibrator": lc,
        "short_model": sm,
        "short_calibrator": sc,
        "horizon_minutes": horizon_minutes,
        "take_profit": tp,
        "stop_loss": sl,
        "confidence_threshold": threshold,
    }, MODEL_FILE)

    summary = {
        "research_only": True,
        "selection_policy": (
            "Regime definition thresholds come from training data. "
            "Model/regime/barrier/threshold selected on validation only; "
            "later test evaluated once."
        ),
        "regimes_tested": list(regimes.keys()),
        "selected_validation": selected.to_dict(),
        "later_test": test_row,
        "qualified_for_forward_paper": bool(
            selected["strict_target_met"]
            and test_row["strict_target_met"]
        ),
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )

    print("\nVALIDATION WINNER")
    print(selected.to_string())
    print("\nLATER TEST")
    print(pd.Series(test_row).to_string())
    print("\nQualified:", summary["qualified_for_forward_paper"])


if __name__ == "__main__":
    main()
