from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import math
import random
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_DIR = PROJECT_ROOT / "backend" / "services" / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from research_v3_barrier_opportunity import (
    HIGH_COST,
    MID_COST,
    TEST_START,
    VALIDATION_START,
    barrier_outcomes,
    evaluate,
    load_dataset,
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "v4_transformer_tuning"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
RESULT_FILE = OUTPUT_DIR / "validation_search.csv"
TEST_FILE = OUTPUT_DIR / "selected_test_result.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
MODEL_FILE = MODEL_DIR / "v4_transformer_candidate.pt"
SCALER_FILE = MODEL_DIR / "v4_transformer_scaler.joblib"

HORIZON_MINUTES = 60
HORIZON_BARS = 12
TAKE_PROFIT = 0.0035
STOP_LOSS = 0.0030

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
TARGET_CONFIDENCE = 0.80
TARGET_WIN_RATE_PCT = 80.0
TARGET_TRADES_PER_DAY = 4.0
TARGET_MEDIAN_GAP_HOURS = 2.0
MIN_TARGET_TRADES = 50

CONFIGS = [
    dict(name="small_fast", seq=12, d_model=64, nhead=4, layers=2, ff=128, dropout=0.10, lr=4e-4, wd=1e-4, epochs=6, stride=2, weight_mode="none"),
    dict(name="medium", seq=24, d_model=96, nhead=4, layers=3, ff=192, dropout=0.10, lr=3e-4, wd=1e-4, epochs=7, stride=2, weight_mode="none"),
    dict(name="medium_dropout", seq=24, d_model=96, nhead=4, layers=3, ff=192, dropout=0.20, lr=3e-4, wd=3e-4, epochs=7, stride=2, weight_mode="sqrt"),
    dict(name="long_context", seq=48, d_model=96, nhead=4, layers=3, ff=192, dropout=0.15, lr=2e-4, wd=2e-4, epochs=7, stride=3, weight_mode="sqrt"),
]
BATCH_SIZE = 512
SEED = 42


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class SeqDataset(Dataset):
    def __init__(self, features, rows, targets, seq):
        self.features = features
        self.rows = rows
        self.targets = targets
        self.seq = seq

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = int(self.rows[idx])
        x = self.features[row-self.seq+1:row+1]
        return torch.from_numpy(x).float(), torch.from_numpy(self.targets[row]).float()


class TradeTransformer(nn.Module):
    def __init__(self, feature_count, cfg):
        super().__init__()
        self.seq = int(cfg["seq"])
        d_model = int(cfg["d_model"])
        self.proj = nn.Linear(feature_count, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.seq, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(cfg["nhead"]),
            dim_feedforward=int(cfg["ff"]),
            dropout=float(cfg["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(cfg["layers"]))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        return self.head(self.norm(x[:, -1, :]))


def valid_rows(rows, valid, seq):
    rows = rows[rows >= seq - 1]
    return rows[valid[rows]].astype(np.int64)


def split_prevalidation(rows):
    a = int(len(rows) * 0.70)
    b = int(len(rows) * 0.85)
    return rows[:a], rows[a:b], rows[b:]


def pos_weight(targets, rows, mode, device):
    y = targets[rows]
    pos = y.sum(axis=0)
    neg = len(y) - pos
    ratio = neg / np.maximum(pos, 1.0)
    if mode == "none":
        ratio = np.ones_like(ratio)
    elif mode == "sqrt":
        ratio = np.sqrt(ratio)
    return torch.tensor(ratio, dtype=torch.float32, device=device)


def infer(model, features, rows, targets, seq, device):
    ds = SeqDataset(features, rows, targets, seq)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type=="cuda")
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            logits = model(x.to(device, non_blocking=True))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0)


def validation_loss(model, features, rows, targets, seq, device):
    ds = SeqDataset(features, rows, targets, seq)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    loss_fn = nn.BCEWithLogitsLoss()
    vals = []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            vals.append(float(loss_fn(model(x.to(device)), y.to(device)).cpu()))
    return float(np.mean(vals))


def train_one(features, targets, train_rows, early_rows, cfg, device):
    model = TradeTransformer(features.shape[1], cfg).to(device)
    ds = SeqDataset(features, train_rows[::int(cfg["stride"])], targets, int(cfg["seq"]))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=device.type=="cuda")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight(targets, train_rows, cfg["weight_mode"], device))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    best_state = None
    best_loss = math.inf
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses = []
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        ev = validation_loss(model, features, early_rows, targets, int(cfg["seq"]), device)
        print(f"  epoch {epoch}: train={np.mean(losses):.5f} early={ev:.5f}")
        if ev < best_loss:
            best_loss = ev
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_loss


def calibrate(raw, y):
    calibrated = raw.copy()
    calibrators = []
    briers = []
    for side in range(2):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw[:, side], y[:, side])
        calibrated[:, side] = iso.transform(raw[:, side])
        calibrators.append(iso)
        briers.append(float(brier_score_loss(y[:, side], calibrated[:, side])))
    return calibrators, calibrated, briers


def apply_cal(raw, calibrators):
    out = raw.copy()
    for side in range(2):
        out[:, side] = calibrators[side].transform(raw[:, side])
    return out


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


def rank_score(threshold, m):
    if strict(threshold, m):
        tier = 4
    elif threshold >= .80 and m["trades"] >= 50 and m["net_high_sum_pct"] > 0:
        tier = 3
    elif m["trades"] >= 50 and m["net_high_sum_pct"] > 0:
        tier = 2
    elif threshold >= .80 and m["trades"] >= 50:
        tier = 1
    else:
        tier = 0
    gap = m["median_gap_hours"] if np.isfinite(m["median_gap_hours"]) else 999.0
    penalty = (
        max(0, TARGET_CONFIDENCE-threshold)*200
        + max(0, TARGET_WIN_RATE_PCT-m["win_rate_mid_pct"])*1.5
        + max(0, TARGET_TRADES_PER_DAY-m["trades_per_day"])*12
        + max(0, gap-TARGET_MEDIAN_GAP_HOURS)*2
        + max(0, MIN_TARGET_TRADES-m["trades"])*.25
        + (50 if m["net_high_sum_pct"] <= 0 else 0)
    )
    reward = m["win_rate_mid_pct"]*.25 + min(m["trades_per_day"], 12)*2 + max(min(m["net_high_sum_pct"],100),-100)*.1
    return tier, reward-penalty


def main():
    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("TRADEMIND V4 TRANSFORMER TUNER")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    frame, features = load_dataset()
    outcomes = barrier_outcomes(frame, TAKE_PROFIT, STOP_LOSS, HORIZON_BARS)
    valid = outcomes["valid"]

    targets = np.column_stack([
        ((outcomes["long_gross"] - HIGH_COST) > 0).astype(np.float32),
        ((outcomes["short_gross"] - HIGH_COST) > 0).astype(np.float32),
    ])

    pre = np.flatnonzero((frame["open_time"] < VALIDATION_START).to_numpy())
    val = np.flatnonzero(((frame["open_time"] >= VALIDATION_START) & (frame["open_time"] < TEST_START)).to_numpy())
    test = np.flatnonzero((frame["open_time"] >= TEST_START).to_numpy())

    results = []
    artifacts = {}

    for cfg in CONFIGS:
        print("\nCONFIG", cfg["name"])
        seq = int(cfg["seq"])
        pre2 = valid_rows(pre, valid, seq)
        val2 = valid_rows(val, valid, seq)
        train_rows, early_rows, cal_rows = split_prevalidation(pre2)

        scaler = StandardScaler()
        scaler.fit(frame.iloc[train_rows][features])
        scaled = scaler.transform(frame[features]).astype(np.float32)

        model, early_loss = train_one(scaled, targets, train_rows, early_rows, cfg, device)
        raw_cal = infer(model, scaled, cal_rows, targets, seq, device)
        calibrators, _, briers = calibrate(raw_cal, targets[cal_rows])

        raw_val = infer(model, scaled, val2, targets, seq, device)
        val_prob = apply_cal(raw_val, calibrators)

        for threshold in THRESHOLDS:
            m = evaluate(
                frame, val2, val_prob[:,0], val_prob[:,1], outcomes, threshold,
                VALIDATION_START, TEST_START,
            )
            tier, score = rank_score(threshold, m)
            results.append({
                "config": cfg["name"],
                "threshold": threshold,
                "early_loss": early_loss,
                "long_brier": briers[0],
                "short_brier": briers[1],
                **m,
                "strict_target_met": strict(threshold, m),
                "tier": tier,
                "score": score,
            })
        artifacts[cfg["name"]] = (cfg, scaler, model.state_dict(), calibrators)

    df = pd.DataFrame(results).sort_values(
        ["tier","score","win_rate_mid_pct","trades_per_day"],
        ascending=[False,False,False,False],
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_FILE, index=False)

    selected = df.iloc[0]
    cfg, scaler, state_dict, calibrators = artifacts[selected["config"]]
    seq = int(cfg["seq"])
    test2 = valid_rows(test, valid, seq)
    scaled = scaler.transform(frame[features]).astype(np.float32)
    model = TradeTransformer(len(features), cfg).to(device)
    model.load_state_dict(state_dict)
    raw_test = infer(model, scaled, test2, targets, seq, device)
    test_prob = apply_cal(raw_test, calibrators)
    threshold = float(selected["threshold"])
    tm = evaluate(
        frame, test2, test_prob[:,0], test_prob[:,1], outcomes, threshold,
        TEST_START, frame["open_time"].iloc[-1] + pd.Timedelta(minutes=5),
    )
    test_row = {
        "config": cfg["name"],
        "threshold": threshold,
        **tm,
        "strict_target_met": strict(threshold, tm),
    }
    pd.DataFrame([test_row]).to_csv(TEST_FILE, index=False)

    torch.save({
        "research_only": True,
        "feature_columns": features,
        "config": cfg,
        "state_dict": state_dict,
        "horizon_minutes": HORIZON_MINUTES,
        "take_profit": TAKE_PROFIT,
        "stop_loss": STOP_LOSS,
        "threshold": threshold,
    }, MODEL_FILE)
    joblib.dump({"scaler": scaler, "calibrators": calibrators, "feature_columns": features}, SCALER_FILE)

    summary = {
        "research_only": True,
        "selection_policy": "Architecture and threshold selected on validation only; later test evaluated once.",
        "target_label": "profit after high-cost assumption",
        "selected_validation": selected.to_dict(),
        "later_test": test_row,
        "qualified_for_forward_paper": bool(selected["strict_target_met"] and test_row["strict_target_met"]),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print("\nVALIDATION WINNER")
    print(selected.to_string())
    print("\nLATER TEST")
    print(pd.Series(test_row).to_string())
    print("\nQualified:", summary["qualified_for_forward_paper"])


if __name__ == "__main__":
    main()
