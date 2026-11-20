"""Deterministic EUR/USD paper trading engine. No broker orders, no live FX claims."""
import json
import math
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/forex", tags=["forex-paper"])
ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "processed" / "forex_paper_state.json"
LOCK = threading.RLock()
START = datetime(2026, 10, 1, tzinfo=timezone.utc)
INITIAL_BALANCE = 10000.0
DEFAULT = {"running": False, "cursor": 240, "balance": INITIAL_BALANCE,
           "risk_pct": 1.0, "stop_pips": 18, "take_pips": 36,
           "confidence_threshold": 0.58, "position": None, "trades": []}
# IMPORTANT: This is an indicator-based prototype, NOT the BTC pretrained model.
MODEL_STATUS = "Indicator baseline; BTC transfer learning not connected"
SOURCE = "Deterministic simulated EUR/USD candles; not a live market feed"


class Settings(BaseModel):
    risk_pct: float = Field(default=1.0, gt=0, le=1.0)
    stop_pips: int = Field(default=18, ge=2, le=200)
    take_pips: int = Field(default=36, ge=2, le=400)
    confidence_threshold: float = Field(default=0.58, ge=0.5, le=0.99)


def load():
    if STATE_PATH.exists():
        return {**DEFAULT, **json.loads(STATE_PATH.read_text())}
    return {**DEFAULT, "trades": []}


def save(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATE_PATH)


def candle(i):
    """Seeded mathematical series: stable across requests, never presented as real FX."""
    base = 1.0992 + 0.0036 * math.sin(i * 0.045) + 0.0015 * math.sin(i * 0.137)
    jitter = 0.00024 * math.sin(i * 1.783 + 0.41)
    close = base + jitter
    prev = 1.0992 + 0.0036 * math.sin((i - 1) * 0.045) + 0.0015 * math.sin((i - 1) * 0.137) + 0.00024 * math.sin((i - 1) * 1.783 + 0.41)
    return {"time": int((START + timedelta(minutes=15 * i)).timestamp()),
            "open": round(prev, 5), "high": round(max(prev, close) + 0.00018, 5),
            "low": round(min(prev, close) - 0.00018, 5), "close": round(close, 5)}


def ema(values, window):
    alpha = 2 / (window + 1)
    x = values[0]
    for value in values[1:]:
        x = alpha * value + (1 - alpha) * x
    return x


def signal(cursor):
    closes = [candle(i)["close"] for i in range(max(0, cursor - 120), cursor + 1)]
    fast, slow = ema(closes, 20), ema(closes, 50)
    diffs = [b - a for a, b in zip(closes[:-1], closes[1:])]
    gains = sum(max(x, 0) for x in diffs[-14:]) / 14
    losses = sum(max(-x, 0) for x in diffs[-14:]) / 14
    rsi = 100 * gains / (gains + losses) if gains + losses else 50
    spread_pips = (fast - slow) / 0.0001
    # Rule-based heuristic confidence, not calibrated predictive probabilities.
    confidence = min(0.85, 0.50 + abs(spread_pips) / 65)
    side = "LONG" if spread_pips > 1.4 and 49 < rsi < 76 else (
        "SHORT" if spread_pips < -1.4 and 24 < rsi < 51 else "HOLD")
    return {"side": side, "confidence": round(confidence, 3),
            "rsi": round(rsi, 1), "ema20": round(fast, 5),
            "ema50": round(slow, 5),
            "reason": "EMA20/50 trend with RSI regime filter"}


def metrics(state):
    trades = state["trades"]
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    peak = INITIAL_BALANCE
    equity = INITIAL_BALANCE
    max_dd = 0.0
    for trade in trades:
        equity += trade["pnl"]
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)
    return {"trades": len(trades), "wins": len(wins),
            "win_rate": round(100 * len(wins) / len(trades), 2) if trades else 0,
            "net_pnl": round(sum(t["pnl"] for t in trades), 2),
            "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
            "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if sum(losses) < 0 else None,
            "max_drawdown_pct": round(max_dd, 2), "target_win_rate": 80}


def snapshot(state):
    return {"running": state["running"], "cursor": state["cursor"],
            "balance": round(state["balance"], 2), "position": state["position"],
            "settings": {k: state[k] for k in ("risk_pct", "stop_pips", "take_pips", "confidence_threshold")},
            "signal": signal(state["cursor"]), "metrics": metrics(state),
            "trades": state["trades"][-250:][::-1], "source": SOURCE,
            "model_status": MODEL_STATUS, "paper_only": True}


@router.get("/status")
def status():
    with LOCK:
        return snapshot(load())


@router.get("/candles")
def candles(limit: int = 200):
    if not 30 <= limit <= 500:
        raise HTTPException(422, "limit must be 30 to 500")
    with LOCK:
        cursor = load()["cursor"]
        return {"symbol": "EUR/USD", "timeframe": "15m", "source": SOURCE,
                "candles": [candle(i) for i in range(max(0, cursor - limit + 1), cursor + 1)]}


@router.post("/start")
def start():
    with LOCK:
        state = load()
        state["running"] = True
        save(state)
        return snapshot(state)


@router.post("/stop")
def stop():
    with LOCK:
        state = load()
        state["running"] = False
        save(state)
        return snapshot(state)


@router.put("/settings")
def settings(config: Settings):
    with LOCK:
        state = load()
        state.update(config.model_dump())
        save(state)
        return snapshot(state)


@router.post("/step")
def step():
    """Advance one simulated 15m bar. Frontend schedules steps while running."""
    with LOCK:
        state = load()
        if not state["running"]:
            raise HTTPException(409, "Start the paper bot before stepping")
        state["cursor"] += 1
        bar = candle(state["cursor"])
        pos = state["position"]
        if pos:
            stop_hit = bar["low"] <= pos["stop"] if pos["side"] == "LONG" else bar["high"] >= pos["stop"]
            target_hit = bar["high"] >= pos["target"] if pos["side"] == "LONG" else bar["low"] <= pos["target"]
            if stop_hit or target_hit:
                # Conservative collision handling: stop first.
                exit_mid = pos["stop"] if stop_hit else pos["target"]
                # 0.8 pip simulated spread plus 0.2 pip slippage each side.
                exit_price = exit_mid - 0.00005 if pos["side"] == "LONG" else exit_mid + 0.00005
                pnl = (exit_price - pos["entry"]) * pos["units"] * (1 if pos["side"] == "LONG" else -1)
                state["balance"] = round(state["balance"] + pnl, 4)
                state["trades"].append({**pos, "id": len(state["trades"]) + 1,
                    "exit_time": bar["time"], "exit": round(exit_price, 5),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(100 * pnl / pos["balance_at_entry"], 3),
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "reason_exit": "STOP" if stop_hit else "TARGET"})
                state["position"] = None
        if state["position"] is None:
            sig = signal(state["cursor"])
            if sig["side"] != "HOLD" and sig["confidence"] >= state["confidence_threshold"] and state["balance"] > 0:
                risk_dollars = state["balance"] * state["risk_pct"] / 100
                # Cap exposure at 10x equity in paper mode.
                units = min(risk_dollars / ((state["stop_pips"] + 1.0) * 0.0001),
                            10 * state["balance"] / bar["close"])
                if units > 0:
                    direction = 1 if sig["side"] == "LONG" else -1
                    entry = bar["close"] + direction * 0.00005
                    state["position"] = {"side": sig["side"], "entry_time": bar["time"],
                        "entry": round(entry, 5), "units": round(units, 2),
                        "stop": round(entry - direction * state["stop_pips"] * 0.0001, 5),
                        "target": round(entry + direction * state["take_pips"] * 0.0001, 5),
                        "confidence": sig["confidence"], "reason": sig["reason"],
                        "balance_at_entry": round(state["balance"], 2)}
        save(state)
        return snapshot(state)
