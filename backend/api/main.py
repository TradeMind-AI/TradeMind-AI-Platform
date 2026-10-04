from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.services.trading.timeframe_config import (
    SUPPORTED_TIMEFRAMES,
    get_timeframe_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


app = FastAPI(
    title="TradeMind AI API",
    version="0.2.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_float(value, default=0.0):
    if pd.isna(value):
        return default
    return float(value)


def safe_string(value, default=""):
    if pd.isna(value):
        return default
    return str(value)


def display_timeframe(timeframe: str) -> str:
    return "1H" if timeframe == "1h" else timeframe


def get_ready_state(timeframe: str) -> dict:
    config = get_timeframe_config(timeframe)

    return {
        "timeframe": config.key,
        "label": display_timeframe(config.key),
        "marketReady": config.market_file.exists(),
        "featuresReady": config.feature_file.exists(),
        "modelReady": config.model_file.exists(),
        "paperLogReady": config.paper_log.exists(),
        "tradingReady": (
            config.feature_file.exists()
            and config.model_file.exists()
        ),
    }


@app.get("/")
def root():
    return {
        "name": "TradeMind AI API",
        "status": "running",
        "supportedTimeframes": list(
            SUPPORTED_TIMEFRAMES
        ),
    }


@app.get("/api/timeframes")
def get_timeframes():
    return [
        get_ready_state(timeframe)
        for timeframe in SUPPORTED_TIMEFRAMES
    ]


@app.get("/api/dashboard")
def get_dashboard(
    timeframe: str = Query(
        default="1h",
        description=(
            "Strategy timeframe: "
            "1m, 2m, 5m, 10m, 20m, 30m, or 1h"
        ),
    ),
):
    try:
        config = get_timeframe_config(timeframe)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if not config.paper_log.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"{config.key} paper strategy "
                "is not ready yet. Train and lock "
                "its model, then start paper trading."
            ),
        )

    trades = pd.read_csv(config.paper_log)

    if trades.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No {config.key} paper "
                "trading data available."
            ),
        )

    latest = trades.iloc[-1]

    closed = trades[
        trades["status"].astype(str)
        == "CLOSED"
    ]

    wins = closed[
        closed["result"].astype(str)
        == "WIN"
    ]

    win_rate = (
        len(wins) / len(closed) * 100
        if len(closed) > 0
        else 0.0
    )

    if "account_balance_after" in trades.columns:
        balances = pd.to_numeric(
            trades["account_balance_after"],
            errors="coerce",
        ).dropna()

        balance = (
            float(balances.iloc[-1])
            if not balances.empty
            else 10000.0
        )
    else:
        balance = 10000.0

    if "cumulative_net_return_pct" in trades.columns:
        returns = pd.to_numeric(
            trades["cumulative_net_return_pct"],
            errors="coerce",
        ).dropna()

        net_return = (
            float(returns.iloc[-1])
            if not returns.empty
            else 0.0
        )
    else:
        net_return = 0.0

    probabilities = {
        "down": safe_float(
            latest.get("probability_down", 0.0)
        ),
        "neutral": safe_float(
            latest.get("probability_neutral", 0.0)
        ),
        "up": safe_float(
            latest.get("probability_up", 0.0)
        ),
    }

    recent_signals = []

    for _, row in (
        trades
        .tail(10)
        .iloc[::-1]
        .iterrows()
    ):
        recent_signals.append(
            {
                "timestamp": safe_string(
                    row.get("timestamp")
                ),
                "prediction": safe_string(
                    row.get("prediction")
                ),
                "confidence": safe_float(
                    row.get("confidence")
                ),
                "signal": safe_string(
                    row.get("signal")
                ),
                "status": safe_string(
                    row.get("status")
                ),
            }
        )

    ready = get_ready_state(config.key)

    return {
        "symbol": "BTC / USDT",
        "timeframe": display_timeframe(config.key),
        "timeframeKey": config.key,
        "price": safe_float(latest.get("price")),
        "prediction": safe_string(
            latest.get("prediction")
        ),
        "confidence": safe_float(
            latest.get("confidence")
        ),
        "signal": safe_string(latest.get("signal")),
        "probabilities": probabilities,
        "balance": balance,
        "winRate": win_rate,
        "netReturn": net_return,
        "modelName": safe_string(
            latest.get(
                "model_name",
                config.model_file.stem,
            )
        ),
        "featureCount": (
            84 if config.key == "1h" else 0
        ),
        "confidenceThreshold": (
            0.55 if config.key == "1h" else 0.0
        ),
        "latestCandle": safe_string(
            latest.get("timestamp")
        ),
        "systemStatus": (
            "Active"
            if ready["tradingReady"]
            else "Not ready"
        ),
        "recentSignals": recent_signals,
    }


@app.get("/api/market/btc")
def get_btc_market(
    timeframe: str = Query(default="1h"),
    limit: int = Query(
        default=200,
        ge=1,
        le=1000,
    ),
):
    try:
        config = get_timeframe_config(timeframe)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if not config.market_file.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"{config.key} market data is not ready. "
                "Generate it from the base 1-minute "
                "dataset first."
            ),
        )

    df = pd.read_csv(config.market_file)

    required_columns = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing BTC columns: "
                + ", ".join(sorted(missing))
            ),
        )

    df = df.tail(limit).copy()

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["open_time"])

    candles = []

    for _, row in df.iterrows():
        candles.append(
            {
                "time": int(
                    row["open_time"].timestamp()
                ),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    return candles
