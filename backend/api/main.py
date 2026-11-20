from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAPER_TRADES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_trades.csv"
)

BTC_CLEAN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_clean.csv"
)


app = FastAPI(
    title="TradeMind AI API",
    version="0.1.0",
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


@app.get("/")
def root():
    return {
        "name": "TradeMind AI API",
        "status": "running",
    }


@app.get("/api/dashboard")
def get_dashboard():
    if not PAPER_TRADES_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="paper_trades.csv not found",
        )

    trades = pd.read_csv(PAPER_TRADES_FILE)

    if trades.empty:
        raise HTTPException(
            status_code=404,
            detail="No paper trading data available",
        )

    latest = trades.iloc[-1]

    closed = trades[
        trades["status"].astype(str) == "CLOSED"
    ]

    wins = closed[
        closed["result"].astype(str) == "WIN"
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
            latest.get(
                "probability_down",
                0.0,
            )
        ),
        "neutral": safe_float(
            latest.get(
                "probability_neutral",
                0.0,
            )
        ),
        "up": safe_float(
            latest.get(
                "probability_up",
                0.0,
            )
        ),
    }

    recent_signals = []

    for _, row in trades.tail(10).iloc[::-1].iterrows():
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

    return {
        "symbol": "BTC / USDT",
        "timeframe": "1H",
        "price": safe_float(
            latest.get("price")
        ),
        "prediction": safe_string(
            latest.get("prediction")
        ),
        "confidence": safe_float(
            latest.get("confidence")
        ),
        "signal": safe_string(
            latest.get("signal")
        ),
        "probabilities": probabilities,
        "balance": balance,
        "winRate": win_rate,
        "netReturn": net_return,
        "modelName": safe_string(
            latest.get(
                "model_name",
                "Locked XGBoost",
            )
        ),
        "featureCount": 84,
        "confidenceThreshold": 0.55,
        "latestCandle": safe_string(
            latest.get("timestamp")
        ),
        "systemStatus": "Active",
        "recentSignals": recent_signals,
    }


@app.get("/api/market/btc")
def get_btc_market(limit: int = 200):
    if not BTC_CLEAN_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="btcusdt_1h_clean.csv not found",
        )

    if limit < 10:
        limit = 10

    if limit > 1000:
        limit = 1000

    df = pd.read_csv(BTC_CLEAN_FILE)

    required_columns = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing BTC columns: "
                + ", ".join(
                    sorted(missing)
                )
            ),
        )

    df = df.tail(limit).copy()

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    candles = []

    for _, row in df.iterrows():
        candles.append(
            {
                "time": int(
                    row[
                        "open_time"
                    ].timestamp()
                ),
                "open": float(
                    row["open"]
                ),
                "high": float(
                    row["high"]
                ),
                "low": float(
                    row["low"]
                ),
                "close": float(
                    row["close"]
                ),
                "volume": float(
                    row["volume"]
                ),
            }
        )

    return candles