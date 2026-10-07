from pathlib import Path

import joblib
import pandas as pd
import json
import threading

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from backend.services.trading.timeframe_config import (
    SUPPORTED_TIMEFRAMES,
    get_timeframe_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANUAL_PAPER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manual_paper_account.json"
)

MANUAL_PAPER_STARTING_USDT = 10000.0
MANUAL_PAPER_FEE_RATE = 0.0005
MANUAL_PAPER_LOCK = threading.Lock()


class ManualPaperOrderRequest(BaseModel):
    side: str
    amount_usdt: float = Field(
        gt=0,
        le=1_000_000,
    )




app = FastAPI(
    title="TradeMind AI API",
    version="0.2.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
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


def manual_paper_default_state() -> dict:
    return {
        "cash_usdt": MANUAL_PAPER_STARTING_USDT,
        "btc_quantity": 0.0,
        "average_entry_price": 0.0,
        "realized_pnl_usdt": 0.0,
        "orders": [],
    }


def load_manual_paper_state() -> dict:
    if not MANUAL_PAPER_FILE.exists():
        return manual_paper_default_state()

    try:
        return json.loads(
            MANUAL_PAPER_FILE.read_text(
                encoding="utf-8",
            )
        )
    except (
        json.JSONDecodeError,
        OSError,
    ):
        return manual_paper_default_state()


def save_manual_paper_state(
    state: dict,
) -> None:
    MANUAL_PAPER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANUAL_PAPER_FILE.write_text(
        json.dumps(
            state,
            indent=2,
        ),
        encoding="utf-8",
    )


def fetch_live_btc_price() -> float:
    try:
        response = requests.get(
            BINANCE_SPOT_BASE_URL
            + BINANCE_KLINES_ENDPOINT,
            params={
                "symbol": "BTCUSDT",
                "interval": "1m",
                "limit": 1,
            },
            timeout=10,
        )
        response.raise_for_status()
        rows = response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not retrieve live BTC "
                "price for paper order."
            ),
        ) from exc

    if not rows:
        raise HTTPException(
            status_code=502,
            detail="No BTC price returned.",
        )

    return float(rows[-1][4])


def manual_paper_snapshot(
    state: dict,
    price: float,
) -> dict:
    btc_quantity = float(
        state.get(
            "btc_quantity",
            0.0,
        )
    )
    cash = float(
        state.get(
            "cash_usdt",
            MANUAL_PAPER_STARTING_USDT,
        )
    )
    avg_entry = float(
        state.get(
            "average_entry_price",
            0.0,
        )
    )
    btc_value = btc_quantity * price
    unrealized = (
        (price - avg_entry)
        * btc_quantity
        if btc_quantity > 0
        else 0.0
    )

    return {
        "cashUsdt": cash,
        "btcQuantity": btc_quantity,
        "averageEntryPrice": avg_entry,
        "btcMarketValueUsdt": btc_value,
        "equityUsdt": cash + btc_value,
        "realizedPnlUsdt": float(
            state.get(
                "realized_pnl_usdt",
                0.0,
            )
        ),
        "unrealizedPnlUsdt": unrealized,
        "livePrice": price,
        "feeRatePct": (
            MANUAL_PAPER_FEE_RATE
            * 100
        ),
        "recentOrders": list(
            reversed(
                state.get(
                    "orders",
                    [],
                )[-20:]
            )
        ),
    }


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

    ready = get_ready_state(config.key)

    if config.key == "10m":
        latest = trades.iloc[-1]

        closed = trades[
            trades["status"].astype(str)
            == "CLOSED"
        ].copy()

        mid_returns = pd.to_numeric(
            closed.get(
                "mid_cost_net_pct",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).dropna()

        win_rate = (
            float((mid_returns > 0).mean() * 100)
            if not mid_returns.empty
            else 0.0
        )

        net_return = (
            float(mid_returns.sum())
            if not mid_returns.empty
            else 0.0
        )

        price = 0.0

        if config.market_file.exists():
            market = pd.read_csv(
                config.market_file
            )

            if (
                not market.empty
                and "close" in market.columns
            ):
                price = safe_float(
                    market.iloc[-1].get(
                        "close",
                        0.0,
                    )
                )

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

        for _, row in (
            trades
            .tail(10)
            .iloc[::-1]
            .iterrows()
        ):
            recent_signals.append(
                {
                    "timestamp": safe_string(
                        row.get(
                            "signal_time"
                        )
                    ),
                    "prediction": safe_string(
                        row.get(
                            "prediction"
                        )
                    ),
                    "confidence": safe_float(
                        row.get(
                            "confidence"
                        )
                    ),
                    "signal": safe_string(
                        row.get(
                            "signal"
                        )
                    ),
                    "status": safe_string(
                        row.get(
                            "status"
                        )
                    ),
                }
            )

        bundle = {}

        if config.model_file.exists():
            bundle = joblib.load(
                config.model_file
            )

        return {
            "symbol": "BTC / USDT",
            "timeframe": display_timeframe(
                config.key
            ),
            "timeframeKey": config.key,
            "price": price,
            "prediction": safe_string(
                latest.get(
                    "prediction",
                    "NEUTRAL",
                )
            ),
            "confidence": safe_float(
                latest.get(
                    "confidence",
                    0.0,
                )
            ),
            "signal": safe_string(
                latest.get(
                    "signal",
                    "HOLD",
                )
            ),
            "probabilities": probabilities,
            "balance": 10000.0,
            "winRate": win_rate,
            "netReturn": net_return,
            "modelName": safe_string(
                latest.get(
                    "model_name",
                    bundle.get(
                        "model_name",
                        config.model_file.stem,
                    ),
                )
            ),
            "featureCount": len(
                bundle.get(
                    "feature_columns",
                    [],
                )
            ),
            "confidenceThreshold": safe_float(
                bundle.get(
                    "confidence_threshold",
                    0.55,
                )
            ),
            "latestCandle": safe_string(
                latest.get(
                    "signal_time"
                )
            ),
            "systemStatus": (
                "Forward paper collection"
                if ready["tradingReady"]
                else "Not ready"
            ),
            "recentSignals":
                recent_signals,
            "strategyMode":
                "Forward research",
            "researchStatus":
                "Frozen candidate",
            "forecastHorizonMinutes":
                int(
                    bundle.get(
                        "horizon_minutes",
                        60,
                    )
                ),
            "targetThresholdPct":
                safe_float(
                    bundle.get(
                        "target_threshold",
                        0.004,
                    )
                )
                * 100,
            "entryModel":
                "Next 10m bar open",
            "exitModel":
                "Fixed 60m horizon close",
            "primaryCostPct":
                safe_float(
                    bundle.get(
                        "execution_costs",
                        {},
                    ).get(
                        "mid",
                        0.0014,
                    )
                )
                * 100,
            "stressCostPct":
                safe_float(
                    bundle.get(
                        "execution_costs",
                        {},
                    ).get(
                        "high",
                        0.0027,
                    )
                )
                * 100,
            "forwardSignalCount":
                int(len(trades)),
            "closedForwardTrades":
                int(len(closed)),
            "evidenceNote": (
                "Collecting genuinely new forward "
                "paper evidence. Do not tune the "
                "frozen model against these results."
            ),
        }

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
        "strategyMode": "Paper account",
        "researchStatus": "Locked benchmark",
        "forecastHorizonMinutes": 60,
        "targetThresholdPct": 0.20,
        "entryModel": "Signal candle close",
        "exitModel": "1h risk-managed paper exit",
        "primaryCostPct": 0.05,
        "stressCostPct": 0.05,
        "forwardSignalCount": int(len(trades)),
        "closedForwardTrades": int(len(closed)),
        "evidenceNote": (
            "Existing locked 1h benchmark strategy."
        ),
    }


@app.get("/api/indicator/ai")
def get_ai_indicator(
    timeframe: str = Query(
        default="10m",
        description=(
            "AI strategy timeframe. "
            "Currently only the frozen 10m "
            "forward model is exposed."
        ),
    ),
    limit: int = Query(
        default=200,
        ge=1,
        le=500,
    ),
):
    normalized = (
        timeframe.strip().lower()
    )

    if normalized != "10m":
        raise HTTPException(
            status_code=409,
            detail=(
                "AI overlay is currently available "
                "for the frozen 10m strategy only."
            ),
        )

    config = get_timeframe_config(
        normalized
    )

    if (
        not config.model_file.exists()
        or not config.feature_file.exists()
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "10m AI overlay is not ready. "
                "Model or live feature file missing."
            ),
        )

    bundle = joblib.load(
        config.model_file
    )

    features = pd.read_csv(
        config.feature_file
    )

    if features.empty:
        return []

    features["open_time"] = pd.to_datetime(
        features["open_time"],
        utc=True,
        errors="coerce",
    )

    features = (
        features.dropna(
            subset=["open_time"]
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    training_end = pd.to_datetime(
        bundle.get(
            "training_end"
        ),
        utc=True,
        errors="coerce",
    )

    if pd.notna(training_end):
        features = features[
            features["open_time"]
            > training_end
        ].copy()

    if features.empty:
        return []

    features = features.tail(
        limit
    ).copy()

    feature_columns = bundle[
        "feature_columns"
    ]

    missing = [
        column
        for column in feature_columns
        if column not in features.columns
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "AI overlay feature mismatch: "
                + ", ".join(
                    missing[:10]
                )
            ),
        )

    X = features[
        feature_columns
    ]

    model = bundle["model"]
    encoder = bundle["encoder"]

    probabilities = (
        model.predict_proba(X)
    )

    encoded = model.predict(X)

    predictions = (
        encoder.inverse_transform(
            encoded
        )
    )

    class_index = {
        name: index
        for index, name
        in enumerate(
            encoder.classes_
        )
    }

    threshold = float(
        bundle.get(
            "confidence_threshold",
            0.55,
        )
    )

    points = []

    for row_index, (_, row) in enumerate(
        features.iterrows()
    ):
        down = float(
            probabilities[
                row_index,
                class_index["DOWN"],
            ]
        )

        neutral = float(
            probabilities[
                row_index,
                class_index["NEUTRAL"],
            ]
        )

        up = float(
            probabilities[
                row_index,
                class_index["UP"],
            ]
        )

        confidence = float(
            probabilities[
                row_index
            ].max()
        )

        prediction = str(
            predictions[
                row_index
            ]
        )

        signal = "HOLD"

        if confidence >= threshold:
            if prediction == "UP":
                signal = "BUY"
            elif prediction == "DOWN":
                signal = "SELL"

        points.append(
            {
                "time": int(
                    row[
                        "open_time"
                    ].timestamp()
                ),
                "score": (
                    up - down
                ),
                "up": up,
                "neutral": neutral,
                "down": down,
                "confidence":
                    confidence,
                "prediction":
                    prediction,
                "signal": signal,
            }
        )

    return points


CHART_TIMEFRAME_MINUTES = {
    "1m": 1,
    "2m": 2,
    "3m": 3,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "20m": 20,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

BINANCE_SPOT_BASE_URL = (
    "https://data-api.binance.vision"
)

BINANCE_KLINES_ENDPOINT = (
    "/api/v3/klines"
)


NATIVE_BINANCE_CHART_INTERVALS = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
}


def fetch_native_chart_data(
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    response = requests.get(
        BINANCE_SPOT_BASE_URL
        + BINANCE_KLINES_ENDPOINT,
        params={
            "symbol": "BTCUSDT",
            "interval": timeframe,
            "limit": limit,
        },
        timeout=20,
    )

    response.raise_for_status()

    rows = response.json()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return (
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
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def fetch_recent_1m_chart_data(
    required_rows: int,
) -> pd.DataFrame:
    rows = []
    end_time = None

    while len(rows) < required_rows:
        batch_limit = min(
            1000,
            required_rows - len(rows),
        )

        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "limit": batch_limit,
        }

        if end_time is not None:
            params["endTime"] = end_time

        response = requests.get(
            BINANCE_SPOT_BASE_URL
            + BINANCE_KLINES_ENDPOINT,
            params=params,
            timeout=20,
        )

        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        rows = batch + rows

        oldest_open = int(
            batch[0][0]
        )

        end_time = (
            oldest_open - 1
        )

        if len(batch) < batch_limit:
            break

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return (
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
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def resample_chart_data(
    base: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    minutes = (
        CHART_TIMEFRAME_MINUTES[
            timeframe
        ]
    )

    if minutes == 1:
        return base.copy()

    rule = (
        "1d"
        if timeframe == "1d"
        else (
            f"{minutes}min"
        )
    )

    indexed = base.set_index(
        "open_time"
    )

    result = (
        indexed.resample(
            rule,
            label="left",
            closed="left",
            origin="epoch",
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .reset_index()
    )

    return result




@app.get("/api/paper/manual/account")
def get_manual_paper_account():
    with MANUAL_PAPER_LOCK:
        state = load_manual_paper_state()
        price = fetch_live_btc_price()
        return manual_paper_snapshot(
            state,
            price,
        )


@app.post("/api/paper/manual/order")
def place_manual_paper_order(
    request: ManualPaperOrderRequest,
):
    side = request.side.strip().upper()

    if side not in {
        "BUY",
        "SELL",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Paper side must be BUY or SELL."
            ),
        )

    with MANUAL_PAPER_LOCK:
        state = load_manual_paper_state()
        price = fetch_live_btc_price()
        requested_usdt = float(
            request.amount_usdt
        )
        timestamp = pd.Timestamp.now(
            tz="UTC"
        ).isoformat()

        cash = float(
            state.get(
                "cash_usdt",
                MANUAL_PAPER_STARTING_USDT,
            )
        )
        btc_quantity = float(
            state.get(
                "btc_quantity",
                0.0,
            )
        )
        average_entry = float(
            state.get(
                "average_entry_price",
                0.0,
            )
        )
        fee_usdt = (
            requested_usdt
            * MANUAL_PAPER_FEE_RATE
        )

        if side == "BUY":
            total_cost = (
                requested_usdt
                + fee_usdt
            )

            if total_cost > cash:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Not enough paper USDT for "
                        "this BUY order including fee."
                    ),
                )

            quantity = (
                requested_usdt
                / price
            )
            previous_cost_basis = (
                btc_quantity
                * average_entry
            )
            new_quantity = (
                btc_quantity
                + quantity
            )
            new_average_entry = (
                (
                    previous_cost_basis
                    + requested_usdt
                )
                / new_quantity
            )

            state["cash_usdt"] = (
                cash - total_cost
            )
            state["btc_quantity"] = (
                new_quantity
            )
            state["average_entry_price"] = (
                new_average_entry
            )
            realized_pnl = 0.0
        else:
            available_value = (
                btc_quantity
                * price
            )

            if requested_usdt > (
                available_value
                + 1e-9
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Not enough paper BTC for "
                        "this SELL order."
                    ),
                )

            quantity = (
                requested_usdt
                / price
            )
            realized_pnl = (
                (price - average_entry)
                * quantity
                - fee_usdt
            )
            remaining_quantity = max(
                0.0,
                btc_quantity
                - quantity,
            )

            state["cash_usdt"] = (
                cash
                + requested_usdt
                - fee_usdt
            )
            state["btc_quantity"] = (
                remaining_quantity
            )
            state["realized_pnl_usdt"] = (
                float(
                    state.get(
                        "realized_pnl_usdt",
                        0.0,
                    )
                )
                + realized_pnl
            )

            if remaining_quantity <= 1e-12:
                state["btc_quantity"] = 0.0
                state["average_entry_price"] = 0.0

        order = {
            "timestamp": timestamp,
            "side": side,
            "price": price,
            "amountUsdt": requested_usdt,
            "btcQuantity": quantity,
            "feeUsdt": fee_usdt,
            "realizedPnlUsdt": realized_pnl,
        }

        state.setdefault(
            "orders",
            [],
        ).append(order)

        state["orders"] = state[
            "orders"
        ][-500:]

        save_manual_paper_state(
            state
        )

        return manual_paper_snapshot(
            state,
            price,
        )


@app.post("/api/paper/manual/reset")
def reset_manual_paper_account():
    with MANUAL_PAPER_LOCK:
        state = manual_paper_default_state()
        save_manual_paper_state(
            state
        )
        price = fetch_live_btc_price()
        return manual_paper_snapshot(
            state,
            price,
        )


@app.get("/api/market/btc")
def get_btc_market(
    timeframe: str = Query(default="10m"),
    limit: int = Query(
        default=300,
        ge=1,
        le=500,
    ),
):
    normalized = (
        timeframe.strip().lower()
    )

    if (
        normalized
        not in CHART_TIMEFRAME_MINUTES
    ):
        supported = ", ".join(
            CHART_TIMEFRAME_MINUTES
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported chart timeframe "
                f"'{timeframe}'. Supported: "
                f"{supported}"
            ),
        )

    minutes = (
        CHART_TIMEFRAME_MINUTES[
            normalized
        ]
    )

    # Extra source rows ensure the first aggregated
    # chart candle is complete enough for display.
    required_1m_rows = min(
        10000,
        limit * minutes
        + minutes * 2,
    )

    try:
        if (
            normalized
            in NATIVE_BINANCE_CHART_INTERVALS
        ):
            df = fetch_native_chart_data(
                normalized,
                limit,
            )
        else:
            base = fetch_recent_1m_chart_data(
                required_1m_rows
            )

            if base.empty:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "No live BTC chart data "
                        "was returned."
                    ),
                )

            df = resample_chart_data(
                base,
                normalized,
            ).tail(limit)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Live BTC chart data request "
                "to Binance failed."
            ),
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=502,
            detail=(
                "No live BTC chart data "
                "was returned."
            ),
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
