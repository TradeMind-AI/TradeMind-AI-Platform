from pathlib import Path

import joblib
import pandas as pd
import json
import threading
import time

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


MICRO_PAPER_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micro_paper_learning"
)

MICRO_PAPER_ACCOUNT_FILE = (
    MICRO_PAPER_DIR
    / "account.json"
)

MICRO_PAPER_FEATURE_FILE = (
    MICRO_PAPER_DIR
    / "feature_snapshots.csv"
)

MICRO_PAPER_TRADE_FILE = (
    MICRO_PAPER_DIR
    / "trades.csv"
)


MICRO_PAPER_EVALUATION_FILE = (
    MICRO_PAPER_DIR
    / "evaluation_metrics.json"
)

STAGED_FORWARD_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "staged_micro_forward"
)

STAGED_FORWARD_EVALUATION_FILE = (
    STAGED_FORWARD_DIR
    / "forward_evaluation.json"
)

STAGED_FORWARD_MANIFEST_FILE = (
    STAGED_FORWARD_DIR
    / "frozen_candidate_manifest.json"
)

MANUAL_PAPER_STARTING_USDT = 10000.0
MANUAL_PAPER_FEE_RATE = 0.0005
MANUAL_PAPER_LOCK = threading.Lock()
STAGED_FORWARD_REFRESH_LOCK = threading.Lock()
STAGED_FORWARD_LAST_REFRESH_ATTEMPT = 0.0
STAGED_FORWARD_REFRESH_INTERVAL_SECONDS = 60.0


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


def refresh_staged_forward_if_needed() -> None:
    global STAGED_FORWARD_LAST_REFRESH_ATTEMPT

    now = time.time()

    if (
        now
        - STAGED_FORWARD_LAST_REFRESH_ATTEMPT
        < STAGED_FORWARD_REFRESH_INTERVAL_SECONDS
    ):
        return

    snapshot_files = list(
        MICRO_PAPER_DIR.glob(
            "feature_snapshots*.csv"
        )
    )

    if not snapshot_files:
        return

    latest_snapshot_mtime = max(
        path.stat().st_mtime
        for path in snapshot_files
    )

    evaluation_mtime = (
        STAGED_FORWARD_EVALUATION_FILE
        .stat()
        .st_mtime
        if STAGED_FORWARD_EVALUATION_FILE.exists()
        else 0.0
    )

    if (
        evaluation_mtime
        >= latest_snapshot_mtime
    ):
        return

    if not STAGED_FORWARD_REFRESH_LOCK.acquire(
        blocking=False,
    ):
        return

    STAGED_FORWARD_LAST_REFRESH_ATTEMPT = now

    def run_refresh() -> None:
        try:
            from backend.services.ml.evaluate_frozen_staged_micro_forward import (
                main as evaluate_staged_forward,
            )

            evaluate_staged_forward()
        except Exception as exc:
            print(
                "Automatic staged forward "
                f"evaluation failed: {exc}"
            )
        finally:
            STAGED_FORWARD_REFRESH_LOCK.release()

    threading.Thread(
        target=run_refresh,
        daemon=True,
        name="staged-forward-refresh",
    ).start()


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



@app.get("/api/research/staged-forward")
def get_staged_forward_research():
    refresh_staged_forward_if_needed()

    evaluation = {}
    manifest = {}

    if STAGED_FORWARD_EVALUATION_FILE.exists():
        try:
            evaluation = json.loads(
                STAGED_FORWARD_EVALUATION_FILE.read_text(
                    encoding="utf-8",
                )
            )
        except (
            json.JSONDecodeError,
            OSError,
        ):
            evaluation = {}

    if STAGED_FORWARD_MANIFEST_FILE.exists():
        try:
            manifest = json.loads(
                STAGED_FORWARD_MANIFEST_FILE.read_text(
                    encoding="utf-8",
                )
            )
        except (
            json.JSONDecodeError,
            OSError,
        ):
            manifest = {}

    frozen_at = (
        evaluation.get("frozen_at")
        or manifest.get("frozen_at")
        or ""
    )

    minimum_checkpoint = int(
        evaluation.get(
            "minimum_checkpoint_new_snapshots",
            manifest.get(
                "forward_checkpoints",
                {},
            ).get(
                "minimum",
                25000,
            ),
        )
    )

    preferred_checkpoint = int(
        evaluation.get(
            "preferred_checkpoint_new_snapshots",
            manifest.get(
                "forward_checkpoints",
                {},
            ).get(
                "preferred",
                50000,
            ),
        )
    )

    new_snapshots = int(
        evaluation.get(
            "new_raw_snapshots",
            0,
        )
    )

    selected_predictions = int(
        evaluation.get(
            "selected_predictions",
            0,
        )
    )

    qualified_rate = evaluation.get(
        "qualified_trade_rate_pct"
    )

    avg_return = evaluation.get(
        "avg_selected_net_return_pct"
    )

    if not frozen_at:
        verdict = "NOT_FROZEN"
        verdict_note = (
            "No frozen staged candidate is available yet."
        )
    elif selected_predictions < 20:
        verdict = "INSUFFICIENT_EVIDENCE"
        verdict_note = (
            "The candidate is frozen, but there are too few "
            "selected forward trades for a reliable conclusion."
        )
    elif (
        qualified_rate is not None
        and avg_return is not None
        and float(qualified_rate) >= 55.0
        and float(avg_return) > 0.0
    ):
        verdict = "PROMISING"
        verdict_note = (
            "Forward evidence is positive so far, but the "
            "candidate remains research-only until the planned "
            "checkpoint is reached."
        )
    else:
        verdict = "FAILING_SO_FAR"
        verdict_note = (
            "Current forward evidence is not positive. Keep the "
            "candidate frozen and continue collecting data."
        )

    return {
        "available": bool(frozen_at),
        "frozenAt": frozen_at,
        "modelName": "staged_micro_forward_frozen_v1",
        "researchOnly": True,
        "minimumNetProfitPct": safe_float(
            manifest.get(
                "min_net_profit_pct",
                0.00075,
            )
        ) * 100.0,
        "newSnapshots": new_snapshots,
        "evaluableForwardRows": int(
            evaluation.get(
                "evaluable_forward_rows",
                0,
            )
        ),
        "selectedPredictions": selected_predictions,
        "coveragePct": safe_float(
            evaluation.get(
                "coverage_pct",
                0.0,
            )
        ),
        "qualifiedTradeRatePct": (
            None
            if qualified_rate is None
            else safe_float(
                qualified_rate
            )
        ),
        "avgSelectedNetReturnPct": (
            None
            if avg_return is None
            else safe_float(
                avg_return
            )
        ),
        "minimumCheckpoint": minimum_checkpoint,
        "preferredCheckpoint": preferred_checkpoint,
        "minimumCheckpointProgressPct": safe_float(
            evaluation.get(
                "minimum_checkpoint_progress_pct",
                (
                    new_snapshots
                    / minimum_checkpoint
                    * 100.0
                    if minimum_checkpoint > 0
                    else 0.0
                ),
            )
        ),
        "preferredCheckpointProgressPct": safe_float(
            evaluation.get(
                "preferred_checkpoint_progress_pct",
                (
                    new_snapshots
                    / preferred_checkpoint
                    * 100.0
                    if preferred_checkpoint > 0
                    else 0.0
                ),
            )
        ),
        "verdict": verdict,
        "verdictNote": verdict_note,
        "evidenceStatus": (
            "Forward-only, frozen candidate"
            if frozen_at
            else "Not frozen"
        ),
    }


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







@app.get("/api/paper/micro/status")
def get_micro_paper_status():
    if not MICRO_PAPER_ACCOUNT_FILE.exists():
        return {
            "running": False,
            "message": (
                "Micro paper learner has not "
                "written an account yet."
            ),
            "equityUsdt": 10000.0,
            "realizedPnlUsdt": 0.0,
            "closedTrades": 0,
            "wins": 0,
            "losses": 0,
            "winRatePct": 0.0,
            "position": None,
            "latest": None,
            "recentTrades": [],
        }

    try:
        account = json.loads(
            MICRO_PAPER_ACCOUNT_FILE.read_text(
                encoding="utf-8",
            )
        )
    except (
        json.JSONDecodeError,
        OSError,
    ) as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Micro paper account file "
                "could not be read."
            ),
        ) from exc

    latest = None

    if MICRO_PAPER_FEATURE_FILE.exists():
        try:
            snapshots = pd.read_csv(
                MICRO_PAPER_FEATURE_FILE
            )

            if not snapshots.empty:
                row = snapshots.iloc[-1]

                latest = {
                    "timestamp":
                        safe_string(
                            row.get(
                                "timestamp"
                            )
                        ),
                    "price":
                        safe_float(
                            row.get(
                                "price"
                            )
                        ),
                    "prediction":
                        int(
                            safe_float(
                                row.get(
                                    "prediction"
                                )
                            )
                        ),
                    "confidence":
                        safe_float(
                            row.get(
                                "confidence"
                            )
                        ),
                    "probDown":
                        safe_float(
                            row.get(
                                "prob_down"
                            )
                        ),
                    "probNeutral":
                        safe_float(
                            row.get(
                                "prob_neutral"
                            )
                        ),
                    "probUp":
                        safe_float(
                            row.get(
                                "prob_up"
                            )
                        ),
                    "learningSamples":
                        int(
                            safe_float(
                                row.get(
                                    "learning_samples"
                                )
                            )
                        ),
                    "decisionSource":
                        safe_string(
                            row.get(
                                "decision_source"
                            )
                        ),
                    "effectiveEntryThreshold":
                        safe_float(
                            row.get(
                                "effective_entry_threshold"
                            )
                        ),
                    "tradeSelectorProbability":
                        safe_float(
                            row.get(
                                "trade_selector_probability"
                            )
                        ),
                    "tradeSelectorSamples":
                        int(
                            safe_float(
                                row.get(
                                    "trade_selector_samples"
                                )
                            )
                        ),
                    "tradeSelectorReady":
                        bool(
                            int(
                                safe_float(
                                    row.get(
                                        "trade_selector_ready"
                                    )
                                )
                            )
                        ),
                    "tradeDirection":
                        int(
                            safe_float(
                                row.get(
                                    "trade_direction"
                                )
                            )
                        ),
                    "spotDepthImbalance":
                        safe_float(
                            row.get(
                                "spot_depth20_imbalance"
                            )
                        ),
                    "spotTradeFlowImbalance":
                        safe_float(
                            row.get(
                                "spot_trade_flow_imbalance"
                            )
                        ),
                    "spotVelocityBps":
                        safe_float(
                            row.get(
                                "spot_price_velocity_bps"
                            )
                        ),
                    "futuresDepthImbalance":
                        safe_float(
                            row.get(
                                "futures_depth20_imbalance"
                            )
                        ),
                    "futuresTradeFlowImbalance":
                        safe_float(
                            row.get(
                                "futures_trade_flow_imbalance"
                            )
                        ),
                    "futuresVelocityBps":
                        safe_float(
                            row.get(
                                "futures_price_velocity_bps"
                            )
                        ),
                    "fundingRate":
                        safe_float(
                            row.get(
                                "funding_rate"
                            )
                        ),
                    "openInterestChangePct":
                        safe_float(
                            row.get(
                                "open_interest_change_pct"
                            )
                        ),
                }
        except Exception:
            latest = None

    recent_trades = []

    if MICRO_PAPER_TRADE_FILE.exists():
        try:
            trades = pd.read_csv(
                MICRO_PAPER_TRADE_FILE
            )

            if not trades.empty:
                for _, row in (
                    trades
                    .tail(20)
                    .iloc[::-1]
                    .iterrows()
                ):
                    recent_trades.append(
                        {
                            "timestamp":
                                safe_string(
                                    row.get(
                                        "timestamp"
                                    )
                                ),
                            "event":
                                safe_string(
                                    row.get(
                                        "event"
                                    )
                                ),
                            "side":
                                safe_string(
                                    row.get(
                                        "side"
                                    )
                                ),
                            "entryPrice":
                                safe_float(
                                    row.get(
                                        "entry_price"
                                    )
                                ),
                            "exitPrice":
                                safe_float(
                                    row.get(
                                        "exit_price"
                                    )
                                ),
                            "notionalUsdt":
                                safe_float(
                                    row.get(
                                        "notional_usdt"
                                    )
                                ),
                            "netReturnPct":
                                safe_float(
                                    row.get(
                                        "net_return_pct"
                                    )
                                ),
                            "pnlUsdt":
                                safe_float(
                                    row.get(
                                        "pnl_usdt"
                                    )
                                ),
                            "reason":
                                safe_string(
                                    row.get(
                                        "reason"
                                    )
                                ),
                            "modelConfidence":
                                safe_float(
                                    row.get(
                                        "model_confidence"
                                    )
                                ),
                        }
                    )
        except Exception:
            recent_trades = []

    evaluation = {}

    if MICRO_PAPER_EVALUATION_FILE.exists():
        try:
            evaluation = json.loads(
                MICRO_PAPER_EVALUATION_FILE.read_text(
                    encoding="utf-8",
                )
            )
        except (
            json.JSONDecodeError,
            OSError,
        ):
            evaluation = {}

    evaluation_total = int(
        evaluation.get(
            "total",
            0,
        )
    )

    evaluation_correct = int(
        evaluation.get(
            "correct",
            0,
        )
    )

    prequential_accuracy_pct = (
        evaluation_correct
        / evaluation_total
        * 100.0
        if evaluation_total > 0
        else 0.0
    )

    recalls = []

    confusion = evaluation.get(
        "confusion",
        {},
    )

    for label in [
        "-1",
        "0",
        "1",
    ]:
        row_values = confusion.get(
            label,
            {},
        )

        row_total = sum(
            int(value)
            for value in row_values.values()
        )

        if row_total > 0:
            recalls.append(
                int(
                    row_values.get(
                        label,
                        0,
                    )
                )
                / row_total
            )

    balanced_accuracy_pct = (
        float(
            sum(recalls)
            / len(recalls)
            * 100.0
        )
        if recalls
        else 0.0
    )

    high_conf_total = int(
        evaluation.get(
            "high_conf_total",
            0,
        )
    )

    high_conf_correct = int(
        evaluation.get(
            "high_conf_correct",
            0,
        )
    )

    high_conf_accuracy_pct = (
        high_conf_correct
        / high_conf_total
        * 100.0
        if high_conf_total > 0
        else 0.0
    )

    closed_trades = int(
        account.get(
            "closed_trades",
            0,
        )
    )

    wins = int(
        account.get(
            "wins",
            0,
        )
    )

    losses = int(
        account.get(
            "losses",
            0,
        )
    )

    win_rate = (
        wins / closed_trades * 100.0
        if closed_trades > 0
        else 0.0
    )

    position = account.get(
        "position"
    )

    heartbeat_age_seconds = None
    learner_running = False

    if latest is not None:
        try:
            heartbeat_time = pd.Timestamp(
                latest[
                    "timestamp"
                ]
            )

            now_utc = pd.Timestamp.now(
                tz="UTC"
            )

            if heartbeat_time.tzinfo is None:
                heartbeat_time = heartbeat_time.tz_localize(
                    "UTC"
                )
            else:
                heartbeat_time = heartbeat_time.tz_convert(
                    "UTC"
                )

            heartbeat_age_seconds = max(
                0.0,
                float(
                    (
                        now_utc
                        - heartbeat_time
                    ).total_seconds()
                ),
            )

            learner_running = (
                heartbeat_age_seconds
                <= 20.0
            )
        except Exception:
            learner_running = False

    return {
        "running": learner_running,
        "message": (
            "Micro paper learner is actively updating."
            if learner_running
            else (
                "Micro paper learner snapshot is stale. "
                "Start the learner and collector to resume learning."
                if latest is not None
                else "Waiting for learner snapshots."
            )
        ),
        "heartbeatAgeSeconds":
            heartbeat_age_seconds,
        "prequentialAccuracyPct":
            prequential_accuracy_pct,
        "balancedAccuracyPct":
            balanced_accuracy_pct,
        "highConfidenceAccuracyPct":
            high_conf_accuracy_pct,
        "evaluationSamples":
            evaluation_total,
        "equityUsdt":
            safe_float(
                account.get(
                    "equity_usdt",
                    10000.0,
                )
            ),
        "realizedPnlUsdt":
            safe_float(
                account.get(
                    "realized_pnl_usdt",
                    0.0,
                )
            ),
        "closedTrades":
            closed_trades,
        "wins":
            wins,
        "losses":
            losses,
        "winRatePct":
            win_rate,
        "position":
            position,
        "latest":
            latest,
        "recentTrades":
            recent_trades,
    }


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
