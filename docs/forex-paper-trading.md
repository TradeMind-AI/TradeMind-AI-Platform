# Forex Paper Trading Module

This feature adds a separate EUR/USD **simulator** alongside the existing BTC dashboard.

## Run locally

From the repository root:

```bash
pip install fastapi uvicorn pandas pydantic
python -m uvicorn backend.api.main:app --reload --port 8000
```

Then in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173/forex`. The frontend uses `VITE_API_BASE_URL` if configured and otherwise connects to `http://127.0.0.1:8000`.

## Endpoints

- `GET /api/forex/status` — persistent bot state, strategy signal, metrics, closed trades
- `GET /api/forex/candles?limit=200` — deterministic *simulated* EUR/USD 15-minute candles
- `POST /api/forex/start` — enable simulator
- `POST /api/forex/stop` — disable simulator
- `POST /api/forex/step` — advance exactly one synthetic candle while running
- `PUT /api/forex/settings` — risk, SL/TP, and signal-threshold inputs

The browser calls `/step` approximately every 2.5 seconds **only while the Forex page remains open**. Start does not launch an unattended server worker. The JSON ledger is `data/processed/forex_paper_state.json`; do not commit this file. Current ledger is **single-user only**; deployment with authentication/multiuser use requires an actual database and user-scoped state. A production scheduler should own advance events if unattended simulation is required.

## Financial and ML limitations

- Synthetic prices are generated via deterministic oscillations, **not real-time EUR/USD quotes**. There is no Longbridge API integration in the committed runtime.
- The indicator heuristic uses EMA20/EMA50 and RSI. Confidence is an uncalibrated signal-strength proxy; it is **not a probability of winning**.
- The BTC model training code is in `backend/services/ml`. The trained `locked_xgboost_bundle.joblib` is not included in the repo and the BTC feature schema is not directly compatible with FX. **No BTC weights are loaded and no FX fine-tuning is performed** by this MVP.
- Position sizing caps initial risk to 1% of current equity with a 10x notional limit. Spread/slippage are modeled in simulated entries/exits. During a candle that touches stop and target, the stop executes first. Gaps are not modeled.
- An 80% winning-trade threshold is a measurement target only. A win rate can be high while losing money. Validate **profit factor, net P&L, drawdown, and out-of-sample performance** including trading costs.

## Next steps for real FX ML research

1. Obtain licensed historical EUR/USD bid/ask candles and split chronologically into train, validation, untouched holdout (with walk-forward testing).
2. Build FX-specific feature pipelines matching model inputs and avoid future leakage.
3. If transferring BTC knowledge, explicitly map compatible features and retrain/recalibrate with FX examples, then version the serialized model and feature definitions.
4. Compare the model to this simple baseline using win rate, P&L, costs, drawdown and trade count. Only enable inference when validated artifacts exist.
5. Add user-scoped persistence and an authenticated background worker before any persistent hosting.

**Never use this simulator to place live orders.**
