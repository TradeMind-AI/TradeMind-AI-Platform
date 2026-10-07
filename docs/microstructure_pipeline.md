# TradeMind raw microstructure research pipeline

This pipeline is separate from the frozen 10-minute paper strategy.

## Raw exchange sources

The collector stores Binance events before creating ML features.

Spot BTCUSDT:
- aggregate trades
- best bid/ask book ticker
- top-20 order-book depth snapshots at 100 ms

USD-M Futures BTCUSDT perpetual:
- aggregate trades
- best bid/ask book ticker
- top-20 order-book depth snapshots at 100 ms
- mark price and current funding rate
- liquidation stream
- REST open-interest snapshots (default every 10 seconds)

Files are append-only JSON Lines under:

`data/raw/microstructure/`

Each line records TradeMind receive time, source, stream name, and the Binance payload.

## Run

Install the WebSocket dependency:

```powershell
python -m pip install websockets requests
```

Start collection:

```powershell
python backend/services/data_pipeline/collect_binance_microstructure.py
```

Stop with Ctrl+C.

Validate today's UTC files:

```powershell
python backend/services/data_pipeline/validate_microstructure_raw.py
```

Build one-second features:

```powershell
python backend/services/data_pipeline/build_microstructure_features.py --bucket 1s
```

Build five-second features:

```powershell
python backend/services/data_pipeline/build_microstructure_features.py --bucket 5s
```

Use `--date YYYY-MM-DD` to process a specific UTC date.

## Important

Order-book imbalance, spread, trade-flow imbalance, velocity, acceleration, and open-interest changes are derived measurements. Their inputs are the raw exchange events stored above.

The collector does not place orders and does not change the existing paper-trading model.
