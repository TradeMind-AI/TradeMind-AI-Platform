# Multi-timeframe paper trading

TradeMind supports the strategy timeframes:

- 1 minute
- 2 minutes
- 5 minutes
- 10 minutes
- 20 minutes
- 30 minutes
- 1 hour

## Design

The short-interval strategies must not reuse the locked 1-hour model. Each timeframe has a different return distribution and must be trained, validated, and paper-tested independently.

The base historical dataset is BTCUSDT 1-minute spot candles. Higher intervals are resampled from those candles using standard OHLCV aggregation:

- open: first
- high: maximum
- low: minimum
- close: last
- volume: sum

The existing locked 1-hour strategy remains unchanged while shorter strategies are developed.

## Workflow

1. Download a historical 1-minute dataset with `fetch_1m_history.py`.
2. Generate 1m, 2m, 5m, 10m, 20m, 30m, and 1h datasets with `resample_timeframes.py`.
3. Build features for each timeframe.
4. Train and validate a separate model for each timeframe.
5. Lock a model only after after-cost validation.
6. Run each locked strategy in paper-trading mode.
7. Compare strategies using net return, trade count, precision, drawdown, average trade, and cost sensitivity.

## Important

A model trained on 1-hour candles must not be applied to 1-minute or other shorter candles.

Shorter intervals are especially sensitive to fees, spread, and slippage, so model accuracy alone is not sufficient evidence of a viable strategy.
