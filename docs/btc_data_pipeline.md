# BTC Historical Data Pipeline

## 1. Overview

This document describes the historical Bitcoin market-data pipeline used by TradeMind AI.

The pipeline collects historical BTC/USDT market data, stores the original raw data, cleans and validates the data, and produces a reproducible processed dataset for future machine-learning and deep-learning workflows.

The pipeline does not implement trading strategies, trading signals, machine-learning models, AI agents, or live trading.

---

## 2. Data Source

**Source:** Binance Spot Market Data API

**Trading pair:** BTC/USDT

**Interval:** 1 hour

**API endpoint:**

`https://data-api.binance.vision/api/v3/klines`

The endpoint provides public historical candlestick (kline) data and does not require API credentials for this historical-data collection process.

---

## 3. Dataset Coverage

The initial dataset covers:

- Start: 2023-09-25 00:00:00 UTC
- End: 2026-09-24 23:00:00 UTC
- Interval: 1 hour
- Total candles: 26,304

All timestamps are stored in UTC.

---

## 4. Market Data Fields

The Binance kline response contains the following fields:

- `open_time`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `close_time`
- `quote_asset_volume`
- `number_of_trades`
- `taker_buy_base_volume`
- `taker_buy_quote_volume`
- `ignore`

The primary OHLCV fields used by the current pipeline are:

- Open
- High
- Low
- Close
- Volume

Additional Binance kline fields are retained in the raw dataset for future use.

---

## 5. Data Collection

Historical data is collected using:

`backend/services/data_pipeline/fetch_btc_data.py`

Binance limits the number of candles returned by a single request, so the pipeline uses pagination.

The collection process:

1. Sends a request for a batch of candles.
2. Converts Binance timestamps from milliseconds to UTC timestamps.
3. Stores the returned batch.
4. Advances the start time by one hour.
5. Requests the next batch.
6. Continues until the requested end date is reached.
7. Combines all batches.
8. Removes duplicate timestamps.
9. Sorts the final dataset chronologically.

The download process is implemented in:

`backend/services/data_pipeline/download_btc_history.py`

---

## 6. Raw Data Storage

The original collected dataset is stored locally at:

`data/raw/btcusdt_1h.csv`

The raw dataset is preserved and is not modified by the cleaning process.

Raw CSV files are excluded from Git tracking because they are generated datasets rather than source code.

---

## 7. Data Cleaning

Cleaning is implemented in:

`backend/services/data_pipeline/clean_btc_data.py`

The cleaning process:

1. Converts timestamps to UTC.
2. Converts market values to numeric data types.
3. Removes rows with invalid required values.
4. Removes duplicate candle timestamps.
5. Validates OHLC relationships.
6. Removes rows with negative volume.
7. Sorts the data chronologically.
8. Detects missing or irregular hourly intervals.

Missing intervals are detected and reported rather than being filled with fabricated market data.

The raw dataset is never overwritten.

---

## 8. Processed Data Storage

The cleaned dataset is generated at:

`data/processed/btcusdt_1h_clean.csv`

The processed dataset is generated from the raw dataset by running:

`backend/services/data_pipeline/clean_btc_data.py`

Processed CSV files are excluded from Git tracking because they are reproducible generated artifacts.

---

## 9. Data Validation

The validation process checks:

- Duplicate timestamps
- Chronological timestamp ordering
- Missing or irregular hourly intervals
- OHLC relationships
- Negative volume
- Null values

The validation code is implemented in:

`backend/services/data_pipeline/validate_btc_data.py`

Full-dataset validation is performed using:

`backend/services/data_pipeline/validate_full_btc_data.py`

### Validation Results

For the initial 26,304-candle dataset:

| Check | Result |
|---|---:|
| Duplicate timestamps | 0 |
| Timestamps sorted | Yes |
| Missing/irregular hourly intervals | 0 |
| Invalid OHLC rows | 0 |
| Negative volume rows | 0 |
| Null values | 0 |

The processed dataset also passed independent validation with the same results.

---

## 10. Data Quality Results

The cleaning process produced:

- Original rows: 26,304
- Rows after invalid/null removal: 26,304
- Rows after duplicate removal: 26,304
- Rows after OHLC validation: 26,304
- Rows after volume validation: 26,304
- Final rows: 26,304

No rows were removed during the initial cleaning run because the collected dataset passed all quality checks.

---

## 11. Reproducibility

The pipeline is designed so that another team member can reproduce the processed dataset.

The general workflow is:

```text
Binance API
    ↓
fetch_btc_data.py
    ↓
data/raw/btcusdt_1h.csv
    ↓
clean_btc_data.py
    ↓
data/processed/btcusdt_1h_clean.csv
    ↓
validation