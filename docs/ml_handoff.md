# TradeMind AI — ML Handoff

## 1. Purpose

This document describes the finalized ML pipeline and the prediction output that can be consumed by the trading engine and multi-agent system.

---

## 2. Dataset

### Market

- Asset: BTC/USDT
- Exchange: Binance Spot
- Timeframe: 1 hour
- Period: 2023-09-25 to 2026-09-24
- Clean candles: 26,304

### Cleaned dataset

```text
data/processed/btcusdt_1h_clean.csv