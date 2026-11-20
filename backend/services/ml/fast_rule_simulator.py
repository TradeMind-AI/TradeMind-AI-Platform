"""Fast, parity-tested NumPy rule screening for TradeMind research.

Only the candidate screening is vectorized. Execution remains greedy and
non-overlapping, matching the original strategy's exit_index + 1 rule.
"""
import numpy as np
import pandas as pd


def prepare_rule_arrays(df, validation_start, test_start, horizon_bars):
    def numeric(name):
        return df[name].to_numpy(dtype=np.float64, na_value=np.nan)

    prediction = df["ai_prediction"]
    return {
        "time": df["open_time"].tolist(),
        "validation_start": int(df["open_time"].searchsorted(validation_start)),
        "test_start": test_start,
        "horizon": horizon_bars,
        "up": prediction.eq("UP").fillna(False).to_numpy(dtype=bool),
        "down": prediction.eq("DOWN").fillna(False).to_numpy(dtype=bool),
        **{name: numeric(name) for name in (
            "ai_confidence", "rsi14_filter", "ema20_filter",
            "macd_hist_filter", "bb_upper_filter", "bb_lower_filter",
            "relative_volume_filter", "close", "open"
        )},
    }


def simulate_rule_fast(cache, rule, costs):
    h = cache["horizon"]
    n = len(cache["time"])
    start = cache["validation_start"]
    if start >= n - h - 1:
        return []

    up = cache["up"]
    down = cache["down"]
    conf = cache["ai_confidence"]
    mask = (up | down) & ~np.isnan(conf) & (conf >= rule.confidence)

    if rule.rsi_rule is not None:
        rsi = cache["rsi14_filter"]
        low, high = rule.rsi_rule
        mask &= ~np.isnan(rsi) & ~((up & (rsi > low)) | (down & (rsi < high)))

    if rule.ema_alignment:
        ema = cache["ema20_filter"]
        price = cache["close"]
        mask &= ~np.isnan(ema) & ~((up & (price <= ema)) | (down & (price >= ema)))

    if rule.macd_alignment:
        hist = cache["macd_hist_filter"]
        mask &= ~np.isnan(hist) & ~((up & (hist <= 0.0)) | (down & (hist >= 0.0)))

    if rule.bollinger_reversion:
        upper, lower = cache["bb_upper_filter"], cache["bb_lower_filter"]
        price = cache["close"]
        mask &= ~np.isnan(upper) & ~np.isnan(lower)
        mask &= ~((up & (price > lower)) | (down & (price < upper)))

    if rule.relative_volume_min is not None:
        volume = cache["relative_volume_filter"]
        mask &= ~np.isnan(volume) & (volume >= rule.relative_volume_min)

    mask[:start] = False
    mask[n - h - 1:] = False

    trades = []
    next_allowed = start
    times = cache["time"]
    opens, closes = cache["open"], cache["close"]
    rsi = cache["rsi14_filter"]
    vol = cache["relative_volume_filter"]
    test_start = cache["test_start"]

    for i in np.flatnonzero(mask):
        if i < next_allowed:
            continue
        entry_price = float(opens[i + 1])
        exit_price = float(closes[i + h])
        direction = 1 if up[i] else -1
        gross_return = (exit_price / entry_price - 1.0) * direction
        trade = {
            "rule_key": rule.key,
            "rule": rule.label,
            "period": "validation" if times[i] < test_start else "test",
            "signal_time": times[i],
            "direction": "LONG" if direction == 1 else "SHORT",
            "confidence": float(conf[i]),
            "rsi14": rsi[i],
            "relative_volume": vol[i],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return_pct": gross_return * 100.0,
        }
        for name, cost in costs.items():
            trade[f"net_{name}_pct"] = (gross_return - cost) * 100.0
        trades.append(trade)
        next_allowed = i + h + 1
    return trades
