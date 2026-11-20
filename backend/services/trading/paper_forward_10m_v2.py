from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.services.trading.paper_forward_10m import (
    COSTS,
    LOG_COLUMNS,
    has_signal_time,
    overlap_blocked,
    settle_existing,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "locked_xgboost_10m_bundle.joblib"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m_live_features.csv"
)

MARKET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m.csv"
)

LOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_forward_10m_v2.csv"
)

MODEL_NAME = "cost_aware_10m_rsi25_75_bollinger_v2"
CONFIDENCE_THRESHOLD = 0.55
RSI_LONG_MAX = 25.0
RSI_SHORT_MIN = 75.0

V2_EXTRA_COLUMNS = [
    "rsi14_filter",
    "bb_upper_filter",
    "bb_lower_filter",
    "filter_passed",
    "filter_reason",
]

V2_LOG_COLUMNS = [
    *LOG_COLUMNS,
    *V2_EXTRA_COLUMNS,
]


def add_discovery_filters(
    market: pd.DataFrame,
) -> pd.DataFrame:
    result = market.copy()

    close = pd.to_numeric(
        result["close"],
        errors="coerce",
    )

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    average_gain = gain.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=14,
    ).mean()

    average_loss = loss.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = (
        average_gain
        / average_loss.replace(
            0.0,
            np.nan,
        )
    )

    result["rsi14_filter"] = (
        100.0
        - 100.0
        / (1.0 + rs)
    )

    sma20 = close.rolling(
        20,
        min_periods=20,
    ).mean()

    std20 = close.rolling(
        20,
        min_periods=20,
    ).std(ddof=0)

    result["bb_upper_filter"] = (
        sma20
        + 2.0 * std20
    )

    result["bb_lower_filter"] = (
        sma20
        - 2.0 * std20
    )

    return result


def load_log() -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame(
            columns=V2_LOG_COLUMNS
        )

    log = pd.read_csv(LOG_FILE)

    for column in V2_LOG_COLUMNS:
        if column not in log.columns:
            log[column] = pd.NA

    return log[
        V2_LOG_COLUMNS
    ].copy()


def save_log(
    log: pd.DataFrame,
) -> None:
    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log.to_csv(
        LOG_FILE,
        index=False,
    )


def evaluate_v2_filter(
    prediction: str,
    confidence: float,
    close: float,
    rsi: float,
    bb_upper: float,
    bb_lower: float,
) -> tuple[str, bool, str]:
    if confidence < CONFIDENCE_THRESHOLD:
        return (
            "HOLD",
            False,
            "AI confidence below 0.55",
        )

    if prediction == "UP":
        passed = (
            rsi <= RSI_LONG_MAX
            and close <= bb_lower
        )

        return (
            "BUY" if passed else "SKIP_FILTER",
            passed,
            (
                "RSI<=25 and close<=lower Bollinger"
                if passed
                else "LONG rejected by RSI/Bollinger gate"
            ),
        )

    if prediction == "DOWN":
        passed = (
            rsi >= RSI_SHORT_MIN
            and close >= bb_upper
        )

        return (
            "SELL" if passed else "SKIP_FILTER",
            passed,
            (
                "RSI>=75 and close>=upper Bollinger"
                if passed
                else "SHORT rejected by RSI/Bollinger gate"
            ),
        )

    return (
        "HOLD",
        False,
        "Neutral AI prediction",
    )


def main() -> None:
    print(
        "Running TradeMind 10m V2 "
        "forward paper collector..."
    )

    bundle = joblib.load(MODEL_FILE)

    features = pd.read_csv(
        FEATURE_FILE
    )

    features["open_time"] = (
        pd.to_datetime(
            features["open_time"],
            utc=True,
        )
    )

    features = (
        features.sort_values(
            "open_time"
        )
        .reset_index(drop=True)
    )

    market = pd.read_csv(
        MARKET_FILE
    )

    market["open_time"] = (
        pd.to_datetime(
            market["open_time"],
            utc=True,
        )
    )

    market = (
        market.sort_values(
            "open_time"
        )
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    market = add_discovery_filters(
        market
    )

    log = load_log()
    log = settle_existing(
        log,
        market,
    )

    latest = features.iloc[-1]
    signal_time = latest["open_time"]

    if has_signal_time(
        log,
        signal_time,
    ):
        save_log(log)

        print(
            "V2 signal already recorded for "
            f"{signal_time}"
        )
        return

    market_row = market[
        market["open_time"]
        == signal_time
    ]

    if market_row.empty:
        raise RuntimeError(
            "Latest feature candle is missing "
            "from the 10m market file."
        )

    market_latest = market_row.iloc[-1]

    filter_values = {
        "rsi14_filter": float(
            market_latest[
                "rsi14_filter"
            ]
        ),
        "bb_upper_filter": float(
            market_latest[
                "bb_upper_filter"
            ]
        ),
        "bb_lower_filter": float(
            market_latest[
                "bb_lower_filter"
            ]
        ),
    }

    if any(
        not np.isfinite(value)
        for value in filter_values.values()
    ):
        raise RuntimeError(
            "V2 technical filters are not "
            "ready on the latest candle."
        )

    feature_columns = (
        bundle["feature_columns"]
    )

    X = pd.DataFrame(
        [
            latest[
                feature_columns
            ]
        ]
    )

    model = bundle["model"]
    encoder = bundle["encoder"]

    probabilities = (
        model.predict_proba(X)[0]
    )

    encoded = model.predict(X)[0]

    prediction = (
        encoder.inverse_transform(
            [encoded]
        )[0]
    )

    class_index = {
        name: index
        for index, name
        in enumerate(
            encoder.classes_
        )
    }

    confidence = float(
        probabilities.max()
    )

    signal, filter_passed, filter_reason = (
        evaluate_v2_filter(
            prediction=str(prediction),
            confidence=confidence,
            close=float(
                market_latest["close"]
            ),
            rsi=filter_values[
                "rsi14_filter"
            ],
            bb_upper=filter_values[
                "bb_upper_filter"
            ],
            bb_lower=filter_values[
                "bb_lower_filter"
            ],
        )
    )

    status = "NO_TRADE"

    if signal in {
        "BUY",
        "SELL",
    }:
        if overlap_blocked(
            log,
            signal_time,
        ):
            signal = "SKIP_OVERLAP"
            status = "NO_TRADE"
            filter_reason = (
                "Technical gate passed; "
                "blocked by 60m overlap rule"
            )
        else:
            status = "PENDING_ENTRY"

    entry_time = (
        signal_time
        + pd.Timedelta(minutes=10)
    )

    exit_time = (
        signal_time
        + pd.Timedelta(minutes=60)
    )

    row = {
        "signal_time": signal_time,
        "model_name": MODEL_NAME,
        "prediction": prediction,
        "confidence": confidence,
        "probability_down": float(
            probabilities[
                class_index["DOWN"]
            ]
        ),
        "probability_neutral": float(
            probabilities[
                class_index["NEUTRAL"]
            ]
        ),
        "probability_up": float(
            probabilities[
                class_index["UP"]
            ]
        ),
        "signal": signal,
        "status": status,
        "entry_candle_time": entry_time,
        "exit_candle_time": exit_time,
        **filter_values,
        "filter_passed": filter_passed,
        "filter_reason": filter_reason,
    }

    new_row = pd.DataFrame([row])

    for column in V2_LOG_COLUMNS:
        if column not in new_row.columns:
            new_row[column] = pd.NA

    log = pd.concat(
        [
            log,
            new_row[
                V2_LOG_COLUMNS
            ],
        ],
        ignore_index=True,
    )

    log = settle_existing(
        log,
        market,
    )

    save_log(log)

    closed = log[
        log["status"] == "CLOSED"
    ]

    print()
    print(f"Signal candle: {signal_time}")
    print(f"Prediction:    {prediction}")
    print(f"Confidence:    {confidence:.4f}")
    print(
        "RSI14:         "
        f"{filter_values['rsi14_filter']:.2f}"
    )
    print(
        "Bollinger:     "
        f"{filter_values['bb_lower_filter']:.2f} "
        "to "
        f"{filter_values['bb_upper_filter']:.2f}"
    )
    print(f"V2 signal:     {signal}")
    print(f"Filter reason: {filter_reason}")
    print(f"Status:        {status}")
    print()
    print(
        f"V2 forward log rows: {len(log)}"
    )
    print(
        "V2 closed forward trades: "
        f"{len(closed)}"
    )

    if not closed.empty:
        mid = pd.to_numeric(
            closed["mid_cost_net_pct"],
            errors="coerce",
        ).dropna()

        if not mid.empty:
            print(
                "V2 mid-cost win rate: "
                f"{(mid > 0).mean() * 100:.2f}%"
            )
            print(
                "V2 mid-cost cumulative "
                f"net return: {mid.sum():.2f}%"
            )

    print()
    print(f"Saved: {LOG_FILE}")


if __name__ == "__main__":
    main()
