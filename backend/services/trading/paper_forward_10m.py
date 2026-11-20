from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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
    / "paper_forward_10m.csv"
)

LOG_COLUMNS = [
    "signal_time",
    "model_name",
    "prediction",
    "confidence",
    "probability_down",
    "probability_neutral",
    "probability_up",
    "signal",
    "status",
    "entry_candle_time",
    "exit_candle_time",
    "entry_price",
    "exit_price",
    "gross_return_pct",
    "low_cost_net_pct",
    "mid_cost_net_pct",
    "high_cost_net_pct",
]

COSTS = {
    "low": 0.0006,
    "mid": 0.0014,
    "high": 0.0027,
}


def load_log() -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame(
            columns=LOG_COLUMNS
        )

    log = pd.read_csv(
        LOG_FILE
    )

    for column in LOG_COLUMNS:
        if column not in log.columns:
            log[column] = pd.NA

    return log[
        LOG_COLUMNS
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


def settle_existing(
    log: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    if log.empty:
        return log

    market_by_time = (
        market.set_index(
            "open_time"
        )
    )

    for index, row in log.iterrows():
        if row["signal"] not in {
            "BUY",
            "SELL",
        }:
            continue

        if row["status"] == "CLOSED":
            continue

        entry_time = pd.to_datetime(
            row["entry_candle_time"],
            utc=True,
        )

        exit_time = pd.to_datetime(
            row["exit_candle_time"],
            utc=True,
        )

        if (
            pd.isna(
                row["entry_price"]
            )
            and entry_time
            in market_by_time.index
        ):
            log.at[
                index,
                "entry_price",
            ] = float(
                market_by_time.loc[
                    entry_time,
                    "open",
                ]
            )

            log.at[
                index,
                "status",
            ] = "ACTIVE"

        if (
            exit_time
            not in market_by_time.index
        ):
            continue

        entry_price = pd.to_numeric(
            pd.Series(
                [
                    log.at[
                        index,
                        "entry_price",
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if pd.isna(entry_price):
            continue

        exit_price = float(
            market_by_time.loc[
                exit_time,
                "close",
            ]
        )

        if row["signal"] == "BUY":
            gross = (
                exit_price
                / float(entry_price)
                - 1.0
            )
        else:
            gross = (
                float(entry_price)
                / exit_price
                - 1.0
            )

        log.at[
            index,
            "exit_price",
        ] = exit_price

        log.at[
            index,
            "gross_return_pct",
        ] = gross * 100

        log.at[
            index,
            "low_cost_net_pct",
        ] = (
            gross - COSTS["low"]
        ) * 100

        log.at[
            index,
            "mid_cost_net_pct",
        ] = (
            gross - COSTS["mid"]
        ) * 100

        log.at[
            index,
            "high_cost_net_pct",
        ] = (
            gross - COSTS["high"]
        ) * 100

        log.at[
            index,
            "status",
        ] = "CLOSED"

    return log


def has_signal_time(
    log: pd.DataFrame,
    signal_time,
) -> bool:
    if log.empty:
        return False

    times = pd.to_datetime(
        log["signal_time"],
        utc=True,
        errors="coerce",
    )

    return signal_time in set(
        times.dropna()
    )


def overlap_blocked(
    log: pd.DataFrame,
    signal_time,
) -> bool:
    if log.empty:
        return False

    accepted = log[
        log["signal"].isin(
            [
                "BUY",
                "SELL",
            ]
        )
    ].copy()

    if accepted.empty:
        return False

    accepted_times = pd.to_datetime(
        accepted[
            "signal_time"
        ],
        utc=True,
        errors="coerce",
    ).dropna()

    if accepted_times.empty:
        return False

    last_signal = (
        accepted_times.max()
    )

    return (
        signal_time
        < last_signal
        + pd.Timedelta(minutes=60)
    )


def main() -> None:
    print(
        "Running frozen 10m forward "
        "paper collector..."
    )

    bundle = joblib.load(
        MODEL_FILE
    )

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

    log = load_log()

    log = settle_existing(
        log,
        market,
    )

    latest = features.iloc[-1]

    signal_time = latest[
        "open_time"
    ]

    if has_signal_time(
        log,
        signal_time,
    ):
        save_log(log)

        print(
            f"Signal already recorded for "
            f"{signal_time}"
        )

        return

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
        model.predict_proba(
            X
        )[0]
    )

    encoded = model.predict(
        X
    )[0]

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

    signal = "HOLD"
    status = "NO_TRADE"

    if (
        confidence
        >= float(
            bundle[
                "confidence_threshold"
            ]
        )
    ):
        if prediction == "UP":
            signal = "BUY"

        elif prediction == "DOWN":
            signal = "SELL"

    if signal in {
        "BUY",
        "SELL",
    }:
        if overlap_blocked(
            log,
            signal_time,
        ):
            signal = (
                "SKIP_OVERLAP"
            )
            status = (
                "NO_TRADE"
            )
        else:
            status = (
                "PENDING_ENTRY"
            )

    entry_time = (
        signal_time
        + pd.Timedelta(minutes=10)
    )

    exit_time = (
        signal_time
        + pd.Timedelta(minutes=60)
    )

    row = {
        "signal_time":
            signal_time,
        "model_name":
            bundle["model_name"],
        "prediction":
            prediction,
        "confidence":
            confidence,
        "probability_down":
            float(
                probabilities[
                    class_index["DOWN"]
                ]
            ),
        "probability_neutral":
            float(
                probabilities[
                    class_index["NEUTRAL"]
                ]
            ),
        "probability_up":
            float(
                probabilities[
                    class_index["UP"]
                ]
            ),
        "signal":
            signal,
        "status":
            status,
        "entry_candle_time":
            entry_time,
        "exit_candle_time":
            exit_time,
    }

    new_row = pd.DataFrame(
        [row]
    )

    for column in LOG_COLUMNS:
        if column not in new_row.columns:
            new_row[column] = pd.NA

    log = pd.concat(
        [
            log,
            new_row[
                LOG_COLUMNS
            ],
        ],
        ignore_index=True,
    )

    log = settle_existing(
        log,
        market,
    )

    save_log(log)

    print()
    print(
        f"Signal candle: {signal_time}"
    )

    print(
        f"Prediction:    {prediction}"
    )

    print(
        f"Confidence:    {confidence:.4f}"
    )

    print(
        f"Paper signal:  {signal}"
    )

    print(
        f"Status:        {status}"
    )

    closed = log[
        log["status"] == "CLOSED"
    ]

    print()
    print(
        f"Forward log rows: "
        f"{len(log)}"
    )

    print(
        f"Closed forward trades: "
        f"{len(closed)}"
    )

    if not closed.empty:
        mid = pd.to_numeric(
            closed[
                "mid_cost_net_pct"
            ],
            errors="coerce",
        )

        print(
            f"Mid-cost cumulative "
            f"net return: "
            f"{mid.sum():.2f}%"
        )

    print()
    print(
        f"Saved: {LOG_FILE}"
    )


if __name__ == "__main__":
    main()
