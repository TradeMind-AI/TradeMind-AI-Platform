from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_1M_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "btcusdt_1m.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TIMEFRAME_RULES = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "10m": "10min",
    "20m": "20min",
    "30m": "30min",
    "1h": "1h",
}


def load_base_1m(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Base 1-minute dataset not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required 1-minute columns: "
            + ", ".join(sorted(missing))
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
        errors="coerce",
    )

    df = (
        df.dropna(subset=["open_time"])
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def aggregate_timeframe(
    base: pd.DataFrame,
    rule: str,
) -> pd.DataFrame:
    indexed = base.set_index("open_time")

    expected_minutes = int(
        pd.to_timedelta(rule)
        / pd.Timedelta(minutes=1)
    )

    aggregations: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    optional_sum_columns = [
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for column in optional_sum_columns:
        if column in indexed.columns:
            aggregations[column] = "sum"

    grouped = indexed.resample(
        rule,
        label="left",
        closed="left",
        origin="epoch",
    )

    resampled = grouped.agg(
        aggregations
    )

    source_counts = grouped[
        "close"
    ].count()

    complete_mask = (
        source_counts
        == expected_minutes
    )

    incomplete_count = int(
        (
            (source_counts > 0)
            & (~complete_mask)
        ).sum()
    )

    resampled = (
        resampled.loc[
            complete_mask
        ]
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

    if incomplete_count:
        print(
            f"  dropped {incomplete_count:,} "
            f"incomplete {rule} candle(s) "
            f"caused by missing 1m source data"
        )

    if "number_of_trades" not in resampled.columns:
        resampled["number_of_trades"] = 0

    if "taker_buy_base_volume" not in resampled.columns:
        resampled["taker_buy_base_volume"] = 0.0

    if "quote_asset_volume" not in resampled.columns:
        resampled["quote_asset_volume"] = 0.0

    if "taker_buy_quote_volume" not in resampled.columns:
        resampled["taker_buy_quote_volume"] = 0.0

    interval = pd.to_timedelta(rule)

    resampled["close_time"] = (
        resampled["open_time"]
        + interval
        - pd.Timedelta(milliseconds=1)
    )

    resampled["ignore"] = 0

    return resampled


def save_timeframe(
    timeframe: str,
    frame: pd.DataFrame,
) -> Path:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / f"btcusdt_{timeframe}.csv"
    )

    frame.to_csv(
        path,
        index=False,
    )

    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resample BTCUSDT 1-minute candles "
            "into TradeMind strategy timeframes."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_1M_FILE,
    )

    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=list(TIMEFRAME_RULES),
        choices=list(TIMEFRAME_RULES),
    )

    args = parser.parse_args()

    base = load_base_1m(
        args.input
    )

    print(
        f"Loaded {len(base):,} base 1-minute candles."
    )

    for timeframe in args.timeframes:
        rule = TIMEFRAME_RULES[timeframe]

        if timeframe == "1m":
            result = aggregate_timeframe(
                base,
                rule,
            )
        else:
            result = aggregate_timeframe(
                base,
                rule,
            )

        output = save_timeframe(
            timeframe,
            result,
        )

        print(
            f"{timeframe}: "
            f"{len(result):,} candles -> {output}"
        )


if __name__ == "__main__":
    main()
