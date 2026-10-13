from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "technical_strategy_benchmark"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "strategy_results.csv"
)

TRADE_FILE = (
    OUTPUT_DIR
    / "strategy_trades.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "summary.json"
)

RESEARCH_START = pd.Timestamp(
    "2023-03-26",
    tz="UTC",
)

VALIDATION_START = pd.Timestamp(
    "2025-01-01",
    tz="UTC",
)

TEST_START = pd.Timestamp(
    "2025-10-01",
    tz="UTC",
)

HORIZON_BARS = 6

COSTS = {
    "low": 0.0006,
    "mid": 0.0014,
    "high": 0.0027,
}


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str


STRATEGIES = [
    Strategy(
        key="ema_trend",
        name="EMA Trend",
    ),
    Strategy(
        key="rsi_reversion",
        name="RSI Mean Reversion",
    ),
    Strategy(
        key="bollinger_breakout",
        name="Bollinger Breakout",
    ),
    Strategy(
        key="macd_momentum",
        name="MACD Momentum",
    ),
]


def ema(
    series: pd.Series,
    span: int,
) -> pd.Series:
    return series.ewm(
        span=span,
        adjust=False,
        min_periods=span,
    ).mean()


def add_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    close = result["close"]

    result["sma20"] = (
        close
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    result["ema20"] = ema(
        close,
        20,
    )

    delta = close.diff()

    gain = delta.clip(
        lower=0.0,
    )

    loss = (
        -delta.clip(
            upper=0.0,
        )
    )

    average_gain = (
        gain
        .ewm(
            alpha=1.0 / 14.0,
            adjust=False,
            min_periods=14,
        )
        .mean()
    )

    average_loss = (
        loss
        .ewm(
            alpha=1.0 / 14.0,
            adjust=False,
            min_periods=14,
        )
        .mean()
    )

    rs = (
        average_gain
        / average_loss.replace(
            0.0,
            np.nan,
        )
    )

    result["rsi14"] = (
        100.0
        - 100.0
        / (
            1.0
            + rs
        )
    )

    rolling_std = (
        close
        .rolling(
            20,
            min_periods=20,
        )
        .std(
            ddof=0,
        )
    )

    result["bb_upper"] = (
        result["sma20"]
        + 2.0
        * rolling_std
    )

    result["bb_lower"] = (
        result["sma20"]
        - 2.0
        * rolling_std
    )

    ema12 = ema(
        close,
        12,
    )

    ema26 = ema(
        close,
        26,
    )

    result["macd"] = (
        ema12
        - ema26
    )

    result["macd_signal"] = (
        result["macd"]
        .ewm(
            span=9,
            adjust=False,
            min_periods=9,
        )
        .mean()
    )

    result["macd_hist"] = (
        result["macd"]
        - result["macd_signal"]
    )

    return result


def strategy_signal(
    row: pd.Series,
    strategy: str,
) -> int:
    if strategy == "ema_trend":
        if pd.isna(
            row["ema20"]
        ) or pd.isna(
            row["sma20"]
        ):
            return 0

        if (
            row["ema20"]
            > row["sma20"]
        ):
            return 1

        if (
            row["ema20"]
            < row["sma20"]
        ):
            return -1

        return 0

    if strategy == "rsi_reversion":
        if pd.isna(
            row["rsi14"]
        ):
            return 0

        if row["rsi14"] <= 30.0:
            return 1

        if row["rsi14"] >= 70.0:
            return -1

        return 0

    if strategy == "bollinger_breakout":
        if (
            pd.isna(
                row["bb_upper"]
            )
            or pd.isna(
                row["bb_lower"]
            )
        ):
            return 0

        if (
            row["close"]
            > row["bb_upper"]
        ):
            return 1

        if (
            row["close"]
            < row["bb_lower"]
        ):
            return -1

        return 0

    if strategy == "macd_momentum":
        if pd.isna(
            row["macd_hist"]
        ):
            return 0

        if (
            row["macd_hist"]
            > 0.0
        ):
            return 1

        if (
            row["macd_hist"]
            < 0.0
        ):
            return -1

        return 0

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


def simulate(
    df: pd.DataFrame,
    strategy: Strategy,
    period_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> list[dict]:
    trades = []

    index = 0

    while (
        index
        < len(df) - HORIZON_BARS - 1
    ):
        signal_time = (
            df.iloc[index][
                "open_time"
            ]
        )

        if signal_time < start:
            index += 1
            continue

        if (
            end is not None
            and signal_time >= end
        ):
            break

        direction = strategy_signal(
            df.iloc[index],
            strategy.key,
        )

        if direction == 0:
            index += 1
            continue

        entry_index = (
            index + 1
        )

        exit_index = (
            entry_index
            + HORIZON_BARS
            - 1
        )

        if exit_index >= len(df):
            break

        entry_row = (
            df.iloc[
                entry_index
            ]
        )

        exit_row = (
            df.iloc[
                exit_index
            ]
        )

        if (
            end is not None
            and entry_row[
                "open_time"
            ] >= end
        ):
            break

        entry_price = float(
            entry_row[
                "open"
            ]
        )

        exit_price = float(
            exit_row[
                "close"
            ]
        )

        gross_return = (
            (
                exit_price
                / entry_price
                - 1.0
            )
            * direction
        )

        trade = {
            "strategy":
                strategy.name,
            "strategy_key":
                strategy.key,
            "period":
                period_name,
            "signal_time":
                signal_time,
            "entry_time":
                entry_row[
                    "open_time"
                ],
            "exit_time":
                exit_row[
                    "open_time"
                ],
            "direction":
                (
                    "LONG"
                    if direction == 1
                    else "SHORT"
                ),
            "entry_price":
                entry_price,
            "exit_price":
                exit_price,
            "gross_return_pct":
                gross_return
                * 100.0,
        }

        for (
            cost_name,
            cost,
        ) in COSTS.items():
            trade[
                f"net_{cost_name}_pct"
            ] = (
                gross_return
                - cost
            ) * 100.0

        trades.append(
            trade
        )

        # Non-overlapping trades:
        # the next signal is considered
        # only after this 60-minute
        # research trade has completed.
        index = exit_index + 1

    return trades


def summarize(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for strategy in STRATEGIES:
        for period in [
            "validation",
            "test",
        ]:
            subset = trades[
                (
                    trades[
                        "strategy_key"
                    ]
                    == strategy.key
                )
                & (
                    trades[
                        "period"
                    ]
                    == period
                )
            ]

            row = {
                "strategy":
                    strategy.name,
                "strategy_key":
                    strategy.key,
                "period":
                    period,
                "trades":
                    int(
                        len(
                            subset
                        )
                    ),
            }

            if subset.empty:
                for name in COSTS:
                    row[
                        f"win_rate_{name}_pct"
                    ] = 0.0

                    row[
                        f"net_{name}_sum_pct"
                    ] = 0.0

                    row[
                        f"avg_net_{name}_pct"
                    ] = 0.0

                rows.append(
                    row
                )
                continue

            row[
                "gross_sum_pct"
            ] = float(
                subset[
                    "gross_return_pct"
                ].sum()
            )

            row[
                "gross_win_rate_pct"
            ] = float(
                (
                    subset[
                        "gross_return_pct"
                    ]
                    > 0.0
                ).mean()
                * 100.0
            )

            for name in COSTS:
                column = (
                    f"net_{name}_pct"
                )

                row[
                    f"win_rate_{name}_pct"
                ] = float(
                    (
                        subset[
                            column
                        ]
                        > 0.0
                    ).mean()
                    * 100.0
                )

                row[
                    f"net_{name}_sum_pct"
                ] = float(
                    subset[
                        column
                    ].sum()
                )

                row[
                    f"avg_net_{name}_pct"
                ] = float(
                    subset[
                        column
                    ].mean()
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    print(
        "TRADEMIND - TECHNICAL STRATEGY BENCHMARK"
    )
    print()

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing 10m market file: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    if "open_time" not in df.columns:
        raise RuntimeError(
            "Expected an open_time column."
        )

    required = {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
        errors="coerce",
    )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "open_time",
                *required,
            ]
        )
        .sort_values(
            "open_time"
        )
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    df = df[
        df["open_time"]
        >= RESEARCH_START
    ].copy().reset_index(
        drop=True
    )

    df = add_indicators(
        df
    )

    all_trades = []

    for strategy in STRATEGIES:
        all_trades.extend(
            simulate(
                df,
                strategy,
                "validation",
                VALIDATION_START,
                TEST_START,
            )
        )

        all_trades.extend(
            simulate(
                df,
                strategy,
                "test",
                TEST_START,
                None,
            )
        )

    trades = pd.DataFrame(
        all_trades
    )

    results = summarize(
        trades
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        TRADE_FILE,
        index=False,
    )

    results.to_csv(
        RESULT_FILE,
        index=False,
    )

    summary = {
        "timeframe":
            "10m",
        "horizon_minutes":
            60,
        "entry_model":
            "Signal at candle close; enter next 10m open",
        "exit_model":
            "Fixed 60m horizon close",
        "overlap_policy":
            "No overlapping trades",
        "validation_start":
            str(
                VALIDATION_START
            ),
        "test_start":
            str(
                TEST_START
            ),
        "costs_pct": {
            name:
                value * 100.0
            for name, value
            in COSTS.items()
        },
        "strategies": [
            {
                "key":
                    strategy.key,
                "name":
                    strategy.name,
            }
            for strategy
            in STRATEGIES
        ],
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        results[
            [
                "strategy",
                "period",
                "trades",
                "win_rate_mid_pct",
                "net_mid_sum_pct",
                "avg_net_mid_pct",
                "net_high_sum_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Interpretation rule:"
    )
    print(
        "A TradingView-style strategy only helps "
        "if it remains useful AFTER costs and also "
        "holds up in the later test period."
    )

    print()
    print(
        "Saved:"
    )
    print(
        RESULT_FILE
    )
    print(
        TRADE_FILE
    )
    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()
