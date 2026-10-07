from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "microstructure"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "microstructure"
)

DEPTH_LEVELS = 5


def today_utc() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")


def iter_records(
    path: Path,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            try:
                yield json.loads(
                    line
                )
            except json.JSONDecodeError:
                continue


def event_time_ms(
    record: dict,
) -> int:
    payload = record.get(
        "payload",
        {}
    )

    for key in (
        "E",
        "T",
        "time",
    ):
        value = payload.get(
            key
        )

        if isinstance(
            value,
            int,
        ):
            return value

    return int(
        record[
            "received_time_ms"
        ]
    )


def depth_metrics(
    payload: dict,
) -> dict:
    bids = (
        payload.get("bids")
        or payload.get("b")
        or []
    )

    asks = (
        payload.get("asks")
        or payload.get("a")
        or []
    )

    bids = bids[
        :DEPTH_LEVELS
    ]

    asks = asks[
        :DEPTH_LEVELS
    ]

    bid_depth = sum(
        float(row[1])
        for row in bids
        if len(row) >= 2
    )

    ask_depth = sum(
        float(row[1])
        for row in asks
        if len(row) >= 2
    )

    denominator = (
        bid_depth
        + ask_depth
    )

    imbalance = (
        (
            bid_depth
            - ask_depth
        )
        / denominator
        if denominator > 0
        else 0.0
    )

    best_bid = (
        float(bids[0][0])
        if bids
        else np.nan
    )

    best_ask = (
        float(asks[0][0])
        if asks
        else np.nan
    )

    mid = (
        (
            best_bid
            + best_ask
        )
        / 2
        if np.isfinite(best_bid)
        and np.isfinite(best_ask)
        else np.nan
    )

    spread_bps = (
        (
            best_ask
            - best_bid
        )
        / mid
        * 10000
        if np.isfinite(mid)
        and mid > 0
        else np.nan
    )

    return {
        "bid_depth":
            bid_depth,
        "ask_depth":
            ask_depth,
        "order_book_imbalance":
            imbalance,
        "best_bid":
            best_bid,
        "best_ask":
            best_ask,
        "spread_bps":
            spread_bps,
    }


def trade_row(
    record: dict,
    source: str,
) -> dict | None:
    payload = record.get(
        "payload",
        {}
    )

    if "p" not in payload:
        return None

    price = float(
        payload["p"]
    )

    quantity = float(
        payload.get(
            "q",
            0.0,
        )
    )

    buyer_is_maker = bool(
        payload.get(
            "m",
            False,
        )
    )

    aggressive_side = (
        "SELL"
        if buyer_is_maker
        else "BUY"
    )

    return {
        "time_ms":
            event_time_ms(
                record
            ),
        "source":
            source,
        "price":
            price,
        "quantity":
            quantity,
        "aggressive_side":
            aggressive_side,
        "notional":
            price * quantity,
    }


def load_source(
    path: Path,
    source: str,
):
    trades = []
    depth = []
    book = []
    mark = []
    liquidations = []
    open_interest = []

    if not path.exists():
        return (
            trades,
            depth,
            book,
            mark,
            liquidations,
            open_interest,
        )

    for record in iter_records(
        path
    ):
        stream = str(
            record.get(
                "stream",
                "",
            )
        )

        payload = record.get(
            "payload",
            {}
        )

        ts = event_time_ms(
            record
        )

        if stream.endswith(
            "@aggTrade"
        ):
            row = trade_row(
                record,
                source,
            )

            if row is not None:
                trades.append(
                    row
                )

        elif (
            "@depth" in stream
        ):
            metrics = depth_metrics(
                payload
            )

            metrics[
                "time_ms"
            ] = ts

            metrics[
                "source"
            ] = source

            depth.append(
                metrics
            )

        elif stream.endswith(
            "@bookTicker"
        ):
            bid = float(
                payload.get(
                    "b",
                    np.nan,
                )
            )

            ask = float(
                payload.get(
                    "a",
                    np.nan,
                )
            )

            bid_qty = float(
                payload.get(
                    "B",
                    0.0,
                )
            )

            ask_qty = float(
                payload.get(
                    "A",
                    0.0,
                )
            )

            mid = (
                (
                    bid + ask
                )
                / 2
                if np.isfinite(bid)
                and np.isfinite(ask)
                else np.nan
            )

            book.append(
                {
                    "time_ms": ts,
                    "source": source,
                    "best_bid": bid,
                    "best_ask": ask,
                    "best_bid_qty":
                        bid_qty,
                    "best_ask_qty":
                        ask_qty,
                    "spread_bps":
                        (
                            (ask - bid)
                            / mid
                            * 10000
                            if np.isfinite(mid)
                            and mid > 0
                            else np.nan
                        ),
                }
            )

        elif stream.endswith(
            "@markPrice@1s"
        ):
            mark.append(
                {
                    "time_ms": ts,
                    "mark_price":
                        float(
                            payload.get(
                                "p",
                                np.nan,
                            )
                        ),
                    "index_price":
                        float(
                            payload.get(
                                "i",
                                np.nan,
                            )
                        ),
                    "funding_rate":
                        float(
                            payload.get(
                                "r",
                                np.nan,
                            )
                        ),
                    "next_funding_time":
                        payload.get(
                            "T"
                        ),
                }
            )

        elif stream.endswith(
            "@forceOrder"
        ):
            order = payload.get(
                "o",
                {}
            )

            liquidations.append(
                {
                    "time_ms": ts,
                    "side":
                        order.get(
                            "S"
                        ),
                    "price":
                        float(
                            order.get(
                                "p",
                                0.0,
                            )
                        ),
                    "quantity":
                        float(
                            order.get(
                                "q",
                                0.0,
                            )
                        ),
                    "average_price":
                        float(
                            order.get(
                                "ap",
                                0.0,
                            )
                        ),
                }
            )

        elif (
            stream
            == "openInterest"
        ):
            open_interest.append(
                {
                    "time_ms": ts,
                    "open_interest":
                        float(
                            payload.get(
                                "openInterest",
                                np.nan,
                            )
                        ),
                }
            )

    return (
        trades,
        depth,
        book,
        mark,
        liquidations,
        open_interest,
    )


def time_index(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = frame.copy()

    frame["time"] = pd.to_datetime(
        frame["time_ms"],
        unit="ms",
        utc=True,
    )

    return (
        frame
        .sort_values("time")
        .set_index("time")
    )


def aggregate_trades(
    frame: pd.DataFrame,
    rule: str,
    prefix: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    frame = time_index(
        frame
    )

    buy = frame[
        frame["aggressive_side"]
        == "BUY"
    ]

    sell = frame[
        frame["aggressive_side"]
        == "SELL"
    ]

    result = pd.DataFrame(
        index=frame.resample(
            rule
        ).size().index
    )

    result[
        f"{prefix}_last_price"
    ] = (
        frame["price"]
        .resample(rule)
        .last()
    )

    result[
        f"{prefix}_trade_count"
    ] = (
        frame["price"]
        .resample(rule)
        .count()
    )

    result[
        f"{prefix}_buy_qty"
    ] = (
        buy["quantity"]
        .resample(rule)
        .sum()
    )

    result[
        f"{prefix}_sell_qty"
    ] = (
        sell["quantity"]
        .resample(rule)
        .sum()
    )

    result[
        f"{prefix}_buy_notional"
    ] = (
        buy["notional"]
        .resample(rule)
        .sum()
    )

    result[
        f"{prefix}_sell_notional"
    ] = (
        sell["notional"]
        .resample(rule)
        .sum()
    )

    denominator = (
        result[
            f"{prefix}_buy_qty"
        ]
        + result[
            f"{prefix}_sell_qty"
        ]
    )

    result[
        f"{prefix}_trade_flow_imbalance"
    ] = (
        (
            result[
                f"{prefix}_buy_qty"
            ]
            - result[
                f"{prefix}_sell_qty"
            ]
        )
        / denominator.replace(
            0,
            np.nan,
        )
    )

    return result


def aggregate_last(
    rows: list[dict],
    rule: str,
    prefix: str,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    frame = time_index(
        pd.DataFrame(
            rows
        )
    )

    numeric = (
        frame
        .select_dtypes(
            include=[
                np.number,
            ]
        )
        .drop(
            columns=[
                "time_ms",
            ],
            errors="ignore",
        )
    )

    result = (
        numeric.resample(
            rule
        )
        .last()
    )

    return result.add_prefix(
        f"{prefix}_"
    )


def aggregate_liquidations(
    rows: list[dict],
    rule: str,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    frame = time_index(
        pd.DataFrame(
            rows
        )
    )

    frame["notional"] = (
        np.where(
            frame["average_price"]
            > 0,
            frame["average_price"],
            frame["price"],
        )
        * frame["quantity"]
    )

    buy = frame[
        frame["side"]
        == "BUY"
    ]

    sell = frame[
        frame["side"]
        == "SELL"
    ]

    index = (
        frame.resample(
            rule
        )
        .size()
        .index
    )

    result = pd.DataFrame(
        index=index
    )

    result[
        "futures_liquidation_buy_notional"
    ] = (
        buy["notional"]
        .resample(rule)
        .sum()
    )

    result[
        "futures_liquidation_sell_notional"
    ] = (
        sell["notional"]
        .resample(rule)
        .sum()
    )

    return result


def build_features(
    date: str,
    rule: str,
) -> pd.DataFrame:
    spot_path = (
        RAW_DIR
        / f"spot_{date}.jsonl"
    )

    futures_path = (
        RAW_DIR
        / f"futures_{date}.jsonl"
    )

    rest_path = (
        RAW_DIR
        / (
            "futures_rest_"
            f"{date}.jsonl"
        )
    )

    (
        spot_trades,
        spot_depth,
        spot_book,
        _,
        _,
        _,
    ) = load_source(
        spot_path,
        "spot",
    )

    (
        futures_trades,
        futures_depth,
        futures_book,
        futures_mark,
        futures_liquidations,
        _,
    ) = load_source(
        futures_path,
        "futures",
    )

    (
        _,
        _,
        _,
        _,
        _,
        open_interest,
    ) = load_source(
        rest_path,
        "futures_rest",
    )

    frames = [
        aggregate_trades(
            pd.DataFrame(
                spot_trades
            ),
            rule,
            "spot",
        ),
        aggregate_trades(
            pd.DataFrame(
                futures_trades
            ),
            rule,
            "futures",
        ),
        aggregate_last(
            spot_depth,
            rule,
            "spot_depth",
        ),
        aggregate_last(
            futures_depth,
            rule,
            "futures_depth",
        ),
        aggregate_last(
            spot_book,
            rule,
            "spot_book",
        ),
        aggregate_last(
            futures_book,
            rule,
            "futures_book",
        ),
        aggregate_last(
            futures_mark,
            rule,
            "futures_mark",
        ),
        aggregate_last(
            open_interest,
            rule,
            "futures",
        ),
        aggregate_liquidations(
            futures_liquidations,
            rule,
        ),
    ]

    frames = [
        frame
        for frame in frames
        if not frame.empty
    ]

    if not frames:
        raise RuntimeError(
            "No raw microstructure events "
            "were available to aggregate."
        )

    result = pd.concat(
        frames,
        axis=1,
    ).sort_index()

    price_column = (
        "spot_last_price"
        if "spot_last_price"
        in result.columns
        else "futures_last_price"
    )

    if price_column in result.columns:
        result[
            "price_return_1_bucket"
        ] = (
            result[
                price_column
            ]
            .pct_change()
        )

        result[
            "price_velocity"
        ] = (
            result[
                price_column
            ]
            .diff()
        )

        result[
            "price_acceleration"
        ] = (
            result[
                "price_velocity"
            ]
            .diff()
        )

    for column in (
        "futures_open_interest",
        "futures_mark_funding_rate",
    ):
        if column in result.columns:
            result[
                f"{column}_change"
            ] = (
                result[column]
                .diff()
            )

    result = (
        result
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    result.index.name = (
        "bucket_time"
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        default=today_utc(),
    )

    parser.add_argument(
        "--bucket",
        choices=[
            "1s",
            "5s",
        ],
        default="1s",
    )

    args = parser.parse_args()

    print(
        "TRADEMIND - BUILD "
        "MICROSTRUCTURE FEATURES"
    )

    print(
        f"Date: {args.date} UTC"
    )

    print(
        f"Bucket: {args.bucket}"
    )

    features = build_features(
        args.date,
        args.bucket,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        OUTPUT_DIR
        / (
            "btcusdt_microstructure_"
            f"{args.bucket}_"
            f"{args.date}.csv"
        )
    )

    features.to_csv(
        output
    )

    print(
        f"Rows: {len(features):,}"
    )

    print(
        f"Columns: "
        f"{len(features.columns):,}"
    )

    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()
