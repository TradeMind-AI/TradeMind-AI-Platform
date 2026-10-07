from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import websockets


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "live_microstructure"
)

SPOT_WS = (
    "wss://stream.binance.com:9443/stream?"
    "streams="
    "btcusdt@aggTrade/"
    "btcusdt@bookTicker/"
    "btcusdt@depth20@100ms"
)

FUTURES_WS = (
    "wss://fstream.binance.com/stream?"
    "streams="
    "btcusdt@aggTrade/"
    "btcusdt@bookTicker/"
    "btcusdt@depth20@100ms/"
    "btcusdt@markPrice@1s/"
    "btcusdt@forceOrder"
)

FUTURES_OI_URL = (
    "https://fapi.binance.com/fapi/v1/openInterest"
)

SYMBOL = "BTCUSDT"

HEADERS: dict[str, list[str]] = {
    "spot_agg_trade": [
        "received_time",
        "event_time",
        "trade_time",
        "aggregate_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "buyer_is_maker",
    ],
    "spot_book_ticker": [
        "received_time",
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
    ],
    "spot_depth20": [
        "received_time",
        "event_time",
        "last_update_id",
        "bids_json",
        "asks_json",
    ],
    "futures_agg_trade": [
        "received_time",
        "event_time",
        "trade_time",
        "aggregate_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "buyer_is_maker",
    ],
    "futures_book_ticker": [
        "received_time",
        "event_time",
        "transaction_time",
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
    ],
    "futures_depth20": [
        "received_time",
        "event_time",
        "transaction_time",
        "first_update_id",
        "final_update_id",
        "previous_final_update_id",
        "bids_json",
        "asks_json",
    ],
    "futures_mark_price": [
        "received_time",
        "event_time",
        "mark_price",
        "index_price",
        "estimated_settle_price",
        "funding_rate",
        "next_funding_time",
    ],
    "futures_liquidation": [
        "received_time",
        "event_time",
        "symbol",
        "side",
        "order_type",
        "time_in_force",
        "original_quantity",
        "price",
        "average_price",
        "order_status",
        "last_filled_quantity",
        "filled_accumulated_quantity",
        "trade_time",
    ],
    "futures_open_interest": [
        "received_time",
        "symbol",
        "open_interest",
        "server_time",
    ],
}


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
    )


def output_file(
    stream_name: str,
) -> Path:
    date = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")

    return (
        OUTPUT_DIR
        / f"{stream_name}_{date}.csv"
    )


def append_row(
    stream_name: str,
    row: dict[str, Any],
) -> None:
    path = output_file(
        stream_name
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=HEADERS[
                stream_name
            ],
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            {
                key: row.get(
                    key,
                    "",
                )
                for key in HEADERS[
                    stream_name
                ]
            }
        )


def handle_spot(
    stream: str,
    data: dict[str, Any],
) -> None:
    received = utc_now_iso()

    if stream.endswith(
        "@aggTrade"
    ):
        append_row(
            "spot_agg_trade",
            {
                "received_time":
                    received,
                "event_time":
                    data.get("E"),
                "trade_time":
                    data.get("T"),
                "aggregate_trade_id":
                    data.get("a"),
                "price":
                    data.get("p"),
                "quantity":
                    data.get("q"),
                "first_trade_id":
                    data.get("f"),
                "last_trade_id":
                    data.get("l"),
                "buyer_is_maker":
                    data.get("m"),
            },
        )
        return

    if stream.endswith(
        "@bookTicker"
    ):
        append_row(
            "spot_book_ticker",
            {
                "received_time":
                    received,
                "update_id":
                    data.get("u"),
                "best_bid_price":
                    data.get("b"),
                "best_bid_qty":
                    data.get("B"),
                "best_ask_price":
                    data.get("a"),
                "best_ask_qty":
                    data.get("A"),
            },
        )
        return

    if "@depth20" in stream:
        append_row(
            "spot_depth20",
            {
                "received_time":
                    received,
                "event_time":
                    data.get("E"),
                "last_update_id":
                    data.get(
                        "lastUpdateId",
                        data.get("u"),
                    ),
                "bids_json":
                    json.dumps(
                        data.get("bids",
                            data.get("b", []))
                    ),
                "asks_json":
                    json.dumps(
                        data.get("asks",
                            data.get("a", []))
                    ),
            },
        )


def handle_futures(
    stream: str,
    data: dict[str, Any],
) -> None:
    received = utc_now_iso()

    if stream.endswith(
        "@aggTrade"
    ):
        append_row(
            "futures_agg_trade",
            {
                "received_time":
                    received,
                "event_time":
                    data.get("E"),
                "trade_time":
                    data.get("T"),
                "aggregate_trade_id":
                    data.get("a"),
                "price":
                    data.get("p"),
                "quantity":
                    data.get("q"),
                "first_trade_id":
                    data.get("f"),
                "last_trade_id":
                    data.get("l"),
                "buyer_is_maker":
                    data.get("m"),
            },
        )
        return

    if stream.endswith(
        "@bookTicker"
    ):
        append_row(
            "futures_book_ticker",
            {
                "received_time":
                    received,
                "event_time":
                    data.get("E"),
                "transaction_time":
                    data.get("T"),
                "update_id":
                    data.get("u"),
                "best_bid_price":
                    data.get("b"),
                "best_bid_qty":
                    data.get("B"),
                "best_ask_price":
                    data.get("a"),
                "best_ask_qty":
                    data.get("A"),
            },
        )
        return

    if "@depth20" in stream:
        append_row(
            "futures_depth20",
            {
                "received_time":
                    received,
                "event_time":
                    data.get("E"),
                "transaction_time":
                    data.get("T"),
                "first_update_id":
                    data.get("U"),
                "final_update_id":
                    data.get("u"),
                "previous_final_update_id":
                    data.get("pu"),
                "bids_json":
                    json.dumps(
                        data.get("b", [])
                    ),
                "asks_json":
                    json.dumps(
                        data.get("a", [])
                    ),
            },
        )
        return

    if "@markPrice" in stream:
        append_row(
            "futures_mark_price",
            {
                "received_time":
                    received,
                "event_time":
                    data.get("E"),
                "mark_price":
                    data.get("p"),
                "index_price":
                    data.get("i"),
                "estimated_settle_price":
                    data.get("P"),
                "funding_rate":
                    data.get("r"),
                "next_funding_time":
                    data.get("T"),
            },
        )
        return

    if stream.endswith(
        "@forceOrder"
    ):
        order = data.get(
            "o",
            {},
        )

        append_row(
            "futures_liquidation",
            {
                "received_time":
                    received,
                "event_time":
                    data.get("E"),
                "symbol":
                    order.get("s"),
                "side":
                    order.get("S"),
                "order_type":
                    order.get("o"),
                "time_in_force":
                    order.get("f"),
                "original_quantity":
                    order.get("q"),
                "price":
                    order.get("p"),
                "average_price":
                    order.get("ap"),
                "order_status":
                    order.get("X"),
                "last_filled_quantity":
                    order.get("l"),
                "filled_accumulated_quantity":
                    order.get("z"),
                "trade_time":
                    order.get("T"),
            },
        )


async def websocket_loop(
    name: str,
    url: str,
    handler,
) -> None:
    while True:
        try:
            print(
                f"Connecting {name}..."
            )

            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_queue=10000,
            ) as socket:
                print(
                    f"{name} connected."
                )

                async for message in socket:
                    payload = json.loads(
                        message
                    )

                    stream = str(
                        payload.get(
                            "stream",
                            "",
                        )
                    )

                    data = payload.get(
                        "data",
                        {},
                    )

                    handler(
                        stream,
                        data,
                    )
        except (
            asyncio.CancelledError
        ):
            raise
        except Exception as exc:
            print(
                f"{name} error: {exc}. "
                "Reconnecting in 5s..."
            )

            await asyncio.sleep(
                5
            )


async def open_interest_loop() -> None:
    session = requests.Session()

    while True:
        try:
            response = await asyncio.to_thread(
                session.get,
                FUTURES_OI_URL,
                params={
                    "symbol":
                        SYMBOL,
                },
                timeout=10,
            )

            response.raise_for_status()

            data = response.json()

            append_row(
                "futures_open_interest",
                {
                    "received_time":
                        utc_now_iso(),
                    "symbol":
                        data.get(
                            "symbol",
                            SYMBOL,
                        ),
                    "open_interest":
                        data.get(
                            "openInterest"
                        ),
                    "server_time":
                        data.get("time"),
                },
            )
        except (
            asyncio.CancelledError
        ):
            raise
        except Exception as exc:
            print(
                "Open-interest error: "
                f"{exc}"
            )

        await asyncio.sleep(
            5
        )


async def main() -> None:
    print(
        "TRADEMIND RAW MICROSTRUCTURE "
        "COLLECTOR"
    )

    print(
        "Spot + USDT-M Futures BTCUSDT"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        "Press Ctrl+C to stop cleanly."
    )

    await asyncio.gather(
        websocket_loop(
            "spot",
            SPOT_WS,
            handle_spot,
        ),
        websocket_loop(
            "futures",
            FUTURES_WS,
            handle_futures,
        ),
        open_interest_loop(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(
            main()
        )
    except KeyboardInterrupt:
        print(
            "\nCollector stopped."
        )
