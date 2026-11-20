from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import websockets


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "microstructure"
)

SYMBOL = "BTCUSDT"
STREAM_SYMBOL = SYMBOL.lower()

SPOT_WS = (
    "wss://stream.binance.com:9443/stream?streams="
    f"{STREAM_SYMBOL}@aggTrade/"
    f"{STREAM_SYMBOL}@bookTicker/"
    f"{STREAM_SYMBOL}@depth20@100ms"
)

FUTURES_WS = (
    "wss://fstream.binance.com/stream?streams="
    f"{STREAM_SYMBOL}@aggTrade/"
    f"{STREAM_SYMBOL}@bookTicker/"
    f"{STREAM_SYMBOL}@depth20@100ms/"
    f"{STREAM_SYMBOL}@markPrice@1s/"
    f"{STREAM_SYMBOL}@forceOrder"
)

OPEN_INTEREST_URL = (
    "https://fapi.binance.com/fapi/v1/openInterest"
)


def utc_now_ms() -> int:
    return int(
        time.time() * 1000
    )


def utc_day(
    timestamp_ms: int | None = None,
) -> str:
    if timestamp_ms is None:
        timestamp_ms = utc_now_ms()

    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d")


class RawEventWriter:
    def __init__(
        self,
        output_dir: Path,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._handles: dict[
            tuple[str, str],
            Any,
        ] = {}

    def _handle(
        self,
        source: str,
        day: str,
    ):
        key = (
            source,
            day,
        )

        if key not in self._handles:
            path = (
                self.output_dir
                / f"{source}_{day}.jsonl"
            )

            self._handles[key] = open(
                path,
                "a",
                encoding="utf-8",
                buffering=1,
            )

        return self._handles[key]

    def write(
        self,
        source: str,
        stream: str,
        payload: Any,
    ) -> None:
        received_ms = utc_now_ms()

        record = {
            "received_time_ms":
                received_ms,
            "source":
                source,
            "stream":
                stream,
            "payload":
                payload,
        }

        handle = self._handle(
            source,
            utc_day(
                received_ms
            ),
        )

        handle.write(
            json.dumps(
                record,
                separators=(
                    ",",
                    ":",
                ),
            )
            + "\n"
        )

    def close(self) -> None:
        for handle in (
            self._handles.values()
        ):
            handle.close()

        self._handles.clear()


async def websocket_collector(
    name: str,
    url: str,
    writer: RawEventWriter,
    stop_event: asyncio.Event,
) -> None:
    backoff = 1

    while not stop_event.is_set():
        try:
            print(
                f"[{name}] connecting..."
            )

            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_queue=10000,
            ) as socket:
                print(
                    f"[{name}] connected"
                )

                backoff = 1

                async for message in socket:
                    if stop_event.is_set():
                        break

                    envelope = json.loads(
                        message
                    )

                    writer.write(
                        source=name,
                        stream=str(
                            envelope.get(
                                "stream",
                                "unknown",
                            )
                        ),
                        payload=envelope.get(
                            "data",
                            envelope,
                        ),
                    )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            print(
                f"[{name}] disconnected: "
                f"{exc}"
            )

            if stop_event.is_set():
                break

            await asyncio.sleep(
                backoff
            )

            backoff = min(
                backoff * 2,
                30,
            )


def fetch_open_interest() -> dict:
    response = requests.get(
        OPEN_INTEREST_URL,
        params={
            "symbol": SYMBOL,
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


async def open_interest_collector(
    writer: RawEventWriter,
    stop_event: asyncio.Event,
    poll_seconds: float,
) -> None:
    while not stop_event.is_set():
        started = time.monotonic()

        try:
            payload = await asyncio.to_thread(
                fetch_open_interest
            )

            writer.write(
                source="futures_rest",
                stream="openInterest",
                payload=payload,
            )

        except Exception as exc:
            print(
                "[futures_rest] "
                "open interest error: "
                f"{exc}"
            )

        elapsed = (
            time.monotonic()
            - started
        )

        wait_for = max(
            0.1,
            poll_seconds
            - elapsed,
        )

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=wait_for,
            )
        except asyncio.TimeoutError:
            pass


async def status_loop(
    writer: RawEventWriter,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(30)

        day = utc_day()

        spot = (
            RAW_DIR
            / f"spot_{day}.jsonl"
        )

        futures = (
            RAW_DIR
            / f"futures_{day}.jsonl"
        )

        futures_rest = (
            RAW_DIR
            / (
                "futures_rest_"
                f"{day}.jsonl"
            )
        )

        def size_mb(
            path: Path,
        ) -> float:
            if not path.exists():
                return 0.0

            return (
                path.stat().st_size
                / 1024
                / 1024
            )

        print(
            "[status] raw MB "
            f"spot={size_mb(spot):.1f} "
            f"futures={size_mb(futures):.1f} "
            f"rest={size_mb(futures_rest):.3f}"
        )


async def main_async(
    poll_seconds: float,
) -> None:
    print(
        "TRADEMIND - BINANCE RAW "
        "MICROSTRUCTURE COLLECTOR"
    )

    print(
        "Collecting raw exchange events. "
        "No ML features are created here."
    )

    print(
        f"Output: {RAW_DIR}"
    )

    writer = RawEventWriter(
        RAW_DIR
    )

    stop_event = asyncio.Event()

    loop = (
        asyncio.get_running_loop()
    )

    def request_stop() -> None:
        if not stop_event.is_set():
            print(
                "\nStopping collector..."
            )
            stop_event.set()

    for sig in (
        signal.SIGINT,
        signal.SIGTERM,
    ):
        try:
            loop.add_signal_handler(
                sig,
                request_stop,
            )
        except (
            NotImplementedError,
            RuntimeError,
        ):
            pass

    tasks = [
        asyncio.create_task(
            websocket_collector(
                "spot",
                SPOT_WS,
                writer,
                stop_event,
            )
        ),
        asyncio.create_task(
            websocket_collector(
                "futures",
                FUTURES_WS,
                writer,
                stop_event,
            )
        ),
        asyncio.create_task(
            open_interest_collector(
                writer,
                stop_event,
                poll_seconds,
            )
        ),
        asyncio.create_task(
            status_loop(
                writer,
                stop_event,
            )
        ),
    ]

    try:
        await stop_event.wait()

    except KeyboardInterrupt:
        request_stop()

    finally:
        for task in tasks:
            task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        writer.close()

        print(
            "Collector stopped cleanly."
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--open-interest-seconds",
        type=float,
        default=10.0,
        help=(
            "Seconds between Binance "
            "Futures open-interest REST "
            "snapshots."
        ),
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            main_async(
                args.open_interest_seconds
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
