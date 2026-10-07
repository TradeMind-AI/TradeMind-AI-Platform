from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "microstructure"
)


def today_utc() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")


def validate_file(
    path: Path,
) -> dict:
    lines = 0
    bad_json = 0
    streams = Counter()
    first_ms = None
    last_ms = None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            lines += 1

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError:
                bad_json += 1
                continue

            stream = str(
                record.get(
                    "stream",
                    "missing",
                )
            )

            streams[stream] += 1

            received = record.get(
                "received_time_ms"
            )

            if isinstance(
                received,
                int,
            ):
                if first_ms is None:
                    first_ms = received

                last_ms = received

    return {
        "file": path.name,
        "lines": lines,
        "bad_json": bad_json,
        "first_ms": first_ms,
        "last_ms": last_ms,
        "streams": streams,
    }


def format_time(
    timestamp_ms: int | None,
) -> str:
    if timestamp_ms is None:
        return "-"

    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    ).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        default=today_utc(),
        help=(
            "UTC date in YYYY-MM-DD "
            "format."
        ),
    )

    args = parser.parse_args()

    print(
        "TRADEMIND - RAW "
        "MICROSTRUCTURE VALIDATOR"
    )

    files = [
        RAW_DIR
        / f"spot_{args.date}.jsonl",
        RAW_DIR
        / f"futures_{args.date}.jsonl",
        RAW_DIR
        / (
            "futures_rest_"
            f"{args.date}.jsonl"
        ),
    ]

    found = False

    for path in files:
        if not path.exists():
            print()
            print(
                f"MISSING: {path}"
            )
            continue

        found = True

        result = validate_file(
            path
        )

        print()
        print(
            f"FILE: {result['file']}"
        )

        print(
            f"  rows: "
            f"{result['lines']:,}"
        )

        print(
            f"  malformed JSON: "
            f"{result['bad_json']:,}"
        )

        print(
            f"  first receive: "
            f"{format_time(result['first_ms'])}"
        )

        print(
            f"  last receive: "
            f"{format_time(result['last_ms'])}"
        )

        print(
            "  streams:"
        )

        for stream, count in (
            result["streams"]
            .most_common()
        ):
            print(
                f"    {stream}: "
                f"{count:,}"
            )

    if not found:
        raise SystemExit(
            "No raw microstructure files "
            "found for that UTC date."
        )


if __name__ == "__main__":
    main()
