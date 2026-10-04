from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RESEARCH_START = "2023-03-26"
DEFAULT_VALIDATION_START = "2025-01-01"
DEFAULT_KNOWN_RECENT_START = "2025-10-01"


def validate_frame(
    df: pd.DataFrame,
) -> None:
    if not df[
        "open_time"
    ].is_monotonic_increasing:
        raise ValueError(
            "open_time is not sorted."
        )

    if df[
        "open_time"
    ].duplicated().any():
        raise ValueError(
            "Duplicate open_time values found."
        )

    if df.isna().any().any():
        raise ValueError(
            "NaN values found."
        )

    numeric = df.select_dtypes(
        include=[np.number]
    )

    if np.isinf(
        numeric.to_numpy()
    ).any():
        raise ValueError(
            "Infinite values found."
        )


def describe_split(
    name: str,
    frame: pd.DataFrame,
) -> None:
    print()
    print(name)
    print(
        f"Rows:  {len(frame):,}"
    )

    if frame.empty:
        print("EMPTY")
        return

    print(
        f"Start: "
        f"{frame['open_time'].iloc[0]}"
    )

    print(
        f"End:   "
        f"{frame['open_time'].iloc[-1]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create date-based extended-history "
            "research splits without pretending the "
            "already-seen recent period is a fresh test."
        )
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        choices=[
            "2m",
            "10m",
        ],
    )

    parser.add_argument(
        "--research-start",
        default=DEFAULT_RESEARCH_START,
    )

    parser.add_argument(
        "--validation-start",
        default=DEFAULT_VALIDATION_START,
    )

    parser.add_argument(
        "--known-recent-start",
        default=DEFAULT_KNOWN_RECENT_START,
    )

    args = parser.parse_args()

    input_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"btcusdt_{args.timeframe}_features.csv"
    )

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"extended_research_splits_{args.timeframe}"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: "
            f"{input_file}"
        )

    print(
        f"Loading {args.timeframe} "
        f"extended feature dataset..."
    )

    df = pd.read_csv(
        input_file
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    validate_frame(df)

    research_start = pd.Timestamp(
        args.research_start,
        tz="UTC",
    )

    validation_start = pd.Timestamp(
        args.validation_start,
        tz="UTC",
    )

    known_recent_start = pd.Timestamp(
        args.known_recent_start,
        tz="UTC",
    )

    if not (
        research_start
        < validation_start
        < known_recent_start
    ):
        raise ValueError(
            "Expected research_start < "
            "validation_start < "
            "known_recent_start."
        )

    # Start after the March 24, 2023 spot outage
    # plus enough warm-up so rolling features do not
    # bridge the preserved exchange-data gap.
    research_df = df[
        df["open_time"]
        >= research_start
    ].copy()

    train = research_df[
        (
            research_df["open_time"]
            < validation_start
        )
    ].copy()

    validation = research_df[
        (
            research_df["open_time"]
            >= validation_start
        )
        & (
            research_df["open_time"]
            < known_recent_start
        )
    ].copy()

    known_recent = research_df[
        research_df["open_time"]
        >= known_recent_start
    ].copy()

    if train.empty:
        raise RuntimeError(
            "Training split is empty."
        )

    if validation.empty:
        raise RuntimeError(
            "Validation split is empty."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        output_dir / "train.csv",
        index=False,
    )

    validation.to_csv(
        output_dir / "validation.csv",
        index=False,
    )

    known_recent.to_csv(
        output_dir / "known_recent.csv",
        index=False,
    )

    print(
        f"Total source rows: "
        f"{len(df):,}"
    )

    describe_split(
        "TRAIN",
        train,
    )

    describe_split(
        "VALIDATION",
        validation,
    )

    describe_split(
        "KNOWN_RECENT",
        known_recent,
    )

    print()
    print(
        "KNOWN_RECENT is NOT a fresh test set. "
        "Dates from 2025-10-01 onward were already "
        "examined during earlier research."
    )

    print()
    print(
        "A genuinely fresh final test must come "
        "from new future candles collected after "
        "the current research cutoff."
    )

    print()
    print("Saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
