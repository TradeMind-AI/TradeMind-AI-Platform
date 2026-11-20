from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1m_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_1m"
)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def main() -> None:
    print(
        "Loading 1-minute feature dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    if (
        not df["open_time"]
        .is_monotonic_increasing
    ):
        raise ValueError(
            "Timestamps are not chronological."
        )

    duplicate_count = (
        df["open_time"]
        .duplicated()
        .sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} "
            "duplicate timestamps."
        )

    nan_count = int(
        df.isna().sum().sum()
    )

    if nan_count:
        raise ValueError(
            f"Found {nan_count} NaN values."
        )

    numeric = df.select_dtypes(
        include="number"
    )

    if np.isinf(
        numeric.to_numpy()
    ).any():
        raise ValueError(
            "Infinite numeric values found."
        )

    if (
        "future_return_1m"
        not in df.columns
    ):
        raise ValueError(
            "future_return_1m is missing."
        )

    total_rows = len(df)

    train_end = int(
        total_rows
        * TRAIN_RATIO
    )

    validation_end = (
        train_end
        + int(
            total_rows
            * VALIDATION_RATIO
        )
    )

    train_df = (
        df.iloc[
            :train_end
        ]
        .copy()
    )

    validation_df = (
        df.iloc[
            train_end:
            validation_end
        ]
        .copy()
    )

    test_df = (
        df.iloc[
            validation_end:
        ]
        .copy()
    )

    if (
        train_df[
            "open_time"
        ].max()
        >= validation_df[
            "open_time"
        ].min()
    ):
        raise ValueError(
            "Train/validation overlap."
        )

    if (
        validation_df[
            "open_time"
        ].max()
        >= test_df[
            "open_time"
        ].min()
    ):
        raise ValueError(
            "Validation/test overlap."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = {
        "train.csv":
            train_df,
        "validation.csv":
            validation_df,
        "test.csv":
            test_df,
    }

    for filename, frame in (
        splits.items()
    ):
        frame.to_csv(
            OUTPUT_DIR
            / filename,
            index=False,
        )

    print(
        f"Total rows:      "
        f"{total_rows:,}"
    )

    for name, frame in [
        ("TRAIN", train_df),
        (
            "VALIDATION",
            validation_df,
        ),
        ("TEST", test_df),
    ]:
        print(
            f"\n{name}"
        )

        print(
            f"Rows:  "
            f"{len(frame):,}"
        )

        print(
            f"Start: "
            f"{frame['open_time'].iloc[0]}"
        )

        print(
            f"End:   "
            f"{frame['open_time'].iloc[-1]}"
        )

    print(
        f"\nSplits saved to:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        "\nIMPORTANT: do not use "
        "test.csv during tuning."
    )


if __name__ == "__main__":
    main()
