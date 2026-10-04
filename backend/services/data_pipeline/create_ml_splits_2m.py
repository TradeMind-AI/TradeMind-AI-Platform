from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_2m_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_2m"
)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def main() -> None:
    print(
        "Loading 2-minute feature dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

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

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:
        validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    print(
        f"Total rows:      "
        f"{total_rows:,}"
    )

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        print()
        print(name)
        print(
            f"Rows:  {len(frame):,}"
        )
        print(
            f"Start: "
            f"{frame['open_time'].iloc[0]}"
        )
        print(
            f"End:   "
            f"{frame['open_time'].iloc[-1]}"
        )

    print()
    print("Splits saved to:")
    print(OUTPUT_DIR)

    print()
    print(
        "IMPORTANT: do not use "
        "2m test.csv during tuning."
    )


if __name__ == "__main__":
    main()
