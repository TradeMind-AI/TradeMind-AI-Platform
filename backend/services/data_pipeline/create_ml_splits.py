from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_features_enriched_v3.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
)


# Split ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def load_dataset() -> pd.DataFrame:
    """Load the ML-ready feature dataset."""

    df = pd.read_csv(INPUT_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        utc=True,
    )

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Run basic checks before splitting."""

    print("\nRunning pre-split validation...")

    # Row count
    print(f"Rows: {len(df)}")

    # Timestamp ordering
    if not df["open_time"].is_monotonic_increasing:
        raise ValueError(
            "Dataset timestamps are not sorted."
        )

    print("Timestamp ordering: OK")

    # Duplicate timestamps
    duplicate_count = (
        df["open_time"].duplicated().sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate timestamps."
        )

    print("Duplicate timestamps: 0")

    # NaN check
    nan_count = df.isna().sum().sum()

    if nan_count != 0:
        raise ValueError(
            f"Dataset contains {nan_count} NaN values."
        )

    print("NaN values: 0")

    # Infinite values
    numeric_columns = df.select_dtypes(
        include="number"
    )

    infinite_count = (
        numeric_columns
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    if infinite_count != 0:
        raise ValueError(
            f"Found {infinite_count} infinite values."
        )

    print("Infinite values: 0")

    # Target existence
    required_columns = {
        "future_return_1h",
        "target",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing target columns: {missing_columns}"
        )

    print("Target columns: OK")


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset chronologically."""

    total_rows = len(df)

    train_end = int(
        total_rows * TRAIN_RATIO
    )

    validation_end = train_end + int(
        total_rows * VALIDATION_RATIO
    )

    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    return (
        train_df,
        validation_df,
        test_df,
    )


def validate_split_boundaries(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Verify that the chronological split has no overlap."""

    print("\nChecking split boundaries...")

    if (
        train_df["open_time"].max()
        >= validation_df["open_time"].min()
    ):
        raise ValueError(
            "Train and validation periods overlap."
        )

    if (
        validation_df["open_time"].max()
        >= test_df["open_time"].min()
    ):
        raise ValueError(
            "Validation and test periods overlap."
        )

    print("Chronological ordering: OK")
    print("No temporal overlap: OK")


def print_split_summary(
    name: str,
    df: pd.DataFrame,
) -> None:
    """Print information about a dataset split."""

    print(f"\n{name}")
    print("-" * len(name))

    print(f"Rows: {len(df)}")

    print(
        f"Start: {df['open_time'].min()}"
    )

    print(
        f"End:   {df['open_time'].max()}"
    )

    print("\nTarget distribution:")

    print(
        df["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTarget percentages:")

    percentages = (
        df["target"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print(
        percentages.to_string()
    )


def save_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Save train, validation, and test datasets."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_file = (
        OUTPUT_DIR / "train.csv"
    )

    validation_file = (
        OUTPUT_DIR / "validation.csv"
    )

    test_file = (
        OUTPUT_DIR / "test.csv"
    )

    train_df.to_csv(
        train_file,
        index=False,
    )

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    test_df.to_csv(
        test_file,
        index=False,
    )

    print("\nFiles saved:")

    print(train_file)
    print(validation_file)
    print(test_file)


def main() -> None:
    """Create chronological ML train/validation/test splits."""

    print("Loading ML-ready dataset...")

    df = load_dataset()

    validate_dataset(df)

    train_df, validation_df, test_df = (
        split_dataset(df)
    )

    validate_split_boundaries(
        train_df,
        validation_df,
        test_df,
    )

    print_split_summary(
        "TRAIN",
        train_df,
    )

    print_split_summary(
        "VALIDATION",
        validation_df,
    )

    print_split_summary(
        "TEST",
        test_df,
    )

    save_splits(
        train_df,
        validation_df,
        test_df,
    )


if __name__ == "__main__":
    main()