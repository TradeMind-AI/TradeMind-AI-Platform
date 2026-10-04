from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_validation_predictions.csv"
)


# --------------------------------------------------
# Load validation predictions
# --------------------------------------------------

print("Loading validation predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# --------------------------------------------------
# Confidence analysis function
# --------------------------------------------------

def analyze_confidence(
    data,
    probability_column,
    direction,
):
    """
    Analyze actual future returns for predictions
    grouped by model confidence.
    """

    analysis_df = data.copy()

    # Create confidence buckets.
    analysis_df["confidence_bucket"] = pd.cut(
        analysis_df[probability_column],
        bins=[
            0.0,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            1.00,
        ],
        include_lowest=True,
    )

    grouped = (
        analysis_df
        .groupby(
            "confidence_bucket",
            observed=False,
        )
        .agg(
            samples=(
                "future_return_1h",
                "count",
            ),
            average_return=(
                "future_return_1h",
                "mean",
            ),
            median_return=(
                "future_return_1h",
                "median",
            ),
            positive_returns=(
                "future_return_1h",
                lambda x: (
                    x > 0
                ).mean(),
            ),
        )
        .reset_index()
    )

    grouped["positive_return_pct"] = (
        grouped["positive_returns"] * 100
    )

    grouped["average_return_pct"] = (
        grouped["average_return"] * 100
    )

    grouped["median_return_pct"] = (
        grouped["median_return"] * 100
    )

    grouped = grouped.drop(
        columns=["positive_returns"]
    )

    print("\n" + "=" * 80)
    print(
        f"{direction} CONFIDENCE VS ACTUAL RETURN"
    )
    print("=" * 80)

    print(
        grouped.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return grouped


# --------------------------------------------------
# UP confidence
# --------------------------------------------------

up_analysis = analyze_confidence(
    df,
    "probability_up",
    "UP",
)


# --------------------------------------------------
# DOWN confidence
# --------------------------------------------------

down_analysis = analyze_confidence(
    df,
    "probability_down",
    "DOWN",
)


# --------------------------------------------------
# Model confidence
# --------------------------------------------------

df["maximum_probability"] = (
    df[
        [
            "probability_down",
            "probability_neutral",
            "probability_up",
        ]
    ].max(axis=1)
)

df["predicted_confidence"] = (
    df["maximum_probability"]
)


confidence_summary = (
    df.groupby(
        pd.cut(
            df["predicted_confidence"],
            bins=[
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                1.00,
            ],
            include_lowest=True,
        ),
        observed=False,
    )
    .agg(
        samples=(
            "future_return_1h",
            "count",
        ),
        average_return=(
            "future_return_1h",
            "mean",
        ),
    )
    .reset_index()
)


confidence_summary[
    "average_return_pct"
] = (
    confidence_summary["average_return"]
    * 100
)


print("\n" + "=" * 80)
print("OVERALL MODEL CONFIDENCE VS ACTUAL RETURN")
print("=" * 80)

print(
    confidence_summary[
        [
            "predicted_confidence",
            "samples",
            "average_return_pct",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\nAnalysis complete.")