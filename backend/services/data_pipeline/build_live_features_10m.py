from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PIPELINE_DIR = (
    PROJECT_ROOT
    / "backend"
    / "services"
    / "data_pipeline"
)

if str(
    DATA_PIPELINE_DIR
) not in sys.path:
    sys.path.insert(
        0,
        str(DATA_PIPELINE_DIR),
    )

from feature_engineering_10m import create_features


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m_live_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "locked_xgboost_10m_bundle.joblib"
)


def main() -> None:
    print(
        "Building live 10m features..."
    )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"10m model bundle not found: "
            f"{MODEL_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        utc=True,
    )

    df = (
        df.sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    expected_latest = (
        df["open_time"].iloc[-1]
    )

    df = create_features(
        df
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    bundle = joblib.load(
        MODEL_FILE
    )

    feature_columns = (
        bundle["feature_columns"]
    )

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Live 10m features are "
            "missing model columns: "
            + ", ".join(missing)
        )

    df = df.dropna(
        subset=feature_columns
    ).copy()

    if df.empty:
        raise RuntimeError(
            "Live 10m feature dataset "
            "is empty."
        )

    actual_latest = (
        df["open_time"].iloc[-1]
    )

    if actual_latest != expected_latest:
        raise RuntimeError(
            "Latest 10m candle was lost "
            "during feature construction."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"Latest ready candle: "
        f"{actual_latest}"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
