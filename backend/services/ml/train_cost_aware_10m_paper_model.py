from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "locked_xgboost_10m_bundle.joblib"
)

HORIZON_BARS = 6
HORIZON_MINUTES = 60
TARGET_THRESHOLD = 0.0040
CONFIDENCE_THRESHOLD = 0.55

NON_FEATURE_COLUMNS = {
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
    "future_return_10m",
    "target",
}


def make_target(
    returns: pd.Series,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                returns < -TARGET_THRESHOLD,
                returns > TARGET_THRESHOLD,
            ],
            [
                "DOWN",
                "UP",
            ],
            default="NEUTRAL",
        ),
        index=returns.index,
    )


def main() -> None:
    print(
        "Training frozen COST_AWARE_10M_V1 "
        "for forward paper collection..."
    )

    df = pd.read_csv(
        FEATURE_FILE
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
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

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    future_return = (
        df["close"].shift(-HORIZON_BARS)
        / df["close"]
        - 1.0
    )

    train_mask = future_return.notna()

    X = df.loc[
        train_mask,
        feature_columns,
    ]

    y_text = make_target(
        future_return.loc[
            train_mask
        ]
    )

    encoder = LabelEncoder()

    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    y = encoder.transform(
        y_text
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X,
        y,
    )

    bundle = {
        "model":
            model,
        "encoder":
            encoder,
        "feature_columns":
            feature_columns,
        "confidence_threshold":
            CONFIDENCE_THRESHOLD,
        "target_threshold":
            TARGET_THRESHOLD,
        "horizon_bars":
            HORIZON_BARS,
        "horizon_minutes":
            HORIZON_MINUTES,
        "timeframe":
            "10m",
        "model_name":
            "cost_aware_10m_v1",
        "paper_mode":
            "forward_fixed_horizon",
        "training_start":
            str(
                df.loc[
                    train_mask,
                    "open_time",
                ].iloc[0]
            ),
        "training_end":
            str(
                df.loc[
                    train_mask,
                    "open_time",
                ].iloc[-1]
            ),
        "execution_costs":
            {
                "low": 0.0006,
                "mid": 0.0014,
                "high": 0.0027,
            },
        "research_status":
            (
                "Frozen research candidate. "
                "Forward paper evidence required."
            ),
    }

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        MODEL_FILE,
    )

    print(
        f"Rows used: "
        f"{int(train_mask.sum()):,}"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"Training start: "
        f"{bundle['training_start']}"
    )

    print(
        f"Training end:   "
        f"{bundle['training_end']}"
    )

    print(
        "Frozen settings: "
        "10m / 60m horizon / ±0.40% / "
        "confidence 0.55"
    )

    print()
    print("Saved model bundle:")
    print(MODEL_FILE)


if __name__ == "__main__":
    main()
