from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from backend.services.data_pipeline.feature_engineering_1m import (
    create_candle_features,
    create_lag_features,
    create_momentum_features,
    create_return_features,
    create_target,
    create_time_features,
    create_trend_features,
    create_volatility_features,
    create_volume_features,
    load_data,
)


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1m_features_breakout.csv"
)


def create_breakout_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    features = {}

    for window in [
        10,
        20,
        60,
    ]:
        prior_high = (
            df["high"]
            .rolling(window)
            .max()
            .shift(1)
        )

        prior_low = (
            df["low"]
            .rolling(window)
            .min()
            .shift(1)
        )

        rolling_range = (
            prior_high
            - prior_low
        )

        features[
            f"distance_to_{window}m_high"
        ] = (
            df["close"]
            / prior_high
            - 1
        )

        features[
            f"distance_to_{window}m_low"
        ] = (
            df["close"]
            / prior_low
            - 1
        )

        features[
            f"breakout_{window}m"
        ] = (
            df["close"]
            > prior_high
        ).astype(int)

        features[
            f"breakdown_{window}m"
        ] = (
            df["close"]
            < prior_low
        ).astype(int)

        features[
            f"range_position_{window}m"
        ] = np.where(
            rolling_range > 0,
            (
                df["close"]
                - prior_low
            )
            / rolling_range,
            0.5,
        )

    breakout_df = pd.DataFrame(
        features,
        index=df.index,
    )

    return pd.concat(
        [
            df,
            breakout_df,
        ],
        axis=1,
    )


def main() -> None:
    print(
        "Loading BTCUSDT 1-minute data "
        "for breakout ablation..."
    )

    df = load_data()

    print(
        f"Input rows: {len(df):,}"
    )

    # Exact V1 baseline feature set.
    df = create_return_features(df)
    df = create_trend_features(df)
    df = create_momentum_features(df)
    df = create_volatility_features(df)
    df = create_candle_features(df)
    df = create_volume_features(df)
    df = create_time_features(df)
    df = create_lag_features(df)

    # Add only support/resistance + breakout features.
    df = create_breakout_features(df)

    df = create_target(df)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    before = len(df)

    df = df.dropna().copy()

    excluded = {
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
        "future_return_1m",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    print(
        f"Rows removed for warm-up/NaN: "
        f"{before - len(df):,}"
    )

    print(
        f"Final ML-ready rows: "
        f"{len(df):,}"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"Breakout features added: "
        f"{len(feature_columns) - 81}"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
