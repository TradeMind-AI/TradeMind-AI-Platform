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
    / "btcusdt_1m_features_vwap.csv"
)


def create_vwap_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    price_volume = (
        typical_price
        * df["volume"]
    )

    features = {}

    for window in [
        20,
        60,
        240,
    ]:
        volume_sum = (
            df["volume"]
            .rolling(window)
            .sum()
        )

        price_volume_sum = (
            price_volume
            .rolling(window)
            .sum()
        )

        vwap = (
            price_volume_sum
            / volume_sum
        )

        features[
            f"vwap_{window}"
        ] = vwap

        features[
            f"distance_to_vwap_{window}"
        ] = (
            df["close"]
            / vwap
            - 1
        )

        features[
            f"above_vwap_{window}"
        ] = (
            df["close"]
            > vwap
        ).astype(int)

    vwap_df = pd.DataFrame(
        features,
        index=df.index,
    )

    return pd.concat(
        [
            df,
            vwap_df,
        ],
        axis=1,
    )


def main() -> None:
    print(
        "Loading BTCUSDT 1-minute data "
        "for VWAP ablation..."
    )

    df = load_data()

    print(
        f"Input rows: {len(df):,}"
    )

    # Exact V1 baseline.
    df = create_return_features(df)
    df = create_trend_features(df)
    df = create_momentum_features(df)
    df = create_volatility_features(df)
    df = create_candle_features(df)
    df = create_volume_features(df)
    df = create_time_features(df)
    df = create_lag_features(df)

    # Add only VWAP-family features.
    df = create_vwap_features(df)

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
        f"VWAP features added: "
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
