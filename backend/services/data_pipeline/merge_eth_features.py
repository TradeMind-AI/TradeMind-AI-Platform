from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BTC_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_features_enriched.csv"
)

ETH_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ethusdt_1h.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_features_enriched_v2.csv"
)


def main():
    print("Loading BTC enriched feature dataset...")
    btc = pd.read_csv(BTC_FEATURE_FILE)

    print("Loading ETH history...")
    eth = pd.read_csv(ETH_FILE)

    btc["open_time"] = pd.to_datetime(
        btc["open_time"],
        utc=True,
    )

    eth["open_time"] = pd.to_datetime(
        eth["open_time"],
        utc=True,
    )

    eth["close"] = pd.to_numeric(
        eth["close"],
        errors="coerce",
    )

    eth["volume"] = pd.to_numeric(
        eth["volume"],
        errors="coerce",
    )

    eth = eth.sort_values(
        "open_time"
    ).reset_index(drop=True)

    print("Creating ETH features...")

    eth["eth_return_1h"] = (
        eth["close"].pct_change(1)
    )

    eth["eth_return_3h"] = (
        eth["close"].pct_change(3)
    )

    eth["eth_return_6h"] = (
        eth["close"].pct_change(6)
    )

    eth["eth_return_24h"] = (
        eth["close"].pct_change(24)
    )

    eth["eth_volatility_24h"] = (
        eth["eth_return_1h"]
        .rolling(
            window=24,
            min_periods=24,
        )
        .std()
    )

    eth["eth_volume_change_1h"] = (
        eth["volume"].pct_change(1)
    )

    eth_features = eth[
        [
            "open_time",
            "eth_return_1h",
            "eth_return_3h",
            "eth_return_6h",
            "eth_return_24h",
            "eth_volatility_24h",
            "eth_volume_change_1h",
        ]
    ].copy()

    merged = btc.merge(
        eth_features,
        on="open_time",
        how="left",
        validate="one_to_one",
    )

    print(
        f"BTC rows before merge: {len(btc)}"
    )

    print(
        f"Rows after merge:      {len(merged)}"
    )

    missing = (
        merged[
            [
                "eth_return_1h",
                "eth_return_3h",
                "eth_return_6h",
                "eth_return_24h",
                "eth_volatility_24h",
                "eth_volume_change_1h",
            ]
        ]
        .isna()
        .sum()
    )

    print("\nMissing ETH feature values:")
    print(missing)

    # BTC relative performance vs ETH
    if "return_1h" in merged.columns:
        merged["btc_eth_relative_return_1h"] = (
            merged["return_1h"]
            - merged["eth_return_1h"]
        )

    if "return_24h" in merged.columns:
        merged["btc_eth_relative_return_24h"] = (
            merged["return_24h"]
            - merged["eth_return_24h"]
        )

    merged = merged.dropna().copy()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nETH feature merge complete.")
    print(f"Final rows: {len(merged)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()