from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BTC_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_features_enriched_v2.csv"
)

BNB_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bnbusdt_1h.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_features_enriched_v3.csv"
)


def main():
    print("Loading BTC enriched dataset...")
    btc = pd.read_csv(BTC_FEATURE_FILE)

    print("Loading BNB history...")
    bnb = pd.read_csv(BNB_FILE)

    btc["open_time"] = pd.to_datetime(
        btc["open_time"],
        utc=True,
    )

    bnb["open_time"] = pd.to_datetime(
        bnb["open_time"],
        utc=True,
    )

    bnb["close"] = pd.to_numeric(
        bnb["close"],
        errors="coerce",
    )

    bnb["volume"] = pd.to_numeric(
        bnb["volume"],
        errors="coerce",
    )

    bnb = bnb.sort_values(
        "open_time"
    ).reset_index(drop=True)

    print("Creating BNB features...")

    bnb["bnb_return_1h"] = (
        bnb["close"].pct_change(1)
    )

    bnb["bnb_return_3h"] = (
        bnb["close"].pct_change(3)
    )

    bnb["bnb_return_6h"] = (
        bnb["close"].pct_change(6)
    )

    bnb["bnb_return_24h"] = (
        bnb["close"].pct_change(24)
    )

    bnb["bnb_volatility_24h"] = (
        bnb["bnb_return_1h"]
        .rolling(
            window=24,
            min_periods=24,
        )
        .std()
    )

    bnb["bnb_volume_change_1h"] = (
        bnb["volume"].pct_change(1)
    )

    bnb_features = bnb[
        [
            "open_time",
            "bnb_return_1h",
            "bnb_return_3h",
            "bnb_return_6h",
            "bnb_return_24h",
            "bnb_volatility_24h",
            "bnb_volume_change_1h",
        ]
    ].copy()

    merged = btc.merge(
        bnb_features,
        on="open_time",
        how="left",
        validate="one_to_one",
    )

    if "return_1h" in merged.columns:
        merged["btc_bnb_relative_return_1h"] = (
            merged["return_1h"]
            - merged["bnb_return_1h"]
        )

    if "return_24h" in merged.columns:
        merged["btc_bnb_relative_return_24h"] = (
            merged["return_24h"]
            - merged["bnb_return_24h"]
        )

    print(
        f"Rows before merge: {len(btc)}"
    )

    print(
        f"Rows after merge:  {len(merged)}"
    )

    missing = merged[
        [
            "bnb_return_1h",
            "bnb_return_3h",
            "bnb_return_6h",
            "bnb_return_24h",
            "bnb_volatility_24h",
            "bnb_volume_change_1h",
        ]
    ].isna().sum()

    print("\nMissing BNB feature values:")
    print(missing)

    merged = merged.dropna().copy()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nBNB feature merge complete.")
    print(f"Final rows: {len(merged)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()