from pathlib import Path

import joblib
import pandas as pd

# Reuse the EXACT BTC feature formulas used during training.
from feature_engineering import (
    create_features,
    remove_warmup_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# Input files
# ============================================================

BTC_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_clean.csv"
)

DERIVATIVES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_derivatives_1h.csv"
)

ETH_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ethusdt_1h.csv"
)

BNB_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bnbusdt_1h.csv"
)


# ============================================================
# Output
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_live_features.csv"
)


# ============================================================
# Locked model
# ============================================================

MODEL_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "locked_xgboost_bundle.joblib"
)


# ============================================================
# BTC features
# ============================================================

def build_btc_features() -> pd.DataFrame:
    print("Loading cleaned BTC data...")

    df = pd.read_csv(BTC_FILE)

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

    print(
        f"BTC rows: {len(df)}"
    )

    print(
        f"Latest BTC candle: "
        f"{df['open_time'].iloc[-1]}"
    )

    print(
        "\nCreating the same BTC features "
        "used during model training..."
    )

    # IMPORTANT:
    # This calls the existing training feature functions.
    # We intentionally DO NOT call create_target().
    df = create_features(df)

    # Same historical warm-up rule as training.
    df = remove_warmup_rows(df)

    print(
        f"BTC feature rows: {len(df)}"
    )

    return df


# ============================================================
# Funding features
# ============================================================

def build_funding_features() -> pd.DataFrame:
    print(
        "\nLoading BTC funding data..."
    )

    funding = pd.read_csv(
        DERIVATIVES_FILE
    )

    # Support both historical formats.
    if "timestamp" in funding.columns:
        funding = funding.rename(
            columns={
                "timestamp": "open_time"
            }
        )

    elif "open_time" not in funding.columns:
        raise ValueError(
            "Funding file must contain either "
            "'timestamp' or 'open_time'.\n"
            f"Columns found: {list(funding.columns)}"
        )

    if "funding_rate" not in funding.columns:
        raise ValueError(
            "Funding file is missing funding_rate."
        )

    funding["open_time"] = pd.to_datetime(
        funding["open_time"],
        utc=True,
    )

    funding["funding_rate"] = pd.to_numeric(
        funding["funding_rate"],
        errors="coerce",
    )

    funding = (
        funding[
            [
                "open_time",
                "funding_rate",
            ]
        ]
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # EXACT funding feature set used in v1 enrichment.
    # --------------------------------------------------------

    funding[
        "funding_rate_change_1h"
    ] = (
        funding["funding_rate"]
        .diff()
    )

    funding[
        "funding_rate_change_8h"
    ] = (
        funding["funding_rate"]
        - funding["funding_rate"].shift(8)
    )

    funding[
        "funding_rate_mean_24h"
    ] = (
        funding["funding_rate"]
        .rolling(
            window=24,
            min_periods=24,
        )
        .mean()
    )

    funding[
        "funding_rate_abs"
    ] = (
        funding["funding_rate"]
        .abs()
    )

    print(
        f"Funding rows: {len(funding)}"
    )

    print(
        f"Latest funding timestamp: "
        f"{funding['open_time'].iloc[-1]}"
    )

    return funding


# ============================================================
# ETH features
# ============================================================

def build_eth_features() -> pd.DataFrame:
    print(
        "\nLoading ETH data..."
    )

    eth = pd.read_csv(
        ETH_FILE
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

    eth = (
        eth.sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Same ETH features used during training.
    # --------------------------------------------------------

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

    eth = eth[
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

    print(
        f"ETH rows: {len(eth)}"
    )

    print(
        f"Latest ETH candle: "
        f"{eth['open_time'].iloc[-1]}"
    )

    return eth


# ============================================================
# BNB features
# ============================================================

def build_bnb_features() -> pd.DataFrame:
    print(
        "\nLoading BNB data..."
    )

    bnb = pd.read_csv(
        BNB_FILE
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

    bnb = (
        bnb.sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Same BNB features used during training.
    # --------------------------------------------------------

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

    bnb = bnb[
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

    print(
        f"BNB rows: {len(bnb)}"
    )

    print(
        f"Latest BNB candle: "
        f"{bnb['open_time'].iloc[-1]}"
    )

    return bnb


# ============================================================
# Merge everything
# ============================================================

def merge_all_features(
    btc: pd.DataFrame,
    funding: pd.DataFrame,
    eth: pd.DataFrame,
    bnb: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "\nMerging funding features..."
    )

    df = btc.merge(
        funding,
        on="open_time",
        how="left",
        validate="one_to_one",
    )

    funding_columns = [
        "funding_rate",
        "funding_rate_change_1h",
        "funding_rate_change_8h",
        "funding_rate_mean_24h",
        "funding_rate_abs",
    ]

    print(
        "Missing funding values:"
    )

    print(
        df[
            funding_columns
        ].isna().sum()
    )

    # We only forward fill.
    # Never backward fill future information.
    df[
        funding_columns
    ] = (
        df[
            funding_columns
        ].ffill()
    )

    print(
        "\nMerging ETH features..."
    )

    df = df.merge(
        eth,
        on="open_time",
        how="left",
        validate="one_to_one",
    )

    # Same relative-return features used during training.
    df[
        "btc_eth_relative_return_1h"
    ] = (
        df["return_1h"]
        - df["eth_return_1h"]
    )

    df[
        "btc_eth_relative_return_24h"
    ] = (
        df["return_24h"]
        - df["eth_return_24h"]
    )

    print(
        "\nMerging BNB features..."
    )

    df = df.merge(
        bnb,
        on="open_time",
        how="left",
        validate="one_to_one",
    )

    df[
        "btc_bnb_relative_return_1h"
    ] = (
        df["return_1h"]
        - df["bnb_return_1h"]
    )

    df[
        "btc_bnb_relative_return_24h"
    ] = (
        df["return_24h"]
        - df["bnb_return_24h"]
    )

    return df


# ============================================================
# Model compatibility validation
# ============================================================

def validate_against_locked_model(
    df: pd.DataFrame,
):
    print(
        "\nValidating live features against "
        "locked model..."
    )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Locked model not found:\n{MODEL_FILE}"
        )

    bundle = joblib.load(
        MODEL_FILE
    )

    model_features = (
        bundle["feature_columns"]
    )

    missing_features = [
        column
        for column in model_features
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "\nLive dataset is missing model features:\n"
            + "\n".join(missing_features)
        )

    latest = df.iloc[-1]

    missing_latest = [
        column
        for column in model_features
        if pd.isna(
            latest[column]
        )
    ]

    if missing_latest:
        raise ValueError(
            "\nLatest live row contains NaN "
            "model features:\n"
            + "\n".join(missing_latest)
        )

    print(
        f"Locked model features: "
        f"{len(model_features)}"
    )

    print(
        "Missing model features: 0"
    )

    print(
        "Latest-row NaN model features: 0"
    )

    if len(model_features) != 84:
        print(
            "WARNING: locked model does not "
            "contain the expected 84 features."
        )
    else:
        print(
            "84-feature model compatibility: OK"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "BUILDING LIVE ML FEATURES"
    )

    print(
        "=" * 70
    )

    btc = build_btc_features()

    expected_latest_timestamp = (
        btc["open_time"].max()
    )

    funding = build_funding_features()
    eth = build_eth_features()
    bnb = build_bnb_features()

    df = merge_all_features(
        btc,
        funding,
        eth,
        bnb,
    )

    # --------------------------------------------------------
    # Only remove rows where required historical/model
    # features are unavailable.
    #
    # DO NOT remove the newest row because it has no future
    # target. There is intentionally no target in this file.
    # --------------------------------------------------------

    bundle = joblib.load(
        MODEL_FILE
    )

    model_features = (
        bundle["feature_columns"]
    )

    before = len(df)

    df = df.dropna(
        subset=model_features
    ).copy()

    rows_removed = (
        before - len(df)
    )

    df = (
        df.sort_values(
            "open_time"
        )
        .reset_index(drop=True)
    )

    print(
        f"\nRows removed for unavailable "
        f"historical features: {rows_removed}"
    )

    print(
        f"Final live feature rows: "
        f"{len(df)}"
    )

    if df.empty:
        raise ValueError(
            "Live feature dataset is empty."
        )

    actual_latest_timestamp = (
        df["open_time"].iloc[-1]
    )

    print(
        f"\nExpected latest BTC candle: "
        f"{expected_latest_timestamp}"
    )

    print(
        f"Latest live feature candle: "
        f"{actual_latest_timestamp}"
    )

    # --------------------------------------------------------
    # Critical timestamp check.
    # --------------------------------------------------------

    if (
        actual_latest_timestamp
        != expected_latest_timestamp
    ):
        raise ValueError(
            "\nTIMESTAMP ALIGNMENT FAILED\n"
            f"BTC latest:  "
            f"{expected_latest_timestamp}\n"
            f"Feature latest: "
            f"{actual_latest_timestamp}"
        )

    print(
        "Live timestamp alignment: OK"
    )

    validate_against_locked_model(
        df
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
        "\n"
        + "=" * 70
    )

    print(
        "LIVE FEATURE BUILD COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Output:\n{OUTPUT_FILE}"
    )

    print(
        f"\nLatest candle ready for prediction:"
        f"\n{actual_latest_timestamp}"
    )


if __name__ == "__main__":
    main()