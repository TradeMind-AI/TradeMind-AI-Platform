import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(PROJECT_ROOT))


from backend.services.data_pipeline.fetch_btc_data import (
    fetch_historical_data,
)


START_DATE = "2023-09-25"
END_DATE = "2026-09-25"

OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "btcusdt_1h.csv"


def main():
    print("Starting BTC historical data download...")
    print(f"Start date: {START_DATE}")
    print(f"End date:   {END_DATE}")

    df = fetch_historical_data(
        start_time=START_DATE,
        end_time=END_DATE,
        batch_size=1000,
    )

    if df.empty:
        print("No data was downloaded.")
        return

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nDownload complete.")
    print(f"Total candles: {len(df)}")
    print(f"First candle: {df['open_time'].iloc[0]}")
    print(f"Last candle:  {df['open_time'].iloc[-1]}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()