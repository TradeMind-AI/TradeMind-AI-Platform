from pathlib import Path
import pandas as pd

from validate_btc_data import validate_btc_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "btcusdt_1h.csv"


def main():
    print("Loading BTC historical dataset...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["open_time", "close_time"],
    )

    print(f"Dataset: {DATA_FILE}")
    print(f"Total candles: {len(df)}")
    print(f"First candle: {df['open_time'].iloc[0]}")
    print(f"Last candle: {df['open_time'].iloc[-1]}")

    print()

    validate_btc_data(df)


if __name__ == "__main__":
    main()