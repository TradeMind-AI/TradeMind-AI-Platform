from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SIGNALS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trading_signals.csv"
)

SENTIMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btc_news_sentiment_hourly.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trading_signals_with_sentiment.csv"
)


MIN_NEWS_COUNT = 3

POSITIVE_SENTIMENT_THRESHOLD = 0.10
NEGATIVE_SENTIMENT_THRESHOLD = -0.10


def apply_sentiment_filter(row: pd.Series) -> str:
    signal = row["signal"]

    news_count = row["news_count"]
    avg_sentiment = row["avg_sentiment"]

    if pd.isna(news_count) or pd.isna(avg_sentiment):
        return signal

    if news_count < MIN_NEWS_COUNT:
        return signal

    if signal == "BUY":
        if avg_sentiment <= NEGATIVE_SENTIMENT_THRESHOLD:
            return "HOLD"

        return "BUY"

    if signal == "SELL":
        if avg_sentiment >= POSITIVE_SENTIMENT_THRESHOLD:
            return "HOLD"

        return "SELL"

    return "HOLD"


def main():
    print("Loading trading signals...")
    signals = pd.read_csv(SIGNALS_FILE)

    print("Loading news sentiment...")
    sentiment = pd.read_csv(SENTIMENT_FILE)

    signals["timestamp"] = pd.to_datetime(
        signals["timestamp"],
        utc=True,
    )

    sentiment["open_time"] = pd.to_datetime(
        sentiment["open_time"],
        utc=True,
    )

    sentiment = sentiment.rename(
        columns={
            "open_time": "timestamp"
        }
    )

    sentiment_columns = [
        "timestamp",
        "news_count",
        "avg_sentiment",
        "positive_news_count",
        "negative_news_count",
        "sentiment_net",
    ]

    merged = signals.merge(
        sentiment[sentiment_columns],
        on="timestamp",
        how="left",
    )

    merged["filtered_signal"] = merged.apply(
        apply_sentiment_filter,
        axis=1,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSentiment filtering complete.")

    print("\nOriginal signal distribution:")
    print(
        merged["signal"]
        .value_counts()
    )

    print("\nFiltered signal distribution:")
    print(
        merged["filtered_signal"]
        .value_counts()
    )

    changed = (
        merged["signal"]
        != merged["filtered_signal"]
    ).sum()

    print(
        f"\nSignals changed by sentiment: "
        f"{changed}"
    )

    sentiment_matches = (
        merged["avg_sentiment"]
        .notna()
        .sum()
    )

    print(
        f"Rows with sentiment data: "
        f"{sentiment_matches}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()