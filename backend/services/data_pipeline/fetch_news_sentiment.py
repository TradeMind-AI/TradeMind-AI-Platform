from pathlib import Path
from datetime import datetime, timedelta, timezone
import time

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btc_news_sentiment_hourly.csv"
)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

QUERY = '(bitcoin OR BTC OR cryptocurrency OR crypto)'

MAX_RECORDS = 250

# Start with recent data first.
LOOKBACK_DAYS = 30


analyzer = SentimentIntensityAnalyzer()


def fetch_articles(
    start_time: datetime,
    end_time: datetime,
) -> list[dict]:
    params = {
        "query": QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "sort": "DateAsc",
        "startdatetime": start_time.strftime(
            "%Y%m%d%H%M%S"
        ),
        "enddatetime": end_time.strftime(
            "%Y%m%d%H%M%S"
        ),
    }

    response = requests.get(
        GDELT_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    return payload.get("articles", [])


def score_title(title: str) -> float:
    if not title:
        return 0.0

    scores = analyzer.polarity_scores(
        title
    )

    return scores["compound"]


def main():
    end_time = datetime.now(
        timezone.utc
    ).replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    start_time = (
        end_time
        - timedelta(days=LOOKBACK_DAYS)
    )

    print(
        f"Collecting BTC news from "
        f"{start_time} to {end_time}"
    )

    all_rows = []

    current = start_time

    # Fetch in 6-hour windows to avoid giant responses.
    while current < end_time:
        window_end = min(
            current + timedelta(hours=6),
            end_time,
        )

        print(
            f"Fetching {current} -> "
            f"{window_end}"
        )

        try:
            articles = fetch_articles(
                current,
                window_end,
            )
        except Exception as exc:
            print(
                f"Request failed: {exc}"
            )

            current = window_end

            time.sleep(2)

            continue

        for article in articles:
            title = article.get(
                "title",
                "",
            )

            published = article.get(
                "seendate"
            )

            if not published:
                continue

            try:
                timestamp = pd.to_datetime(
                    published,
                    utc=True,
                )
            except Exception:
                continue

            sentiment = score_title(
                title
            )

            all_rows.append(
                {
                    "timestamp": timestamp,
                    "title": title,
                    "sentiment_score": sentiment,
                    "domain": article.get(
                        "domain"
                    ),
                    "url": article.get(
                        "url"
                    ),
                }
            )

        print(
            f"Articles collected: "
            f"{len(all_rows)}"
        )

        current = window_end

        time.sleep(1)

    if not all_rows:
        raise ValueError(
            "No BTC news articles were collected."
        )

    articles_df = pd.DataFrame(
        all_rows
    )

    articles_df = (
        articles_df
        .drop_duplicates(
            subset=["url"]
        )
        .sort_values("timestamp")
    )

    articles_df["hour"] = (
        articles_df["timestamp"]
        .dt.floor("h")
    )

    articles_df["positive"] = (
        articles_df[
            "sentiment_score"
        ]
        > 0.05
    ).astype(int)

    articles_df["negative"] = (
        articles_df[
            "sentiment_score"
        ]
        < -0.05
    ).astype(int)

    hourly = (
        articles_df
        .groupby("hour")
        .agg(
            news_count=(
                "sentiment_score",
                "size",
            ),
            avg_sentiment=(
                "sentiment_score",
                "mean",
            ),
            max_sentiment=(
                "sentiment_score",
                "max",
            ),
            min_sentiment=(
                "sentiment_score",
                "min",
            ),
            positive_news_count=(
                "positive",
                "sum",
            ),
            negative_news_count=(
                "negative",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "hour": "open_time"
            }
        )
    )

    hourly[
        "sentiment_abs_mean"
    ] = (
        hourly[
            "avg_sentiment"
        ]
        .abs()
    )

    hourly[
        "sentiment_net"
    ] = (
        hourly[
            "positive_news_count"
        ]
        - hourly[
            "negative_news_count"
        ]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nNews sentiment complete.")
    print(
        f"Articles: "
        f"{len(articles_df)}"
    )
    print(
        f"Hourly rows: "
        f"{len(hourly)}"
    )
    print(
        f"Output: "
        f"{OUTPUT_FILE}"
    )

    print("\nFirst rows:")
    print(
        hourly.head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()