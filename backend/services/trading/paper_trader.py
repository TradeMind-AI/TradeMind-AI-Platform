from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_live_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "locked_xgboost_bundle.joblib"
)

PAPER_LOG = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_trades.csv"
)


def load_model_bundle():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}\n"
            "Run train_locked_model.py first."
        )

    return joblib.load(MODEL_FILE)


def load_latest_market_row(feature_columns):
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{FEATURE_FILE}"
        )

    df = pd.read_csv(FEATURE_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df = (
        df.sort_values("open_time")
        .reset_index(drop=True)
    )

    latest = df.iloc[-1].copy()

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    X_latest = pd.DataFrame(
        [
            latest[
                feature_columns
            ]
        ]
    )

    if X_latest.isna().any().any():
        bad_columns = X_latest.columns[
            X_latest.isna().any()
        ].tolist()

        raise ValueError(
            "Latest row contains NaN values "
            f"in features: {bad_columns}"
        )

    return latest, X_latest


def already_logged(timestamp):
    if not PAPER_LOG.exists():
        return False

    log = pd.read_csv(PAPER_LOG)

    if log.empty:
        return False

    log["timestamp"] = pd.to_datetime(
        log["timestamp"],
        utc=True,
    )

    timestamp = pd.to_datetime(
        timestamp,
        utc=True,
    )

    return timestamp in set(
        log["timestamp"]
    )


def save_signal(row):
    PAPER_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_row = pd.DataFrame([row])

    if PAPER_LOG.exists():
        new_row.to_csv(
            PAPER_LOG,
            mode="a",
            header=False,
            index=False,
        )
    else:
        new_row.to_csv(
            PAPER_LOG,
            index=False,
        )


def main():
    print(
        "Loading locked paper-trading model..."
    )

    bundle = load_model_bundle()

    model = bundle["model"]
    encoder = bundle["encoder"]
    feature_columns = bundle["feature_columns"]

    confidence_threshold = (
        bundle["confidence_threshold"]
    )

    latest, X_latest = load_latest_market_row(
        feature_columns
    )

    timestamp = latest["open_time"]

    print(
        f"\nLatest candle: {timestamp}"
    )

    if already_logged(timestamp):
        print(
            "Signal already generated "
            "for this candle."
        )
        return

    probabilities = model.predict_proba(
        X_latest
    )[0]

    prediction_encoded = model.predict(
        X_latest
    )[0]

    prediction = (
        encoder.inverse_transform(
            [prediction_encoded]
        )[0]
    )

    class_index = {
        class_name: index
        for index, class_name
        in enumerate(encoder.classes_)
    }

    probability_down = float(
        probabilities[
            class_index["DOWN"]
        ]
    )

    probability_neutral = float(
        probabilities[
            class_index["NEUTRAL"]
        ]
    )

    probability_up = float(
        probabilities[
            class_index["UP"]
        ]
    )

    confidence = float(
        probabilities.max()
    )

    signal = "HOLD"

    if confidence >= confidence_threshold:
        if prediction == "UP":
            signal = "BUY"

        elif prediction == "DOWN":
            signal = "SELL"

    price = float(
        latest["close"]
    )

    print("\n" + "=" * 60)
    print("PAPER TRADING SIGNAL")
    print("=" * 60)

    print(
        f"Timestamp:            {timestamp}"
    )

    print(
        f"BTC price:            {price:.2f}"
    )

    print(
        f"Prediction:           {prediction}"
    )

    print(
        f"Confidence:           {confidence:.4f}"
    )

    print(
        f"Probability DOWN:     {probability_down:.4f}"
    )

    print(
        f"Probability NEUTRAL:  {probability_neutral:.4f}"
    )

    print(
        f"Probability UP:       {probability_up:.4f}"
    )

    print(
        f"Signal:               {signal}"
    )

    print("=" * 60)

    log_row = {
        "timestamp": timestamp,
        "price": price,
        "prediction": prediction,
        "confidence": confidence,
        "probability_down":
            probability_down,
        "probability_neutral":
            probability_neutral,
        "probability_up":
            probability_up,
        "signal": signal,
        "model_name":
            bundle["model_name"],
        "status": "OPEN",
    }

    save_signal(log_row)

    print(
        f"\nPaper signal saved to:\n{PAPER_LOG}"
    )


if __name__ == "__main__":
    main()