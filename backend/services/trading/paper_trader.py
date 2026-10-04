from pathlib import Path

import joblib
import pandas as pd

from risk_manager import evaluate_trade


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


LOG_COLUMNS = [
    "timestamp",
    "price",
    "prediction",
    "confidence",
    "probability_down",
    "probability_neutral",
    "probability_up",
    "signal",
    "model_name",
    "status",

    "risk_status",
    "risk_reason",

    "account_balance_before",
    "risk_amount",
    "position_notional",
    "quantity",

    "stop_loss_price",
    "take_profit_price",

    "entry_price",
    "exit_price",
    "exit_reason",

    "gross_return_pct",
    "fee_pct",
    "net_return_pct",

    "account_pnl",
    "account_balance_after",

    "result",
    "cumulative_net_return_pct",
]


def load_model_bundle():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    return joblib.load(
        MODEL_FILE
    )


def load_paper_log():
    if not PAPER_LOG.exists():
        return pd.DataFrame(
            columns=LOG_COLUMNS
        )

    log = pd.read_csv(
        PAPER_LOG
    )

    for column in LOG_COLUMNS:
        if column not in log.columns:
            log[column] = pd.NA

    return log[
        LOG_COLUMNS
    ].copy()


def load_latest_market_row(
    feature_columns,
):
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Live feature dataset not found:\n"
            f"{FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

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

        bad = X_latest.columns[
            X_latest.isna().any()
        ].tolist()

        raise ValueError(
            "Latest row contains NaN values "
            f"in features: {bad}"
        )

    return latest, X_latest


def already_logged(
    log: pd.DataFrame,
    timestamp,
):
    if log.empty:
        return False

    timestamps = pd.to_datetime(
        log["timestamp"],
        utc=True,
        errors="coerce",
    )

    timestamp = pd.to_datetime(
        timestamp,
        utc=True,
    )

    return timestamp in set(
        timestamps.dropna()
    )


def save_log(
    log: pd.DataFrame,
    row: dict,
):
    new_row = pd.DataFrame(
        [row]
    )

    for column in LOG_COLUMNS:

        if column not in new_row.columns:
            new_row[column] = pd.NA

    new_row = new_row[
        LOG_COLUMNS
    ]

    combined = pd.concat(
        [
            log,
            new_row,
        ],
        ignore_index=True,
    )

    PAPER_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
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

    feature_columns = (
        bundle["feature_columns"]
    )

    confidence_threshold = float(
        bundle["confidence_threshold"]
    )

    latest, X_latest = (
        load_latest_market_row(
            feature_columns
        )
    )

    timestamp = latest[
        "open_time"
    ]

    print(
        f"\nLatest candle: {timestamp}"
    )

    log = load_paper_log()

    if already_logged(
        log,
        timestamp,
    ):
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
        name: index
        for index, name
        in enumerate(
            encoder.classes_
        )
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

    model_signal = "HOLD"

    if confidence >= confidence_threshold:

        if prediction == "UP":
            model_signal = "BUY"

        elif prediction == "DOWN":
            model_signal = "SELL"

    price = float(
        latest["close"]
    )

    # --------------------------------------------------------
    # Risk management
    # --------------------------------------------------------

    risk = evaluate_trade(
        signal=model_signal,
        entry_price=price,
        timestamp=timestamp,
        paper_log=log,
    )

    final_signal = model_signal

    if (
        model_signal in {"BUY", "SELL"}
        and not risk.allowed
    ):
        final_signal = "HOLD"

    if final_signal in {"BUY", "SELL"}:
        status = "OPEN"
        risk_status = "APPROVED"
    else:
        status = "NO_TRADE"

        if model_signal == "HOLD":
            risk_status = "NOT_REQUIRED"
        else:
            risk_status = "BLOCKED"

    print("\n" + "=" * 70)
    print("PAPER TRADING SIGNAL")
    print("=" * 70)

    print(
        f"Timestamp:              {timestamp}"
    )

    print(
        f"BTC price:              {price:.2f}"
    )

    print(
        f"Prediction:             {prediction}"
    )

    print(
        f"Confidence:             {confidence:.4f}"
    )

    print(
        f"Probability DOWN:       {probability_down:.4f}"
    )

    print(
        f"Probability NEUTRAL:    {probability_neutral:.4f}"
    )

    print(
        f"Probability UP:         {probability_up:.4f}"
    )

    print(
        f"Model signal:           {model_signal}"
    )

    print(
        f"Final signal:           {final_signal}"
    )

    print("\n" + "-" * 70)
    print("RISK MANAGEMENT")
    print("-" * 70)

    print(
        f"Account balance:        ${risk.balance:.2f}"
    )

    print(
        f"Risk status:            {risk_status}"
    )

    print(
        f"Reason:                 {risk.reason}"
    )

    if final_signal in {"BUY", "SELL"}:

        print(
            f"Position notional:      "
            f"${risk.position_notional:.2f}"
        )

        print(
            f"BTC quantity:           "
            f"{risk.quantity:.8f}"
        )

        print(
            f"Maximum risk amount:    "
            f"${risk.risk_amount:.2f}"
        )

        print(
            f"Stop loss:              "
            f"${risk.stop_loss_price:.2f}"
        )

        print(
            f"Take profit:            "
            f"${risk.take_profit_price:.2f}"
        )

    print("=" * 70)

    row = {
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

        "signal": final_signal,

        "model_name":
            bundle["model_name"],

        "status": status,

        "risk_status":
            risk_status,

        "risk_reason":
            risk.reason,

        "account_balance_before":
            risk.balance,

        "risk_amount":
            risk.risk_amount
            if final_signal
            in {"BUY", "SELL"}
            else pd.NA,

        "position_notional":
            risk.position_notional
            if final_signal
            in {"BUY", "SELL"}
            else pd.NA,

        "quantity":
            risk.quantity
            if final_signal
            in {"BUY", "SELL"}
            else pd.NA,

        "stop_loss_price":
            risk.stop_loss_price
            if final_signal
            in {"BUY", "SELL"}
            else pd.NA,

        "take_profit_price":
            risk.take_profit_price
            if final_signal
            in {"BUY", "SELL"}
            else pd.NA,

        "entry_price":
            price
            if final_signal
            in {"BUY", "SELL"}
            else pd.NA,
    }

    save_log(
        log,
        row,
    )

    print(
        f"\nPaper signal saved to:\n"
        f"{PAPER_LOG}"
    )


if __name__ == "__main__":
    main()