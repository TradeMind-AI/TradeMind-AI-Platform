from __future__ import annotations

import csv
import json
import random
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "live_microstructure"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micro_paper_learning"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "micro_online_paper_model.joblib"
)

ACCOUNT_FILE = (
    OUTPUT_DIR
    / "account.json"
)

TRADE_LOG = (
    OUTPUT_DIR
    / "trades.csv"
)

LEARNING_LOG = (
    OUTPUT_DIR
    / "learning_samples.csv"
)

FEATURE_LOG = (
    OUTPUT_DIR
    / "feature_snapshots.csv"
)

LOOP_SECONDS = 5
LABEL_HORIZON_SECONDS = 30
LABEL_THRESHOLD = 0.0005

STARTING_EQUITY = 10000.0
POSITION_FRACTION = 0.05
FEE_RATE = 0.0005
TAKE_PROFIT_PCT = 0.0020
STOP_LOSS_PCT = 0.0012
MAX_HOLD_SECONDS = 60

MODEL_ENTRY_CONFIDENCE = 0.55
EXPLORATION_RATE = 0.15
MIN_TRAINED_SAMPLES = 100

CLASSES = np.array(
    [-1, 0, 1],
    dtype=int,
)

FEATURE_NAMES = [
    "spot_spread_bps",
    "spot_top_imbalance",
    "spot_depth20_imbalance",
    "spot_trade_flow_imbalance",
    "spot_price_velocity_bps",
    "futures_spread_bps",
    "futures_top_imbalance",
    "futures_depth20_imbalance",
    "futures_trade_flow_imbalance",
    "futures_price_velocity_bps",
    "funding_rate",
    "open_interest_change_pct",
    "spot_futures_mid_basis_bps",
]


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def utc_now_iso() -> str:
    return utc_now().isoformat()


def today_file(
    prefix: str,
) -> Path:
    date = utc_now().strftime(
        "%Y-%m-%d"
    )

    return (
        RAW_DIR
        / f"{prefix}_{date}.csv"
    )


def tail_csv(
    path: Path,
    max_rows: int,
) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        header = handle.readline()

        if not header:
            return []

        lines = deque(
            handle,
            maxlen=max_rows,
        )

    if not lines:
        return []

    reader = csv.DictReader(
        [header, *lines]
    )

    return list(reader)


def number(
    value,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)

        if np.isfinite(result):
            return result
    except (
        TypeError,
        ValueError,
    ):
        pass

    return default


def bool_value(
    value,
) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def imbalance(
    bid: float,
    ask: float,
) -> float:
    total = bid + ask

    if total <= 0:
        return 0.0

    return (
        bid - ask
    ) / total


def latest_book_features(
    prefix: str,
) -> tuple[float, float, float]:
    rows = tail_csv(
        today_file(
            f"{prefix}_book_ticker"
        ),
        2,
    )

    if not rows:
        return (
            0.0,
            0.0,
            0.0,
        )

    row = rows[-1]

    bid = number(
        row.get(
            "best_bid_price"
        )
    )
    ask = number(
        row.get(
            "best_ask_price"
        )
    )
    bid_qty = number(
        row.get(
            "best_bid_qty"
        )
    )
    ask_qty = number(
        row.get(
            "best_ask_qty"
        )
    )

    mid = (
        (bid + ask) / 2.0
        if bid > 0 and ask > 0
        else max(
            bid,
            ask,
        )
    )

    spread_bps = (
        (
            (ask - bid)
            / mid
            * 10000.0
        )
        if mid > 0
        else 0.0
    )

    return (
        mid,
        spread_bps,
        imbalance(
            bid_qty,
            ask_qty,
        ),
    )


def latest_depth_imbalance(
    prefix: str,
) -> float:
    rows = tail_csv(
        today_file(
            f"{prefix}_depth20"
        ),
        1,
    )

    if not rows:
        return 0.0

    row = rows[-1]

    try:
        bids = json.loads(
            row.get(
                "bids_json",
                "[]",
            )
        )
        asks = json.loads(
            row.get(
                "asks_json",
                "[]",
            )
        )
    except json.JSONDecodeError:
        return 0.0

    bid_qty = sum(
        number(level[1])
        for level in bids
        if len(level) >= 2
    )

    ask_qty = sum(
        number(level[1])
        for level in asks
        if len(level) >= 2
    )

    return imbalance(
        bid_qty,
        ask_qty,
    )


def trade_flow_features(
    prefix: str,
) -> tuple[float, float]:
    rows = tail_csv(
        today_file(
            f"{prefix}_agg_trade"
        ),
        500,
    )

    if not rows:
        return (
            0.0,
            0.0,
        )

    cutoff_ms = int(
        (
            time.time()
            - 5.0
        )
        * 1000
    )

    recent = [
        row
        for row in rows
        if number(
            row.get(
                "trade_time"
            )
        ) >= cutoff_ms
    ]

    if not recent:
        recent = rows[-50:]

    aggressive_buy = 0.0
    aggressive_sell = 0.0

    for row in recent:
        quantity = number(
            row.get(
                "quantity"
            )
        )

        if bool_value(
            row.get(
                "buyer_is_maker"
            )
        ):
            aggressive_sell += (
                quantity
            )
        else:
            aggressive_buy += (
                quantity
            )

    flow = imbalance(
        aggressive_buy,
        aggressive_sell,
    )

    first_price = number(
        recent[0].get(
            "price"
        )
    )
    last_price = number(
        recent[-1].get(
            "price"
        )
    )

    velocity_bps = (
        (
            last_price
            / first_price
            - 1.0
        )
        * 10000.0
        if first_price > 0
        else 0.0
    )

    return (
        flow,
        velocity_bps,
    )


def latest_funding_rate() -> float:
    rows = tail_csv(
        today_file(
            "futures_mark_price"
        ),
        1,
    )

    if not rows:
        return 0.0

    return number(
        rows[-1].get(
            "funding_rate"
        )
    )


def latest_oi_change_pct() -> float:
    rows = tail_csv(
        today_file(
            "futures_open_interest"
        ),
        20,
    )

    if len(rows) < 2:
        return 0.0

    first = number(
        rows[0].get(
            "open_interest"
        )
    )
    last = number(
        rows[-1].get(
            "open_interest"
        )
    )

    if first <= 0:
        return 0.0

    return (
        last / first
        - 1.0
    ) * 100.0


def build_snapshot() -> tuple[
    float,
    np.ndarray,
    dict[str, float],
]:
    (
        spot_mid,
        spot_spread,
        spot_top_imbalance,
    ) = latest_book_features(
        "spot"
    )

    (
        futures_mid,
        futures_spread,
        futures_top_imbalance,
    ) = latest_book_features(
        "futures"
    )

    spot_depth = (
        latest_depth_imbalance(
            "spot"
        )
    )

    futures_depth = (
        latest_depth_imbalance(
            "futures"
        )
    )

    (
        spot_flow,
        spot_velocity,
    ) = trade_flow_features(
        "spot"
    )

    (
        futures_flow,
        futures_velocity,
    ) = trade_flow_features(
        "futures"
    )

    funding = (
        latest_funding_rate()
    )

    oi_change = (
        latest_oi_change_pct()
    )

    basis_bps = (
        (
            futures_mid
            / spot_mid
            - 1.0
        )
        * 10000.0
        if (
            spot_mid > 0
            and futures_mid > 0
        )
        else 0.0
    )

    values = {
        "spot_spread_bps":
            spot_spread,
        "spot_top_imbalance":
            spot_top_imbalance,
        "spot_depth20_imbalance":
            spot_depth,
        "spot_trade_flow_imbalance":
            spot_flow,
        "spot_price_velocity_bps":
            spot_velocity,
        "futures_spread_bps":
            futures_spread,
        "futures_top_imbalance":
            futures_top_imbalance,
        "futures_depth20_imbalance":
            futures_depth,
        "futures_trade_flow_imbalance":
            futures_flow,
        "futures_price_velocity_bps":
            futures_velocity,
        "funding_rate":
            funding,
        "open_interest_change_pct":
            oi_change,
        "spot_futures_mid_basis_bps":
            basis_bps,
    }

    vector = np.array(
        [
            values[name]
            for name in FEATURE_NAMES
        ],
        dtype=np.float64,
    )

    vector = np.nan_to_num(
        vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return (
        spot_mid,
        vector,
        values,
    )


class OnlineModel:
    def __init__(
        self,
    ) -> None:
        self.scaler = StandardScaler()
        self.model = SGDClassifier(
            loss="log_loss",
            alpha=0.0005,
            random_state=42,
        )
        self.scaler_ready = False
        self.model_ready = False
        self.samples = 0

        if MODEL_FILE.exists():
            try:
                saved = joblib.load(
                    MODEL_FILE
                )
                self.scaler = saved[
                    "scaler"
                ]
                self.model = saved[
                    "model"
                ]
                self.scaler_ready = bool(
                    saved.get(
                        "scaler_ready",
                        False,
                    )
                )
                self.model_ready = bool(
                    saved.get(
                        "model_ready",
                        False,
                    )
                )
                self.samples = int(
                    saved.get(
                        "samples",
                        0,
                    )
                )
            except Exception:
                pass

    def update(
        self,
        features: np.ndarray,
        label: int,
    ) -> None:
        X = features.reshape(
            1,
            -1,
        )

        self.scaler.partial_fit(
            X
        )
        self.scaler_ready = True

        transformed = (
            self.scaler.transform(
                X
            )
        )

        if not self.model_ready:
            self.model.partial_fit(
                transformed,
                np.array(
                    [label]
                ),
                classes=CLASSES,
            )
            self.model_ready = True
        else:
            self.model.partial_fit(
                transformed,
                np.array(
                    [label]
                ),
            )

        self.samples += 1
        self.save()

    def predict(
        self,
        features: np.ndarray,
    ) -> tuple[
        int,
        float,
        dict[int, float],
    ]:
        if (
            not self.scaler_ready
            or not self.model_ready
        ):
            return (
                0,
                0.0,
                {
                    -1: 0.0,
                    0: 1.0,
                    1: 0.0,
                },
            )

        X = self.scaler.transform(
            features.reshape(
                1,
                -1,
            )
        )

        probabilities = (
            self.model.predict_proba(
                X
            )[0]
        )

        mapping = {
            int(label): float(prob)
            for label, prob in zip(
                self.model.classes_,
                probabilities,
            )
        }

        prediction = max(
            mapping,
            key=mapping.get,
        )

        return (
            int(prediction),
            float(
                mapping[
                    prediction
                ]
            ),
            mapping,
        )

    def save(
        self,
    ) -> None:
        MODEL_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "scaler":
                    self.scaler,
                "model":
                    self.model,
                "scaler_ready":
                    self.scaler_ready,
                "model_ready":
                    self.model_ready,
                "samples":
                    self.samples,
                "feature_names":
                    FEATURE_NAMES,
                "label_horizon_seconds":
                    LABEL_HORIZON_SECONDS,
                "label_threshold":
                    LABEL_THRESHOLD,
            },
            MODEL_FILE,
        )


def heuristic_direction(
    values: dict[str, float],
) -> int:
    score = (
        0.20
        * values[
            "spot_depth20_imbalance"
        ]
        + 0.20
        * values[
            "spot_trade_flow_imbalance"
        ]
        + 0.15
        * values[
            "futures_depth20_imbalance"
        ]
        + 0.15
        * values[
            "futures_trade_flow_imbalance"
        ]
        + 0.15
        * np.tanh(
            values[
                "spot_price_velocity_bps"
            ]
            / 5.0
        )
        + 0.10
        * np.tanh(
            values[
                "futures_price_velocity_bps"
            ]
            / 5.0
        )
        + 0.05
        * np.tanh(
            values[
                "open_interest_change_pct"
            ]
            * 10.0
        )
    )

    if score > 0.08:
        return 1

    if score < -0.08:
        return -1

    return 0


def load_account() -> dict:
    if ACCOUNT_FILE.exists():
        try:
            return json.loads(
                ACCOUNT_FILE.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            pass

    return {
        "equity_usdt":
            STARTING_EQUITY,
        "realized_pnl_usdt":
            0.0,
        "position": None,
        "closed_trades": 0,
        "wins": 0,
        "losses": 0,
    }


def save_account(
    account: dict,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ACCOUNT_FILE.write_text(
        json.dumps(
            account,
            indent=2,
        ),
        encoding="utf-8",
    )


def append_csv(
    path: Path,
    row: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                row.keys()
            ),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            row
        )


def close_position(
    account: dict,
    price: float,
    reason: str,
) -> None:
    position = account.get(
        "position"
    )

    if not position:
        return

    direction = int(
        position["direction"]
    )
    entry_price = float(
        position["entry_price"]
    )
    notional = float(
        position["notional_usdt"]
    )

    gross_return = (
        (
            price / entry_price
            - 1.0
        )
        * direction
    )

    net_return = (
        gross_return
        - 2.0 * FEE_RATE
    )

    pnl = (
        notional
        * net_return
    )

    account[
        "equity_usdt"
    ] = float(
        account[
            "equity_usdt"
        ]
    ) + pnl

    account[
        "realized_pnl_usdt"
    ] = float(
        account[
            "realized_pnl_usdt"
        ]
    ) + pnl

    account[
        "closed_trades"
    ] = int(
        account[
            "closed_trades"
        ]
    ) + 1

    if pnl > 0:
        account[
            "wins"
        ] = int(
            account[
                "wins"
            ]
        ) + 1
    else:
        account[
            "losses"
        ] = int(
            account[
                "losses"
            ]
        ) + 1

    append_csv(
        TRADE_LOG,
        {
            "timestamp":
                utc_now_iso(),
            "event":
                "CLOSE",
            "side":
                (
                    "LONG"
                    if direction == 1
                    else "SHORT"
                ),
            "entry_price":
                entry_price,
            "exit_price":
                price,
            "notional_usdt":
                notional,
            "gross_return_pct":
                gross_return
                * 100.0,
            "net_return_pct":
                net_return
                * 100.0,
            "pnl_usdt":
                pnl,
            "reason":
                reason,
            "model_confidence":
                position.get(
                    "model_confidence",
                    0.0,
                ),
            "learning_samples":
                position.get(
                    "learning_samples",
                    0,
                ),
        },
    )

    account["position"] = None


def maybe_manage_position(
    account: dict,
    price: float,
    direction: int,
) -> None:
    position = account.get(
        "position"
    )

    if not position:
        return

    entry_price = float(
        position["entry_price"]
    )
    position_direction = int(
        position["direction"]
    )

    gross_return = (
        (
            price / entry_price
            - 1.0
        )
        * position_direction
    )

    opened_at = datetime.fromisoformat(
        position[
            "opened_at"
        ]
    )

    held_seconds = (
        utc_now()
        - opened_at
    ).total_seconds()

    if gross_return >= (
        TAKE_PROFIT_PCT
    ):
        close_position(
            account,
            price,
            "TAKE_PROFIT",
        )
    elif gross_return <= (
        -STOP_LOSS_PCT
    ):
        close_position(
            account,
            price,
            "STOP_LOSS",
        )
    elif (
        direction != 0
        and direction
        != position_direction
    ):
        close_position(
            account,
            price,
            "MODEL_REVERSAL",
        )
    elif held_seconds >= (
        MAX_HOLD_SECONDS
    ):
        close_position(
            account,
            price,
            "MAX_HOLD",
        )


def maybe_open_position(
    account: dict,
    price: float,
    direction: int,
    confidence: float,
    samples: int,
    source: str,
) -> None:
    if (
        account.get(
            "position"
        )
        or direction == 0
        or price <= 0
    ):
        return

    notional = (
        float(
            account[
                "equity_usdt"
            ]
        )
        * POSITION_FRACTION
    )

    account["position"] = {
        "direction":
            direction,
        "entry_price":
            price,
        "notional_usdt":
            notional,
        "opened_at":
            utc_now_iso(),
        "model_confidence":
            confidence,
        "learning_samples":
            samples,
        "decision_source":
            source,
    }

    append_csv(
        TRADE_LOG,
        {
            "timestamp":
                utc_now_iso(),
            "event":
                "OPEN",
            "side":
                (
                    "LONG"
                    if direction == 1
                    else "SHORT"
                ),
            "entry_price":
                price,
            "exit_price":
                "",
            "notional_usdt":
                notional,
            "gross_return_pct":
                "",
            "net_return_pct":
                "",
            "pnl_usdt":
                "",
            "reason":
                source,
            "model_confidence":
                confidence,
            "learning_samples":
                samples,
        },
    )


def label_from_return(
    future_return: float,
) -> int:
    if future_return > (
        LABEL_THRESHOLD
    ):
        return 1

    if future_return < (
        -LABEL_THRESHOLD
    ):
        return -1

    return 0


def main() -> None:
    random.seed(42)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = OnlineModel()
    account = load_account()

    pending: deque[dict] = (
        deque()
    )

    print(
        "TRADEMIND MICRO PAPER LEARNER"
    )
    print(
        "Paper only - no real orders."
    )
    print(
        "Reads the live microstructure "
        "collector files every 5 seconds."
    )
    print(
        "Learning label: BTC move after "
        f"{LABEL_HORIZON_SECONDS}s with "
        f"+/-{LABEL_THRESHOLD * 100:.3f}% "
        "threshold."
    )
    print(
        "Research entry confidence: "
        f"{MODEL_ENTRY_CONFIDENCE:.2f}; "
        "production target remains >= 0.80."
    )
    print(
        "Press Ctrl+C to stop."
    )

    try:
        while True:
            now = time.time()

            (
                price,
                features,
                values,
            ) = build_snapshot()

            if price <= 0:
                print(
                    "Waiting for live "
                    "microstructure data..."
                )
                time.sleep(
                    LOOP_SECONDS
                )
                continue

            while (
                pending
                and now
                - pending[0][
                    "created_at"
                ]
                >= LABEL_HORIZON_SECONDS
            ):
                sample = (
                    pending.popleft()
                )

                future_return = (
                    price
                    / sample[
                        "price"
                    ]
                    - 1.0
                )

                label = (
                    label_from_return(
                        future_return
                    )
                )

                model.update(
                    sample[
                        "features"
                    ],
                    label,
                )

                append_csv(
                    LEARNING_LOG,
                    {
                        "timestamp":
                            utc_now_iso(),
                        "sample_time":
                            sample[
                                "sample_time"
                            ],
                        "entry_price":
                            sample[
                                "price"
                            ],
                        "label_price":
                            price,
                        "future_return_pct":
                            future_return
                            * 100.0,
                        "label":
                            label,
                        "total_samples":
                            model.samples,
                    },
                )

            pending.append(
                {
                    "created_at":
                        now,
                    "sample_time":
                        utc_now_iso(),
                    "price":
                        price,
                    "features":
                        features.copy(),
                }
            )

            (
                prediction,
                confidence,
                probabilities,
            ) = model.predict(
                features
            )

            heuristic = (
                heuristic_direction(
                    values
                )
            )

            source = "MODEL"
            trade_direction = 0

            if (
                model.samples
                >= MIN_TRAINED_SAMPLES
                and confidence
                >= MODEL_ENTRY_CONFIDENCE
                and prediction != 0
            ):
                trade_direction = (
                    prediction
                )
            else:
                source = "EXPLORATION"

                if (
                    heuristic != 0
                    and random.random()
                    < EXPLORATION_RATE
                ):
                    trade_direction = (
                        heuristic
                    )

            maybe_manage_position(
                account,
                price,
                trade_direction,
            )

            maybe_open_position(
                account,
                price,
                trade_direction,
                confidence,
                model.samples,
                source,
            )

            save_account(
                account
            )

            snapshot_row = {
                "timestamp":
                    utc_now_iso(),
                "price":
                    price,
                "prediction":
                    prediction,
                "confidence":
                    confidence,
                "prob_down":
                    probabilities.get(
                        -1,
                        0.0,
                    ),
                "prob_neutral":
                    probabilities.get(
                        0,
                        0.0,
                    ),
                "prob_up":
                    probabilities.get(
                        1,
                        0.0,
                    ),
                "heuristic_direction":
                    heuristic,
                "trade_direction":
                    trade_direction,
                "decision_source":
                    source,
                "learning_samples":
                    model.samples,
                "equity_usdt":
                    account[
                        "equity_usdt"
                    ],
                "closed_trades":
                    account[
                        "closed_trades"
                    ],
                "wins":
                    account[
                        "wins"
                    ],
                "losses":
                    account[
                        "losses"
                    ],
                **values,
            }

            append_csv(
                FEATURE_LOG,
                snapshot_row,
            )

            win_rate = (
                (
                    account[
                        "wins"
                    ]
                    / account[
                        "closed_trades"
                    ]
                    * 100.0
                )
                if account[
                    "closed_trades"
                ] > 0
                else 0.0
            )

            position = (
                account.get(
                    "position"
                )
            )

            position_text = (
                "FLAT"
                if not position
                else (
                    "LONG"
                    if int(
                        position[
                            "direction"
                        ]
                    ) == 1
                    else "SHORT"
                )
            )

            print(
                f"price={price:.2f} "
                f"samples={model.samples} "
                f"pred={prediction:+d} "
                f"conf={confidence:.3f} "
                f"position={position_text} "
                f"trades={account['closed_trades']} "
                f"win={win_rate:.1f}% "
                f"equity_usdt={account['equity_usdt']:.2f}"
            )

            time.sleep(
                LOOP_SECONDS
            )

    except KeyboardInterrupt:
        save_account(
            account
        )

        print(
            "\nMicro paper learner stopped."
        )


if __name__ == "__main__":
    main()
