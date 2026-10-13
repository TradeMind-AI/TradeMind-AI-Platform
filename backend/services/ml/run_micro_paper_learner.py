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

TRADE_SELECTOR_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "micro_trade_selector.joblib"
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

TRADE_OUTCOME_LOG = (
    OUTPUT_DIR
    / "trade_outcome_learning.csv"
)

EVALUATION_LOG = (
    OUTPUT_DIR
    / "prequential_evaluation.csv"
)

EVALUATION_FILE = (
    OUTPUT_DIR
    / "evaluation_metrics.json"
)

LOOP_SECONDS = 5
LABEL_HORIZON_SECONDS = 30
LABEL_THRESHOLD = 0.0005

# Use non-overlapping learning examples. The bot may
# make a decision every 5s, but it only trains on one
# independent 30s sample at a time.
TRAIN_SAMPLE_INTERVAL_SECONDS = 30
MODEL_VERSION = 2

STARTING_EQUITY = 10000.0
POSITION_FRACTION = 0.05
FEE_RATE = 0.0005
TAKE_PROFIT_PCT = 0.0020
STOP_LOSS_PCT = 0.0012

# Trailing stop activates only after the trade first
# moves in our favour. It then follows the best price
# seen and exits if price gives back too much.
TRAILING_STOP_ACTIVATION_PCT = 0.0010
TRAILING_STOP_DISTANCE_PCT = 0.0008

MAX_HOLD_SECONDS = 60

MODEL_ENTRY_CONFIDENCE = 0.55
EXPLORATION_RATE = 0.05
MIN_TRAINED_SAMPLES = 100

TRADE_SELECTOR_MIN_SAMPLES = 50
TRADE_SELECTOR_THRESHOLD = 0.55

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
            alpha=0.001,
            penalty="l2",
            average=True,
            random_state=42,
        )
        self.scaler_ready = False
        self.model_ready = False
        self.samples = 0
        self.class_counts = {
            -1: 0,
            0: 0,
            1: 0,
        }

        if MODEL_FILE.exists():
            try:
                saved = joblib.load(
                    MODEL_FILE
                )
                if int(
                    saved.get(
                        "model_version",
                        1,
                    )
                ) != MODEL_VERSION:
                    raise ValueError(
                        "Old online model version"
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

                saved_counts = saved.get(
                    "class_counts",
                    {},
                )

                self.class_counts = {
                    -1: int(
                        saved_counts.get(
                            -1,
                            saved_counts.get(
                                "-1",
                                0,
                            ),
                        )
                    ),
                    0: int(
                        saved_counts.get(
                            0,
                            saved_counts.get(
                                "0",
                                0,
                            ),
                        )
                    ),
                    1: int(
                        saved_counts.get(
                            1,
                            saved_counts.get(
                                "1",
                                0,
                            ),
                        )
                    ),
                }
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

        total_seen = max(
            1,
            sum(
                self.class_counts.values()
            ),
        )

        class_seen = max(
            1,
            self.class_counts.get(
                int(label),
                0,
            ),
        )

        # Mild inverse-frequency weighting reduces
        # UP/DOWN/NEUTRAL class bias without allowing
        # rare classes to dominate the online learner.
        sample_weight = float(
            np.clip(
                np.sqrt(
                    total_seen
                    / (
                        3.0
                        * class_seen
                    )
                ),
                0.5,
                3.0,
            )
        )

        if not self.model_ready:
            self.model.partial_fit(
                transformed,
                np.array(
                    [label]
                ),
                classes=CLASSES,
                sample_weight=np.array(
                    [sample_weight]
                ),
            )
            self.model_ready = True
        else:
            self.model.partial_fit(
                transformed,
                np.array(
                    [label]
                ),
                sample_weight=np.array(
                    [sample_weight]
                ),
            )

        self.class_counts[
            int(label)
        ] += 1

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
                "model_version":
                    MODEL_VERSION,
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
                "class_counts":
                    self.class_counts,
                "feature_names":
                    FEATURE_NAMES,
                "label_horizon_seconds":
                    LABEL_HORIZON_SECONDS,
                "label_threshold":
                    LABEL_THRESHOLD,
            },
            MODEL_FILE,
        )


class TradeOutcomeSelector:
    """
    Online meta-model that learns whether a proposed
    paper trade is likely to finish net-profitable.

    It is intentionally separate from the direction
    predictor. The direction model learns market
    movement; this model learns execution quality from
    completed good and bad paper trades.
    """

    def __init__(
        self,
    ) -> None:
        self.scaler = StandardScaler()
        self.model = SGDClassifier(
            loss="log_loss",
            alpha=0.001,
            penalty="l2",
            average=True,
            random_state=84,
        )
        self.scaler_ready = False
        self.model_ready = False
        self.samples = 0
        self.profitable = 0
        self.unprofitable = 0

        if TRADE_SELECTOR_FILE.exists():
            try:
                saved = joblib.load(
                    TRADE_SELECTOR_FILE
                )
                self.scaler = saved["scaler"]
                self.model = saved["model"]
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
                self.profitable = int(
                    saved.get(
                        "profitable",
                        0,
                    )
                )
                self.unprofitable = int(
                    saved.get(
                        "unprofitable",
                        0,
                    )
                )
            except Exception:
                pass

    def predict(
        self,
        features: np.ndarray,
    ) -> float:
        if (
            not self.scaler_ready
            or not self.model_ready
        ):
            return 0.5

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

        class_map = {
            int(label): float(prob)
            for label, prob in zip(
                self.model.classes_,
                probabilities,
            )
        }

        return float(
            class_map.get(
                1,
                0.5,
            )
        )

    def update(
        self,
        features: np.ndarray,
        profitable: bool,
        net_return: float,
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

        label = int(
            profitable
        )

        # Larger wins/losses carry somewhat more
        # information, but cap the influence so one
        # extreme trade cannot dominate the learner.
        magnitude_weight = float(
            np.clip(
                1.0
                + abs(
                    net_return
                )
                / 0.001,
                1.0,
                4.0,
            )
        )

        if not self.model_ready:
            self.model.partial_fit(
                transformed,
                np.array(
                    [label]
                ),
                classes=np.array(
                    [0, 1],
                    dtype=int,
                ),
                sample_weight=np.array(
                    [magnitude_weight]
                ),
            )
            self.model_ready = True
        else:
            self.model.partial_fit(
                transformed,
                np.array(
                    [label]
                ),
                sample_weight=np.array(
                    [magnitude_weight]
                ),
            )

        self.samples += 1

        if profitable:
            self.profitable += 1
        else:
            self.unprofitable += 1

        self.save()

    def save(
        self,
    ) -> None:
        TRADE_SELECTOR_FILE.parent.mkdir(
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
                "profitable":
                    self.profitable,
                "unprofitable":
                    self.unprofitable,
            },
            TRADE_SELECTOR_FILE,
        )


def build_trade_selector_features(
    market_features: np.ndarray,
    direction: int,
    confidence: float,
    probabilities: dict[int, float],
    entry_threshold: float,
) -> np.ndarray:
    extras = np.array(
        [
            float(direction),
            float(confidence),
            float(
                probabilities.get(
                    -1,
                    0.0,
                )
            ),
            float(
                probabilities.get(
                    0,
                    0.0,
                )
            ),
            float(
                probabilities.get(
                    1,
                    0.0,
                )
            ),
            float(
                entry_threshold
            ),
        ],
        dtype=np.float64,
    )

    return np.concatenate(
        [
            market_features.astype(
                np.float64
            ),
            extras,
        ]
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
        "trade_feedback": {
            "LONG": {
                "count": 0,
                "wins": 0,
                "net_return_sum": 0.0,
            },
            "SHORT": {
                "count": 0,
                "wins": 0,
                "net_return_sum": 0.0,
            },
        },
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


def ensure_trade_feedback(
    account: dict,
) -> dict:
    feedback = account.setdefault(
        "trade_feedback",
        {},
    )

    for side in [
        "LONG",
        "SHORT",
    ]:
        feedback.setdefault(
            side,
            {
                "count": 0,
                "wins": 0,
                "net_return_sum": 0.0,
            },
        )

    return feedback


def adaptive_entry_threshold(
    account: dict,
    direction: int,
) -> float:
    feedback = ensure_trade_feedback(
        account
    )

    side = (
        "LONG"
        if direction == 1
        else "SHORT"
    )

    stats = feedback[side]
    count = int(
        stats.get(
            "count",
            0,
        )
    )

    if count < 20:
        return MODEL_ENTRY_CONFIDENCE

    wins = int(
        stats.get(
            "wins",
            0,
        )
    )

    win_rate = wins / count
    avg_net_return = (
        float(
            stats.get(
                "net_return_sum",
                0.0,
            )
        )
        / count
    )

    penalty = 0.0

    if win_rate < 0.50:
        penalty += min(
            0.10,
            (
                0.50
                - win_rate
            )
            * 0.25,
        )

    if avg_net_return < 0:
        penalty += min(
            0.05,
            abs(
                avg_net_return
            )
            * 20.0,
        )

    return float(
        np.clip(
            MODEL_ENTRY_CONFIDENCE
            + penalty,
            MODEL_ENTRY_CONFIDENCE,
            0.75,
        )
    )


def close_position(
    account: dict,
    price: float,
    reason: str,
    trade_selector:
        TradeOutcomeSelector,
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

    feedback = ensure_trade_feedback(
        account
    )

    side_key = (
        "LONG"
        if direction == 1
        else "SHORT"
    )

    side_stats = feedback[
        side_key
    ]

    side_stats["count"] = int(
        side_stats.get(
            "count",
            0,
        )
    ) + 1

    if pnl > 0:
        side_stats["wins"] = int(
            side_stats.get(
                "wins",
                0,
            )
        ) + 1

    side_stats[
        "net_return_sum"
    ] = float(
        side_stats.get(
            "net_return_sum",
            0.0,
        )
    ) + net_return

    selector_values = position.get(
        "trade_selector_features"
    )

    if selector_values:
        selector_features = np.array(
            selector_values,
            dtype=np.float64,
        )

        trade_selector.update(
            selector_features,
            pnl > 0,
            net_return,
        )

        append_csv(
            TRADE_OUTCOME_LOG,
            {
                "timestamp":
                    utc_now_iso(),
                "side":
                    side_key,
                "entry_price":
                    entry_price,
                "exit_price":
                    price,
                "net_return_pct":
                    net_return
                    * 100.0,
                "pnl_usdt":
                    pnl,
                "profitable":
                    int(
                        pnl > 0
                    ),
                "selector_probability":
                    position.get(
                        "selector_probability",
                        0.5,
                    ),
                "selector_samples_before":
                    position.get(
                        "selector_samples",
                        0,
                    ),
                "selector_samples_after":
                    trade_selector.samples,
                "reason":
                    reason,
            },
        )

    account["position"] = None


def maybe_manage_position(
    account: dict,
    price: float,
    direction: int,
    trade_selector:
        TradeOutcomeSelector,
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

    best_price = float(
        position.get(
            "best_price",
            entry_price,
        )
    )

    trailing_active = bool(
        position.get(
            "trailing_active",
            False,
        )
    )

    if position_direction == 1:
        best_price = max(
            best_price,
            price,
        )

        best_return = (
            best_price / entry_price
            - 1.0
        )

        if (
            not trailing_active
            and best_return
            >= TRAILING_STOP_ACTIVATION_PCT
        ):
            trailing_active = True

        trailing_stop_price = (
            best_price
            * (
                1.0
                - TRAILING_STOP_DISTANCE_PCT
            )
            if trailing_active
            else 0.0
        )

        trailing_hit = (
            trailing_active
            and price
            <= trailing_stop_price
        )
    else:
        best_price = min(
            best_price,
            price,
        )

        best_return = (
            entry_price / best_price
            - 1.0
            if best_price > 0
            else 0.0
        )

        if (
            not trailing_active
            and best_return
            >= TRAILING_STOP_ACTIVATION_PCT
        ):
            trailing_active = True

        trailing_stop_price = (
            best_price
            * (
                1.0
                + TRAILING_STOP_DISTANCE_PCT
            )
            if trailing_active
            else 0.0
        )

        trailing_hit = (
            trailing_active
            and price
            >= trailing_stop_price
        )

    position["best_price"] = (
        best_price
    )
    position["trailing_active"] = (
        trailing_active
    )
    position["trailing_stop_price"] = (
        trailing_stop_price
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
            trade_selector,
        )
    elif gross_return <= (
        -STOP_LOSS_PCT
    ):
        close_position(
            account,
            price,
            "STOP_LOSS",
            trade_selector,
        )
    elif trailing_hit:
        close_position(
            account,
            price,
            "TRAILING_STOP",
            trade_selector,
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
            trade_selector,
        )
    elif held_seconds >= (
        MAX_HOLD_SECONDS
    ):
        close_position(
            account,
            price,
            "MAX_HOLD",
            trade_selector,
        )


def maybe_open_position(
    account: dict,
    price: float,
    direction: int,
    confidence: float,
    samples: int,
    source: str,
    trade_selector_features:
        np.ndarray,
    selector_probability: float,
    selector_samples: int,
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
        "best_price":
            price,
        "trailing_active":
            False,
        "trailing_stop_price":
            0.0,
        "trade_selector_features":
            trade_selector_features.tolist(),
        "selector_probability":
            selector_probability,
        "selector_samples":
            selector_samples,
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





def load_evaluation() -> dict:
    if EVALUATION_FILE.exists():
        try:
            return json.loads(
                EVALUATION_FILE.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            pass

    return {
        "total": 0,
        "correct": 0,
        "confusion": {
            "-1": {
                "-1": 0,
                "0": 0,
                "1": 0,
            },
            "0": {
                "-1": 0,
                "0": 0,
                "1": 0,
            },
            "1": {
                "-1": 0,
                "0": 0,
                "1": 0,
            },
        },
        "high_conf_total": 0,
        "high_conf_correct": 0,
        "prediction_counts": {
            "-1": 0,
            "0": 0,
            "1": 0,
        },
        "actual_counts": {
            "-1": 0,
            "0": 0,
            "1": 0,
        },
    }


def save_evaluation(
    evaluation: dict,
) -> None:
    EVALUATION_FILE.write_text(
        json.dumps(
            evaluation,
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluation_summary(
    evaluation: dict,
) -> tuple[
    float,
    float,
    float,
]:
    total = int(
        evaluation.get(
            "total",
            0,
        )
    )

    accuracy = (
        float(
            evaluation.get(
                "correct",
                0,
            )
        )
        / total
        if total > 0
        else 0.0
    )

    recalls = []

    confusion = evaluation[
        "confusion"
    ]

    for label in [
        "-1",
        "0",
        "1",
    ]:
        row_total = sum(
            int(value)
            for value in confusion[
                label
            ].values()
        )

        if row_total > 0:
            recalls.append(
                int(
                    confusion[
                        label
                    ][label]
                )
                / row_total
            )

    balanced_accuracy = (
        float(
            np.mean(
                recalls
            )
        )
        if recalls
        else 0.0
    )

    high_total = int(
        evaluation.get(
            "high_conf_total",
            0,
        )
    )

    high_accuracy = (
        float(
            evaluation.get(
                "high_conf_correct",
                0,
            )
        )
        / high_total
        if high_total > 0
        else 0.0
    )

    return (
        accuracy,
        balanced_accuracy,
        high_accuracy,
    )


def record_prequential_result(
    evaluation: dict,
    sample: dict,
    label: int,
    future_return: float,
) -> None:
    prediction = int(
        sample[
            "prediction"
        ]
    )

    confidence = float(
        sample[
            "confidence"
        ]
    )

    actual_key = str(
        label
    )

    prediction_key = str(
        prediction
    )

    evaluation["total"] = int(
        evaluation.get(
            "total",
            0,
        )
    ) + 1

    if prediction == label:
        evaluation["correct"] = int(
            evaluation.get(
                "correct",
                0,
            )
        ) + 1

    evaluation[
        "confusion"
    ][actual_key][
        prediction_key
    ] += 1

    evaluation[
        "prediction_counts"
    ][prediction_key] += 1

    evaluation[
        "actual_counts"
    ][actual_key] += 1

    if confidence >= 0.80:
        evaluation[
            "high_conf_total"
        ] += 1

        if prediction == label:
            evaluation[
                "high_conf_correct"
            ] += 1

    append_csv(
        EVALUATION_LOG,
        {
            "timestamp":
                utc_now_iso(),
            "sample_time":
                sample[
                    "sample_time"
                ],
            "prediction":
                prediction,
            "confidence":
                confidence,
            "actual_label":
                label,
            "correct":
                int(
                    prediction
                    == label
                ),
            "future_return_pct":
                future_return
                * 100.0,
            "prob_down":
                sample[
                    "probabilities"
                ].get(
                    -1,
                    0.0,
                ),
            "prob_neutral":
                sample[
                    "probabilities"
                ].get(
                    0,
                    0.0,
                ),
            "prob_up":
                sample[
                    "probabilities"
                ].get(
                    1,
                    0.0,
                ),
        },
    )

    save_evaluation(
        evaluation
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
    trade_selector = TradeOutcomeSelector()
    account = load_account()
    evaluation = load_evaluation()

    pending: deque[dict] = (
        deque()
    )

    last_training_sample_at = 0.0

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
        "Anti-overfit mode: prequential "
        "test-before-train evaluation + "
        f"{TRAIN_SAMPLE_INTERVAL_SECONDS}s "
        "non-overlapping training samples."
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

                # Evaluate the prediction that was made
                # BEFORE this future outcome was known.
                # Only after scoring it do we let the
                # model learn from the sample.
                record_prequential_result(
                    evaluation,
                    sample,
                    label,
                    future_return,
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

            (
                prediction,
                confidence,
                probabilities,
            ) = model.predict(
                features
            )

            if (
                now
                - last_training_sample_at
                >= TRAIN_SAMPLE_INTERVAL_SECONDS
            ):
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
                        "prediction":
                            prediction,
                        "confidence":
                            confidence,
                        "probabilities":
                            dict(
                                probabilities
                            ),
                    }
                )

                last_training_sample_at = (
                    now
                )

            heuristic = (
                heuristic_direction(
                    values
                )
            )

            source = "MODEL"
            trade_direction = 0

            effective_entry_threshold = (
                adaptive_entry_threshold(
                    account,
                    prediction,
                )
                if prediction != 0
                else MODEL_ENTRY_CONFIDENCE
            )

            proposed_direction = (
                prediction
                if prediction != 0
                else heuristic
            )

            selector_features = (
                build_trade_selector_features(
                    features,
                    proposed_direction,
                    confidence,
                    probabilities,
                    effective_entry_threshold,
                )
            )

            selector_probability = (
                trade_selector.predict(
                    selector_features
                )
            )

            selector_ready = (
                trade_selector.samples
                >= TRADE_SELECTOR_MIN_SAMPLES
            )

            selector_allows = (
                not selector_ready
                or selector_probability
                >= TRADE_SELECTOR_THRESHOLD
            )

            if (
                model.samples
                >= MIN_TRAINED_SAMPLES
                and confidence
                >= effective_entry_threshold
                and prediction != 0
                and selector_allows
            ):
                trade_direction = (
                    prediction
                )
                source = "MODEL+TRADE_SELECTOR"
            else:
                source = "EXPLORATION"

                # Small exploration rate is retained so
                # the selector keeps receiving examples
                # from market conditions it would
                # otherwise reject completely.
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
                trade_selector,
            )

            maybe_open_position(
                account,
                price,
                trade_direction,
                confidence,
                model.samples,
                source,
                selector_features,
                selector_probability,
                trade_selector.samples,
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
                "effective_entry_threshold":
                    effective_entry_threshold,
                "trade_selector_probability":
                    selector_probability,
                "trade_selector_samples":
                    trade_selector.samples,
                "trade_selector_ready":
                    int(
                        selector_ready
                    ),
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

            (
                model_accuracy,
                balanced_accuracy,
                high_conf_accuracy,
            ) = evaluation_summary(
                evaluation
            )

            print(
                f"price={price:.2f} "
                f"samples={model.samples} "
                f"pred={prediction:+d} "
                f"conf={confidence:.3f} "
                f"gate={effective_entry_threshold:.3f} "
                f"trade_p={selector_probability:.3f} "
                f"trade_n={trade_selector.samples} "
                f"preq_acc={model_accuracy * 100:.1f}% "
                f"bal_acc={balanced_accuracy * 100:.1f}% "
                f"80conf_acc={high_conf_accuracy * 100:.1f}% "
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
