from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Install with: "
        "python -m pip install -r requirements-ml-deep.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformer_trade_selector_10m"
)

RESULT_FILE = OUTPUT_DIR / "selector_search.csv"
TRADE_FILE = OUTPUT_DIR / "selected_trades.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"

RESEARCH_START = pd.Timestamp("2023-03-26", tz="UTC")
VALIDATION_START = pd.Timestamp("2025-01-01", tz="UTC")
TEST_START = pd.Timestamp("2025-10-01", tz="UTC")

HORIZON_BARS = 3
HORIZON_MINUTES = 30
TARGET_MOVE = 0.0020

MID_COST = 0.0014
HIGH_COST = 0.0027

SEQUENCE_LENGTH = 36
TRAIN_STRIDE = 2
BATCH_SIZE = 512
EPOCHS = 6
LEARNING_RATE = 3e-4
RANDOM_SEED = 42

D_MODEL = 96
NHEAD = 4
NUM_LAYERS = 2
FEEDFORWARD_DIM = 192
DROPOUT = 0.10

PROPOSER_CONFIDENCE_THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
]

SELECTOR_THRESHOLDS = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
]

TARGET_SELECTOR_CONFIDENCE = 0.80
TARGET_WIN_RATE_PCT = 80.0
TARGET_TRADES_PER_DAY = 8.0
TARGET_MEDIAN_GAP_HOURS = 2.0
MIN_TARGET_TRADES = 50

NON_FEATURE_COLUMNS = {
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
    "future_return_10m",
    "target",
}


def set_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def make_direction_target(
    future_return: pd.Series,
) -> np.ndarray:
    return np.select(
        [
            future_return < -TARGET_MOVE,
            future_return > TARGET_MOVE,
        ],
        [
            0,
            2,
        ],
        default=1,
    ).astype(np.int64)


def build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )


def realized_return(
    df: pd.DataFrame,
    row_index: int,
    direction: int,
) -> float:
    entry_index = row_index + 1
    exit_index = row_index + HORIZON_BARS

    entry = float(
        df.iloc[entry_index]["open"]
    )
    exit_price = float(
        df.iloc[exit_index]["close"]
    )

    if direction == 2:
        return (
            exit_price / entry - 1.0
        )

    return (
        entry / exit_price - 1.0
    )


def max_drawdown_pct(
    net_returns: np.ndarray,
) -> float:
    if len(net_returns) == 0:
        return 0.0

    equity = np.cumprod(
        1.0 + net_returns
    )
    peak = np.maximum.accumulate(
        equity
    )
    dd = equity / peak - 1.0
    return float(dd.min() * 100.0)


class SelectorDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        row_indices: np.ndarray,
        directions: np.ndarray,
        proposer_confidence: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.features = features
        self.row_indices = row_indices
        self.directions = directions
        self.proposer_confidence = proposer_confidence
        self.targets = targets

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, item: int):
        end = int(
            self.row_indices[item]
        )
        start = end - SEQUENCE_LENGTH + 1

        sequence = self.features[
            start:end + 1
        ]

        context = np.array(
            [
                1.0 if self.directions[item] == 2 else -1.0,
                float(self.proposer_confidence[item]),
            ],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(sequence).float(),
            torch.from_numpy(context).float(),
            torch.tensor(
                float(self.targets[item]),
                dtype=torch.float32,
            ),
        )


class TradeSelectorTransformer(nn.Module):
    def __init__(
        self,
        feature_count: int,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(
            feature_count,
            D_MODEL,
        )

        self.position = nn.Parameter(
            torch.zeros(
                1,
                SEQUENCE_LENGTH,
                D_MODEL,
            )
        )

        nn.init.normal_(
            self.position,
            mean=0.0,
            std=0.02,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=NUM_LAYERS,
        )

        self.norm = nn.LayerNorm(D_MODEL)

        self.head = nn.Sequential(
            nn.Linear(D_MODEL + 2, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        encoded = (
            self.input_projection(sequence)
            + self.position
        )

        encoded = self.encoder(encoded)

        last = self.norm(
            encoded[:, -1, :]
        )

        combined = torch.cat(
            [
                last,
                context,
            ],
            dim=1,
        )

        return self.head(
            combined
        ).squeeze(-1)


def train_selector(
    model: nn.Module,
    dataset: SelectorDataset,
    device: torch.device,
) -> nn.Module:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = model.to(device)

    targets = np.asarray(
        dataset.targets,
        dtype=float,
    )

    positive = float(
        targets.sum()
    )
    negative = float(
        len(targets) - positive
    )

    pos_weight = torch.tensor(
        [
            negative
            / max(positive, 1.0)
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    print(
        "Selector labels: "
        f"wins={int(positive):,} "
        f"losses={int(negative):,} "
        "win-rate="
        f"{positive / max(len(targets), 1) * 100:.2f}%"
    )

    best_state = None
    best_loss = np.inf

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()
        losses = []

        for (
            sequence,
            context,
            target,
        ) in loader:
            sequence = sequence.to(
                device,
                non_blocking=True,
            )
            context = context.to(
                device,
                non_blocking=True,
            )
            target = target.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                sequence,
                context,
            )

            loss = criterion(
                logits,
                target,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        epoch_loss = float(
            np.mean(losses)
        )

        print(
            f"selector epoch {epoch}/{EPOCHS} "
            f"loss={epoch_loss:.5f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = deepcopy(
                model.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "No selector model state produced."
        )

    model.load_state_dict(
        best_state
    )

    return model


def selector_probabilities(
    model: nn.Module,
    dataset: SelectorDataset,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    values = []

    model.eval()

    with torch.no_grad():
        for (
            sequence,
            context,
            _,
        ) in loader:
            logits = model(
                sequence.to(
                    device,
                    non_blocking=True,
                ),
                context.to(
                    device,
                    non_blocking=True,
                ),
            )

            values.append(
                torch.sigmoid(
                    logits
                )
                .cpu()
                .numpy()
            )

    return np.concatenate(values)


def oof_proposals(
    df: pd.DataFrame,
    feature_columns: list[str],
    direction_targets: np.ndarray,
    train_rows: np.ndarray,
) -> pd.DataFrame:
    boundaries = np.linspace(
        0,
        len(train_rows),
        5,
        dtype=int,
    )

    frames = []

    for fold in range(
        1,
        4,
    ):
        fit_rows = train_rows[
            :boundaries[fold]
        ]

        eval_rows = train_rows[
            boundaries[fold]:
            boundaries[fold + 1]
        ]

        fit_rows = fit_rows[
            fit_rows
            + HORIZON_BARS
            < len(df)
        ]

        eval_rows = eval_rows[
            (
                eval_rows
                >= SEQUENCE_LENGTH - 1
            )
            & (
                eval_rows
                + HORIZON_BARS
                < len(df)
            )
        ]

        model = build_xgb()

        model.fit(
            df.iloc[
                fit_rows
            ][feature_columns],
            direction_targets[
                fit_rows
            ],
        )

        probabilities = (
            model.predict_proba(
                df.iloc[
                    eval_rows
                ][feature_columns]
            )
        )

        direction = probabilities.argmax(
            axis=1
        )

        confidence = probabilities.max(
            axis=1
        )

        keep = (
            direction != 1
        )

        frame = pd.DataFrame(
            {
                "row_index":
                    eval_rows[keep],
                "direction":
                    direction[keep],
                "proposer_confidence":
                    confidence[keep],
            }
        )

        frames.append(frame)

        print(
            f"OOF fold {fold}: "
            f"{len(frame):,} directional proposals"
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def make_selector_targets(
    df: pd.DataFrame,
    proposals: pd.DataFrame,
) -> np.ndarray:
    labels = []

    for row in proposals.itertuples(
        index=False
    ):
        gross = realized_return(
            df,
            int(row.row_index),
            int(row.direction),
        )

        labels.append(
            1.0
            if (
                gross - MID_COST
            ) > 0.0
            else 0.0
        )

    return np.asarray(
        labels,
        dtype=np.float32,
    )


def proposal_frame(
    df: pd.DataFrame,
    model: XGBClassifier,
    feature_columns: list[str],
    rows: np.ndarray,
) -> pd.DataFrame:
    probabilities = model.predict_proba(
        df.iloc[
            rows
        ][feature_columns]
    )

    direction = probabilities.argmax(
        axis=1
    )

    confidence = probabilities.max(
        axis=1
    )

    keep = (
        direction != 1
    )

    return pd.DataFrame(
        {
            "row_index":
                rows[keep],
            "direction":
                direction[keep],
            "proposer_confidence":
                confidence[keep],
        }
    )


def frequency_metrics(
    signal_times: pd.Series,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> tuple[float, float, float]:
    span_days = max(
        (
            period_end
            - period_start
        ).total_seconds()
        / 86400.0,
        1e-9,
    )

    trades_per_day = (
        len(signal_times)
        / span_days
    )

    if len(signal_times) < 2:
        return (
            trades_per_day,
            float("inf"),
            float("inf"),
        )

    ordered = pd.to_datetime(
        signal_times,
        utc=True,
    ).sort_values()

    gaps = (
        ordered.diff()
        .dropna()
        .dt.total_seconds()
        / 3600.0
    )

    return (
        trades_per_day,
        float(gaps.median()),
        float(gaps.quantile(0.90)),
    )


def search(
    df: pd.DataFrame,
    proposals: pd.DataFrame,
    selector_probability: np.ndarray,
    period: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> tuple[list[dict], list[dict]]:
    result_rows = []
    trade_rows = []

    proposals = proposals.copy()
    proposals[
        "selector_probability"
    ] = selector_probability

    for proposer_threshold in (
        PROPOSER_CONFIDENCE_THRESHOLDS
    ):
        for selector_threshold in (
            SELECTOR_THRESHOLDS
        ):
            candidate = proposals[
                (
                    proposals[
                        "proposer_confidence"
                    ]
                    >= proposer_threshold
                )
                & (
                    proposals[
                        "selector_probability"
                    ]
                    >= selector_threshold
                )
            ].copy()

            accepted = []
            next_available = -1

            for row in candidate.itertuples(
                index=False
            ):
                index = int(
                    row.row_index
                )

                if index < next_available:
                    continue

                gross = realized_return(
                    df,
                    index,
                    int(
                        row.direction
                    ),
                )

                accepted.append(
                    {
                        "period": period,
                        "row_index": index,
                        "signal_time":
                            df.iloc[
                                index
                            ]["open_time"],
                        "direction":
                            "UP"
                            if int(
                                row.direction
                            ) == 2
                            else "DOWN",
                        "proposer_confidence":
                            float(
                                row.proposer_confidence
                            ),
                        "selector_probability":
                            float(
                                row.selector_probability
                            ),
                        "gross_return":
                            gross,
                        "mid_net_return":
                            gross - MID_COST,
                        "high_net_return":
                            gross - HIGH_COST,
                        "proposer_threshold":
                            proposer_threshold,
                        "selector_threshold":
                            selector_threshold,
                    }
                )

                next_available = (
                    index
                    + HORIZON_BARS
                )

            trades = pd.DataFrame(
                accepted
            )

            if trades.empty:
                trade_count = 0
                win_rate = 0.0
                avg_mid = 0.0
                net_mid = 0.0
                net_high = 0.0
                drawdown = 0.0
                mean_selector = 0.0
                trades_per_day = 0.0
                median_gap = float("inf")
                p90_gap = float("inf")
            else:
                mid = trades[
                    "mid_net_return"
                ].to_numpy(
                    dtype=float
                )

                high = trades[
                    "high_net_return"
                ].to_numpy(
                    dtype=float
                )

                trade_count = len(trades)

                win_rate = float(
                    (mid > 0).mean()
                    * 100.0
                )

                avg_mid = float(
                    mid.mean()
                    * 100.0
                )

                net_mid = float(
                    mid.sum()
                    * 100.0
                )

                net_high = float(
                    high.sum()
                    * 100.0
                )

                drawdown = (
                    max_drawdown_pct(mid)
                )

                mean_selector = float(
                    trades[
                        "selector_probability"
                    ].mean()
                    * 100.0
                )

                (
                    trades_per_day,
                    median_gap,
                    p90_gap,
                ) = frequency_metrics(
                    trades[
                        "signal_time"
                    ],
                    period_start,
                    period_end,
                )

                trade_rows.extend(
                    trades.to_dict(
                        "records"
                    )
                )

            target_met = bool(
                selector_threshold
                >= TARGET_SELECTOR_CONFIDENCE
                and trade_count
                >= MIN_TARGET_TRADES
                and win_rate
                >= TARGET_WIN_RATE_PCT
                and trades_per_day
                >= TARGET_TRADES_PER_DAY
                and median_gap
                <= TARGET_MEDIAN_GAP_HOURS
                and net_mid > 0.0
                and net_high > 0.0
            )

            result_rows.append(
                {
                    "period": period,
                    "proposer_threshold":
                        proposer_threshold,
                    "selector_threshold":
                        selector_threshold,
                    "trades": trade_count,
                    "trades_per_day":
                        trades_per_day,
                    "median_gap_hours":
                        median_gap,
                    "p90_gap_hours":
                        p90_gap,
                    "mean_selector_confidence_pct":
                        mean_selector,
                    "win_rate_mid_pct":
                        win_rate,
                    "avg_net_mid_pct":
                        avg_mid,
                    "net_mid_sum_pct":
                        net_mid,
                    "net_high_sum_pct":
                        net_high,
                    "max_drawdown_mid_pct":
                        drawdown,
                    "target_met":
                        target_met,
                }
            )

    return (
        result_rows,
        trade_rows,
    )


def main() -> None:
    set_seed()

    print(
        "TRADEMIND - FREQUENCY-AWARE "
        "TRANSFORMER TRADE SELECTOR"
    )

    print()
    print(
        "Design:"
    )
    print(
        "  XGBoost proposes frequent direction trades"
    )
    print(
        "  Transformer predicts whether each proposal "
        "will be profitable after mid-cost execution"
    )
    print()
    print(
        "Research target (not forced):"
    )
    print(
        "  selector confidence >= 80%"
    )
    print(
        "  win rate >= 80%"
    )
    print(
        f"  trades/day >= {TARGET_TRADES_PER_DAY:.1f}"
    )
    print(
        "  median time between trades <= "
        f"{TARGET_MEDIAN_GAP_HOURS:.1f}h"
    )
    print(
        f"  at least {MIN_TARGET_TRADES} trades"
    )
    print(
        "  positive net result at mid and high costs"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"PyTorch device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU: "
            + torch.cuda.get_device_name(0)
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=["open_time"]
        )
        .sort_values(
            "open_time"
        )
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    df = df[
        df["open_time"]
        >= RESEARCH_START
    ].reset_index(drop=True)

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    future_return = (
        df["close"].shift(
            -HORIZON_BARS
        )
        / df["close"]
        - 1.0
    )

    direction_targets = (
        make_direction_target(
            future_return.fillna(0.0)
        )
    )

    valid_target = future_return.notna()

    train_mask = (
        (
            df["open_time"]
            < VALIDATION_START
        )
        & valid_target
    ).to_numpy()

    validation_mask = (
        (
            df["open_time"]
            >= VALIDATION_START
        )
        & (
            df["open_time"]
            < TEST_START
        )
        & valid_target
    ).to_numpy()

    test_mask = (
        (
            df["open_time"]
            >= TEST_START
        )
        & valid_target
    ).to_numpy()

    train_rows = np.flatnonzero(
        train_mask
    )

    validation_rows = np.flatnonzero(
        validation_mask
    )

    test_rows = np.flatnonzero(
        test_mask
    )

    print()
    print(
        f"Rows: {len(df):,}"
    )
    print(
        f"Features: {len(feature_columns)}"
    )
    print(
        f"Train rows: {len(train_rows):,}"
    )
    print(
        "Validation rows: "
        f"{len(validation_rows):,}"
    )
    print(
        f"Test rows: {len(test_rows):,}"
    )

    print()
    print(
        "Building leakage-safe OOF direction proposals..."
    )

    oof = oof_proposals(
        df,
        feature_columns,
        direction_targets,
        train_rows,
    )

    oof = oof[
        oof["row_index"]
        >= SEQUENCE_LENGTH - 1
    ].reset_index(drop=True)

    oof_targets = make_selector_targets(
        df,
        oof,
    )

    scaler = StandardScaler()

    scaler.fit(
        df.iloc[
            train_rows
        ][feature_columns]
    )

    scaled = (
        scaler.transform(
            df[
                feature_columns
            ]
        )
        .astype(
            np.float32
        )
    )

    selector_dataset = SelectorDataset(
        scaled,
        oof[
            "row_index"
        ].to_numpy(
            dtype=np.int64
        ),
        oof[
            "direction"
        ].to_numpy(
            dtype=np.int64
        ),
        oof[
            "proposer_confidence"
        ].to_numpy(
            dtype=np.float32
        ),
        oof_targets,
    )

    selector = TradeSelectorTransformer(
        len(feature_columns)
    )

    print()
    print(
        "Training Transformer trade-success selector..."
    )

    selector = train_selector(
        selector,
        selector_dataset,
        device,
    )

    print()
    print(
        "Training final direction proposer..."
    )

    proposer = build_xgb()

    proposer.fit(
        df.iloc[
            train_rows
        ][feature_columns],
        direction_targets[
            train_rows
        ],
    )

    all_result_rows = []
    all_trade_rows = []

    for (
        period,
        rows,
        period_start,
        period_end,
    ) in [
        (
            "validation",
            validation_rows,
            VALIDATION_START,
            TEST_START,
        ),
        (
            "test",
            test_rows,
            TEST_START,
            df[
                "open_time"
            ].iloc[-1]
            + pd.Timedelta(minutes=10),
        ),
    ]:
        proposals = proposal_frame(
            df,
            proposer,
            feature_columns,
            rows,
        )

        proposals = proposals[
            proposals["row_index"]
            >= SEQUENCE_LENGTH - 1
        ].reset_index(drop=True)

        dummy_targets = make_selector_targets(
            df,
            proposals,
        )

        dataset = SelectorDataset(
            scaled,
            proposals[
                "row_index"
            ].to_numpy(
                dtype=np.int64
            ),
            proposals[
                "direction"
            ].to_numpy(
                dtype=np.int64
            ),
            proposals[
                "proposer_confidence"
            ].to_numpy(
                dtype=np.float32
            ),
            dummy_targets,
        )

        selector_probability = (
            selector_probabilities(
                selector,
                dataset,
                device,
            )
        )

        (
            result_rows,
            trade_rows,
        ) = search(
            df,
            proposals,
            selector_probability,
            period,
            period_start,
            period_end,
        )

        all_result_rows.extend(
            result_rows
        )

        all_trade_rows.extend(
            trade_rows
        )

    results = pd.DataFrame(
        all_result_rows
    )

    trades = pd.DataFrame(
        all_trade_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULT_FILE,
        index=False,
    )

    trades.to_csv(
        TRADE_FILE,
        index=False,
    )

    validation_ranked = (
        results[
            results[
                "period"
            ] == "validation"
        ]
        .copy()
    )

    validation_ranked[
        "frequency_shortfall"
    ] = np.maximum(
        0.0,
        TARGET_TRADES_PER_DAY
        - validation_ranked[
            "trades_per_day"
        ],
    )

    validation_ranked[
        "win_shortfall"
    ] = np.maximum(
        0.0,
        TARGET_WIN_RATE_PCT
        - validation_ranked[
            "win_rate_mid_pct"
        ],
    )

    validation_ranked[
        "gap_penalty"
    ] = np.maximum(
        0.0,
        validation_ranked[
            "median_gap_hours"
        ].replace(
            np.inf,
            999.0,
        )
        - TARGET_MEDIAN_GAP_HOURS,
    )

    validation_ranked[
        "research_score"
    ] = (
        validation_ranked[
            "win_rate_mid_pct"
        ]
        + validation_ranked[
            "trades_per_day"
        ] * 3.0
        + validation_ranked[
            "avg_net_mid_pct"
        ] * 10.0
        - validation_ranked[
            "frequency_shortfall"
        ] * 5.0
        - validation_ranked[
            "win_shortfall"
        ] * 0.5
        - validation_ranked[
            "gap_penalty"
        ] * 2.0
    )

    best = validation_ranked.sort_values(
        [
            "target_met",
            "research_score",
        ],
        ascending=[
            False,
            False,
        ],
    ).iloc[0]

    matching_test = results[
        (
            results["period"]
            == "test"
        )
        & (
            results[
                "proposer_threshold"
            ]
            == best[
                "proposer_threshold"
            ]
        )
        & (
            results[
                "selector_threshold"
            ]
            == best[
                "selector_threshold"
            ]
        )
    ].iloc[0]

    summary = {
        "research_only": True,
        "architecture": (
            "XGBoost direction proposer + "
            "Transformer trade-success selector"
        ),
        "horizon_minutes":
            HORIZON_MINUTES,
        "target_move_pct":
            TARGET_MOVE * 100.0,
        "targets": {
            "selector_confidence":
                TARGET_SELECTOR_CONFIDENCE,
            "win_rate_pct":
                TARGET_WIN_RATE_PCT,
            "trades_per_day":
                TARGET_TRADES_PER_DAY,
            "median_gap_hours":
                TARGET_MEDIAN_GAP_HOURS,
            "minimum_trades":
                MIN_TARGET_TRADES,
        },
        "validation_selected": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (bool, np.bool_),
                )
                else float(value)
                if isinstance(
                    value,
                    (float, np.floating),
                )
                else int(value)
                if isinstance(
                    value,
                    (int, np.integer),
                )
                else value
            )
            for key, value
            in best.to_dict().items()
            if key not in {
                "frequency_shortfall",
                "win_shortfall",
                "gap_penalty",
                "research_score",
            }
        },
        "test_result": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (bool, np.bool_),
                )
                else float(value)
                if isinstance(
                    value,
                    (float, np.floating),
                )
                else int(value)
                if isinstance(
                    value,
                    (int, np.integer),
                )
                else value
            )
            for key, value
            in matching_test.to_dict().items()
        },
        "warning": (
            "This is research only. Any promising result "
            "must be frozen and forward paper tested before "
            "it can replace an existing candidate."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 145)
    print(
        "FREQUENCY-AWARE SELECTOR RESULT"
    )
    print("=" * 145)
    print()
    print(
        "Validation-selected candidate:"
    )
    print(
        best.to_string()
    )
    print()
    print(
        "Same thresholds on later test:"
    )
    print(
        matching_test.to_string()
    )
    print()
    print(
        "Target met on validation: "
        f"{bool(best['target_met'])}"
    )
    print(
        "Target met on test: "
        f"{bool(matching_test['target_met'])}"
    )
    print()
    print("Saved:")
    print(RESULT_FILE)
    print(TRADE_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
