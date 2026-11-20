from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import math
import random
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import (
        DataLoader,
        Dataset,
    )
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Install with: "
        "python -m pip install -r "
        "requirements-ml-deep.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ML_DIR = (
    PROJECT_ROOT
    / "backend"
    / "services"
    / "ml"
)

if str(ML_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ML_DIR),
    )

from research_v3_barrier_opportunity import (
    HIGH_COST,
    MID_COST,
    TEST_START,
    VALIDATION_START,
    barrier_outcomes,
    evaluate,
    load_dataset,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v3_transformer_barrier"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
)

RESULT_FILE = OUTPUT_DIR / "threshold_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
MODEL_FILE = MODEL_DIR / "v3_transformer_barrier_research.pt"
SCALER_FILE = MODEL_DIR / "v3_transformer_barrier_scaler.joblib"

BASELINE_RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v3_barrier_opportunity"
    / "candidate_results.csv"
)

HORIZON_MINUTES = 30
HORIZON_BARS = 6
TAKE_PROFIT = 0.0050
STOP_LOSS = 0.0030

SEQUENCE_LENGTH = 24
TRAIN_STRIDE = 2
BATCH_SIZE = 512
EPOCHS = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

D_MODEL = 96
NHEAD = 4
NUM_LAYERS = 3
FEEDFORWARD_DIM = 192
DROPOUT = 0.10

RANDOM_SEED = 42

CONFIDENCE_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

TARGET_CONFIDENCE = 0.80
TARGET_WIN_RATE_PCT = 80.0
TARGET_TRADES_PER_DAY = 4.0
TARGET_MEDIAN_GAP_HOURS = 2.0
MIN_TARGET_TRADES = 50


def set_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


class TradeOutcomeDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        rows: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.features = features
        self.rows = rows
        self.targets = targets

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        item: int,
    ):
        row = int(
            self.rows[item]
        )

        start = (
            row
            - SEQUENCE_LENGTH
            + 1
        )

        sequence = self.features[
            start:row + 1
        ]

        return (
            torch.from_numpy(
                sequence
            ).float(),
            torch.from_numpy(
                self.targets[row]
            ).float(),
        )


class BarrierTransformer(nn.Module):
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

        self.norm = nn.LayerNorm(
            D_MODEL
        )

        self.head = nn.Sequential(
            nn.Linear(
                D_MODEL,
                64,
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                64,
                2,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            self.input_projection(x)
            + self.position
        )

        x = self.encoder(x)

        last = self.norm(
            x[:, -1, :]
        )

        return self.head(
            last
        )


def valid_sequence_rows(
    rows: np.ndarray,
    valid: np.ndarray,
    stride: int = 1,
) -> np.ndarray:
    rows = rows[
        rows
        >= (
            SEQUENCE_LENGTH
            - 1
        )
    ]

    rows = rows[
        valid[
            rows
        ]
    ]

    if stride > 1:
        rows = rows[
            ::stride
        ]

    return rows.astype(
        np.int64
    )


def split_prevalidation(
    rows: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    first = int(
        len(rows)
        * 0.70
    )

    second = int(
        len(rows)
        * 0.85
    )

    return (
        rows[:first],
        rows[first:second],
        rows[second:],
    )


def loss_pos_weight(
    targets: np.ndarray,
    rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    y = targets[
        rows
    ]

    positives = y.sum(
        axis=0
    )

    negatives = (
        len(y)
        - positives
    )

    weights = (
        negatives
        / np.maximum(
            positives,
            1.0,
        )
    )

    print(
        "Training label rates: "
        f"LONG={positives[0] / len(y) * 100:.2f}% "
        f"SHORT={positives[1] / len(y) * 100:.2f}%"
    )

    print(
        "Positive loss weights: "
        f"LONG={weights[0]:.3f} "
        f"SHORT={weights[1]:.3f}"
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def dataset_loss(
    model: nn.Module,
    features: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> float:
    dataset = TradeOutcomeDataset(
        features,
        rows,
        targets,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    values = []

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            logits = model(
                x.to(
                    device,
                    non_blocking=True,
                )
            )

            loss = criterion(
                logits,
                y.to(
                    device,
                    non_blocking=True,
                ),
            )

            values.append(
                float(
                    loss.cpu()
                )
            )

    return float(
        np.mean(values)
    )


def train_model(
    model: nn.Module,
    features: np.ndarray,
    train_rows: np.ndarray,
    early_stop_rows: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> nn.Module:
    dataset = TradeOutcomeDataset(
        features,
        train_rows[
            ::TRAIN_STRIDE
        ],
        targets,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    model = model.to(
        device
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=
                loss_pos_weight(
                    targets,
                    train_rows,
                    device,
                )
        )
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=
                WEIGHT_DECAY,
        )
    )

    best_state = None
    best_loss = math.inf

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        losses = []

        for x, y in loader:
            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = criterion(
                logits,
                y,
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

        early_loss = dataset_loss(
            model,
            features,
            early_stop_rows,
            targets,
            device,
        )

        print(
            f"epoch {epoch}/{EPOCHS} "
            f"train={np.mean(losses):.5f} "
            f"early={early_loss:.5f}"
        )

        if early_loss < best_loss:
            best_loss = early_loss
            best_state = deepcopy(
                model.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "Transformer produced no "
            "valid state."
        )

    model.load_state_dict(
        best_state
    )

    print(
        "Best early-stop loss: "
        f"{best_loss:.5f}"
    )

    return model


def raw_probabilities(
    model: nn.Module,
    features: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    dataset = TradeOutcomeDataset(
        features,
        rows,
        targets,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    values = []

    model.eval()

    with torch.no_grad():
        for x, _ in loader:
            logits = model(
                x.to(
                    device,
                    non_blocking=True,
                )
            )

            values.append(
                torch.sigmoid(
                    logits
                )
                .cpu()
                .numpy()
            )

    return np.concatenate(
        values,
        axis=0,
    )


def fit_calibrators(
    raw: np.ndarray,
    y: np.ndarray,
) -> tuple[
    list[IsotonicRegression | None],
    list[float],
]:
    calibrators = []
    briers = []

    for side in range(2):
        probability = raw[
            :,
            side
        ]

        target = y[
            :,
            side
        ]

        calibrator = None

        if (
            len(
                np.unique(
                    probability
                )
            )
            >= 2
            and len(
                np.unique(
                    target
                )
            )
            >= 2
        ):
            calibrator = (
                IsotonicRegression(
                    out_of_bounds=
                        "clip",
                )
            )

            calibrator.fit(
                probability,
                target,
            )

            calibrated = (
                calibrator.transform(
                    probability
                )
            )
        else:
            calibrated = probability

        brier = float(
            brier_score_loss(
                target,
                calibrated,
            )
        )

        calibrators.append(
            calibrator
        )

        briers.append(
            brier
        )

    return (
        calibrators,
        briers,
    )


def calibrate(
    raw: np.ndarray,
    calibrators:
        list[
            IsotonicRegression
            | None
        ],
) -> np.ndarray:
    result = raw.copy()

    for side in range(2):
        calibrator = (
            calibrators[
                side
            ]
        )

        if calibrator is not None:
            result[
                :,
                side
            ] = (
                calibrator.transform(
                    raw[
                        :,
                        side
                    ]
                )
            )

    return result


def strict_target(
    threshold: float,
    metrics: dict,
) -> bool:
    return bool(
        threshold
        >= TARGET_CONFIDENCE
        and metrics[
            "trades"
        ]
        >= MIN_TARGET_TRADES
        and metrics[
            "win_rate_mid_pct"
        ]
        >= TARGET_WIN_RATE_PCT
        and metrics[
            "trades_per_day"
        ]
        >= TARGET_TRADES_PER_DAY
        and metrics[
            "median_gap_hours"
        ]
        <= TARGET_MEDIAN_GAP_HOURS
        and metrics[
            "net_mid_sum_pct"
        ]
        > 0.0
        and metrics[
            "net_high_sum_pct"
        ]
        > 0.0
    )


def main() -> None:
    set_seed()

    print(
        "TRADEMIND V3 - TRANSFORMER "
        "BARRIER TRADE-SUCCESS BENCHMARK"
    )

    print()
    print(
        "Fixed research configuration:"
    )

    print(
        "  execution: 5m"
    )

    print(
        "  context: closed 10m + 1h"
    )

    print(
        f"  max hold: "
        f"{HORIZON_MINUTES}m"
    )

    print(
        f"  TP: "
        f"{TAKE_PROFIT * 100:.2f}%"
    )

    print(
        f"  SL: "
        f"{STOP_LOSS * 100:.2f}%"
    )

    print(
        "  output 1: probability LONG "
        "finishes profitable"
    )

    print(
        "  output 2: probability SHORT "
        "finishes profitable"
    )

    print()
    print(
        "Strict target remains:"
    )

    print(
        "  confidence >= 80%, "
        "win rate >= 80%, "
        ">= 4 trades/day, "
        "positive mid/high-cost net"
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
            + torch.cuda.get_device_name(
                0
            )
        )

    (
        frame,
        feature_columns,
    ) = load_dataset()

    outcomes = barrier_outcomes(
        frame,
        TAKE_PROFIT,
        STOP_LOSS,
        HORIZON_BARS,
    )

    valid = outcomes[
        "valid"
    ]

    targets = np.column_stack(
        [
            (
                outcomes[
                    "long_gross"
                ]
                - MID_COST
                > 0.0
            ).astype(
                np.float32
            ),
            (
                outcomes[
                    "short_gross"
                ]
                - MID_COST
                > 0.0
            ).astype(
                np.float32
            ),
        ]
    )

    prevalidation_rows = (
        valid_sequence_rows(
            np.flatnonzero(
                (
                    frame[
                        "open_time"
                    ]
                    < VALIDATION_START
                ).to_numpy()
            ),
            valid,
        )
    )

    validation_rows = (
        valid_sequence_rows(
            np.flatnonzero(
                (
                    (
                        frame[
                            "open_time"
                        ]
                        >= VALIDATION_START
                    )
                    & (
                        frame[
                            "open_time"
                        ]
                        < TEST_START
                    )
                ).to_numpy()
            ),
            valid,
        )
    )

    test_rows = (
        valid_sequence_rows(
            np.flatnonzero(
                (
                    frame[
                        "open_time"
                    ]
                    >= TEST_START
                ).to_numpy()
            ),
            valid,
        )
    )

    (
        train_rows,
        early_stop_rows,
        calibration_rows,
    ) = split_prevalidation(
        prevalidation_rows
    )

    print()
    print(
        f"Train sequences: "
        f"{len(train_rows):,}"
    )

    print(
        f"Early-stop sequences: "
        f"{len(early_stop_rows):,}"
    )

    print(
        f"Calibration sequences: "
        f"{len(calibration_rows):,}"
    )

    print(
        f"Validation sequences: "
        f"{len(validation_rows):,}"
    )

    print(
        f"Test sequences: "
        f"{len(test_rows):,}"
    )

    scaler = StandardScaler()

    scaler.fit(
        frame.iloc[
            train_rows
        ][feature_columns]
    )

    scaled = (
        scaler.transform(
            frame[
                feature_columns
            ]
        )
        .astype(
            np.float32
        )
    )

    model = BarrierTransformer(
        len(feature_columns)
    )

    print()
    print(
        "Training Transformer..."
    )

    model = train_model(
        model,
        scaled,
        train_rows,
        early_stop_rows,
        targets,
        device,
    )

    calibration_raw = (
        raw_probabilities(
            model,
            scaled,
            calibration_rows,
            targets,
            device,
        )
    )

    (
        calibrators,
        calibration_briers,
    ) = fit_calibrators(
        calibration_raw,
        targets[
            calibration_rows
        ],
    )

    print()
    print(
        "Calibration Brier: "
        f"LONG={calibration_briers[0]:.4f} "
        f"SHORT={calibration_briers[1]:.4f}"
    )

    result_rows = []

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
            frame[
                "open_time"
            ].iloc[-1]
            + pd.Timedelta(
                minutes=5
            ),
        ),
    ]:
        raw = raw_probabilities(
            model,
            scaled,
            rows,
            targets,
            device,
        )

        probabilities = calibrate(
            raw,
            calibrators,
        )

        for threshold in (
            CONFIDENCE_THRESHOLDS
        ):
            metrics = evaluate(
                frame,
                rows,
                probabilities[
                    :,
                    0
                ],
                probabilities[
                    :,
                    1
                ],
                outcomes,
                threshold,
                period_start,
                period_end,
            )

            result_rows.append(
                {
                    "model":
                        "Transformer",
                    "period":
                        period,
                    "confidence_threshold":
                        threshold,
                    "horizon_minutes":
                        HORIZON_MINUTES,
                    "take_profit_pct":
                        TAKE_PROFIT
                        * 100.0,
                    "stop_loss_pct":
                        STOP_LOSS
                        * 100.0,
                    **metrics,
                    "strict_target_met":
                        strict_target(
                            threshold,
                            metrics,
                        ),
                }
            )

    results = pd.DataFrame(
        result_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULT_FILE,
        index=False,
    )

    validation = (
        results[
            results[
                "period"
            ] == "validation"
        ]
        .copy()
    )

    validation[
        "score"
    ] = (
        validation[
            "win_rate_mid_pct"
        ]
        + validation[
            "trades_per_day"
        ]
        * 6.0
        + validation[
            "avg_net_mid_pct"
        ]
        * 25.0
        + np.where(
            validation[
                "net_high_sum_pct"
            ] > 0.0,
            12.0,
            -12.0,
        )
        - np.maximum(
            0.0,
            TARGET_WIN_RATE_PCT
            - validation[
                "win_rate_mid_pct"
            ],
        )
        * 0.8
        - np.maximum(
            0.0,
            TARGET_TRADES_PER_DAY
            - validation[
                "trades_per_day"
            ],
        )
        * 7.0
    )

    best = (
        validation.sort_values(
            [
                "strict_target_met",
                "score",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]
    )

    test_match = (
        results[
            (
                results[
                    "period"
                ]
                == "test"
            )
            & (
                results[
                    "confidence_threshold"
                ]
                == best[
                    "confidence_threshold"
                ]
            )
        ]
        .iloc[0]
    )

    baseline = None

    if BASELINE_RESULT_FILE.exists():
        baseline_frame = pd.read_csv(
            BASELINE_RESULT_FILE
        )

        baseline_match = baseline_frame[
            (
                baseline_frame[
                    "period"
                ]
                == "test"
            )
            & (
                baseline_frame[
                    "horizon_minutes"
                ]
                == HORIZON_MINUTES
            )
            & (
                np.isclose(
                    baseline_frame[
                        "take_profit_pct"
                    ],
                    TAKE_PROFIT
                    * 100.0,
                )
            )
            & (
                np.isclose(
                    baseline_frame[
                        "stop_loss_pct"
                    ],
                    STOP_LOSS
                    * 100.0,
                )
            )
            & (
                baseline_frame[
                    "confidence_threshold"
                ]
                == best[
                    "confidence_threshold"
                ]
            )
        ]

        if not baseline_match.empty:
            row = (
                baseline_match.iloc[0]
            )

            baseline = {
                "trades":
                    int(
                        row[
                            "trades"
                        ]
                    ),
                "trades_per_day":
                    float(
                        row[
                            "trades_per_day"
                        ]
                    ),
                "win_rate_mid_pct":
                    float(
                        row[
                            "win_rate_mid_pct"
                        ]
                    ),
                "net_mid_sum_pct":
                    float(
                        row[
                            "net_mid_sum_pct"
                        ]
                    ),
                "net_high_sum_pct":
                    float(
                        row[
                            "net_high_sum_pct"
                        ]
                    ),
            }

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "feature_columns":
                feature_columns,
            "sequence_length":
                SEQUENCE_LENGTH,
            "horizon_minutes":
                HORIZON_MINUTES,
            "take_profit":
                TAKE_PROFIT,
            "stop_loss":
                STOP_LOSS,
            "d_model":
                D_MODEL,
            "nhead":
                NHEAD,
            "num_layers":
                NUM_LAYERS,
            "feedforward_dim":
                FEEDFORWARD_DIM,
            "research_only":
                True,
        },
        MODEL_FILE,
    )

    joblib.dump(
        {
            "scaler":
                scaler,
            "feature_columns":
                feature_columns,
            "sequence_length":
                SEQUENCE_LENGTH,
            "calibrators":
                calibrators,
            "research_only":
                True,
        },
        SCALER_FILE,
    )

    summary = {
        "research_only": True,
        "architecture": (
            "5m TransformerEncoder dual-side "
            "profitable-trade classifier with "
            "closed 10m/1h context"
        ),
        "configuration": {
            "horizon_minutes":
                HORIZON_MINUTES,
            "take_profit_pct":
                TAKE_PROFIT
                * 100.0,
            "stop_loss_pct":
                STOP_LOSS
                * 100.0,
            "sequence_length":
                SEQUENCE_LENGTH,
        },
        "calibration_brier": {
            "long":
                calibration_briers[
                    0
                ],
            "short":
                calibration_briers[
                    1
                ],
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
            if key != "score"
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
            in test_match.to_dict().items()
        },
        "xgboost_same_config_test":
            baseline,
        "warning": (
            "Research only. The Transformer is "
            "not deployed unless later test and "
            "new forward paper evidence satisfy "
            "the frequency, win-rate and cost "
            "requirements."
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
    print(
        "=" * 140
    )

    print(
        "TRANSFORMER BARRIER RESULT"
    )

    print(
        "=" * 140
    )

    print()
    print(
        "Validation-selected threshold:"
    )

    print(
        best.to_string()
    )

    print()
    print(
        "Same threshold on later test:"
    )

    print(
        test_match.to_string()
    )

    if baseline is not None:
        print()
        print(
            "XGBoost same-config later test:"
        )

        print(
            json.dumps(
                baseline,
                indent=2,
            )
        )

    print()
    print(
        "Target met on validation: "
        f"{bool(best['strict_target_met'])}"
    )

    print(
        "Target met on later test: "
        f"{bool(test_match['strict_target_met'])}"
    )

    print()
    print("Saved:")
    print(RESULT_FILE)
    print(SUMMARY_FILE)
    print(MODEL_FILE)
    print(SCALER_FILE)


if __name__ == "__main__":
    main()
