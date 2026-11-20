from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import math
import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
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
    / "transformer_10m_cost_aware"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
)

RESULT_FILE = OUTPUT_DIR / "threshold_results.csv"
PREDICTION_FILE = OUTPUT_DIR / "test_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
MODEL_FILE = MODEL_DIR / "transformer_10m_cost_aware_research.pt"
SCALER_FILE = MODEL_DIR / "transformer_10m_cost_aware_scaler.joblib"

RESEARCH_START = pd.Timestamp("2023-03-26", tz="UTC")
VALIDATION_START = pd.Timestamp("2025-01-01", tz="UTC")
TEST_START = pd.Timestamp("2025-10-01", tz="UTC")

HORIZON_BARS = 6
HORIZON_MINUTES = 60
TARGET_THRESHOLD = 0.0040

SEQUENCE_LENGTH = 48
TRAIN_STRIDE = 2
BATCH_SIZE = 512
EPOCHS = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

D_MODEL = 96
NHEAD = 4
NUM_LAYERS = 3
FEEDFORWARD_DIM = 192
DROPOUT = 0.10

CONFIDENCE_THRESHOLDS = [
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
MIN_TARGET_TRADES = 30

COSTS = {
    "low": 0.0006,
    "mid": 0.0014,
    "high": 0.0027,
}

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

CLASS_NAMES = np.array(
    [
        "DOWN",
        "NEUTRAL",
        "UP",
    ],
    dtype=object,
)


def set_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def make_target(
    future_return: pd.Series,
) -> np.ndarray:
    return np.select(
        [
            future_return < -TARGET_THRESHOLD,
            future_return > TARGET_THRESHOLD,
        ],
        [
            0,
            2,
        ],
        default=1,
    ).astype(np.int64)


def max_drawdown_pct(
    net_returns: np.ndarray,
) -> float:
    if len(net_returns) == 0:
        return 0.0

    equity = np.cumprod(
        1.0 + net_returns
    )

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity / running_max
        - 1.0
    )

    return float(
        drawdown.min() * 100.0
    )


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        row_indices: np.ndarray,
    ) -> None:
        self.features = features
        self.targets = targets
        self.row_indices = row_indices

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(
        self,
        item: int,
    ):
        end = int(
            self.row_indices[item]
        )

        start = (
            end
            - SEQUENCE_LENGTH
            + 1
        )

        sequence = self.features[
            start:end + 1
        ]

        return (
            torch.from_numpy(
                sequence
            ).float(),
            torch.tensor(
                int(
                    self.targets[end]
                ),
                dtype=torch.long,
            ),
        )


class TransformerClassifier(nn.Module):
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
            nn.Dropout(DROPOUT),
            nn.Linear(
                64,
                3,
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

        pooled = self.norm(
            x[:, -1, :]
        )

        return self.head(
            pooled
        )


def valid_indices(
    mask: np.ndarray,
    total_rows: int,
    stride: int,
) -> np.ndarray:
    indices = np.flatnonzero(mask)

    indices = indices[
        indices
        >= (
            SEQUENCE_LENGTH - 1
        )
    ]

    indices = indices[
        indices
        + HORIZON_BARS
        < total_rows
    ]

    if stride > 1:
        indices = indices[
            ::stride
        ]

    return indices.astype(
        np.int64
    )


def class_weights(
    targets: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    counts = np.bincount(
        targets[indices],
        minlength=3,
    ).astype(np.float64)

    weights = (
        counts.sum()
        / np.maximum(
            counts,
            1.0,
        )
    )

    weights = (
        weights
        / weights.mean()
    )

    print(
        "Train class counts: "
        + ", ".join(
            f"{CLASS_NAMES[i]}={int(counts[i]):,}"
            for i in range(3)
        )
    )

    print(
        "Loss weights: "
        + ", ".join(
            f"{CLASS_NAMES[i]}={weights[i]:.3f}"
            for i in range(3)
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def predict_transformer(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    dataset = SequenceDataset(
        features,
        targets,
        indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    probabilities = []

    model.eval()

    with torch.no_grad():
        for x_batch, _ in loader:
            logits = model(
                x_batch.to(
                    device,
                    non_blocking=True,
                )
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            probabilities.append(
                probs.cpu().numpy()
            )

    return np.concatenate(
        probabilities,
        axis=0,
    )


def validation_loss(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    dataset = SequenceDataset(
        features,
        targets,
        indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    criterion = nn.CrossEntropyLoss()

    losses = []

    model.eval()

    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(
                x_batch.to(
                    device,
                    non_blocking=True,
                )
            )

            loss = criterion(
                logits,
                y_batch.to(
                    device,
                    non_blocking=True,
                ),
            )

            losses.append(
                float(
                    loss.cpu()
                )
            )

    return float(
        np.mean(losses)
    )


def train_transformer(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
) -> nn.Module:
    dataset = SequenceDataset(
        features,
        targets,
        train_indices,
    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(
            targets,
            train_indices,
            device,
        ),
        label_smoothing=0.02,
    )

    best_state = None
    best_validation_loss = math.inf

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        losses = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(
                device,
                non_blocking=True,
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x_batch)

            loss = criterion(
                logits,
                y_batch,
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

        val_loss = validation_loss(
            model,
            features,
            targets,
            validation_indices,
            device,
        )

        print(
            f"epoch {epoch}/{EPOCHS} "
            f"train_loss={np.mean(losses):.5f} "
            f"val_loss={val_loss:.5f}"
        )

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_state = deepcopy(
                model.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "Transformer training did not "
            "produce a valid model state."
        )

    model.load_state_dict(
        best_state
    )

    print(
        "Best validation loss: "
        f"{best_validation_loss:.5f}"
    )

    return model


def build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.0,
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


def simulate_trades(
    df: pd.DataFrame,
    indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    predictions = probabilities.argmax(
        axis=1
    )

    confidence = probabilities.max(
        axis=1
    )

    rows = []

    next_available = -1

    for position, row_index in enumerate(
        indices
    ):
        prediction = int(
            predictions[position]
        )

        if prediction == 1:
            continue

        if float(
            confidence[position]
        ) < threshold:
            continue

        index = int(row_index)

        if index < next_available:
            continue

        entry_index = index + 1
        exit_index = index + HORIZON_BARS

        if exit_index >= len(df):
            continue

        entry_price = float(
            df.iloc[
                entry_index
            ]["open"]
        )

        exit_price = float(
            df.iloc[
                exit_index
            ]["close"]
        )

        if prediction == 2:
            gross = (
                exit_price
                / entry_price
                - 1.0
            )
            side = "UP"
        else:
            gross = (
                entry_price
                / exit_price
                - 1.0
            )
            side = "DOWN"

        rows.append(
            {
                "row_index": index,
                "signal_time":
                    df.iloc[
                        index
                    ]["open_time"],
                "side": side,
                "confidence":
                    float(
                        confidence[position]
                    ),
                "entry_price":
                    entry_price,
                "exit_price":
                    exit_price,
                "gross_return":
                    gross,
            }
        )

        next_available = (
            index
            + HORIZON_BARS
        )

    return pd.DataFrame(
        rows
    )


def threshold_metrics(
    model_name: str,
    period: str,
    df: pd.DataFrame,
    indices: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict]:
    y_true = make_target(
        (
            df["close"].shift(
                -HORIZON_BARS
            )
            / df["close"]
            - 1.0
        )
    )[indices]

    predictions = probabilities.argmax(
        axis=1
    )

    overall_accuracy = (
        accuracy_score(
            y_true,
            predictions,
        )
        * 100.0
    )

    rows = []

    for threshold in CONFIDENCE_THRESHOLDS:
        trades = simulate_trades(
            df,
            indices,
            probabilities,
            threshold,
        )

        trade_count = len(trades)

        if trade_count > 0:
            gross = trades[
                "gross_return"
            ].to_numpy(
                dtype=float
            )

            mid = gross - COSTS["mid"]
            high = gross - COSTS["high"]

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
                max_drawdown_pct(
                    mid
                )
            )

            mean_confidence = float(
                trades[
                    "confidence"
                ].mean()
                * 100.0
            )
        else:
            win_rate = 0.0
            avg_mid = 0.0
            net_mid = 0.0
            net_high = 0.0
            drawdown = 0.0
            mean_confidence = 0.0

        coverage = (
            trade_count
            / len(indices)
            * 100.0
            if len(indices)
            else 0.0
        )

        target_met = bool(
            threshold >= TARGET_CONFIDENCE
            and trade_count >= MIN_TARGET_TRADES
            and win_rate >= TARGET_WIN_RATE_PCT
            and net_mid > 0.0
            and net_high > 0.0
        )

        rows.append(
            {
                "model": model_name,
                "period": period,
                "confidence_threshold":
                    threshold,
                "overall_accuracy_pct":
                    overall_accuracy,
                "trades": trade_count,
                "coverage_pct": coverage,
                "mean_selected_confidence_pct":
                    mean_confidence,
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
                "target_80_conf_80_win_met":
                    target_met,
            }
        )

    return rows


def main() -> None:
    set_seed()

    print(
        "TRADEMIND - 10M COST-AWARE "
        "TRANSFORMER BENCHMARK"
    )
    print()
    print(
        "Research target (not forced):"
    )
    print(
        "  selected confidence >= 80%"
    )
    print(
        "  mid-cost win rate >= 80%"
    )
    print(
        f"  minimum trades >= {MIN_TARGET_TRADES}"
    )
    print(
        "  positive mid/high-cost net result"
    )
    print()
    print(
        "If the data does not support that target, "
        "the script reports failure rather than "
        "inventing confidence."
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

    valid_target = future_return.notna()

    targets = make_target(
        future_return.fillna(0.0)
    )

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

    train_indices = valid_indices(
        train_mask,
        len(df),
        TRAIN_STRIDE,
    )

    validation_indices = valid_indices(
        validation_mask,
        len(df),
        1,
    )

    test_indices = valid_indices(
        test_mask,
        len(df),
        1,
    )

    print()
    print(
        f"Rows: {len(df):,}"
    )
    print(
        f"Features: {len(feature_columns)}"
    )
    print(
        f"Train sequences: {len(train_indices):,}"
    )
    print(
        "Validation sequences: "
        f"{len(validation_indices):,}"
    )
    print(
        f"Test sequences: {len(test_indices):,}"
    )

    scaler = StandardScaler()

    scaler.fit(
        df.loc[
            train_mask,
            feature_columns,
        ]
    )

    scaled_features = (
        scaler.transform(
            df[
                feature_columns
            ]
        )
        .astype(
            np.float32
        )
    )

    transformer = TransformerClassifier(
        len(feature_columns)
    )

    print()
    print(
        "Training Transformer..."
    )

    transformer = train_transformer(
        transformer,
        scaled_features,
        targets,
        train_indices,
        validation_indices,
        device,
    )

    print()
    print(
        "Training comparable XGBoost baseline..."
    )

    xgb = build_xgb()

    xgb.fit(
        df.loc[
            train_mask,
            feature_columns,
        ],
        targets[
            train_mask
        ],
    )

    print()
    print(
        "Generating validation/test probabilities..."
    )

    transformer_validation = (
        predict_transformer(
            transformer,
            scaled_features,
            targets,
            validation_indices,
            device,
        )
    )

    transformer_test = (
        predict_transformer(
            transformer,
            scaled_features,
            targets,
            test_indices,
            device,
        )
    )

    xgb_validation = xgb.predict_proba(
        df.iloc[
            validation_indices
        ][feature_columns]
    )

    xgb_test = xgb.predict_proba(
        df.iloc[
            test_indices
        ][feature_columns]
    )

    result_rows = []

    for (
        model_name,
        period,
        indices,
        probabilities,
    ) in [
        (
            "Transformer",
            "validation",
            validation_indices,
            transformer_validation,
        ),
        (
            "Transformer",
            "test",
            test_indices,
            transformer_test,
        ),
        (
            "XGBoost",
            "validation",
            validation_indices,
            xgb_validation,
        ),
        (
            "XGBoost",
            "test",
            test_indices,
            xgb_test,
        ),
    ]:
        result_rows.extend(
            threshold_metrics(
                model_name,
                period,
                df,
                indices,
                probabilities,
            )
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

    test_prediction_frame = (
        df.iloc[
            test_indices
        ][
            [
                "open_time",
                "open",
                "close",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    for prefix, probabilities in [
        (
            "transformer",
            transformer_test,
        ),
        (
            "xgb",
            xgb_test,
        ),
    ]:
        test_prediction_frame[
            f"{prefix}_prediction"
        ] = CLASS_NAMES[
            probabilities.argmax(
                axis=1
            )
        ]

        test_prediction_frame[
            f"{prefix}_confidence"
        ] = probabilities.max(
            axis=1
        )

        test_prediction_frame[
            f"{prefix}_p_down"
        ] = probabilities[:, 0]

        test_prediction_frame[
            f"{prefix}_p_neutral"
        ] = probabilities[:, 1]

        test_prediction_frame[
            f"{prefix}_p_up"
        ] = probabilities[:, 2]

    test_prediction_frame.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    torch.save(
        {
            "model_state_dict":
                transformer.state_dict(),
            "feature_columns":
                feature_columns,
            "sequence_length":
                SEQUENCE_LENGTH,
            "d_model":
                D_MODEL,
            "nhead":
                NHEAD,
            "num_layers":
                NUM_LAYERS,
            "feedforward_dim":
                FEEDFORWARD_DIM,
            "target_threshold":
                TARGET_THRESHOLD,
            "horizon_bars":
                HORIZON_BARS,
            "research_only":
                True,
        },
        MODEL_FILE,
    )

    joblib.dump(
        {
            "scaler": scaler,
            "feature_columns":
                feature_columns,
            "sequence_length":
                SEQUENCE_LENGTH,
        },
        SCALER_FILE,
    )

    transformer_test_80 = results[
        (
            results["model"]
            == "Transformer"
        )
        & (
            results["period"]
            == "test"
        )
        & (
            results[
                "confidence_threshold"
            ]
            == TARGET_CONFIDENCE
        )
    ].iloc[0]

    feasible = results[
        (
            results["model"]
            == "Transformer"
        )
        & (
            results["period"]
            == "test"
        )
        & (
            results[
                "target_80_conf_80_win_met"
            ]
        )
    ]

    summary = {
        "research_only": True,
        "model": "10m TransformerEncoder",
        "sequence_length":
            SEQUENCE_LENGTH,
        "horizon_minutes":
            HORIZON_MINUTES,
        "target_move_pct":
            TARGET_THRESHOLD * 100.0,
        "target_confidence":
            TARGET_CONFIDENCE,
        "target_win_rate_pct":
            TARGET_WIN_RATE_PCT,
        "minimum_target_trades":
            MIN_TARGET_TRADES,
        "transformer_test_at_080": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (np.bool_, bool),
                )
                else float(value)
                if isinstance(
                    value,
                    (np.floating, float),
                )
                else int(value)
                if isinstance(
                    value,
                    (np.integer, int),
                )
                else value
            )
            for key, value
            in transformer_test_80.to_dict().items()
        },
        "target_met_on_test":
            bool(
                not feasible.empty
            ),
        "warning": (
            "This is historical research. Any winning "
            "configuration becomes a new candidate and "
            "must be frozen and forward paper tested."
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
    print("=" * 150)
    print(
        "TRANSFORMER VS XGBOOST "
        "COST-AWARE RESULTS"
    )
    print("=" * 150)

    print(
        results[
            results[
                "confidence_threshold"
            ]
            >= 0.70
        ].to_string(
            index=False,
        )
    )

    print()
    print(
        "Transformer test result at "
        "confidence >= 0.80:"
    )
    print(
        transformer_test_80.to_string()
    )

    print()

    if feasible.empty:
        print(
            "80% confidence / 80% win-rate "
            "target NOT proven."
        )
    else:
        print(
            "A historical Transformer row met "
            "the 80/80 research target."
        )
        print(
            "Treat it only as a NEW candidate; "
            "freeze it before forward paper testing."
        )

    print()
    print("Saved:")
    print(RESULT_FILE)
    print(PREDICTION_FILE)
    print(SUMMARY_FILE)
    print(MODEL_FILE)
    print(SCALER_FILE)


if __name__ == "__main__":
    main()
