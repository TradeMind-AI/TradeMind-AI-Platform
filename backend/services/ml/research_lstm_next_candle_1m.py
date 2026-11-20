from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

DATA_PIPELINE_DIR = (
    PROJECT_ROOT
    / "backend"
    / "services"
    / "data_pipeline"
)

if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(DATA_PIPELINE_DIR),
    )

from feature_engineering_1m import (
    create_candle_features,
    create_momentum_features,
    create_return_features,
    create_time_features,
    create_trend_features,
    create_volatility_features,
    create_volume_features,
)


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "btcusdt_1m.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lstm_next_candle_1m"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "validation_results.csv"
)

CONFIDENCE_FILE = (
    OUTPUT_DIR
    / "confidence_results.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "lstm_next_candle_1m_research.pt"
)

SCALER_FILE = (
    MODEL_DIR
    / "lstm_next_candle_1m_scaler.joblib"
)

RESEARCH_START = pd.Timestamp(
    "2024-01-01",
    tz="UTC",
)

VALIDATION_START = pd.Timestamp(
    "2026-01-01",
    tz="UTC",
)

KNOWN_RECENT_START = pd.Timestamp(
    "2026-07-01",
    tz="UTC",
)

SEQUENCE_LENGTH = 60
TRAIN_STRIDE = 2
EVAL_STRIDE = 1
EPOCHS = 5
BATCH_SIZE = 1024
LEARNING_RATE = 8e-4
RANDOM_SEED = 42

CONFIDENCE_THRESHOLDS = [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]

FEATURE_COLUMNS = [
    "return_1m",
    "return_3m",
    "return_5m",
    "return_10m",
    "return_20m",
    "return_30m",
    "return_60m",
    "log_return_1m",
    "close_to_sma_20",
    "close_to_sma_60",
    "close_to_sma_240",
    "rsi_14",
    "macd",
    "macd_histogram",
    "rolling_volatility_10",
    "rolling_volatility_30",
    "rolling_volatility_60",
    "atr_14",
    "bollinger_width_20",
    "range_pct",
    "body_pct",
    "close_location",
    "volume_change",
    "relative_volume_30",
    "relative_volume_60",
    "trade_count_change",
    "taker_buy_ratio",
    "taker_buy_imbalance",
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]


def set_seed() -> None:
    np.random.seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


def build_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    df = raw.copy()

    df = create_return_features(df)
    df = create_trend_features(df)
    df = create_momentum_features(df)
    df = create_volatility_features(df)
    df = create_candle_features(df)
    df = create_volume_features(df)
    df = create_time_features(df)

    df["future_return_1m"] = (
        df["close"].shift(-1)
        / df["close"]
        - 1.0
    )

    df["target_up"] = np.where(
        df["future_return_1m"] > 0,
        1,
        0,
    )

    return (
        df.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=(
                FEATURE_COLUMNS
                + ["future_return_1m"]
            )
        )
        .reset_index(drop=True)
    )


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        row_indices: np.ndarray,
        sequence_length: int,
    ) -> None:
        self.features = features
        self.targets = targets
        self.row_indices = row_indices
        self.sequence_length = (
            sequence_length
        )

    def __len__(self) -> int:
        return len(
            self.row_indices
        )

    def __getitem__(
        self,
        item: int,
    ):
        end = int(
            self.row_indices[item]
        )

        start = (
            end
            - self.sequence_length
            + 1
        )

        sequence = (
            self.features[
                start:end + 1
            ]
        )

        target = (
            self.targets[end]
        )

        return (
            torch.from_numpy(
                sequence
            ).float(),
            torch.tensor(
                target,
                dtype=torch.float32,
            ),
        )


class LSTMModel(nn.Module):
    def __init__(
        self,
        feature_count: int,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_count,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.20,
        )

        self.norm = nn.LayerNorm(
            64
        )

        self.head = nn.Sequential(
            nn.Linear(
                64,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):
        output, _ = self.lstm(
            x
        )

        last = self.norm(
            output[:, -1, :]
        )

        return self.head(
            last
        ).squeeze(-1)


def valid_sequence_indices(
    mask: np.ndarray,
    stride: int,
) -> np.ndarray:
    candidates = np.flatnonzero(
        mask
    )

    candidates = candidates[
        candidates
        >= (
            SEQUENCE_LENGTH - 1
        )
    ]

    if stride > 1:
        candidates = candidates[
            ::stride
        ]

    return candidates


def train_model(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    device: torch.device,
) -> nn.Module:
    dataset = SequenceDataset(
        features,
        targets,
        train_indices,
        SEQUENCE_LENGTH,
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

    model = model.to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=2e-4,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    for epoch in range(
        EPOCHS
    ):
        model.train()
        losses = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                x_batch
            )

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
                    loss.detach()
                    .cpu()
                )
            )

        print(
            f"epoch {epoch + 1}/{EPOCHS} "
            f"loss={np.mean(losses):.5f}"
        )

    return model


def predict_probabilities(
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
        SEQUENCE_LENGTH,
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
                    device
                )
            )

            probability = (
                torch.sigmoid(
                    logits
                )
                .detach()
                .cpu()
                .numpy()
            )

            probabilities.append(
                probability
            )

    return np.concatenate(
        probabilities
    )


def overall_metrics(
    y_true: np.ndarray,
    p_up: np.ndarray,
) -> dict:
    prediction = (
        p_up >= 0.5
    ).astype(int)

    return {
        "model": "LSTM",
        "samples": len(y_true),
        "accuracy_pct":
            accuracy_score(
                y_true,
                prediction,
            )
            * 100,
        "auc":
            roc_auc_score(
                y_true,
                p_up,
            ),
        "up_precision_pct":
            precision_score(
                y_true,
                prediction,
                pos_label=1,
                zero_division=0,
            )
            * 100,
        "down_precision_pct":
            precision_score(
                y_true,
                prediction,
                pos_label=0,
                zero_division=0,
            )
            * 100,
        "up_recall_pct":
            recall_score(
                y_true,
                prediction,
                pos_label=1,
                zero_division=0,
            )
            * 100,
        "down_recall_pct":
            recall_score(
                y_true,
                prediction,
                pos_label=0,
                zero_division=0,
            )
            * 100,
    }


def confidence_metrics(
    y_true: np.ndarray,
    p_up: np.ndarray,
) -> pd.DataFrame:
    prediction = (
        p_up >= 0.5
    ).astype(int)

    confidence = np.maximum(
        p_up,
        1.0 - p_up,
    )

    rows = []

    for threshold in (
        CONFIDENCE_THRESHOLDS
    ):
        selected = (
            confidence
            >= threshold
        )

        count = int(
            selected.sum()
        )

        if count == 0:
            accuracy = np.nan
            coverage = 0.0
            up_precision = np.nan
            down_precision = np.nan
        else:
            selected_y = (
                y_true[
                    selected
                ]
            )

            selected_prediction = (
                prediction[
                    selected
                ]
            )

            accuracy = (
                accuracy_score(
                    selected_y,
                    selected_prediction,
                )
                * 100
            )

            coverage = (
                count
                / len(y_true)
                * 100
            )

            up_precision = (
                precision_score(
                    selected_y,
                    selected_prediction,
                    pos_label=1,
                    zero_division=0,
                )
                * 100
            )

            down_precision = (
                precision_score(
                    selected_y,
                    selected_prediction,
                    pos_label=0,
                    zero_division=0,
                )
                * 100
            )

        rows.append(
            {
                "confidence_threshold":
                    threshold,
                "predictions":
                    count,
                "coverage_pct":
                    coverage,
                "accuracy_pct":
                    accuracy,
                "up_precision_pct":
                    up_precision,
                "down_precision_pct":
                    down_precision,
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    set_seed()

    print(
        "TRADEMIND - LSTM "
        "NEXT-CANDLE 1M RESEARCH"
    )

    print()
    print(
        "Goal: compare LSTM fairly against "
        "the existing GRU/TCN/XGBoost "
        "next-candle experiment."
    )

    print(
        "IMPORTANT: confidence is not treated "
        "as success by itself. We measure "
        "accuracy and coverage at each "
        "confidence threshold."
    )

    print(
        "2026-07-01 onward remains excluded "
        "for a later one-shot check."
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

    raw = pd.read_csv(
        INPUT_FILE
    )

    raw["open_time"] = pd.to_datetime(
        raw["open_time"],
        utc=True,
        errors="coerce",
    )

    raw["close_time"] = pd.to_datetime(
        raw["close_time"],
        utc=True,
        errors="coerce",
    )

    raw = (
        raw.dropna(
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

    warmup_start = (
        RESEARCH_START
        - pd.Timedelta(
            minutes=300
        )
    )

    raw = raw[
        (
            raw["open_time"]
            >= warmup_start
        )
        & (
            raw["open_time"]
            < KNOWN_RECENT_START
        )
    ].copy()

    print(
        f"Raw rows loaded: "
        f"{len(raw):,}"
    )

    df = build_features(
        raw
    )

    df = df[
        df["open_time"]
        >= RESEARCH_START
    ].reset_index(drop=True)

    print(
        f"ML-ready rows: "
        f"{len(df):,}"
    )

    print(
        f"Period: "
        f"{df['open_time'].min()} "
        f"-> "
        f"{df['open_time'].max()}"
    )

    train_mask = (
        df["open_time"]
        < VALIDATION_START
    ).to_numpy()

    validation_mask = (
        (
            df["open_time"]
            >= VALIDATION_START
        )
        & (
            df["open_time"]
            < KNOWN_RECENT_START
        )
    ).to_numpy()

    train_indices = (
        valid_sequence_indices(
            train_mask,
            TRAIN_STRIDE,
        )
    )

    validation_indices = (
        valid_sequence_indices(
            validation_mask,
            EVAL_STRIDE,
        )
    )

    print(
        f"Train sequences: "
        f"{len(train_indices):,}"
    )

    print(
        f"Validation sequences: "
        f"{len(validation_indices):,}"
    )

    scaler = StandardScaler()

    scaler.fit(
        df.loc[
            train_mask,
            FEATURE_COLUMNS,
        ]
    )

    features = (
        scaler.transform(
            df[
                FEATURE_COLUMNS
            ]
        )
        .astype(
            np.float32
        )
    )

    targets = (
        df["target_up"]
        .to_numpy(
            dtype=np.float32
        )
    )

    model = LSTMModel(
        len(
            FEATURE_COLUMNS
        )
    )

    print()
    print(
        "Training LSTM..."
    )

    model = train_model(
        model,
        features,
        targets,
        train_indices,
        device,
    )

    print()
    print(
        "Evaluating chronological "
        "validation window..."
    )

    p_up = predict_probabilities(
        model,
        features,
        targets,
        validation_indices,
        device,
    )

    y_true = (
        targets[
            validation_indices
        ].astype(int)
    )

    result = pd.DataFrame(
        [
            overall_metrics(
                y_true,
                p_up,
            )
        ]
    )

    confidence = (
        confidence_metrics(
            y_true,
            p_up,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        RESULT_FILE,
        index=False,
    )

    confidence.to_csv(
        CONFIDENCE_FILE,
        index=False,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "feature_columns":
                FEATURE_COLUMNS,
            "sequence_length":
                SEQUENCE_LENGTH,
            "validation_start":
                str(
                    VALIDATION_START
                ),
            "known_recent_start":
                str(
                    KNOWN_RECENT_START
                ),
            "research_only":
                True,
        },
        MODEL_FILE,
    )

    import joblib

    joblib.dump(
        {
            "scaler":
                scaler,
            "feature_columns":
                FEATURE_COLUMNS,
            "sequence_length":
                SEQUENCE_LENGTH,
        },
        SCALER_FILE,
    )

    print()
    print("=" * 110)
    print(
        "LSTM VALIDATION RESULT"
    )
    print("=" * 110)

    print(
        result.to_string(
            index=False,
            formatters={
                "accuracy_pct":
                    lambda value:
                        f"{value:.3f}%",
                "auc":
                    lambda value:
                        f"{value:.4f}",
                "up_precision_pct":
                    lambda value:
                        f"{value:.3f}%",
                "down_precision_pct":
                    lambda value:
                        f"{value:.3f}%",
                "up_recall_pct":
                    lambda value:
                        f"{value:.3f}%",
                "down_recall_pct":
                    lambda value:
                        f"{value:.3f}%",
            },
        )
    )

    print()
    print("=" * 110)
    print(
        "CONFIDENCE / COVERAGE CHECK"
    )
    print("=" * 110)

    print(
        confidence.to_string(
            index=False,
            formatters={
                "confidence_threshold":
                    lambda value:
                        f"{value:.2f}",
                "coverage_pct":
                    lambda value:
                        f"{value:.3f}%",
                "accuracy_pct":
                    lambda value:
                        (
                            "n/a"
                            if pd.isna(
                                value
                            )
                            else f"{value:.3f}%"
                        ),
                "up_precision_pct":
                    lambda value:
                        (
                            "n/a"
                            if pd.isna(
                                value
                            )
                            else f"{value:.3f}%"
                        ),
                "down_precision_pct":
                    lambda value:
                        (
                            "n/a"
                            if pd.isna(
                                value
                            )
                            else f"{value:.3f}%"
                        ),
            },
        )
    )

    print()
    print(
        "Saved:"
    )

    print(
        RESULT_FILE
    )

    print(
        CONFIDENCE_FILE
    )

    print(
        MODEL_FILE
    )

    print()
    print(
        "Do NOT connect this LSTM to live "
        "paper execution unless it shows a "
        "real improvement in unseen accuracy "
        "with useful coverage."
    )


if __name__ == "__main__":
    main()
