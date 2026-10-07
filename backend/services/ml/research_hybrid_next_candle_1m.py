from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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
        "python -m pip install torch"
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
    / "hybrid_next_candle_1m"
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

OOF_FILE = (
    OUTPUT_DIR
    / "oof_probabilities.csv"
)

BUNDLE_FILE = (
    MODEL_DIR
    / "hybrid_next_candle_1m_research.joblib"
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
EPOCHS = 4
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

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

META_CONTEXT_COLUMNS = [
    "rolling_volatility_30",
    "relative_volume_30",
    "taker_buy_imbalance",
    "rsi_14",
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


class GRUModel(nn.Module):
    def __init__(
        self,
        feature_count: int,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=feature_count,
            hidden_size=48,
            num_layers=2,
            batch_first=True,
            dropout=0.15,
        )

        self.head = nn.Sequential(
            nn.Linear(
                48,
                24,
            ),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(
                24,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):
        output, _ = self.gru(x)

        return self.head(
            output[:, -1, :]
        ).squeeze(-1)


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
    ) -> None:
        super().__init__()

        padding = dilation * 2

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=padding,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(
            0.15
        )

    def forward(
        self,
        x,
    ):
        residual = x

        y = self.conv1(x)
        y = y[
            :,
            :,
            :x.size(-1),
        ]
        y = self.relu(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = y[
            :,
            :,
            :x.size(-1),
        ]
        y = self.dropout(y)

        return self.relu(
            y + residual
        )


class TCNModel(nn.Module):
    def __init__(
        self,
        feature_count: int,
    ) -> None:
        super().__init__()

        self.input_projection = (
            nn.Conv1d(
                feature_count,
                48,
                kernel_size=1,
            )
        )

        self.network = nn.Sequential(
            ResidualTCNBlock(
                48,
                dilation=1,
            ),
            ResidualTCNBlock(
                48,
                dilation=2,
            ),
            ResidualTCNBlock(
                48,
                dilation=4,
            ),
            ResidualTCNBlock(
                48,
                dilation=8,
            ),
        )

        self.head = nn.Sequential(
            nn.Linear(
                48,
                24,
            ),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(
                24,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):
        x = x.transpose(
            1,
            2,
        )

        x = self.input_projection(
            x
        )

        x = self.network(x)

        x = x[:, :, -1]

        return self.head(
            x
        ).squeeze(-1)


def build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )


def valid_sequence_indices(
    df: pd.DataFrame,
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


def train_neural_model(
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
        weight_decay=1e-4,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    model.train()

    for epoch in range(
        EPOCHS
    ):
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
            f"    epoch "
            f"{epoch + 1}/{EPOCHS} "
            f"loss="
            f"{np.mean(losses):.5f}"
        )

    return model


def predict_neural(
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


def evaluate(
    name: str,
    y_true: np.ndarray,
    p_up: np.ndarray,
) -> dict:
    prediction = (
        p_up >= 0.5
    ).astype(int)

    return {
        "model": name,
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


def main() -> None:
    set_seed()

    print(
        "TRADEMIND - HYBRID "
        "NEXT-CANDLE 1M RESEARCH"
    )

    print()
    print(
        "Models: XGBoost + GRU + TCN "
        "+ logistic meta-model"
    )

    print(
        "Target: direction of the NEXT "
        "completed 1-minute candle."
    )

    print(
        "The 10m frozen trading bot is "
        "NOT changed by this script."
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

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing hybrid features: "
            + ", ".join(missing)
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

    train_rows = np.flatnonzero(
        train_mask
    )

    validation_rows = (
        valid_sequence_indices(
            df,
            validation_mask,
            EVAL_STRIDE,
        )
    )

    print(
        f"Train rows: "
        f"{len(train_rows):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_rows):,}"
    )

    scaler = StandardScaler()

    scaler.fit(
        df.loc[
            train_mask,
            FEATURE_COLUMNS,
        ]
    )

    features = scaler.transform(
        df[
            FEATURE_COLUMNS
        ]
    ).astype(
        np.float32
    )

    targets = (
        df["target_up"]
        .to_numpy(
            dtype=np.float32
        )
    )

    xgb_features = (
        df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    # Three expanding walk-forward OOF folds
    # inside the training period. These OOF
    # predictions are the ONLY data used to
    # train the ensemble meta-model.
    boundaries = np.linspace(
        0,
        len(train_rows),
        5,
        dtype=int,
    )

    oof_frames = []

    for fold in range(
        1,
        4,
    ):
        train_end = (
            boundaries[fold]
        )

        eval_start = (
            boundaries[fold]
        )

        eval_end = (
            boundaries[fold + 1]
        )

        fold_train_mask = (
            train_rows[
                :train_end
            ]
        )

        fold_eval_rows = (
            train_rows[
                eval_start:eval_end
            ]
        )

        fold_train_rows = (
            fold_train_mask[
                fold_train_mask
                >= (
                    SEQUENCE_LENGTH - 1
                )
            ][::TRAIN_STRIDE]
        )

        fold_eval_rows = (
            fold_eval_rows[
                fold_eval_rows
                >= (
                    SEQUENCE_LENGTH - 1
                )
            ]
        )

        print()
        print(
            f"OOF fold {fold}/3"
        )

        print(
            f"  train="
            f"{len(fold_train_rows):,} "
            f"eval="
            f"{len(fold_eval_rows):,}"
        )

        xgb = build_xgb()

        xgb.fit(
            xgb_features[
                fold_train_rows
            ],
            targets[
                fold_train_rows
            ],
        )

        print(
            "  training GRU..."
        )

        gru = train_neural_model(
            GRUModel(
                len(
                    FEATURE_COLUMNS
                )
            ),
            features,
            targets,
            fold_train_rows,
            device,
        )

        print(
            "  training TCN..."
        )

        tcn = train_neural_model(
            TCNModel(
                len(
                    FEATURE_COLUMNS
                )
            ),
            features,
            targets,
            fold_train_rows,
            device,
        )

        p_xgb = (
            xgb.predict_proba(
                xgb_features[
                    fold_eval_rows
                ]
            )[:, 1]
        )

        p_gru = predict_neural(
            gru,
            features,
            targets,
            fold_eval_rows,
            device,
        )

        p_tcn = predict_neural(
            tcn,
            features,
            targets,
            fold_eval_rows,
            device,
        )

        frame = pd.DataFrame(
            {
                "row_index":
                    fold_eval_rows,
                "target":
                    targets[
                        fold_eval_rows
                    ].astype(int),
                "p_xgb": p_xgb,
                "p_gru": p_gru,
                "p_tcn": p_tcn,
            }
        )

        for column in (
            META_CONTEXT_COLUMNS
        ):
            frame[
                column
            ] = (
                df.iloc[
                    fold_eval_rows
                ][column]
                .to_numpy()
            )

        oof_frames.append(
            frame
        )

        del gru
        del tcn

        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof.to_csv(
        OOF_FILE,
        index=False,
    )

    meta_columns = [
        "p_xgb",
        "p_gru",
        "p_tcn",
        *META_CONTEXT_COLUMNS,
    ]

    meta_scaler = (
        StandardScaler()
    )

    meta_X = (
        meta_scaler.fit_transform(
            oof[
                meta_columns
            ]
        )
    )

    meta_model = (
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
    )

    meta_model.fit(
        meta_X,
        oof["target"],
    )

    print()
    print(
        "Training final base models "
        "on all pre-validation data..."
    )

    final_train_rows = (
        valid_sequence_indices(
            df,
            train_mask,
            TRAIN_STRIDE,
        )
    )

    final_xgb = build_xgb()

    final_xgb.fit(
        xgb_features[
            final_train_rows
        ],
        targets[
            final_train_rows
        ],
    )

    print(
        "  final GRU..."
    )

    final_gru = train_neural_model(
        GRUModel(
            len(
                FEATURE_COLUMNS
            )
        ),
        features,
        targets,
        final_train_rows,
        device,
    )

    print(
        "  final TCN..."
    )

    final_tcn = train_neural_model(
        TCNModel(
            len(
                FEATURE_COLUMNS
            )
        ),
        features,
        targets,
        final_train_rows,
        device,
    )

    print()
    print(
        "Evaluating untouched validation "
        "window..."
    )

    val_target = (
        targets[
            validation_rows
        ].astype(int)
    )

    val_p_xgb = (
        final_xgb.predict_proba(
            xgb_features[
                validation_rows
            ]
        )[:, 1]
    )

    val_p_gru = predict_neural(
        final_gru,
        features,
        targets,
        validation_rows,
        device,
    )

    val_p_tcn = predict_neural(
        final_tcn,
        features,
        targets,
        validation_rows,
        device,
    )

    val_meta = pd.DataFrame(
        {
            "p_xgb":
                val_p_xgb,
            "p_gru":
                val_p_gru,
            "p_tcn":
                val_p_tcn,
        }
    )

    for column in (
        META_CONTEXT_COLUMNS
    ):
        val_meta[
            column
        ] = (
            df.iloc[
                validation_rows
            ][column]
            .to_numpy()
        )

    val_p_ensemble = (
        meta_model.predict_proba(
            meta_scaler.transform(
                val_meta[
                    meta_columns
                ]
            )
        )[:, 1]
    )

    results = pd.DataFrame(
        [
            evaluate(
                "XGBoost",
                val_target,
                val_p_xgb,
            ),
            evaluate(
                "GRU",
                val_target,
                val_p_gru,
            ),
            evaluate(
                "TCN",
                val_target,
                val_p_tcn,
            ),
            evaluate(
                "HYBRID",
                val_target,
                val_p_ensemble,
            ),
        ]
    )

    results = (
        results.sort_values(
            "accuracy_pct",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    results.to_csv(
        RESULT_FILE,
        index=False,
    )

    print()
    print("=" * 120)
    print(
        "VALIDATION RESULTS"
    )
    print("=" * 120)

    print(
        results.to_string(
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

    bundle = {
        "research_only": True,
        "model_name":
            "hybrid_next_candle_1m_v1_research",
        "sequence_length":
            SEQUENCE_LENGTH,
        "feature_columns":
            FEATURE_COLUMNS,
        "meta_columns":
            meta_columns,
        "feature_scaler":
            scaler,
        "meta_scaler":
            meta_scaler,
        "xgb_model":
            final_xgb,
        "meta_model":
            meta_model,
        "validation_start":
            str(
                VALIDATION_START
            ),
        "known_recent_start":
            str(
                KNOWN_RECENT_START
            ),
        "note": (
            "GRU/TCN weights are stored "
            "separately as PyTorch state dicts."
        ),
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        BUNDLE_FILE,
    )

    torch.save(
        final_gru.state_dict(),
        MODEL_DIR
        / "hybrid_next_candle_1m_gru.pt",
    )

    torch.save(
        final_tcn.state_dict(),
        MODEL_DIR
        / "hybrid_next_candle_1m_tcn.pt",
    )

    print()
    print(
        "Research artifacts saved:"
    )

    print(
        RESULT_FILE
    )

    print(
        OOF_FILE
    )

    print(
        BUNDLE_FILE
    )

    print()
    print(
        "IMPORTANT: KNOWN_RECENT from "
        "2026-07-01 onward is still not "
        "evaluated here."
    )

    print(
        "Do not connect this model to "
        "paper execution until validation "
        "results justify a separate "
        "one-shot holdout evaluation."
    )


if __name__ == "__main__":
    main()
