from __future__ import annotations

from pathlib import Path
import gc
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Install with: "
        "python -m pip install -r requirements-ml-deep.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_DIR = PROJECT_ROOT / "backend" / "services" / "ml"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from research_hybrid_next_candle_1m import (
    FEATURE_COLUMNS,
    META_CONTEXT_COLUMNS,
    RESEARCH_START,
    VALIDATION_START,
    KNOWN_RECENT_START,
    SEQUENCE_LENGTH,
    TRAIN_STRIDE,
    build_features,
    build_xgb,
    valid_sequence_indices,
    train_neural_model,
    predict_neural,
    GRUModel,
    TCNModel,
)

from research_lstm_next_candle_1m import (
    LSTMModel,
    train_model as train_lstm,
    predict_probabilities as predict_lstm,
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
    / "selective_ensemble_1m"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
)

OOF_FILE = OUTPUT_DIR / "base_oof_predictions.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation_predictions.csv"
SEARCH_FILE = OUTPUT_DIR / "threshold_search.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.csv"
BUNDLE_FILE = MODEL_DIR / "selective_ensemble_1m_research.joblib"

TARGET_ACCURACY_PCT = 82.0
TARGET_COVERAGE_PCT = 8.0

RANDOM_SEED = 42
META_TRAIN_FRACTION = 0.60

DIRECTION_THRESHOLDS = np.arange(
    0.50,
    0.86,
    0.02,
)

SELECTOR_THRESHOLDS = np.arange(
    0.50,
    0.96,
    0.02,
)


def set_seed() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


def fit_scaler(
    df: pd.DataFrame,
    row_indices: np.ndarray,
) -> StandardScaler:
    scaler = StandardScaler()

    scaler.fit(
        df.iloc[
            row_indices
        ][FEATURE_COLUMNS]
    )

    return scaler


def transform_all(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> np.ndarray:
    return (
        scaler.transform(
            df[
                FEATURE_COLUMNS
            ]
        )
        .astype(np.float32)
    )


def context_frame(
    df: pd.DataFrame,
    rows: np.ndarray,
) -> pd.DataFrame:
    result = pd.DataFrame(
        index=np.arange(
            len(rows)
        )
    )

    for column in META_CONTEXT_COLUMNS:
        result[column] = (
            df.iloc[
                rows
            ][column]
            .to_numpy()
        )

    return result


def agreement_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    probabilities = frame[
        [
            "p_xgb",
            "p_gru",
            "p_tcn",
            "p_lstm",
        ]
    ].to_numpy()

    hard = (
        probabilities
        >= 0.5
    ).astype(int)

    vote_up = hard.mean(
        axis=1
    )

    result = pd.DataFrame(
        {
            "mean_probability":
                probabilities.mean(
                    axis=1
                ),
            "probability_std":
                probabilities.std(
                    axis=1
                ),
            "vote_up_fraction":
                vote_up,
            "unanimous":
                (
                    (
                        vote_up == 0.0
                    )
                    | (
                        vote_up == 1.0
                    )
                ).astype(int),
            "agreement_strength":
                np.abs(
                    vote_up
                    - 0.5
                )
                * 2.0,
        }
    )

    return result


def make_meta_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    agreement = agreement_features(
        frame
    )

    result = pd.concat(
        [
            frame[
                [
                    "p_xgb",
                    "p_gru",
                    "p_tcn",
                    "p_lstm",
                    *META_CONTEXT_COLUMNS,
                ]
            ].reset_index(
                drop=True
            ),
            agreement.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return result


def make_selector_features(
    frame: pd.DataFrame,
    meta_probability: np.ndarray,
) -> pd.DataFrame:
    meta_features = make_meta_features(
        frame
    )

    confidence = np.maximum(
        meta_probability,
        1.0 - meta_probability,
    )

    meta_features[
        "meta_probability"
    ] = meta_probability

    meta_features[
        "meta_confidence"
    ] = confidence

    meta_features[
        "meta_margin"
    ] = np.abs(
        meta_probability
        - 0.5
    )

    return meta_features


def train_base_models(
    df: pd.DataFrame,
    targets: np.ndarray,
    train_rows: np.ndarray,
    eval_rows: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    scaler = fit_scaler(
        df,
        train_rows,
    )

    scaled = transform_all(
        df,
        scaler,
    )

    xgb_matrix = (
        df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    xgb = build_xgb()
    xgb.fit(
        xgb_matrix[
            train_rows
        ],
        targets[
            train_rows
        ],
    )

    gru = train_neural_model(
        GRUModel(
            len(
                FEATURE_COLUMNS
            )
        ),
        scaled,
        targets,
        train_rows,
        device,
    )

    tcn = train_neural_model(
        TCNModel(
            len(
                FEATURE_COLUMNS
            )
        ),
        scaled,
        targets,
        train_rows,
        device,
    )

    lstm = train_lstm(
        LSTMModel(
            len(
                FEATURE_COLUMNS
            )
        ),
        scaled,
        targets,
        train_rows,
        device,
    )

    result = pd.DataFrame(
        {
            "row_index":
                eval_rows,
            "target":
                targets[
                    eval_rows
                ].astype(int),
            "p_xgb":
                xgb.predict_proba(
                    xgb_matrix[
                        eval_rows
                    ]
                )[:, 1],
            "p_gru":
                predict_neural(
                    gru,
                    scaled,
                    targets,
                    eval_rows,
                    device,
                ),
            "p_tcn":
                predict_neural(
                    tcn,
                    scaled,
                    targets,
                    eval_rows,
                    device,
                ),
            "p_lstm":
                predict_lstm(
                    lstm,
                    scaled,
                    targets,
                    eval_rows,
                    device,
                ),
        }
    )

    context = context_frame(
        df,
        eval_rows,
    )

    for column in META_CONTEXT_COLUMNS:
        result[column] = (
            context[column]
            .to_numpy()
        )

    del xgb
    del gru
    del tcn
    del lstm
    del scaled
    del xgb_matrix

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def train_oof_base_models(
    df: pd.DataFrame,
    targets: np.ndarray,
    train_rows: np.ndarray,
    device: torch.device,
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
        train_end = boundaries[
            fold
        ]

        eval_start = boundaries[
            fold
        ]

        eval_end = boundaries[
            fold + 1
        ]

        fold_train_rows = (
            train_rows[
                :train_end
            ]
        )

        fold_train_rows = (
            fold_train_rows[
                fold_train_rows
                >= (
                    SEQUENCE_LENGTH
                    - 1
                )
            ][::TRAIN_STRIDE]
        )

        fold_eval_rows = (
            train_rows[
                eval_start:eval_end
            ]
        )

        fold_eval_rows = (
            fold_eval_rows[
                fold_eval_rows
                >= (
                    SEQUENCE_LENGTH
                    - 1
                )
            ]
        )

        print()
        print(
            f"BASE OOF FOLD {fold}/3"
        )
        print(
            f"train={len(fold_train_rows):,} "
            f"eval={len(fold_eval_rows):,}"
        )

        fold_frame = train_base_models(
            df,
            targets,
            fold_train_rows,
            fold_eval_rows,
            device,
        )

        fold_frame[
            "fold"
        ] = fold

        frames.append(
            fold_frame
        )

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            "row_index"
        )
        .reset_index(
            drop=True
        )
    )


def search_thresholds(
    y_true: np.ndarray,
    meta_probability: np.ndarray,
    selector_probability: np.ndarray,
) -> pd.DataFrame:
    meta_prediction = (
        meta_probability
        >= 0.5
    ).astype(int)

    meta_confidence = np.maximum(
        meta_probability,
        1.0 - meta_probability,
    )

    rows = []

    total = len(
        y_true
    )

    for direction_threshold in (
        DIRECTION_THRESHOLDS
    ):
        for selector_threshold in (
            SELECTOR_THRESHOLDS
        ):
            selected = (
                (
                    meta_confidence
                    >= direction_threshold
                )
                & (
                    selector_probability
                    >= selector_threshold
                )
            )

            count = int(
                selected.sum()
            )

            coverage_pct = (
                count
                / total
                * 100.0
                if total > 0
                else 0.0
            )

            accuracy_pct = (
                accuracy_score(
                    y_true[
                        selected
                    ],
                    meta_prediction[
                        selected
                    ],
                )
                * 100.0
                if count > 0
                else np.nan
            )

            target_met = bool(
                count > 0
                and accuracy_pct
                >= TARGET_ACCURACY_PCT
                and coverage_pct
                >= TARGET_COVERAGE_PCT
            )

            rows.append(
                {
                    "direction_confidence_threshold":
                        direction_threshold,
                    "selector_threshold":
                        selector_threshold,
                    "predictions":
                        count,
                    "coverage_pct":
                        coverage_pct,
                    "accuracy_pct":
                        accuracy_pct,
                    "target_met":
                        target_met,
                }
            )

    result = pd.DataFrame(
        rows
    )

    result[
        "coverage_shortfall"
    ] = np.maximum(
        0.0,
        TARGET_COVERAGE_PCT
        - result[
            "coverage_pct"
        ],
    )

    result[
        "accuracy_shortfall"
    ] = np.maximum(
        0.0,
        TARGET_ACCURACY_PCT
        - result[
            "accuracy_pct"
        ].fillna(
            0.0
        ),
    )

    result[
        "target_distance"
    ] = (
        result[
            "coverage_shortfall"
        ]
        + result[
            "accuracy_shortfall"
        ]
    )

    return result.sort_values(
        [
            "target_met",
            "target_distance",
            "accuracy_pct",
            "coverage_pct",
        ],
        ascending=[
            False,
            True,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


def print_summary(
    label: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    prediction = (
        probabilities
        >= 0.5
    ).astype(int)

    accuracy = (
        accuracy_score(
            y_true,
            prediction,
        )
        * 100.0
    )

    print(
        f"{label}: "
        f"accuracy={accuracy:.3f}% "
        f"samples={len(y_true):,}"
    )

    return {
        "model":
            label,
        "samples":
            len(y_true),
        "accuracy_pct":
            accuracy,
    }


def main() -> None:
    set_seed()

    print(
        "TRADEMIND - SELECTIVE ENSEMBLE "
        "/ TRADE FILTER RESEARCH"
    )
    print()
    print(
        "Most promising path: stop asking "
        "a single model to do everything. "
        "This experiment combines "
        "XGBoost + GRU + TCN + LSTM, then "
        "adds a learned selector whose job "
        "is to decide when the ensemble "
        "should abstain."
    )
    print()
    print(
        "Target candidate:"
    )
    print(
        f"  accuracy >= "
        f"{TARGET_ACCURACY_PCT:.1f}%"
    )
    print(
        f"  coverage >= "
        f"{TARGET_COVERAGE_PCT:.1f}%"
    )
    print(
        "The target is NOT forced. If the "
        "validation data does not support "
        "it, the script reports failure."
    )
    print(
        "2026-07-01 onward remains untouched "
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
            subset=[
                "open_time"
            ]
        )
        .sort_values(
            "open_time"
        )
        .drop_duplicates(
            subset=[
                "open_time"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    warmup_start = (
        RESEARCH_START
        - pd.Timedelta(
            minutes=300
        )
    )

    raw = raw[
        (
            raw[
                "open_time"
            ]
            >= warmup_start
        )
        & (
            raw[
                "open_time"
            ]
            < KNOWN_RECENT_START
        )
    ].copy()

    df = build_features(
        raw
    )

    df = df[
        df[
            "open_time"
        ]
        >= RESEARCH_START
    ].reset_index(
        drop=True
    )

    targets = (
        df[
            "target_up"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    train_mask = (
        df[
            "open_time"
        ]
        < VALIDATION_START
    ).to_numpy()

    validation_mask = (
        (
            df[
                "open_time"
            ]
            >= VALIDATION_START
        )
        & (
            df[
                "open_time"
            ]
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
            1,
        )
    )

    print(
        f"ML rows: {len(df):,}"
    )
    print(
        f"train rows: "
        f"{len(train_rows):,}"
    )
    print(
        f"validation rows: "
        f"{len(validation_rows):,}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "STEP 1/5 - Build chronological "
        "out-of-fold base predictions"
    )

    oof = train_oof_base_models(
        df,
        targets,
        train_rows,
        device,
    )

    oof.to_csv(
        OOF_FILE,
        index=False,
    )

    split_index = int(
        len(oof)
        * META_TRAIN_FRACTION
    )

    meta_train = (
        oof.iloc[
            :split_index
        ]
        .reset_index(
            drop=True
        )
    )

    selector_train = (
        oof.iloc[
            split_index:
        ]
        .reset_index(
            drop=True
        )
    )

    meta_columns = list(
        make_meta_features(
            meta_train
        ).columns
    )

    print()
    print(
        "STEP 2/5 - Train stacked direction "
        "meta-model on earlier OOF data"
    )

    meta_scaler = StandardScaler()

    meta_X = meta_scaler.fit_transform(
        make_meta_features(
            meta_train
        )
    )

    meta_model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED,
    )

    meta_model.fit(
        meta_X,
        meta_train[
            "target"
        ],
    )

    selector_meta_probability = (
        meta_model.predict_proba(
            meta_scaler.transform(
                make_meta_features(
                    selector_train
                )
            )
        )[:, 1]
    )

    selector_meta_prediction = (
        selector_meta_probability
        >= 0.5
    ).astype(int)

    selector_correct = (
        selector_meta_prediction
        == selector_train[
            "target"
        ].to_numpy()
    ).astype(int)

    selector_features = (
        make_selector_features(
            selector_train,
            selector_meta_probability,
        )
    )

    selector_columns = list(
        selector_features.columns
    )

    selector_scaler = (
        StandardScaler()
    )

    selector_X = (
        selector_scaler.fit_transform(
            selector_features
        )
    )

    selector_model = (
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    )

    selector_model.fit(
        selector_X,
        selector_correct,
    )

    print(
        "selector training rows: "
        f"{len(selector_train):,}"
    )
    print(
        "selector target correct rate: "
        f"{selector_correct.mean() * 100:.2f}%"
    )

    print()
    print(
        "STEP 3/5 - Train final XGBoost, "
        "GRU, TCN and LSTM on all "
        "pre-validation data"
    )

    final_train_rows = (
        valid_sequence_indices(
            df,
            train_mask,
            TRAIN_STRIDE,
        )
    )

    validation_frame = (
        train_base_models(
            df,
            targets,
            final_train_rows,
            validation_rows,
            device,
        )
    )

    validation_frame.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    val_y = (
        validation_frame[
            "target"
        ]
        .to_numpy(
            dtype=int
        )
    )

    print()
    print(
        "STEP 4/5 - Evaluate base models "
        "and stacked direction model"
    )

    summary_rows = []

    for name, column in [
        (
            "XGBoost",
            "p_xgb",
        ),
        (
            "GRU",
            "p_gru",
        ),
        (
            "TCN",
            "p_tcn",
        ),
        (
            "LSTM",
            "p_lstm",
        ),
    ]:
        summary_rows.append(
            print_summary(
                name,
                val_y,
                validation_frame[
                    column
                ].to_numpy(),
            )
        )

    mean_probability = (
        validation_frame[
            [
                "p_xgb",
                "p_gru",
                "p_tcn",
                "p_lstm",
            ]
        ].mean(
            axis=1
        ).to_numpy()
    )

    summary_rows.append(
        print_summary(
            "Simple average",
            val_y,
            mean_probability,
        )
    )

    val_meta_features = (
        make_meta_features(
            validation_frame
        )
    )

    val_meta_probability = (
        meta_model.predict_proba(
            meta_scaler.transform(
                val_meta_features
            )
        )[:, 1]
    )

    summary_rows.append(
        print_summary(
            "Stacked ensemble",
            val_y,
            val_meta_probability,
        )
    )

    val_selector_features = (
        make_selector_features(
            validation_frame,
            val_meta_probability,
        )
    )

    val_selector_probability = (
        selector_model.predict_proba(
            selector_scaler.transform(
                val_selector_features
            )
        )[:, 1]
    )

    print()
    print(
        "STEP 5/5 - Search selective "
        "confidence / abstention thresholds"
    )

    search = search_thresholds(
        val_y,
        val_meta_probability,
        val_selector_probability,
    )

    search.to_csv(
        SEARCH_FILE,
        index=False,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    feasible = search[
        search[
            "target_met"
        ]
    ]

    print()
    print("=" * 100)
    print(
        "SELECTIVE ENSEMBLE RESULT"
    )
    print("=" * 100)

    if not feasible.empty:
        best = feasible.iloc[
            0
        ]

        print(
            "TARGET FOUND ON VALIDATION "
            "WINDOW"
        )
        print(
            f"direction confidence >= "
            f"{best['direction_confidence_threshold']:.2f}"
        )
        print(
            f"selector probability >= "
            f"{best['selector_threshold']:.2f}"
        )
        print(
            f"accuracy = "
            f"{best['accuracy_pct']:.3f}%"
        )
        print(
            f"coverage = "
            f"{best['coverage_pct']:.3f}%"
        )
        print(
            f"predictions = "
            f"{int(best['predictions']):,}"
        )
        print()
        print(
            "IMPORTANT: this is still a "
            "validation-selected candidate. "
            "It must survive the untouched "
            "2026-07-01+ one-shot check "
            "before any execution use."
        )
    else:
        best = search.iloc[
            0
        ]

        print(
            "TARGET NOT ACHIEVED."
        )
        print(
            "Closest validation candidate:"
        )
        print(
            f"direction confidence >= "
            f"{best['direction_confidence_threshold']:.2f}"
        )
        print(
            f"selector probability >= "
            f"{best['selector_threshold']:.2f}"
        )
        print(
            f"accuracy = "
            f"{best['accuracy_pct']:.3f}%"
        )
        print(
            f"coverage = "
            f"{best['coverage_pct']:.3f}%"
        )
        print(
            f"predictions = "
            f"{int(best['predictions']):,}"
        )
        print()
        print(
            "Do not force the threshold. "
            "The current data does not support "
            "82% accuracy at 8% coverage."
        )

    bundle = {
        "research_only":
            True,
        "model_name":
            "selective_ensemble_1m_v1",
        "base_models":
            [
                "XGBoost",
                "GRU",
                "TCN",
                "LSTM",
            ],
        "meta_columns":
            meta_columns,
        "selector_columns":
            selector_columns,
        "meta_scaler":
            meta_scaler,
        "meta_model":
            meta_model,
        "selector_scaler":
            selector_scaler,
        "selector_model":
            selector_model,
        "target_accuracy_pct":
            TARGET_ACCURACY_PCT,
        "target_coverage_pct":
            TARGET_COVERAGE_PCT,
        "validation_start":
            str(
                VALIDATION_START
            ),
        "known_recent_start":
            str(
                KNOWN_RECENT_START
            ),
        "note":
            (
                "Selective ensemble research "
                "candidate only. Base neural "
                "weights are intentionally not "
                "stored here; rerun/freeze only "
                "after validation justifies a "
                "one-shot holdout check."
            ),
    }

    joblib.dump(
        bundle,
        BUNDLE_FILE,
    )

    print()
    print(
        "Saved:"
    )
    print(
        OOF_FILE
    )
    print(
        VALIDATION_FILE
    )
    print(
        SEARCH_FILE
    )
    print(
        SUMMARY_FILE
    )
    print(
        BUNDLE_FILE
    )


if __name__ == "__main__":
    main()
