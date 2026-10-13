from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SNAPSHOT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micro_paper_learning"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cost_aware_micro_tradeability"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
)

SUMMARY_FILE = OUTPUT_DIR / "summary.json"
VALIDATION_FILE = OUTPUT_DIR / "validation_predictions.csv"
THRESHOLD_FILE = OUTPUT_DIR / "threshold_search.csv"
BUNDLE_FILE = MODEL_DIR / "cost_aware_micro_tradeability_research.joblib"

FEATURE_COLUMNS = [
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

ROLLING_BASE = [
    "spot_top_imbalance",
    "spot_depth20_imbalance",
    "spot_trade_flow_imbalance",
    "spot_price_velocity_bps",
    "futures_top_imbalance",
    "futures_depth20_imbalance",
    "futures_trade_flow_imbalance",
    "futures_price_velocity_bps",
    "spot_futures_mid_basis_bps",
]

# Match the current paper trader as closely as possible.
HORIZON_SECONDS = 60
TAKE_PROFIT_PCT = 0.0020
STOP_LOSS_PCT = 0.0012
TRAILING_ACTIVATION_PCT = 0.0010
TRAILING_DISTANCE_PCT = 0.0008
FEE_RATE_PER_SIDE = 0.0005

TRAIN_FRACTION = 0.60
SELECTOR_FRACTION = 0.20
VALIDATION_FRACTION = 0.20

RANDOM_SEED = 42
MIN_SELECTED = 25

# Search thresholds for a selective policy.
MODEL_CONFIDENCE_THRESHOLDS = np.arange(
    0.40,
    0.96,
    0.02,
)

SELECTOR_THRESHOLDS = np.arange(
    0.50,
    0.96,
    0.02,
)


def load_snapshots() -> pd.DataFrame:
    files = sorted(
        SNAPSHOT_DIR.glob(
            "feature_snapshots*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No snapshot files found in {SNAPSHOT_DIR}"
        )

    frames = []

    for path in files:
        try:
            frame = pd.read_csv(
                path,
                on_bad_lines="skip",
                low_memory=False,
            )
        except Exception as exc:
            print(
                f"Skipping {path.name}: {exc}"
            )
            continue

        required = {
            "timestamp",
            "price",
            *FEATURE_COLUMNS,
        }

        if not required.issubset(
            frame.columns
        ):
            print(
                f"Skipping {path.name}: "
                "incompatible schema"
            )
            continue

        frames.append(
            frame[
                [
                    "timestamp",
                    "price",
                    *FEATURE_COLUMNS,
                ]
            ].copy()
        )

    if not frames:
        raise RuntimeError(
            "No compatible snapshot CSVs were found."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    for column in [
        "price",
        *FEATURE_COLUMNS,
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "timestamp",
                "price",
                *FEATURE_COLUMNS,
            ]
        )
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return df[
        df["price"] > 0
    ].copy()


def add_rolling_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    result = df.copy()
    engineered = []

    for column in ROLLING_BASE:
        for window in [
            3,
            6,
            12,
        ]:
            mean_name = (
                f"{column}_mean_{window}"
            )
            std_name = (
                f"{column}_std_{window}"
            )

            result[mean_name] = (
                result[column]
                .rolling(
                    window,
                    min_periods=window,
                )
                .mean()
            )

            result[std_name] = (
                result[column]
                .rolling(
                    window,
                    min_periods=window,
                )
                .std()
            )

            engineered.extend(
                [
                    mean_name,
                    std_name,
                ]
            )

        delta_name = (
            f"{column}_delta_6"
        )

        result[delta_name] = (
            result[column]
            - result[column].shift(6)
        )

        engineered.append(
            delta_name
        )

    result[
        "flow_agreement"
    ] = (
        result[
            "spot_trade_flow_imbalance"
        ]
        * result[
            "futures_trade_flow_imbalance"
        ]
    )

    result[
        "depth_agreement"
    ] = (
        result[
            "spot_depth20_imbalance"
        ]
        * result[
            "futures_depth20_imbalance"
        ]
    )

    result[
        "velocity_agreement"
    ] = (
        result[
            "spot_price_velocity_bps"
        ]
        * result[
            "futures_price_velocity_bps"
        ]
    )

    engineered.extend(
        [
            "flow_agreement",
            "depth_agreement",
            "velocity_agreement",
        ]
    )

    return (
        result,
        engineered,
    )


def simulate_side(
    entry_price: float,
    future_prices: np.ndarray,
    direction: int,
) -> tuple[
    float,
    str,
]:
    best_price = entry_price
    trailing_active = False

    if len(
        future_prices
    ) == 0:
        return (
            -2.0
            * FEE_RATE_PER_SIDE,
            "NO_FUTURE_DATA",
        )

    exit_price = float(
        future_prices[-1]
    )
    reason = "MAX_HOLD"

    for value in future_prices:
        price = float(
            value
        )

        gross_return = (
            (
                price
                / entry_price
                - 1.0
            )
            * direction
        )

        if gross_return >= (
            TAKE_PROFIT_PCT
        ):
            exit_price = price
            reason = "TAKE_PROFIT"
            break

        if gross_return <= (
            -STOP_LOSS_PCT
        ):
            exit_price = price
            reason = "STOP_LOSS"
            break

        if direction == 1:
            best_price = max(
                best_price,
                price,
            )

            best_return = (
                best_price
                / entry_price
                - 1.0
            )

            if (
                not trailing_active
                and best_return
                >= TRAILING_ACTIVATION_PCT
            ):
                trailing_active = True

            if trailing_active:
                trailing_stop = (
                    best_price
                    * (
                        1.0
                        - TRAILING_DISTANCE_PCT
                    )
                )

                if price <= trailing_stop:
                    exit_price = price
                    reason = "TRAILING_STOP"
                    break

        else:
            best_price = min(
                best_price,
                price,
            )

            best_return = (
                entry_price
                / best_price
                - 1.0
                if best_price > 0
                else 0.0
            )

            if (
                not trailing_active
                and best_return
                >= TRAILING_ACTIVATION_PCT
            ):
                trailing_active = True

            if trailing_active:
                trailing_stop = (
                    best_price
                    * (
                        1.0
                        + TRAILING_DISTANCE_PCT
                    )
                )

                if price >= trailing_stop:
                    exit_price = price
                    reason = "TRAILING_STOP"
                    break

    gross_return = (
        (
            exit_price
            / entry_price
            - 1.0
        )
        * direction
    )

    net_return = (
        gross_return
        - 2.0
        * FEE_RATE_PER_SIDE
    )

    return (
        net_return,
        reason,
    )


def add_tradeability_labels(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    timestamps_ns = (
        result[
            "timestamp"
        ]
        .astype("int64")
        .to_numpy()
    )

    prices = (
        result[
            "price"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    horizon_ns = int(
        HORIZON_SECONDS
        * 1_000_000_000
    )

    labels = np.zeros(
        len(
            result
        ),
        dtype=int,
    )

    long_returns = np.full(
        len(
            result
        ),
        np.nan,
        dtype=float,
    )

    short_returns = np.full(
        len(
            result
        ),
        np.nan,
        dtype=float,
    )

    chosen_returns = np.full(
        len(
            result
        ),
        np.nan,
        dtype=float,
    )

    long_reasons = [
        ""
        for _ in range(
            len(
                result
            )
        )
    ]

    short_reasons = [
        ""
        for _ in range(
            len(
                result
            )
        )
    ]

    valid_future = np.zeros(
        len(
            result
        ),
        dtype=bool,
    )

    for index in range(
        len(
            result
        )
    ):
        end_time = (
            timestamps_ns[index]
            + horizon_ns
        )

        end_index = int(
            np.searchsorted(
                timestamps_ns,
                end_time,
                side="right",
            )
        )

        if end_index <= (
            index
            + 1
        ):
            continue

        future_prices = prices[
            index + 1:
            end_index
        ]

        if len(
            future_prices
        ) == 0:
            continue

        valid_future[
            index
        ] = True

        long_return, long_reason = (
            simulate_side(
                prices[
                    index
                ],
                future_prices,
                1,
            )
        )

        short_return, short_reason = (
            simulate_side(
                prices[
                    index
                ],
                future_prices,
                -1,
            )
        )

        long_returns[
            index
        ] = long_return

        short_returns[
            index
        ] = short_return

        long_reasons[
            index
        ] = long_reason

        short_reasons[
            index
        ] = short_reason

        if (
            long_return > 0
            and long_return
            > short_return
        ):
            labels[
                index
            ] = 1

            chosen_returns[
                index
            ] = long_return

        elif (
            short_return > 0
            and short_return
            > long_return
        ):
            labels[
                index
            ] = -1

            chosen_returns[
                index
            ] = short_return

        else:
            labels[
                index
            ] = 0

            chosen_returns[
                index
            ] = max(
                long_return,
                short_return,
            )

    result[
        "tradeability_label"
    ] = labels

    result[
        "long_net_return"
    ] = long_returns

    result[
        "short_net_return"
    ] = short_returns

    result[
        "chosen_net_return"
    ] = chosen_returns

    result[
        "long_exit_reason"
    ] = long_reasons

    result[
        "short_exit_reason"
    ] = short_reasons

    result[
        "valid_future_path"
    ] = valid_future

    return result


def split_rows(
    total: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    train_end = int(
        total
        * TRAIN_FRACTION
    )

    selector_end = int(
        total
        * (
            TRAIN_FRACTION
            + SELECTOR_FRACTION
        )
    )

    return (
        np.arange(
            0,
            train_end,
        ),
        np.arange(
            train_end,
            selector_end,
        ),
        np.arange(
            selector_end,
            total,
        ),
    )


def selector_feature_frame(
    base_X: np.ndarray,
    probabilities: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    confidence = probabilities.max(
        axis=1
    )

    sorted_probability = np.sort(
        probabilities,
        axis=1,
    )

    margin = (
        sorted_probability[
            :,
            -1
        ]
        - sorted_probability[
            :,
            -2
        ]
    )

    entropy = -np.sum(
        np.clip(
            probabilities,
            1e-12,
            1.0,
        )
        * np.log(
            np.clip(
                probabilities,
                1e-12,
                1.0,
            )
        ),
        axis=1,
    )

    frame = pd.DataFrame(
        {
            "prediction":
                prediction,
            "confidence":
                confidence,
            "margin":
                margin,
            "entropy":
                entropy,
            "p_down":
                probabilities[
                    :,
                    0
                ],
            "p_neutral":
                probabilities[
                    :,
                    1
                ],
            "p_up":
                probabilities[
                    :,
                    2
                ],
        }
    )

    # Add a compact subset of current market regime
    # features so the selector can learn when otherwise
    # similar probabilities are trustworthy.
    regime_count = min(
        20,
        base_X.shape[
            1
        ],
    )

    for index in range(
        regime_count
    ):
        frame[
            f"regime_{index}"
        ] = base_X[
            :,
            index
        ]

    return frame


def threshold_search(
    y_true: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
    selector_probability: np.ndarray,
    chosen_net_return: np.ndarray,
) -> pd.DataFrame:
    confidence = probabilities.max(
        axis=1
    )

    rows = []

    total = len(
        y_true
    )

    for model_threshold in (
        MODEL_CONFIDENCE_THRESHOLDS
    ):
        for selector_threshold in (
            SELECTOR_THRESHOLDS
        ):
            selected = (
                (
                    confidence
                    >= model_threshold
                )
                & (
                    selector_probability
                    >= selector_threshold
                )
                & (
                    prediction
                    != 0
                )
            )

            count = int(
                selected.sum()
            )

            if count == 0:
                accuracy = np.nan
                coverage = 0.0
                win_rate = np.nan
                avg_return = np.nan
            else:
                accuracy = (
                    accuracy_score(
                        y_true[
                            selected
                        ],
                        prediction[
                            selected
                        ],
                    )
                    * 100.0
                )

                coverage = (
                    count
                    / total
                    * 100.0
                )

                selected_returns = (
                    chosen_net_return[
                        selected
                    ]
                )

                win_rate = (
                    np.mean(
                        selected_returns
                        > 0
                    )
                    * 100.0
                )

                avg_return = (
                    np.nanmean(
                        selected_returns
                    )
                    * 100.0
                )

            rows.append(
                {
                    "model_confidence_threshold":
                        model_threshold,
                    "selector_threshold":
                        selector_threshold,
                    "predictions":
                        count,
                    "coverage_pct":
                        coverage,
                    "label_accuracy_pct":
                        accuracy,
                    "profitable_path_pct":
                        win_rate,
                    "avg_chosen_net_return_pct":
                        avg_return,
                }
            )

    result = pd.DataFrame(
        rows
    )

    result[
        "score"
    ] = (
        result[
            "profitable_path_pct"
        ].fillna(
            0.0
        )
        + result[
            "label_accuracy_pct"
        ].fillna(
            0.0
        )
        + np.minimum(
            result[
                "coverage_pct"
            ],
            8.0,
        )
        * 2.0
    )

    return result.sort_values(
        [
            "score",
            "profitable_path_pct",
            "coverage_pct",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


def main() -> None:
    print(
        "TRADEMIND - COST-AWARE MICRO "
        "TRADEABILITY RESEARCH"
    )

    print()
    print(
        "Target classes:"
    )
    print(
        "  -1 = SHORT setup with positive "
        "simulated net return"
    )
    print(
        "   0 = NO TRADE"
    )
    print(
        "  +1 = LONG setup with positive "
        "simulated net return"
    )

    print()
    print(
        "Label simulation matches the current "
        "paper policy as closely as possible:"
    )
    print(
        f"  take profit = "
        f"{TAKE_PROFIT_PCT * 100:.3f}%"
    )
    print(
        f"  stop loss = "
        f"{STOP_LOSS_PCT * 100:.3f}%"
    )
    print(
        f"  trailing activates = "
        f"{TRAILING_ACTIVATION_PCT * 100:.3f}%"
    )
    print(
        f"  trailing distance = "
        f"{TRAILING_DISTANCE_PCT * 100:.3f}%"
    )
    print(
        f"  round-trip fee = "
        f"{2 * FEE_RATE_PER_SIDE * 100:.3f}%"
    )
    print(
        f"  max horizon = "
        f"{HORIZON_SECONDS}s"
    )

    df = load_snapshots()

    print()
    print(
        f"compatible snapshots: "
        f"{len(df):,}"
    )

    df, engineered = (
        add_rolling_features(
            df
        )
    )

    print(
        "Simulating future trade paths..."
    )

    df = add_tradeability_labels(
        df
    )

    all_features = [
        *FEATURE_COLUMNS,
        *engineered,
    ]

    df = (
        df.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                *all_features,
                "long_net_return",
                "short_net_return",
                "chosen_net_return",
            ]
        )
        .loc[
            lambda frame:
                frame[
                    "valid_future_path"
                ]
        ]
        .reset_index(
            drop=True
        )
    )

    if len(
        df
    ) < 1000:
        raise RuntimeError(
            "Not enough compatible history. "
            "Keep the collector and learner "
            "running until at least 1,000 "
            "labeled rows are available."
        )

    counts = (
        df[
            "tradeability_label"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        f"ML-ready rows: "
        f"{len(df):,}"
    )

    print(
        "class counts:"
    )

    print(
        f"  SHORT (-1): "
        f"{int(counts.get(-1, 0)):,}"
    )
    print(
        f"  NO TRADE (0): "
        f"{int(counts.get(0, 0)):,}"
    )
    print(
        f"  LONG (+1): "
        f"{int(counts.get(1, 0)):,}"
    )

    (
        train_rows,
        selector_rows,
        validation_rows,
    ) = split_rows(
        len(
            df
        )
    )

    print(
        "chronological split: "
        f"train={len(train_rows):,}, "
        f"selector={len(selector_rows):,}, "
        f"validation={len(validation_rows):,}"
    )

    X = (
        df[
            all_features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_raw = (
        df[
            "tradeability_label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    # XGBoost uses 0,1,2 class IDs.
    label_to_id = {
        -1: 0,
        0: 1,
        1: 2,
    }

    id_to_label = np.array(
        [
            -1,
            0,
            1,
        ],
        dtype=int,
    )

    y = np.array(
        [
            label_to_id[
                int(
                    value
                )
            ]
            for value in y_raw
        ],
        dtype=int,
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=6,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    class_counts = np.bincount(
        y[
            train_rows
        ],
        minlength=3,
    )

    total_train = len(
        train_rows
    )

    class_weights = {
        class_id:
            (
                total_train
                / (
                    3.0
                    * count
                )
                if count > 0
                else 1.0
            )
        for class_id, count
        in enumerate(
            class_counts
        )
    }

    sample_weight = np.array(
        [
            class_weights[
                int(
                    label
                )
            ]
            for label in y[
                train_rows
            ]
        ],
        dtype=float,
    )

    print()
    print(
        "Training 3-class cost-aware "
        "microstructure XGBoost..."
    )

    model.fit(
        X[
            train_rows
        ],
        y[
            train_rows
        ],
        sample_weight=sample_weight,
    )

    selector_probabilities = (
        model.predict_proba(
            X[
                selector_rows
            ]
        )
    )

    selector_prediction_id = (
        np.argmax(
            selector_probabilities,
            axis=1,
        )
    )

    selector_prediction = (
        id_to_label[
            selector_prediction_id
        ]
    )

    selector_correct = (
        selector_prediction
        == y_raw[
            selector_rows
        ]
    ).astype(int)

    selector_frame = (
        selector_feature_frame(
            X[
                selector_rows
            ],
            selector_probabilities,
            selector_prediction,
        )
    )

    selector_scaler = (
        StandardScaler()
    )

    selector_X = (
        selector_scaler.fit_transform(
            selector_frame
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

    validation_probabilities = (
        model.predict_proba(
            X[
                validation_rows
            ]
        )
    )

    validation_prediction_id = (
        np.argmax(
            validation_probabilities,
            axis=1,
        )
    )

    validation_prediction = (
        id_to_label[
            validation_prediction_id
        ]
    )

    validation_y = y_raw[
        validation_rows
    ]

    validation_selector_frame = (
        selector_feature_frame(
            X[
                validation_rows
            ],
            validation_probabilities,
            validation_prediction,
        )
    )

    validation_selector_probability = (
        selector_model.predict_proba(
            selector_scaler.transform(
                validation_selector_frame
            )
        )[:, 1]
    )

    overall_accuracy = (
        accuracy_score(
            validation_y,
            validation_prediction,
        )
        * 100.0
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            validation_y,
            validation_prediction,
        )
        * 100.0
    )

    report = classification_report(
        validation_y,
        validation_prediction,
        labels=[
            -1,
            0,
            1,
        ],
        target_names=[
            "SHORT",
            "NO_TRADE",
            "LONG",
        ],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        validation_y,
        validation_prediction,
        labels=[
            -1,
            0,
            1,
        ],
    )

    chosen_returns = (
        df.iloc[
            validation_rows
        ][
            "chosen_net_return"
        ]
        .to_numpy(
            dtype=float
        )
    )

    search = threshold_search(
        validation_y,
        validation_prediction,
        validation_probabilities,
        validation_selector_probability,
        chosen_returns,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    search.to_csv(
        THRESHOLD_FILE,
        index=False,
    )

    validation_output = pd.DataFrame(
        {
            "timestamp":
                df.iloc[
                    validation_rows
                ][
                    "timestamp"
                ].to_numpy(),
            "actual_label":
                validation_y,
            "predicted_label":
                validation_prediction,
            "p_short":
                validation_probabilities[
                    :,
                    0
                ],
            "p_no_trade":
                validation_probabilities[
                    :,
                    1
                ],
            "p_long":
                validation_probabilities[
                    :,
                    2
                ],
            "model_confidence":
                validation_probabilities.max(
                    axis=1
                ),
            "selector_probability":
                validation_selector_probability,
            "long_net_return":
                df.iloc[
                    validation_rows
                ][
                    "long_net_return"
                ].to_numpy(),
            "short_net_return":
                df.iloc[
                    validation_rows
                ][
                    "short_net_return"
                ].to_numpy(),
            "chosen_net_return":
                chosen_returns,
        }
    )

    validation_output.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    best = search.iloc[
        0
    ]

    summary = {
        "rows":
            int(
                len(
                    df
                )
            ),
        "validation_rows":
            int(
                len(
                    validation_rows
                )
            ),
        "overall_accuracy_pct":
            float(
                overall_accuracy
            ),
        "balanced_accuracy_pct":
            float(
                balanced_accuracy
            ),
        "class_counts": {
            "SHORT":
                int(
                    counts.get(
                        -1,
                        0,
                    )
                ),
            "NO_TRADE":
                int(
                    counts.get(
                        0,
                        0,
                    )
                ),
            "LONG":
                int(
                    counts.get(
                        1,
                        0,
                    )
                ),
        },
        "classification_report":
            report,
        "confusion_matrix_labels": [
            "SHORT",
            "NO_TRADE",
            "LONG",
        ],
        "confusion_matrix":
            matrix.tolist(),
        "best_selective_candidate": {
            key:
                (
                    None
                    if pd.isna(
                        value
                    )
                    else float(
                        value
                    )
                    if isinstance(
                        value,
                        (
                            np.floating,
                            float,
                        ),
                    )
                    else int(
                        value
                    )
                    if isinstance(
                        value,
                        (
                            np.integer,
                            int,
                        ),
                    )
                    else value
                )
            for key, value
            in best.to_dict().items()
        },
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    bundle = {
        "research_only":
            True,
        "model_name":
            "cost_aware_micro_tradeability_v1",
        "feature_columns":
            all_features,
        "label_mapping": {
            0:
                -1,
            1:
                0,
            2:
                1,
        },
        "horizon_seconds":
            HORIZON_SECONDS,
        "take_profit_pct":
            TAKE_PROFIT_PCT,
        "stop_loss_pct":
            STOP_LOSS_PCT,
        "trailing_activation_pct":
            TRAILING_ACTIVATION_PCT,
        "trailing_distance_pct":
            TRAILING_DISTANCE_PCT,
        "fee_rate_per_side":
            FEE_RATE_PER_SIDE,
        "model":
            model,
        "selector_scaler":
            selector_scaler,
        "selector_model":
            selector_model,
        "note":
            (
                "Research only. Labels are based "
                "on simulated future trade paths "
                "using current paper TP/SL/trailing "
                "and fee assumptions."
            ),
    }

    joblib.dump(
        bundle,
        BUNDLE_FILE,
    )

    print()
    print("=" * 100)
    print(
        "COST-AWARE VALIDATION"
    )
    print("=" * 100)

    print(
        f"overall accuracy = "
        f"{overall_accuracy:.3f}%"
    )

    print(
        f"balanced accuracy = "
        f"{balanced_accuracy:.3f}%"
    )

    print()
    print(
        "per-class precision / recall:"
    )

    for label in [
        "SHORT",
        "NO_TRADE",
        "LONG",
    ]:
        stats = report[
            label
        ]

        print(
            f"  {label:<8} "
            f"precision="
            f"{stats['precision'] * 100:.2f}% "
            f"recall="
            f"{stats['recall'] * 100:.2f}% "
            f"support="
            f"{int(stats['support']):,}"
        )

    print()
    print(
        "Confusion matrix "
        "[SHORT, NO_TRADE, LONG]:"
    )

    print(
        matrix
    )

    print()
    print("=" * 100)
    print(
        "BEST SELECTIVE TRADE CANDIDATE"
    )
    print("=" * 100)

    print(
        f"model confidence >= "
        f"{best['model_confidence_threshold']:.2f}"
    )

    print(
        f"selector probability >= "
        f"{best['selector_threshold']:.2f}"
    )

    print(
        f"predictions = "
        f"{int(best['predictions']):,}"
    )

    print(
        f"coverage = "
        f"{best['coverage_pct']:.3f}%"
    )

    print(
        f"label accuracy = "
        f"{best['label_accuracy_pct']:.3f}%"
    )

    print(
        f"profitable path rate = "
        f"{best['profitable_path_pct']:.3f}%"
    )

    print(
        f"avg selected net return = "
        f"{best['avg_chosen_net_return_pct']:.4f}%"
    )

    print()
    print(
        "Saved:"
    )
    print(
        SUMMARY_FILE
    )
    print(
        VALIDATION_FILE
    )
    print(
        THRESHOLD_FILE
    )
    print(
        BUNDLE_FILE
    )

    print()
    print(
        "Do not connect this model to the "
        "paper trader yet. First compare its "
        "selective profitable-path rate and "
        "coverage against the existing "
        "direction-only experiments."
    )


if __name__ == "__main__":
    main()
