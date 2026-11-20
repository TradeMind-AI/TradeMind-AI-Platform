from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
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
    / "staged_profit_target_sweep"
)

RESULT_FILE = OUTPUT_DIR / "profit_target_results.csv"
BEST_FILE = OUTPUT_DIR / "best_candidates.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"

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

HORIZON_SECONDS = 60
TAKE_PROFIT_PCT = 0.0020
STOP_LOSS_PCT = 0.0012
TRAILING_ACTIVATION_PCT = 0.0010
TRAILING_DISTANCE_PCT = 0.0008
FEE_RATE_PER_SIDE = 0.0005

PROFIT_TARGETS = [
    0.0005,
    0.00075,
    0.0010,
    0.00125,
    0.0015,
]

TRAIN_FRACTION = 0.60
SELECTOR_FRACTION = 0.20
RANDOM_SEED = 42

TRADE_THRESHOLDS = np.arange(
    0.40,
    0.91,
    0.02,
)
DIRECTION_THRESHOLDS = np.arange(
    0.50,
    0.91,
    0.02,
)
SELECTOR_THRESHOLDS = np.arange(
    0.50,
    0.91,
    0.02,
)


def load_snapshots() -> pd.DataFrame:
    files = sorted(
        SNAPSHOT_DIR.glob(
            "feature_snapshots*.csv"
        )
    )

    frames = []

    for path in files:
        try:
            frame = pd.read_csv(
                path,
                on_bad_lines="skip",
                low_memory=False,
            )
        except Exception:
            continue

        required = {
            "timestamp",
            "price",
            *FEATURE_COLUMNS,
        }

        if not required.issubset(
            frame.columns
        ):
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
            "No compatible microstructure "
            "snapshot files found."
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


def add_features(
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
) -> float:
    if len(future_prices) == 0:
        return np.nan

    best_price = entry_price
    trailing_active = False
    exit_price = float(
        future_prices[-1]
    )

    for value in future_prices:
        price = float(value)

        gross_return = (
            (
                price / entry_price
                - 1.0
            )
            * direction
        )

        if gross_return >= TAKE_PROFIT_PCT:
            exit_price = price
            break

        if gross_return <= -STOP_LOSS_PCT:
            exit_price = price
            break

        if direction == 1:
            best_price = max(
                best_price,
                price,
            )

            if (
                not trailing_active
                and (
                    best_price / entry_price
                    - 1.0
                )
                >= TRAILING_ACTIVATION_PCT
            ):
                trailing_active = True

            if (
                trailing_active
                and price
                <= best_price
                * (
                    1.0
                    - TRAILING_DISTANCE_PCT
                )
            ):
                exit_price = price
                break
        else:
            best_price = min(
                best_price,
                price,
            )

            if (
                not trailing_active
                and (
                    entry_price / best_price
                    - 1.0
                )
                >= TRAILING_ACTIVATION_PCT
            ):
                trailing_active = True

            if (
                trailing_active
                and price
                >= best_price
                * (
                    1.0
                    + TRAILING_DISTANCE_PCT
                )
            ):
                exit_price = price
                break

    gross_return = (
        (
            exit_price / entry_price
            - 1.0
        )
        * direction
    )

    return (
        gross_return
        - 2.0
        * FEE_RATE_PER_SIDE
    )


def add_future_returns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    times = (
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
            dtype=float
        )
    )

    horizon_ns = int(
        HORIZON_SECONDS
        * 1_000_000_000
    )

    long_returns = np.full(
        len(result),
        np.nan,
    )

    short_returns = np.full(
        len(result),
        np.nan,
    )

    for index in range(
        len(result)
    ):
        end = int(
            np.searchsorted(
                times,
                times[index]
                + horizon_ns,
                side="right",
            )
        )

        if end <= index + 1:
            continue

        future = prices[
            index + 1:
            end
        ]

        long_returns[index] = (
            simulate_side(
                prices[index],
                future,
                1,
            )
        )

        short_returns[index] = (
            simulate_side(
                prices[index],
                future,
                -1,
            )
        )

    result[
        "long_net_return"
    ] = long_returns

    result[
        "short_net_return"
    ] = short_returns

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


def class_weights(
    labels: np.ndarray,
) -> np.ndarray:
    counts = np.bincount(
        labels.astype(int),
        minlength=2,
    )

    total = len(labels)

    mapping = {
        index:
            (
                total
                / (
                    2.0
                    * count
                )
                if count > 0
                else 1.0
            )
        for index, count
        in enumerate(counts)
    }

    return np.array(
        [
            mapping[
                int(label)
            ]
            for label in labels
        ],
        dtype=float,
    )


def model_trade() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=450,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=6,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )


def model_direction() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=450,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.10,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED + 1,
        n_jobs=-1,
        tree_method="hist",
    )


def selector_frame(
    X: np.ndarray,
    p_trade: np.ndarray,
    p_long: np.ndarray,
    proposed_direction: np.ndarray,
) -> pd.DataFrame:
    direction_confidence = np.maximum(
        p_long,
        1.0 - p_long,
    )

    frame = pd.DataFrame(
        {
            "p_trade":
                p_trade,
            "p_long":
                p_long,
            "direction_confidence":
                direction_confidence,
            "proposed_direction":
                proposed_direction,
            "joint_confidence":
                p_trade
                * direction_confidence,
        }
    )

    for index in range(
        min(
            20,
            X.shape[1],
        )
    ):
        frame[
            f"regime_{index}"
        ] = X[
            :,
            index
        ]

    return frame


def evaluate_target(
    df: pd.DataFrame,
    X: np.ndarray,
    target: float,
    train_rows: np.ndarray,
    selector_rows: np.ndarray,
    validation_rows: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    long_return = (
        df[
            "long_net_return"
        ]
        .to_numpy(
            dtype=float
        )
    )

    short_return = (
        df[
            "short_net_return"
        ]
        .to_numpy(
            dtype=float
        )
    )

    best_return = np.maximum(
        long_return,
        short_return,
    )

    tradeable = (
        best_return
        >= target
    ).astype(int)

    best_direction = np.where(
        long_return
        >= short_return,
        1,
        -1,
    )

    stage1 = model_trade()

    stage1_train = (
        tradeable[
            train_rows
        ]
    )

    stage1.fit(
        X[
            train_rows
        ],
        stage1_train,
        sample_weight=class_weights(
            stage1_train
        ),
    )

    direction_train = (
        train_rows[
            tradeable[
                train_rows
            ]
            == 1
        ]
    )

    if len(direction_train) < 100:
        return (
            {
                "target_pct":
                    target
                    * 100.0,
                "status":
                    "insufficient_direction_rows",
            },
            pd.DataFrame(),
        )

    stage2 = model_direction()

    stage2_train = (
        best_direction[
            direction_train
        ]
        == 1
    ).astype(int)

    stage2.fit(
        X[
            direction_train
        ],
        stage2_train,
        sample_weight=class_weights(
            stage2_train
        ),
    )

    selector_p_trade = (
        stage1.predict_proba(
            X[
                selector_rows
            ]
        )[:, 1]
    )

    selector_p_long = (
        stage2.predict_proba(
            X[
                selector_rows
            ]
        )[:, 1]
    )

    selector_direction = np.where(
        selector_p_long
        >= 0.5,
        1,
        -1,
    )

    selector_return = np.where(
        selector_direction
        == 1,
        long_return[
            selector_rows
        ],
        short_return[
            selector_rows
        ],
    )

    selector_target = (
        selector_return
        >= target
    ).astype(int)

    selector_data = selector_frame(
        X[
            selector_rows
        ],
        selector_p_trade,
        selector_p_long,
        selector_direction,
    )

    scaler = StandardScaler()

    selector_X = scaler.fit_transform(
        selector_data
    )

    stage3 = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    stage3.fit(
        selector_X,
        selector_target,
    )

    val_p_trade = (
        stage1.predict_proba(
            X[
                validation_rows
            ]
        )[:, 1]
    )

    val_p_long = (
        stage2.predict_proba(
            X[
                validation_rows
            ]
        )[:, 1]
    )

    val_direction = np.where(
        val_p_long
        >= 0.5,
        1,
        -1,
    )

    val_selector_data = selector_frame(
        X[
            validation_rows
        ],
        val_p_trade,
        val_p_long,
        val_direction,
    )

    val_p_selector = (
        stage3.predict_proba(
            scaler.transform(
                val_selector_data
            )
        )[:, 1]
    )

    val_tradeable = (
        tradeable[
            validation_rows
        ]
    )

    val_best_direction = (
        best_direction[
            validation_rows
        ]
    )

    val_return = np.where(
        val_direction
        == 1,
        long_return[
            validation_rows
        ],
        short_return[
            validation_rows
        ],
    )

    stage1_prediction = (
        val_p_trade
        >= 0.5
    ).astype(int)

    stage1_accuracy = (
        accuracy_score(
            val_tradeable,
            stage1_prediction,
        )
        * 100.0
    )

    stage1_balanced = (
        balanced_accuracy_score(
            val_tradeable,
            stage1_prediction,
        )
        * 100.0
    )

    stage1_precision = (
        precision_score(
            val_tradeable,
            stage1_prediction,
            zero_division=0,
        )
        * 100.0
    )

    stage1_recall = (
        recall_score(
            val_tradeable,
            stage1_prediction,
            zero_division=0,
        )
        * 100.0
    )

    actual_tradeable_mask = (
        val_tradeable
        == 1
    )

    direction_accuracy = (
        np.mean(
            val_direction[
                actual_tradeable_mask
            ]
            == val_best_direction[
                actual_tradeable_mask
            ]
        )
        * 100.0
        if actual_tradeable_mask.any()
        else 0.0
    )

    direction_confidence = np.maximum(
        val_p_long,
        1.0 - val_p_long,
    )

    rows = []

    total = len(
        validation_rows
    )

    for trade_threshold in (
        TRADE_THRESHOLDS
    ):
        for direction_threshold in (
            DIRECTION_THRESHOLDS
        ):
            for selector_threshold in (
                SELECTOR_THRESHOLDS
            ):
                selected = (
                    (
                        val_p_trade
                        >= trade_threshold
                    )
                    & (
                        direction_confidence
                        >= direction_threshold
                    )
                    & (
                        val_p_selector
                        >= selector_threshold
                    )
                )

                count = int(
                    selected.sum()
                )

                if count == 0:
                    coverage = 0.0
                    qualified_rate = np.nan
                    avg_return = np.nan
                else:
                    selected_return = (
                        val_return[
                            selected
                        ]
                    )

                    coverage = (
                        count
                        / total
                        * 100.0
                    )

                    qualified_rate = (
                        np.mean(
                            selected_return
                            >= target
                        )
                        * 100.0
                    )

                    avg_return = (
                        np.mean(
                            selected_return
                        )
                        * 100.0
                    )

                rows.append(
                    {
                        "target_pct":
                            target
                            * 100.0,
                        "trade_threshold":
                            trade_threshold,
                        "direction_confidence_threshold":
                            direction_threshold,
                        "selector_threshold":
                            selector_threshold,
                        "predictions":
                            count,
                        "coverage_pct":
                            coverage,
                        "qualified_trade_pct":
                            qualified_rate,
                        "avg_net_return_pct":
                            avg_return,
                    }
                )

    search = pd.DataFrame(
        rows
    )

    search[
        "score"
    ] = (
        search[
            "qualified_trade_pct"
        ].fillna(0.0)
        + np.clip(
            search[
                "avg_net_return_pct"
            ].fillna(-1.0)
            * 100.0,
            -20.0,
            20.0,
        )
        + np.minimum(
            search[
                "coverage_pct"
            ],
            8.0,
        )
        * 2.0
    )

    search = search.sort_values(
        [
            "score",
            "qualified_trade_pct",
            "avg_net_return_pct",
            "coverage_pct",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    best = search.iloc[
        0
    ]

    summary = {
        "target_pct":
            target
            * 100.0,
        "status":
            "ok",
        "tradeable_rate_pct":
            tradeable.mean()
            * 100.0,
        "stage1_accuracy_pct":
            stage1_accuracy,
        "stage1_balanced_accuracy_pct":
            stage1_balanced,
        "stage1_precision_pct":
            stage1_precision,
        "stage1_recall_pct":
            stage1_recall,
        "stage2_direction_accuracy_pct":
            direction_accuracy,
        "best_predictions":
            int(
                best[
                    "predictions"
                ]
            ),
        "best_coverage_pct":
            float(
                best[
                    "coverage_pct"
                ]
            ),
        "best_qualified_trade_pct":
            float(
                best[
                    "qualified_trade_pct"
                ]
            )
            if pd.notna(
                best[
                    "qualified_trade_pct"
                ]
            )
            else None,
        "best_avg_net_return_pct":
            float(
                best[
                    "avg_net_return_pct"
                ]
            )
            if pd.notna(
                best[
                    "avg_net_return_pct"
                ]
            )
            else None,
        "best_trade_threshold":
            float(
                best[
                    "trade_threshold"
                ]
            ),
        "best_direction_threshold":
            float(
                best[
                    "direction_confidence_threshold"
                ]
            ),
        "best_selector_threshold":
            float(
                best[
                    "selector_threshold"
                ]
            ),
    }

    return (
        summary,
        search.head(
            25
        ),
    )


def main() -> None:
    print(
        "TRADEMIND - STAGED PROFIT TARGET "
        "SWEEP"
    )

    print()
    print(
        "Testing minimum NET profit labels:"
    )

    print(
        ", ".join(
            f"{value * 100:.3f}%"
            for value in PROFIT_TARGETS
        )
    )

    print(
        "Same chronological data and same "
        "staged architecture are used for "
        "every target."
    )

    df = load_snapshots()

    print()
    print(
        f"compatible snapshots: "
        f"{len(df):,}"
    )

    df, engineered = add_features(
        df
    )

    print(
        "Simulating 60-second future paths "
        "once for all target levels..."
    )

    df = add_future_returns(
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
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(df) < 1000:
        raise RuntimeError(
            "Not enough compatible history."
        )

    print(
        f"ML-ready rows: "
        f"{len(df):,}"
    )

    X = (
        df[
            all_features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    (
        train_rows,
        selector_rows,
        validation_rows,
    ) = split_rows(
        len(df)
    )

    print(
        "chronological split: "
        f"train={len(train_rows):,}, "
        f"selector={len(selector_rows):,}, "
        f"validation={len(validation_rows):,}"
    )

    summaries = []
    best_frames = []

    for target in PROFIT_TARGETS:
        print()
        print(
            "=" * 90
        )
        print(
            "MIN NET PROFIT TARGET "
            f"{target * 100:.3f}%"
        )
        print(
            "=" * 90
        )

        summary, best = evaluate_target(
            df,
            X,
            target,
            train_rows,
            selector_rows,
            validation_rows,
        )

        summaries.append(
            summary
        )

        if not best.empty:
            best_frames.append(
                best
            )

        if summary[
            "status"
        ] != "ok":
            print(
                summary[
                    "status"
                ]
            )
            continue

        print(
            f"tradeable rate = "
            f"{summary['tradeable_rate_pct']:.2f}%"
        )
        print(
            f"Stage 1 precision = "
            f"{summary['stage1_precision_pct']:.2f}%"
        )
        print(
            f"Stage 1 recall = "
            f"{summary['stage1_recall_pct']:.2f}%"
        )
        print(
            f"Stage 1 balanced accuracy = "
            f"{summary['stage1_balanced_accuracy_pct']:.2f}%"
        )
        print(
            f"Stage 2 direction accuracy = "
            f"{summary['stage2_direction_accuracy_pct']:.2f}%"
        )
        print(
            f"best predictions = "
            f"{summary['best_predictions']:,}"
        )
        print(
            f"best coverage = "
            f"{summary['best_coverage_pct']:.3f}%"
        )
        print(
            f"best qualified-trade rate = "
            f"{summary['best_qualified_trade_pct']:.3f}%"
        )
        print(
            f"best avg net return = "
            f"{summary['best_avg_net_return_pct']:.4f}%"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = pd.DataFrame(
        summaries
    )

    result.to_csv(
        RESULT_FILE,
        index=False,
    )

    if best_frames:
        pd.concat(
            best_frames,
            ignore_index=True,
        ).to_csv(
            BEST_FILE,
            index=False,
        )

    valid = result[
        result[
            "status"
        ]
        == "ok"
    ].copy()

    if not valid.empty:
        valid[
            "selection_score"
        ] = (
            valid[
                "best_qualified_trade_pct"
            ].fillna(0.0)
            + valid[
                "best_avg_net_return_pct"
            ].fillna(-1.0)
            * 100.0
            + np.minimum(
                valid[
                    "best_coverage_pct"
                ],
                8.0,
            )
            * 2.0
        )

        winner = valid.sort_values(
            [
                "selection_score",
                "best_qualified_trade_pct",
                "best_coverage_pct",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        ).iloc[
            0
        ].to_dict()
    else:
        winner = None

    summary_json = {
        "profit_targets_pct": [
            value
            * 100.0
            for value in PROFIT_TARGETS
        ],
        "winner":
            winner,
        "note":
            (
                "Research-only threshold sweep. "
                "The winning target must still "
                "have enough forward coverage "
                "before it is considered for "
                "paper execution."
            ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary_json,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "PROFIT TARGET SWEEP SUMMARY"
    )
    print("=" * 100)

    display_columns = [
        "target_pct",
        "tradeable_rate_pct",
        "stage1_precision_pct",
        "stage1_recall_pct",
        "stage1_balanced_accuracy_pct",
        "stage2_direction_accuracy_pct",
        "best_predictions",
        "best_coverage_pct",
        "best_qualified_trade_pct",
        "best_avg_net_return_pct",
    ]

    print(
        result[
            [
                column
                for column
                in display_columns
                if column
                in result.columns
            ]
        ].to_string(
            index=False
        )
    )

    if winner is not None:
        print()
        print(
            "Best research candidate:"
        )
        print(
            f"minimum net target = "
            f"{winner['target_pct']:.3f}%"
        )
        print(
            f"coverage = "
            f"{winner['best_coverage_pct']:.3f}%"
        )
        print(
            f"qualified-trade rate = "
            f"{winner['best_qualified_trade_pct']:.3f}%"
        )
        print(
            f"avg net return = "
            f"{winner['best_avg_net_return_pct']:.4f}%"
        )
        print(
            f"selected predictions = "
            f"{int(winner['best_predictions']):,}"
        )

    print()
    print(
        "Saved:"
    )
    print(
        RESULT_FILE
    )
    print(
        BEST_FILE
    )
    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()
