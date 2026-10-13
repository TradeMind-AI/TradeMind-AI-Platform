from __future__ import annotations

from pathlib import Path
import json

import joblib
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
    / "staged_micro_tradeability"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
)

SUMMARY_FILE = OUTPUT_DIR / "summary.json"
VALIDATION_FILE = OUTPUT_DIR / "validation_predictions.csv"
THRESHOLD_FILE = OUTPUT_DIR / "threshold_search.csv"
BUNDLE_FILE = MODEL_DIR / "staged_micro_tradeability_research.joblib"

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

# A setup is only considered genuinely tradeable if
# the simulated side finishes at least this much NET
# positive after fees and the current TP/SL/trailing
# policy. This removes tiny borderline "wins" that are
# too fragile to be useful.
MIN_NET_PROFIT_PCT = 0.0005

TRAIN_FRACTION = 0.60
SELECTOR_FRACTION = 0.20
VALIDATION_FRACTION = 0.20

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
) -> tuple[float, str]:
    if len(future_prices) == 0:
        return (
            -2.0 * FEE_RATE_PER_SIDE,
            "NO_FUTURE_DATA",
        )

    best_price = entry_price
    trailing_active = False
    exit_price = float(
        future_prices[-1]
    )
    reason = "MAX_HOLD"

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
            reason = "TAKE_PROFIT"
            break

        if gross_return <= -STOP_LOSS_PCT:
            exit_price = price
            reason = "STOP_LOSS"
            break

        if direction == 1:
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
                >= TRAILING_ACTIVATION_PCT
            ):
                trailing_active = True

            if trailing_active:
                stop_price = (
                    best_price
                    * (
                        1.0
                        - TRAILING_DISTANCE_PCT
                    )
                )

                if price <= stop_price:
                    exit_price = price
                    reason = "TRAILING_STOP"
                    break
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
                >= TRAILING_ACTIVATION_PCT
            ):
                trailing_active = True

            if trailing_active:
                stop_price = (
                    best_price
                    * (
                        1.0
                        + TRAILING_DISTANCE_PCT
                    )
                )

                if price >= stop_price:
                    exit_price = price
                    reason = "TRAILING_STOP"
                    break

    gross_return = (
        (
            exit_price / entry_price
            - 1.0
        )
        * direction
    )

    net_return = (
        gross_return
        - 2.0 * FEE_RATE_PER_SIDE
    )

    return (
        net_return,
        reason,
    )


def add_future_path_labels(
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

    tradeable = np.zeros(
        len(result),
        dtype=int,
    )

    direction = np.zeros(
        len(result),
        dtype=int,
    )

    valid = np.zeros(
        len(result),
        dtype=bool,
    )

    for index in range(
        len(result)
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

        if end_index <= index + 1:
            continue

        future_prices = prices[
            index + 1:
            end_index
        ]

        if len(future_prices) == 0:
            continue

        valid[index] = True

        long_return, _ = (
            simulate_side(
                prices[index],
                future_prices,
                1,
            )
        )

        short_return, _ = (
            simulate_side(
                prices[index],
                future_prices,
                -1,
            )
        )

        long_returns[index] = (
            long_return
        )

        short_returns[index] = (
            short_return
        )

        best_return = max(
            long_return,
            short_return,
        )

        if best_return >= MIN_NET_PROFIT_PCT:
            tradeable[index] = 1

            direction[index] = (
                1
                if long_return
                >= short_return
                else -1
            )

    result[
        "long_net_return"
    ] = long_returns

    result[
        "short_net_return"
    ] = short_returns

    result[
        "tradeable"
    ] = tradeable

    result[
        "best_direction"
    ] = direction

    result[
        "valid_future_path"
    ] = valid

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


def build_stage1() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.035,
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


def build_stage2() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.035,
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


def balance_binary_weights(
    labels: np.ndarray,
) -> np.ndarray:
    counts = np.bincount(
        labels.astype(int),
        minlength=2,
    )

    total = len(
        labels
    )

    weights = {
        class_id:
            (
                total
                / (
                    2.0
                    * count
                )
                if count > 0
                else 1.0
            )
        for class_id, count
        in enumerate(counts)
    }

    return np.array(
        [
            weights[
                int(label)
            ]
            for label in labels
        ],
        dtype=float,
    )


def selector_features(
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

    regime_count = min(
        24,
        X.shape[1],
    )

    for index in range(
        regime_count
    ):
        frame[
            f"regime_{index}"
        ] = X[
            :,
            index
        ]

    return frame


def predicted_side_returns(
    direction: np.ndarray,
    long_return: np.ndarray,
    short_return: np.ndarray,
) -> np.ndarray:
    return np.where(
        direction == 1,
        long_return,
        short_return,
    )


def search_thresholds(
    p_trade: np.ndarray,
    p_long: np.ndarray,
    selector_probability: np.ndarray,
    proposed_direction: np.ndarray,
    actual_tradeable: np.ndarray,
    actual_direction: np.ndarray,
    predicted_return: np.ndarray,
) -> pd.DataFrame:
    direction_confidence = np.maximum(
        p_long,
        1.0 - p_long,
    )

    rows = []

    total = len(
        p_trade
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
                        p_trade
                        >= trade_threshold
                    )
                    & (
                        direction_confidence
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

                if count == 0:
                    coverage = 0.0
                    profitable_rate = np.nan
                    direction_accuracy = np.nan
                    avg_net_return = np.nan
                    tradeability_precision = np.nan
                else:
                    coverage = (
                        count
                        / total
                        * 100.0
                    )

                    selected_returns = (
                        predicted_return[
                            selected
                        ]
                    )

                    profitable_rate = (
                        np.mean(
                            selected_returns
                            >= MIN_NET_PROFIT_PCT
                        )
                        * 100.0
                    )

                    avg_net_return = (
                        np.mean(
                            selected_returns
                        )
                        * 100.0
                    )

                    tradeability_precision = (
                        np.mean(
                            actual_tradeable[
                                selected
                            ]
                            == 1
                        )
                        * 100.0
                    )

                    valid_direction = (
                        selected
                        & (
                            actual_tradeable
                            == 1
                        )
                    )

                    direction_count = int(
                        valid_direction.sum()
                    )

                    direction_accuracy = (
                        np.mean(
                            proposed_direction[
                                valid_direction
                            ]
                            == actual_direction[
                                valid_direction
                            ]
                        )
                        * 100.0
                        if direction_count > 0
                        else np.nan
                    )

                rows.append(
                    {
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
                        "profitable_path_pct":
                            profitable_rate,
                        "avg_predicted_side_net_return_pct":
                            avg_net_return,
                        "tradeability_precision_pct":
                            tradeability_precision,
                        "direction_accuracy_when_tradeable_pct":
                            direction_accuracy,
                    }
                )

    result = pd.DataFrame(
        rows
    )

    # Favor profitable-path rate first, then actual
    # average net return, then useful coverage.
    result[
        "research_score"
    ] = (
        result[
            "profitable_path_pct"
        ].fillna(0.0)
        + np.clip(
            result[
                "avg_predicted_side_net_return_pct"
            ].fillna(-1.0)
            * 100.0,
            -20.0,
            20.0,
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
            "research_score",
            "profitable_path_pct",
            "avg_predicted_side_net_return_pct",
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


def main() -> None:
    print(
        "TRADEMIND - STAGED COST-AWARE "
        "MICRO TRADEABILITY RESEARCH"
    )

    print()
    print(
        "Stage 1: HIGH-QUALITY TRADE vs SKIP"
    )
    print(
        "Stage 2: LONG vs SHORT "
        "(only on tradeable training rows)"
    )
    print(
        "Stage 3: PROFITABILITY SELECTOR "
        "(learns whether the proposed trade "
        "actually clears the minimum net-profit "
        "requirement)"
    )

    print()
    print(
        "Paper-policy assumptions:"
    )
    print(
        f"TP={TAKE_PROFIT_PCT * 100:.3f}% "
        f"SL={STOP_LOSS_PCT * 100:.3f}% "
        f"trail_on="
        f"{TRAILING_ACTIVATION_PCT * 100:.3f}% "
        f"trail_dist="
        f"{TRAILING_DISTANCE_PCT * 100:.3f}% "
        f"round_trip_fee="
        f"{2 * FEE_RATE_PER_SIDE * 100:.3f}% "
        f"min_net_profit="
        f"{MIN_NET_PROFIT_PCT * 100:.3f}% "
        f"horizon={HORIZON_SECONDS}s"
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
        "Simulating future LONG and SHORT "
        "paper paths..."
    )

    df = add_future_path_labels(
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

    if len(df) < 1000:
        raise RuntimeError(
            "Not enough compatible history yet."
        )

    tradeable_rate = (
        df[
            "tradeable"
        ].mean()
        * 100.0
    )

    tradeable_rows = df[
        df[
            "tradeable"
        ]
        == 1
    ]

    long_share = (
        (
            tradeable_rows[
                "best_direction"
            ]
            == 1
        ).mean()
        * 100.0
        if len(tradeable_rows)
        else 0.0
    )

    print(
        f"ML-ready rows: "
        f"{len(df):,}"
    )

    print(
        f"tradeable rows: "
        f"{int(df['tradeable'].sum()):,} "
        f"({tradeable_rate:.2f}%)"
    )

    print(
        f"among tradeable rows: "
        f"LONG={long_share:.2f}% "
        f"SHORT={100.0 - long_share:.2f}%"
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

    X = (
        df[
            all_features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_trade = (
        df[
            "tradeable"
        ]
        .to_numpy(
            dtype=int
        )
    )

    y_direction = (
        df[
            "best_direction"
        ]
        .to_numpy(
            dtype=int
        )
    )

    stage1 = build_stage1()

    print()
    print(
        "Training Stage 1 "
        "(HIGH-QUALITY TRADE vs SKIP)..."
    )

    stage1_train_y = (
        y_trade[
            train_rows
        ]
    )

    stage1.fit(
        X[
            train_rows
        ],
        stage1_train_y,
        sample_weight=balance_binary_weights(
            stage1_train_y
        ),
    )

    direction_train_rows = (
        train_rows[
            y_trade[
                train_rows
            ]
            == 1
        ]
    )

    if len(
        direction_train_rows
    ) < 100:
        raise RuntimeError(
            "Too few tradeable training rows "
            "for Stage 2."
        )

    stage2 = build_stage2()

    stage2_train_y = (
        y_direction[
            direction_train_rows
        ]
        == 1
    ).astype(int)

    print(
        "Training Stage 2 "
        "(LONG vs SHORT)..."
    )

    stage2.fit(
        X[
            direction_train_rows
        ],
        stage2_train_y,
        sample_weight=balance_binary_weights(
            stage2_train_y
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

    selector_predicted_return = (
        predicted_side_returns(
            selector_direction,
            df.iloc[
                selector_rows
            ][
                "long_net_return"
            ].to_numpy(
                dtype=float
            ),
            df.iloc[
                selector_rows
            ][
                "short_net_return"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    selector_target = (
        selector_predicted_return
        >= MIN_NET_PROFIT_PCT
    ).astype(int)

    selector_frame = (
        selector_features(
            X[
                selector_rows
            ],
            selector_p_trade,
            selector_p_long,
            selector_direction,
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

    stage3 = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    print(
        "Training Stage 3 "
        "(proposed-trade profitability)..."
    )

    stage3.fit(
        selector_X,
        selector_target,
    )

    validation_p_trade = (
        stage1.predict_proba(
            X[
                validation_rows
            ]
        )[:, 1]
    )

    validation_p_long = (
        stage2.predict_proba(
            X[
                validation_rows
            ]
        )[:, 1]
    )

    validation_direction = np.where(
        validation_p_long
        >= 0.5,
        1,
        -1,
    )

    validation_selector_frame = (
        selector_features(
            X[
                validation_rows
            ],
            validation_p_trade,
            validation_p_long,
            validation_direction,
        )
    )

    validation_selector_probability = (
        stage3.predict_proba(
            selector_scaler.transform(
                validation_selector_frame
            )
        )[:, 1]
    )

    validation_tradeable = (
        y_trade[
            validation_rows
        ]
    )

    validation_actual_direction = (
        y_direction[
            validation_rows
        ]
    )

    validation_long_return = (
        df.iloc[
            validation_rows
        ][
            "long_net_return"
        ]
        .to_numpy(
            dtype=float
        )
    )

    validation_short_return = (
        df.iloc[
            validation_rows
        ][
            "short_net_return"
        ]
        .to_numpy(
            dtype=float
        )
    )

    validation_predicted_return = (
        predicted_side_returns(
            validation_direction,
            validation_long_return,
            validation_short_return,
        )
    )

    stage1_prediction = (
        validation_p_trade
        >= 0.5
    ).astype(int)

    stage1_accuracy = (
        accuracy_score(
            validation_tradeable,
            stage1_prediction,
        )
        * 100.0
    )

    stage1_balanced = (
        balanced_accuracy_score(
            validation_tradeable,
            stage1_prediction,
        )
        * 100.0
    )

    stage1_precision = (
        precision_score(
            validation_tradeable,
            stage1_prediction,
            zero_division=0,
        )
        * 100.0
    )

    stage1_recall = (
        recall_score(
            validation_tradeable,
            stage1_prediction,
            zero_division=0,
        )
        * 100.0
    )

    direction_mask = (
        validation_tradeable
        == 1
    )

    direction_accuracy = (
        np.mean(
            validation_direction[
                direction_mask
            ]
            == validation_actual_direction[
                direction_mask
            ]
        )
        * 100.0
        if direction_mask.any()
        else 0.0
    )

    base_profitable_rate = (
        np.mean(
            validation_predicted_return
            >= MIN_NET_PROFIT_PCT
        )
        * 100.0
    )

    base_avg_return = (
        np.mean(
            validation_predicted_return
        )
        * 100.0
    )

    search = search_thresholds(
        validation_p_trade,
        validation_p_long,
        validation_selector_probability,
        validation_direction,
        validation_tradeable,
        validation_actual_direction,
        validation_predicted_return,
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
            "actual_tradeable":
                validation_tradeable,
            "actual_best_direction":
                validation_actual_direction,
            "p_trade":
                validation_p_trade,
            "p_long":
                validation_p_long,
            "proposed_direction":
                validation_direction,
            "selector_probability":
                validation_selector_probability,
            "long_net_return":
                validation_long_return,
            "short_net_return":
                validation_short_return,
            "predicted_side_net_return":
                validation_predicted_return,
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
                len(df)
            ),
        "validation_rows":
            int(
                len(validation_rows)
            ),
        "tradeable_rate_pct":
            float(
                tradeable_rate
            ),
        "stage1_trade_accuracy_pct":
            float(
                stage1_accuracy
            ),
        "stage1_trade_balanced_accuracy_pct":
            float(
                stage1_balanced
            ),
        "stage1_trade_precision_pct":
            float(
                stage1_precision
            ),
        "stage1_trade_recall_pct":
            float(
                stage1_recall
            ),
        "stage2_direction_accuracy_on_tradeable_pct":
            float(
                direction_accuracy
            ),
        "unfiltered_proposed_trade_profitable_pct":
            float(
                base_profitable_rate
            ),
        "unfiltered_avg_predicted_side_net_return_pct":
            float(
                base_avg_return
            ),
        "best_selective_candidate":
            {
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
            "staged_micro_tradeability_v1",
        "feature_columns":
            all_features,
        "stage1_trade_model":
            stage1,
        "stage2_direction_model":
            stage2,
        "stage3_selector_scaler":
            selector_scaler,
        "stage3_profitability_selector":
            stage3,
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
        "min_net_profit_pct":
            MIN_NET_PROFIT_PCT,
        "note":
            (
                "Research only. Stage 1 learns "
                "whether either side is profitable, "
                "Stage 2 learns LONG vs SHORT only "
                "on tradeable examples, and Stage 3 "
                "learns whether the proposed side "
                "actually finishes net-positive."
            ),
    }

    joblib.dump(
        bundle,
        BUNDLE_FILE,
    )

    print()
    print("=" * 100)
    print(
        "STAGED VALIDATION"
    )
    print("=" * 100)

    print(
        f"Stage 1 HIGH-QUALITY TRADE accuracy = "
        f"{stage1_accuracy:.3f}%"
    )
    print(
        f"Stage 1 balanced accuracy = "
        f"{stage1_balanced:.3f}%"
    )
    print(
        f"Stage 1 HIGH-QUALITY TRADE precision = "
        f"{stage1_precision:.3f}%"
    )
    print(
        f"Stage 1 HIGH-QUALITY TRADE recall = "
        f"{stage1_recall:.3f}%"
    )
    print(
        f"Stage 2 direction accuracy "
        f"on actually tradeable rows = "
        f"{direction_accuracy:.3f}%"
    )
    print(
        f"Unfiltered proposed-trade "
        f"profitable rate = "
        f"{base_profitable_rate:.3f}%"
    )
    print(
        f"Unfiltered avg predicted-side "
        f"net return = "
        f"{base_avg_return:.4f}%"
    )

    print()
    print("=" * 100)
    print(
        "BEST STAGED SELECTIVE CANDIDATE"
    )
    print("=" * 100)

    print(
        f"Stage 1 p(TRADE) >= "
        f"{best['trade_threshold']:.2f}"
    )
    print(
        f"Stage 2 direction confidence >= "
        f"{best['direction_confidence_threshold']:.2f}"
    )
    print(
        f"Stage 3 profitability >= "
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
        f"profitable path rate = "
        f"{best['profitable_path_pct']:.3f}%"
    )
    print(
        f"avg predicted-side net return = "
        f"{best['avg_predicted_side_net_return_pct']:.4f}%"
    )
    print(
        f"tradeability precision = "
        f"{best['tradeability_precision_pct']:.3f}%"
    )
    print(
        "direction accuracy when actually "
        "tradeable = "
        f"{best['direction_accuracy_when_tradeable_pct']:.3f}%"
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
        "paper trader yet. We first need to "
        "see whether the staged design improves "
        "profitable-path rate AND keeps enough "
        "coverage on chronological validation."
    )


if __name__ == "__main__":
    main()
