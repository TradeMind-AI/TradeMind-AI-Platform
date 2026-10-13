from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

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

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "staged_micro_forward"
)

BUNDLE_FILE = (
    MODEL_DIR
    / "staged_micro_forward_frozen.joblib"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "frozen_candidate_manifest.json"
)

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

# Frozen research choice from the target sweep.
MIN_NET_PROFIT_PCT = 0.00075

# Fixed forward gates. Do not retune these on forward data.
STAGE1_TRADE_THRESHOLD = 0.50
STAGE2_DIRECTION_THRESHOLD = 0.50
STAGE3_PROFITABILITY_THRESHOLD = 0.50

RANDOM_SEED = 42


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
            "No compatible microstructure snapshots found."
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

    return (
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
        .loc[
            lambda frame:
                frame["price"] > 0
        ]
        .reset_index(drop=True)
    )


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
            mean_name = f"{column}_mean_{window}"
            std_name = f"{column}_std_{window}"

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

        delta_name = f"{column}_delta_6"

        result[delta_name] = (
            result[column]
            - result[column].shift(6)
        )

        engineered.append(
            delta_name
        )

    result["flow_agreement"] = (
        result["spot_trade_flow_imbalance"]
        * result["futures_trade_flow_imbalance"]
    )

    result["depth_agreement"] = (
        result["spot_depth20_imbalance"]
        * result["futures_depth20_imbalance"]
    )

    result["velocity_agreement"] = (
        result["spot_price_velocity_bps"]
        * result["futures_price_velocity_bps"]
    )

    engineered.extend(
        [
            "flow_agreement",
            "depth_agreement",
            "velocity_agreement",
        ]
    )

    return result, engineered


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
        result["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    prices = (
        result["price"]
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

    for index in range(len(result)):
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

    result["long_net_return"] = (
        long_returns
    )
    result["short_net_return"] = (
        short_returns
    )

    return result


def binary_weights(
    labels: np.ndarray,
) -> np.ndarray:
    counts = np.bincount(
        labels.astype(int),
        minlength=2,
    )

    total = len(labels)

    mapping = {
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
            mapping[int(label)]
            for label in labels
        ],
        dtype=float,
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

    for index in range(
        min(
            24,
            X.shape[1],
        )
    ):
        frame[f"regime_{index}"] = (
            X[:, index]
        )

    return frame


def main() -> None:
    print(
        "TRADEMIND - FREEZE STAGED "
        "MICRO FORWARD CANDIDATE"
    )

    print()
    print(
        "Frozen minimum net target: "
        f"{MIN_NET_PROFIT_PCT * 100:.3f}%"
    )

    print(
        "After this script, forward data "
        "must not be used to retune the "
        "candidate."
    )

    df = load_snapshots()
    df, engineered = add_features(df)
    df = add_future_returns(df)

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
        .reset_index(drop=True)
    )

    if len(df) < 1000:
        raise RuntimeError(
            "Not enough history to freeze candidate."
        )

    X = (
        df[
            all_features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    long_return = (
        df["long_net_return"]
        .to_numpy(
            dtype=float
        )
    )

    short_return = (
        df["short_net_return"]
        .to_numpy(
            dtype=float
        )
    )

    best_return = np.maximum(
        long_return,
        short_return,
    )

    y_trade = (
        best_return
        >= MIN_NET_PROFIT_PCT
    ).astype(int)

    y_direction = np.where(
        long_return
        >= short_return,
        1,
        -1,
    )

    # Use the first 80% for Stage 1/2 and the last
    # 20% only to train Stage 3, preserving order.
    split = int(
        len(df)
        * 0.80
    )

    base_rows = np.arange(
        0,
        split,
    )

    selector_rows = np.arange(
        split,
        len(df),
    )

    stage1 = build_stage1()

    stage1.fit(
        X[
            base_rows
        ],
        y_trade[
            base_rows
        ],
        sample_weight=binary_weights(
            y_trade[
                base_rows
            ]
        ),
    )

    direction_rows = (
        base_rows[
            y_trade[
                base_rows
            ]
            == 1
        ]
    )

    if len(direction_rows) < 100:
        raise RuntimeError(
            "Too few high-quality trade rows "
            "for Stage 2."
        )

    stage2 = build_stage2()

    y_long = (
        y_direction[
            direction_rows
        ]
        == 1
    ).astype(int)

    stage2.fit(
        X[
            direction_rows
        ],
        y_long,
        sample_weight=binary_weights(
            y_long
        ),
    )

    p_trade = (
        stage1.predict_proba(
            X[
                selector_rows
            ]
        )[:, 1]
    )

    p_long = (
        stage2.predict_proba(
            X[
                selector_rows
            ]
        )[:, 1]
    )

    proposed_direction = np.where(
        p_long >= 0.5,
        1,
        -1,
    )

    proposed_return = np.where(
        proposed_direction == 1,
        long_return[
            selector_rows
        ],
        short_return[
            selector_rows
        ],
    )

    selector_target = (
        proposed_return
        >= MIN_NET_PROFIT_PCT
    ).astype(int)

    selector_frame = selector_features(
        X[
            selector_rows
        ],
        p_trade,
        p_long,
        proposed_direction,
    )

    selector_scaler = StandardScaler()

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

    stage3.fit(
        selector_X,
        selector_target,
    )

    freeze_timestamp = (
        df["timestamp"]
        .max()
        .isoformat()
    )

    bundle = {
        "model_name":
            "staged_micro_forward_frozen_v1",
        "frozen":
            True,
        "frozen_at":
            freeze_timestamp,
        "feature_columns":
            all_features,
        "raw_feature_columns":
            FEATURE_COLUMNS,
        "rolling_base":
            ROLLING_BASE,
        "stage1_trade_model":
            stage1,
        "stage2_direction_model":
            stage2,
        "stage3_selector_scaler":
            selector_scaler,
        "stage3_profitability_selector":
            stage3,
        "min_net_profit_pct":
            MIN_NET_PROFIT_PCT,
        "stage1_trade_threshold":
            STAGE1_TRADE_THRESHOLD,
        "stage2_direction_threshold":
            STAGE2_DIRECTION_THRESHOLD,
        "stage3_profitability_threshold":
            STAGE3_PROFITABILITY_THRESHOLD,
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
        "training_rows":
            int(len(df)),
        "note":
            (
                "Frozen forward-research candidate. "
                "Do not retrain or retune using "
                "post-freeze observations."
            ),
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        BUNDLE_FILE,
    )

    manifest = {
        "frozen_at":
            freeze_timestamp,
        "training_rows":
            int(len(df)),
        "min_net_profit_pct":
            MIN_NET_PROFIT_PCT,
        "min_net_profit_display":
            f"{MIN_NET_PROFIT_PCT * 100:.3f}%",
        "forward_checkpoints": {
            "minimum":
                25000,
            "preferred":
                50000,
        },
        "fixed_thresholds": {
            "stage1_trade":
                STAGE1_TRADE_THRESHOLD,
            "stage2_direction":
                STAGE2_DIRECTION_THRESHOLD,
            "stage3_profitability":
                STAGE3_PROFITABILITY_THRESHOLD,
        },
        "bundle":
            str(BUNDLE_FILE),
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Candidate frozen."
    )

    print(
        f"freeze timestamp: "
        f"{freeze_timestamp}"
    )

    print(
        f"training rows: "
        f"{len(df):,}"
    )

    print(
        "Forward checkpoints:"
    )

    print(
        "  25,000 new snapshots = "
        "minimum checkpoint"
    )

    print(
        "  50,000 new snapshots = "
        "preferred checkpoint"
    )

    print()
    print(
        "Saved:"
    )

    print(
        BUNDLE_FILE
    )

    print(
        MANIFEST_FILE
    )


if __name__ == "__main__":
    main()
