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

RESULT_FILE = (
    OUTPUT_DIR
    / "forward_evaluation.json"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "forward_predictions.csv"
)


def load_snapshots(
    raw_features: list[str],
) -> pd.DataFrame:
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
            *raw_features,
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
                    *raw_features,
                ]
            ].copy()
        )

    if not frames:
        raise RuntimeError(
            "No compatible snapshots found."
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
        *raw_features,
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
                *raw_features,
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
    rolling_base: list[str],
) -> pd.DataFrame:
    result = df.copy()

    for column in rolling_base:
        for window in [
            3,
            6,
            12,
        ]:
            result[
                f"{column}_mean_{window}"
            ] = (
                result[column]
                .rolling(
                    window,
                    min_periods=window,
                )
                .mean()
            )

            result[
                f"{column}_std_{window}"
            ] = (
                result[column]
                .rolling(
                    window,
                    min_periods=window,
                )
                .std()
            )

        result[
            f"{column}_delta_6"
        ] = (
            result[column]
            - result[column].shift(6)
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

    return result


def simulate_side(
    entry_price: float,
    future_prices: np.ndarray,
    direction: int,
    bundle: dict,
) -> float:
    if len(future_prices) == 0:
        return np.nan

    tp = float(
        bundle["take_profit_pct"]
    )

    sl = float(
        bundle["stop_loss_pct"]
    )

    trail_on = float(
        bundle[
            "trailing_activation_pct"
        ]
    )

    trail_distance = float(
        bundle[
            "trailing_distance_pct"
        ]
    )

    fee = float(
        bundle["fee_rate_per_side"]
    )

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

        if gross_return >= tp:
            exit_price = price
            break

        if gross_return <= -sl:
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
                >= trail_on
            ):
                trailing_active = True

            if (
                trailing_active
                and price
                <= best_price
                * (
                    1.0
                    - trail_distance
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
                >= trail_on
            ):
                trailing_active = True

            if (
                trailing_active
                and price
                >= best_price
                * (
                    1.0
                    + trail_distance
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
        * fee
    )


def main() -> None:
    if not BUNDLE_FILE.exists():
        raise FileNotFoundError(
            "Frozen bundle missing. Run "
            "freeze_staged_micro_forward_candidate.py "
            "first."
        )

    bundle = joblib.load(
        BUNDLE_FILE
    )

    frozen_at = pd.Timestamp(
        bundle["frozen_at"]
    )

    raw_features = list(
        bundle[
            "raw_feature_columns"
        ]
    )

    rolling_base = list(
        bundle[
            "rolling_base"
        ]
    )

    feature_columns = list(
        bundle[
            "feature_columns"
        ]
    )

    df = load_snapshots(
        raw_features
    )

    total_new_raw = int(
        (
            df["timestamp"]
            > frozen_at
        ).sum()
    )

    df = add_features(
        df,
        rolling_base,
    )

    # Keep enough pre-freeze rows to construct rolling
    # features, but evaluate only genuinely new rows.
    forward = df[
        df["timestamp"]
        > frozen_at
    ].copy()

    forward = (
        forward.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=feature_columns
        )
        .reset_index(drop=True)
    )

    if len(forward) == 0:
        print(
            "No post-freeze snapshots yet."
        )
        return

    horizon_seconds = int(
        bundle[
            "horizon_seconds"
        ]
    )

    all_times = (
        df["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    all_prices = (
        df["price"]
        .to_numpy(
            dtype=float
        )
    )

    horizon_ns = int(
        horizon_seconds
        * 1_000_000_000
    )

    rows = []

    for _, row in forward.iterrows():
        timestamp = pd.Timestamp(
            row["timestamp"]
        )

        timestamp_ns = int(
            timestamp.value
        )

        start_index = int(
            np.searchsorted(
                all_times,
                timestamp_ns,
                side="left",
            )
        )

        end_index = int(
            np.searchsorted(
                all_times,
                timestamp_ns
                + horizon_ns,
                side="right",
            )
        )

        if end_index <= start_index + 1:
            continue

        future_prices = all_prices[
            start_index + 1:
            end_index
        ]

        if len(future_prices) == 0:
            continue

        X = np.array(
            [
                float(
                    row[column]
                )
                for column
                in feature_columns
            ],
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        )

        p_trade = float(
            bundle[
                "stage1_trade_model"
            ].predict_proba(
                X
            )[0, 1]
        )

        p_long = float(
            bundle[
                "stage2_direction_model"
            ].predict_proba(
                X
            )[0, 1]
        )

        direction = (
            1
            if p_long >= 0.5
            else -1
        )

        direction_confidence = max(
            p_long,
            1.0 - p_long,
        )

        selector_frame = pd.DataFrame(
            {
                "p_trade":
                    [p_trade],
                "p_long":
                    [p_long],
                "direction_confidence":
                    [
                        direction_confidence
                    ],
                "proposed_direction":
                    [direction],
                "joint_confidence":
                    [
                        p_trade
                        * direction_confidence
                    ],
            }
        )

        for index in range(
            min(
                24,
                X.shape[1],
            )
        ):
            selector_frame[
                f"regime_{index}"
            ] = [
                float(
                    X[
                        0,
                        index
                    ]
                )
            ]

        p_profit = float(
            bundle[
                "stage3_profitability_selector"
            ].predict_proba(
                bundle[
                    "stage3_selector_scaler"
                ].transform(
                    selector_frame
                )
            )[0, 1]
        )

        selected = bool(
            p_trade
            >= float(
                bundle[
                    "stage1_trade_threshold"
                ]
            )
            and direction_confidence
            >= float(
                bundle[
                    "stage2_direction_threshold"
                ]
            )
            and p_profit
            >= float(
                bundle[
                    "stage3_profitability_threshold"
                ]
            )
        )

        proposed_return = simulate_side(
            float(
                row["price"]
            ),
            future_prices,
            direction,
            bundle,
        )

        qualifies = bool(
            proposed_return
            >= float(
                bundle[
                    "min_net_profit_pct"
                ]
            )
        )

        rows.append(
            {
                "timestamp":
                    timestamp,
                "price":
                    float(
                        row["price"]
                    ),
                "p_trade":
                    p_trade,
                "p_long":
                    p_long,
                "direction":
                    direction,
                "direction_confidence":
                    direction_confidence,
                "p_profit":
                    p_profit,
                "selected":
                    int(
                        selected
                    ),
                "net_return":
                    proposed_return,
                "qualifies":
                    int(
                        qualifies
                    ),
            }
        )

    if not rows:
        print(
            "Post-freeze snapshots exist, but "
            "none have a complete 60-second "
            "future path yet."
        )
        return

    result = pd.DataFrame(
        rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    selected = result[
        result["selected"]
        == 1
    ]

    selected_count = len(
        selected
    )

    selected_qualified = (
        (
            selected["qualifies"]
            == 1
        ).mean()
        * 100.0
        if selected_count
        else None
    )

    selected_avg_return = (
        selected["net_return"].mean()
        * 100.0
        if selected_count
        else None
    )

    summary = {
        "frozen_at":
            str(
                frozen_at
            ),
        "new_raw_snapshots":
            total_new_raw,
        "evaluable_forward_rows":
            int(
                len(result)
            ),
        "selected_predictions":
            int(
                selected_count
            ),
        "coverage_pct":
            (
                selected_count
                / len(result)
                * 100.0
                if len(result)
                else 0.0
            ),
        "qualified_trade_rate_pct":
            (
                None
                if selected_qualified
                is None
                else float(
                    selected_qualified
                )
            ),
        "avg_selected_net_return_pct":
            (
                None
                if selected_avg_return
                is None
                else float(
                    selected_avg_return
                )
            ),
        "minimum_checkpoint_new_snapshots":
            25000,
        "preferred_checkpoint_new_snapshots":
            50000,
        "minimum_checkpoint_progress_pct":
            min(
                total_new_raw
                / 25000
                * 100.0,
                100.0,
            ),
        "preferred_checkpoint_progress_pct":
            min(
                total_new_raw
                / 50000
                * 100.0,
                100.0,
            ),
    }

    RESULT_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "TRADEMIND - FROZEN STAGED "
        "MICRO FORWARD EVALUATION"
    )

    print()
    print(
        f"Frozen at: "
        f"{frozen_at}"
    )

    print(
        f"New snapshots: "
        f"{total_new_raw:,}"
    )

    print(
        f"Evaluable forward rows: "
        f"{len(result):,}"
    )

    print(
        "25k checkpoint progress: "
        f"{summary['minimum_checkpoint_progress_pct']:.2f}%"
    )

    print(
        "50k checkpoint progress: "
        f"{summary['preferred_checkpoint_progress_pct']:.2f}%"
    )

    print()
    print(
        f"Selected predictions: "
        f"{selected_count:,}"
    )

    print(
        f"Coverage: "
        f"{summary['coverage_pct']:.3f}%"
    )

    if selected_count:
        print(
            "Qualified-trade rate: "
            f"{selected_qualified:.3f}%"
        )

        print(
            "Average selected net return: "
            f"{selected_avg_return:.4f}%"
        )
    else:
        print(
            "No selected forward trades yet."
        )

    print()
    print(
        "This script does not retrain or "
        "retune the frozen candidate."
    )


if __name__ == "__main__":
    main()
