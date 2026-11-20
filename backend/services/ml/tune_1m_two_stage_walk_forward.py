from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_1m"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "1m_two_stage_walk_forward.csv"
)


FORECAST_HORIZON = 3
MOVE_THRESHOLD = 0.0010
ROUND_TRIP_COST = 0.0010
VALIDATION_FOLDS = 4

STAGE1_THRESHOLDS = [
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
]

STAGE2_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
]


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
    "future_return_1m",
}


def future_return(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        df["close"].shift(
            -FORECAST_HORIZON
        )
        / df["close"]
        - 1
    )


def create_move_target(
    returns: pd.Series,
) -> pd.Series:
    return (
        returns.abs()
        >= MOVE_THRESHOLD
    ).astype(int)


def create_direction_target(
    returns: pd.Series,
) -> pd.Series:
    return (
        returns > 0
    ).astype(int)


def build_stage1_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def build_stage2_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def remove_overlaps(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    accepted = []
    next_available = -1

    for _, row in (
        trades
        .sort_values("row_index")
        .iterrows()
    ):
        row_index = int(
            row["row_index"]
        )

        if row_index < next_available:
            continue

        accepted.append(row)

        next_available = (
            row_index
            + FORECAST_HORIZON
        )

    if not accepted:
        return (
            trades
            .iloc[0:0]
            .copy()
        )

    return (
        pd.DataFrame(accepted)
        .reset_index(drop=True)
    )


def main() -> None:
    print(
        "Running 1m two-stage "
        "walk-forward experiment..."
    )

    train_df = pd.read_csv(
        SPLIT_DIR / "train.csv"
    )

    validation_df = pd.read_csv(
        SPLIT_DIR / "validation.csv"
    )

    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    print()
    print(
        "Frozen setup:"
    )
    print(
        "81 features, 3m horizon, "
        "±0.10% move threshold, "
        "0.10% round-trip cost"
    )
    print(
        "Stage 1: TRADE vs NO TRADE"
    )
    print(
        "Stage 2: UP vs DOWN"
    )

    boundaries = np.linspace(
        0,
        len(validation_df),
        VALIDATION_FOLDS + 1,
        dtype=int,
    )

    fold_payloads = []

    for fold in range(
        VALIDATION_FOLDS
    ):
        start = boundaries[fold]
        end = boundaries[fold + 1]

        eval_df = (
            validation_df.iloc[
                start:end
            ]
            .copy()
            .reset_index(drop=True)
        )

        if fold == 0:
            expanding_train = (
                train_df
                .copy()
                .reset_index(drop=True)
            )
        else:
            expanding_train = (
                pd.concat(
                    [
                        train_df,
                        validation_df.iloc[
                            :start
                        ],
                    ],
                    ignore_index=True,
                )
            )

        train_returns = future_return(
            expanding_train
        )

        eval_returns = future_return(
            eval_df
        )

        train_mask = (
            train_returns.notna()
        )

        eval_mask = (
            eval_returns.notna()
        )

        X_train_all = (
            expanding_train.loc[
                train_mask,
                feature_columns,
            ]
        )

        train_returns_valid = (
            train_returns.loc[
                train_mask
            ]
        )

        move_target = (
            create_move_target(
                train_returns_valid
            )
        )

        directional_train_mask = (
            train_returns_valid.abs()
            >= MOVE_THRESHOLD
        )

        X_train_direction = (
            X_train_all.loc[
                directional_train_mask
            ]
        )

        direction_target = (
            create_direction_target(
                train_returns_valid.loc[
                    directional_train_mask
                ]
            )
        )

        X_eval = (
            eval_df.loc[
                eval_mask,
                feature_columns,
            ]
        )

        print(
            f"Training fold {fold + 1}: "
            f"{len(X_train_all):,} stage1 rows, "
            f"{len(X_train_direction):,} stage2 rows, "
            f"{len(X_eval):,} eval rows..."
        )

        stage1 = build_stage1_model()

        stage1.fit(
            X_train_all,
            move_target,
        )

        stage2 = build_stage2_model()

        stage2.fit(
            X_train_direction,
            direction_target,
        )

        stage1_prob_trade = (
            stage1.predict_proba(
                X_eval
            )[:, 1]
        )

        stage2_prob_up = (
            stage2.predict_proba(
                X_eval
            )[:, 1]
        )

        fold_payloads.append(
            {
                "returns":
                    eval_returns.loc[
                        eval_mask
                    ].to_numpy(),
                "prob_trade":
                    stage1_prob_trade,
                "prob_up":
                    stage2_prob_up,
            }
        )

    results = []

    for stage1_threshold in (
        STAGE1_THRESHOLDS
    ):
        for stage2_threshold in (
            STAGE2_THRESHOLDS
        ):
            fold_returns = []
            total_trades = 0
            total_wins = 0
            all_net_returns = []
            total_direction_correct = 0
            total_direction_predictions = 0

            for payload in fold_payloads:
                rows = pd.DataFrame(
                    {
                        "future_return":
                            payload["returns"],
                        "prob_trade":
                            payload["prob_trade"],
                        "prob_up":
                            payload["prob_up"],
                        "row_index":
                            np.arange(
                                len(
                                    payload[
                                        "returns"
                                    ]
                                )
                            ),
                    }
                )

                rows[
                    "direction_confidence"
                ] = np.maximum(
                    rows["prob_up"],
                    1
                    - rows["prob_up"],
                )

                rows[
                    "prediction"
                ] = np.where(
                    rows["prob_up"]
                    >= 0.5,
                    "UP",
                    "DOWN",
                )

                candidates = rows[
                    (
                        rows["prob_trade"]
                        >= stage1_threshold
                    )
                    & (
                        rows[
                            "direction_confidence"
                        ]
                        >= stage2_threshold
                    )
                ].copy()

                candidates[
                    "gross_return"
                ] = np.where(
                    candidates[
                        "prediction"
                    ] == "UP",
                    candidates[
                        "future_return"
                    ],
                    -candidates[
                        "future_return"
                    ],
                )

                trades = remove_overlaps(
                    candidates
                )

                trades[
                    "net_return"
                ] = (
                    trades[
                        "gross_return"
                    ]
                    - ROUND_TRIP_COST
                )

                fold_net = (
                    trades[
                        "net_return"
                    ].sum()
                    * 100
                )

                fold_returns.append(
                    fold_net
                )

                total_trades += len(
                    trades
                )

                total_wins += int(
                    (
                        trades[
                            "net_return"
                        ] > 0
                    ).sum()
                )

                if not trades.empty:
                    actual_up = (
                        trades[
                            "future_return"
                        ] > 0
                    )

                    predicted_up = (
                        trades[
                            "prediction"
                        ] == "UP"
                    )

                    total_direction_correct += int(
                        (
                            actual_up
                            == predicted_up
                        ).sum()
                    )

                    total_direction_predictions += len(
                        trades
                    )

                all_net_returns.extend(
                    trades[
                        "net_return"
                    ].tolist()
                )

            if total_trades > 0:
                win_rate = (
                    total_wins
                    / total_trades
                    * 100
                )

                avg_net = (
                    np.mean(
                        all_net_returns
                    )
                    * 100
                )

                median_net = (
                    np.median(
                        all_net_returns
                    )
                    * 100
                )
            else:
                win_rate = 0.0
                avg_net = 0.0
                median_net = 0.0

            if (
                total_direction_predictions
                > 0
            ):
                direction_accuracy = (
                    total_direction_correct
                    / total_direction_predictions
                    * 100
                )
            else:
                direction_accuracy = 0.0

            results.append(
                {
                    "stage1_trade_threshold":
                        stage1_threshold,
                    "stage2_direction_threshold":
                        stage2_threshold,
                    "trades":
                        total_trades,
                    "win_rate_pct":
                        win_rate,
                    "direction_accuracy_pct":
                        direction_accuracy,
                    "net_return_pct":
                        sum(
                            fold_returns
                        ),
                    "avg_net_trade_pct":
                        avg_net,
                    "median_net_trade_pct":
                        median_net,
                    "positive_folds":
                        sum(
                            value > 0
                            for value in fold_returns
                        ),
                    "fold_1_net_pct":
                        fold_returns[0],
                    "fold_2_net_pct":
                        fold_returns[1],
                    "fold_3_net_pct":
                        fold_returns[2],
                    "fold_4_net_pct":
                        fold_returns[3],
                }
            )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by=[
                "positive_folds",
                "net_return_pct",
                "avg_net_trade_pct",
                "trades",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "=" * 155
    )
    print(
        "1M TWO-STAGE WALK-FORWARD RESULTS"
    )
    print(
        "=" * 155
    )

    print(
        results_df
        .head(30)
        .to_string(
            index=False,
            formatters={
                "stage1_trade_threshold":
                    lambda value:
                        f"{value:.2f}",
                "stage2_direction_threshold":
                    lambda value:
                        f"{value:.2f}",
                "win_rate_pct":
                    lambda value:
                        f"{value:.2f}%",
                "direction_accuracy_pct":
                    lambda value:
                        f"{value:.2f}%",
                "net_return_pct":
                    lambda value:
                        f"{value:.2f}%",
                "avg_net_trade_pct":
                    lambda value:
                        f"{value:.4f}%",
                "median_net_trade_pct":
                    lambda value:
                        f"{value:.4f}%",
                "fold_1_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "fold_2_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "fold_3_net_pct":
                    lambda value:
                        f"{value:.2f}%",
                "fold_4_net_pct":
                    lambda value:
                        f"{value:.2f}%",
            },
        )
    )

    print()
    print(
        "TEST SPLIT WAS NOT USED."
    )

    print()
    print(
        "Selection rule: prefer "
        "3/4 or 4/4 positive folds, "
        "positive after-cost expectancy, "
        "and sufficient trade count."
    )

    print()
    print(
        f"Saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
