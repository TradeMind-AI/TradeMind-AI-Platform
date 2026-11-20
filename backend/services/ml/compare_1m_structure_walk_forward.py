from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASELINE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_1m"
)

STRUCTURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_1m_structure"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "1m_structure_walk_forward_comparison.csv"
)


FORECAST_HORIZON = 3
TARGET_THRESHOLD = 0.0010
CONFIDENCE_THRESHOLD = 0.45
ROUND_TRIP_COST = 0.0010
VALIDATION_FOLDS = 4


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


def future_return(df: pd.DataFrame) -> pd.Series:
    return (
        df["close"].shift(
            -FORECAST_HORIZON
        )
        / df["close"]
        - 1
    )


def create_target(
    returns: pd.Series,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                returns < -TARGET_THRESHOLD,
                returns > TARGET_THRESHOLD,
            ],
            [
                "DOWN",
                "UP",
            ],
            default="NEUTRAL",
        ),
        index=returns.index,
    )


def build_model() -> XGBClassifier:
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


def evaluate_version(
    version: str,
    split_dir: Path,
) -> dict:
    train_df = pd.read_csv(
        split_dir / "train.csv"
    )

    validation_df = pd.read_csv(
        split_dir / "validation.csv"
    )

    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    encoder = LabelEncoder()

    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    boundaries = np.linspace(
        0,
        len(validation_df),
        VALIDATION_FOLDS + 1,
        dtype=int,
    )

    fold_returns = []
    total_trades = 0
    total_wins = 0
    all_net_returns = []

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

        X_train = (
            expanding_train.loc[
                train_mask,
                feature_columns,
            ]
        )

        y_train = create_target(
            train_returns.loc[
                train_mask
            ]
        )

        X_eval = (
            eval_df.loc[
                eval_mask,
                feature_columns,
            ]
        )

        model = build_model()

        print(
            f"{version} fold "
            f"{fold + 1}: "
            f"{len(X_train):,} train rows, "
            f"{len(X_eval):,} eval rows..."
        )

        model.fit(
            X_train,
            encoder.transform(
                y_train
            ),
        )

        probabilities = (
            model.predict_proba(
                X_eval
            )
        )

        predictions = (
            encoder.inverse_transform(
                model.predict(
                    X_eval
                )
            )
        )

        rows = pd.DataFrame(
            {
                "future_return":
                    eval_returns.loc[
                        eval_mask
                    ].to_numpy(),
                "prediction":
                    predictions,
                "confidence":
                    probabilities.max(
                        axis=1
                    ),
                "row_index":
                    np.arange(
                        len(X_eval)
                    ),
            }
        )

        trades = rows[
            rows[
                "prediction"
            ].isin(
                [
                    "UP",
                    "DOWN",
                ]
            )
            & (
                rows[
                    "confidence"
                ]
                >= CONFIDENCE_THRESHOLD
            )
        ].copy()

        trades[
            "gross_return"
        ] = np.where(
            trades[
                "prediction"
            ] == "UP",
            trades[
                "future_return"
            ],
            -trades[
                "future_return"
            ],
        )

        trades = remove_overlaps(
            trades
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

        total_trades += len(trades)

        total_wins += int(
            (
                trades[
                    "net_return"
                ] > 0
            ).sum()
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
    else:
        win_rate = 0.0
        avg_net = 0.0

    return {
        "version": version,
        "feature_count":
            len(feature_columns),
        "trades":
            total_trades,
        "win_rate_pct":
            win_rate,
        "net_return_pct":
            sum(fold_returns),
        "avg_net_trade_pct":
            avg_net,
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


def main() -> None:
    print(
        "Running controlled market-structure "
        "feature ablation..."
    )

    print()
    print(
        "Frozen settings: "
        "3m horizon, ±0.10% target, "
        "0.45 confidence, 0.10% cost"
    )

    results = [
        evaluate_version(
            "V1_BASELINE",
            BASELINE_DIR,
        ),
        evaluate_version(
            "V1_PLUS_STRUCTURE",
            STRUCTURE_DIR,
        ),
    ]

    results_df = pd.DataFrame(
        results
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
        "=" * 130
    )
    print(
        "1M MARKET-STRUCTURE FEATURE "
        "WALK-FORWARD COMPARISON"
    )
    print(
        "=" * 130
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "win_rate_pct":
                    lambda value:
                        f"{value:.2f}%",
                "net_return_pct":
                    lambda value:
                        f"{value:.2f}%",
                "avg_net_trade_pct":
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

    baseline = results_df.iloc[0]
    structure = results_df.iloc[1]

    print()
    print(
        "Market-structure change vs baseline:"
    )

    print(
        f"Net return: "
        f"{structure['net_return_pct'] - baseline['net_return_pct']:+.2f} percentage points"
    )

    print(
        f"Average net trade: "
        f"{structure['avg_net_trade_pct'] - baseline['avg_net_trade_pct']:+.4f} percentage points"
    )

    print(
        f"Trade count: "
        f"{int(structure['trades'] - baseline['trades']):+d}"
    )

    print(
        f"Positive folds: "
        f"{int(structure['positive_folds'] - baseline['positive_folds']):+d}"
    )

    print()
    print(
        "TEST SPLITS WERE NOT USED."
    )

    print()
    print(
        f"Saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
