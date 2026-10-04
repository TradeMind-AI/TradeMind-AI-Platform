from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_10M = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_10m"
)

SPLIT_2M = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_2m"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "10m_2m_confirmation_walk_forward.csv"
)


VALIDATION_FOLDS = 4
ROUND_TRIP_COST = 0.0010

TEN_MIN_HORIZON_BARS = 3
TEN_MIN_TARGET = 0.0020
TEN_MIN_CONFIDENCE = 0.45

TWO_MIN_HORIZON_BARS = 3
TWO_MIN_TARGET = 0.0010
TWO_MIN_CONFIDENCE = 0.50


NON_FEATURE_10M = {
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
    "future_return_10m",
}

NON_FEATURE_2M = {
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
    "future_return_2m",
}


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


def future_return(
    df: pd.DataFrame,
    horizon_bars: int,
) -> pd.Series:
    return (
        df["close"].shift(-horizon_bars)
        / df["close"]
        - 1
    )


def make_target(
    returns: pd.Series,
    threshold: float,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                returns < -threshold,
                returns > threshold,
            ],
            [
                "DOWN",
                "UP",
            ],
            default="NEUTRAL",
        ),
        index=returns.index,
    )


def remove_overlaps(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    accepted = []
    next_available = -1

    for _, row in trades.sort_values(
        "row_index"
    ).iterrows():
        row_index = int(
            row["row_index"]
        )

        if row_index < next_available:
            continue

        accepted.append(row)

        next_available = (
            row_index
            + TEN_MIN_HORIZON_BARS
        )

    if not accepted:
        return trades.iloc[0:0].copy()

    return (
        pd.DataFrame(accepted)
        .reset_index(drop=True)
    )


def load_split_dir(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(
        path / "train.csv"
    )
    validation = pd.read_csv(
        path / "validation.csv"
    )

    for frame in [
        train,
        validation,
    ]:
        frame["open_time"] = pd.to_datetime(
            frame["open_time"],
            utc=True,
        )
        frame["close_time"] = pd.to_datetime(
            frame["close_time"],
            utc=True,
        )

    return train, validation


def train_predict(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    horizon_bars: int,
    target_threshold: float,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    train_returns = future_return(
        train_df,
        horizon_bars,
    )

    train_mask = (
        train_returns.notna()
    )

    X_train = train_df.loc[
        train_mask,
        feature_columns,
    ]

    y_train = make_target(
        train_returns.loc[
            train_mask
        ],
        target_threshold,
    )

    model = build_model()

    model.fit(
        X_train,
        encoder.transform(
            y_train
        ),
    )

    probabilities = model.predict_proba(
        eval_df[feature_columns]
    )

    predictions = (
        encoder.inverse_transform(
            model.predict(
                eval_df[feature_columns]
            )
        )
    )

    result = eval_df[
        [
            "open_time",
            "close_time",
            "close",
        ]
    ].copy()

    result["prediction"] = predictions
    result["confidence"] = (
        probabilities.max(axis=1)
    )

    return result


def score_variant(
    rows: pd.DataFrame,
    variant: str,
) -> dict:
    primary = rows[
        rows["ten_signal"]
    ].copy()

    if variant == "TEN_MIN_ONLY":
        selected = primary.copy()

    elif variant == "STRICT_AGREEMENT":
        selected = primary[
            (
                primary["two_signal"]
            )
            & (
                primary["two_prediction"]
                == primary["ten_prediction"]
            )
        ].copy()

    elif variant == "TWO_MIN_VETO":
        opposite_high_conf = (
            primary["two_signal"]
            & (
                primary["two_prediction"]
                != primary["ten_prediction"]
            )
        )

        selected = primary[
            ~opposite_high_conf
        ].copy()

    else:
        raise ValueError(
            f"Unknown variant: {variant}"
        )

    selected[
        "gross_return"
    ] = np.where(
        selected[
            "ten_prediction"
        ] == "UP",
        selected[
            "future_return_10m_horizon"
        ],
        -selected[
            "future_return_10m_horizon"
        ],
    )

    selected = remove_overlaps(
        selected
    )

    selected[
        "net_return"
    ] = (
        selected["gross_return"]
        - ROUND_TRIP_COST
    )

    trades = len(selected)
    wins = int(
        (
            selected["net_return"] > 0
        ).sum()
    )

    net_return_pct = (
        selected["net_return"].sum()
        * 100
    )

    if trades:
        win_rate_pct = (
            wins / trades * 100
        )
        avg_net_trade_pct = (
            selected[
                "net_return"
            ].mean()
            * 100
        )
        median_net_trade_pct = (
            selected[
                "net_return"
            ].median()
            * 100
        )
    else:
        win_rate_pct = 0.0
        avg_net_trade_pct = 0.0
        median_net_trade_pct = 0.0

    return {
        "trades": trades,
        "wins": wins,
        "win_rate_pct": win_rate_pct,
        "net_return_pct": net_return_pct,
        "avg_net_trade_pct":
            avg_net_trade_pct,
        "median_net_trade_pct":
            median_net_trade_pct,
    }


def main() -> None:
    print(
        "Running 10m primary + 2m "
        "confirmation walk-forward test..."
    )

    train_10m, val_10m = load_split_dir(
        SPLIT_10M
    )

    train_2m, val_2m = load_split_dir(
        SPLIT_2M
    )

    all_2m = (
        pd.concat(
            [
                train_2m,
                val_2m,
            ],
            ignore_index=True,
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    features_10m = [
        column
        for column in train_10m.columns
        if column not in NON_FEATURE_10M
    ]

    features_2m = [
        column
        for column in train_2m.columns
        if column not in NON_FEATURE_2M
    ]

    encoder = LabelEncoder()
    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    print()
    print(
        "10m primary: "
        "30m horizon, ±0.20% target, "
        "confidence >= 0.45"
    )
    print(
        "2m confirmation: "
        "6m horizon, ±0.10% target, "
        "confidence >= 0.50"
    )
    print(
        "Cost: 0.10% round trip"
    )
    print(
        "Alignment rule: only fully closed "
        "2m candles available by each 10m "
        "candle close are used."
    )

    boundaries = np.linspace(
        0,
        len(val_10m),
        VALIDATION_FOLDS + 1,
        dtype=int,
    )

    variants = [
        "TEN_MIN_ONLY",
        "STRICT_AGREEMENT",
        "TWO_MIN_VETO",
    ]

    fold_results = {
        variant: []
        for variant in variants
    }

    for fold in range(
        VALIDATION_FOLDS
    ):
        start = boundaries[fold]
        end = boundaries[fold + 1]

        eval_10m = (
            val_10m.iloc[
                start:end
            ]
            .copy()
            .reset_index(drop=True)
        )

        eval_start = (
            eval_10m[
                "open_time"
            ].iloc[0]
        )

        eval_end_close = (
            eval_10m[
                "close_time"
            ].iloc[-1]
        )

        if fold == 0:
            expanding_10m = (
                train_10m.copy()
                .reset_index(drop=True)
            )
        else:
            expanding_10m = (
                pd.concat(
                    [
                        train_10m,
                        val_10m.iloc[
                            :start
                        ],
                    ],
                    ignore_index=True,
                )
            )

        expanding_2m = (
            all_2m[
                all_2m["open_time"]
                < eval_start
            ]
            .copy()
            .reset_index(drop=True)
        )

        eval_2m = (
            all_2m[
                (
                    all_2m["open_time"]
                    >= eval_start
                )
                & (
                    all_2m["close_time"]
                    <= eval_end_close
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        ten_eval_returns = future_return(
            eval_10m,
            TEN_MIN_HORIZON_BARS,
        )

        ten_eval_mask = (
            ten_eval_returns.notna()
        )

        eval_10m_scored = (
            eval_10m.loc[
                ten_eval_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        ten_eval_returns = (
            ten_eval_returns.loc[
                ten_eval_mask
            ]
            .reset_index(drop=True)
        )

        print()
        print(
            f"Fold {fold + 1}: "
            f"{len(expanding_10m):,} 10m train, "
            f"{len(expanding_2m):,} 2m train, "
            f"{len(eval_10m_scored):,} 10m eval"
        )

        ten_pred = train_predict(
            expanding_10m,
            eval_10m_scored,
            features_10m,
            TEN_MIN_HORIZON_BARS,
            TEN_MIN_TARGET,
            encoder,
        )

        two_pred = train_predict(
            expanding_2m,
            eval_2m,
            features_2m,
            TWO_MIN_HORIZON_BARS,
            TWO_MIN_TARGET,
            encoder,
        )

        two_pred = (
            two_pred[
                [
                    "close_time",
                    "prediction",
                    "confidence",
                ]
            ]
            .rename(
                columns={
                    "prediction":
                        "two_prediction",
                    "confidence":
                        "two_confidence",
                }
            )
            .sort_values("close_time")
        )

        rows = (
            ten_pred.rename(
                columns={
                    "prediction":
                        "ten_prediction",
                    "confidence":
                        "ten_confidence",
                }
            )
            .sort_values("close_time")
            .reset_index(drop=True)
        )

        rows[
            "future_return_10m_horizon"
        ] = ten_eval_returns.to_numpy()

        rows = pd.merge_asof(
            rows,
            two_pred,
            on="close_time",
            direction="backward",
            allow_exact_matches=True,
        )

        if rows[
            "two_prediction"
        ].isna().any():
            raise ValueError(
                "Missing aligned 2m signal."
            )

        rows["row_index"] = np.arange(
            len(rows)
        )

        rows["ten_signal"] = (
            rows["ten_prediction"].isin(
                [
                    "UP",
                    "DOWN",
                ]
            )
            & (
                rows["ten_confidence"]
                >= TEN_MIN_CONFIDENCE
            )
        )

        rows["two_signal"] = (
            rows["two_prediction"].isin(
                [
                    "UP",
                    "DOWN",
                ]
            )
            & (
                rows["two_confidence"]
                >= TWO_MIN_CONFIDENCE
            )
        )

        for variant in variants:
            metrics = score_variant(
                rows,
                variant,
            )
            metrics["fold"] = (
                fold + 1
            )
            fold_results[
                variant
            ].append(metrics)

            print(
                f"  {variant}: "
                f"{metrics['trades']} trades, "
                f"net "
                f"{metrics['net_return_pct']:.2f}%"
            )

    summary_rows = []

    for variant in variants:
        folds = fold_results[
            variant
        ]

        total_trades = sum(
            row["trades"]
            for row in folds
        )

        total_wins = sum(
            row["wins"]
            for row in folds
        )

        net_return = sum(
            row["net_return_pct"]
            for row in folds
        )

        if total_trades:
            win_rate = (
                total_wins
                / total_trades
                * 100
            )
        else:
            win_rate = 0.0

        weighted_avg_net = (
            net_return
            / total_trades
            if total_trades
            else 0.0
        )

        summary_rows.append(
            {
                "variant": variant,
                "trades": total_trades,
                "win_rate_pct":
                    win_rate,
                "net_return_pct":
                    net_return,
                "avg_net_trade_pct":
                    weighted_avg_net,
                "positive_folds": sum(
                    row[
                        "net_return_pct"
                    ] > 0
                    for row in folds
                ),
                "fold_1_net_pct":
                    folds[0][
                        "net_return_pct"
                    ],
                "fold_2_net_pct":
                    folds[1][
                        "net_return_pct"
                    ],
                "fold_3_net_pct":
                    folds[2][
                        "net_return_pct"
                    ],
                "fold_4_net_pct":
                    folds[3][
                        "net_return_pct"
                    ],
            }
        )

    results = pd.DataFrame(
        summary_rows
    )

    results = (
        results.sort_values(
            by=[
                "positive_folds",
                "net_return_pct",
                "avg_net_trade_pct",
            ],
            ascending=[
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

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "=" * 145
    )
    print(
        "10M + 2M CONFIRMATION "
        "WALK-FORWARD RESULTS"
    )
    print(
        "=" * 145
    )

    print(
        results.to_string(
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

    print()
    print(
        "TEST SPLITS WERE NOT USED."
    )

    print()
    print(
        "Interpretation: the combined "
        "variant must improve robustness "
        "or expectancy without collapsing "
        "the trade count."
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
