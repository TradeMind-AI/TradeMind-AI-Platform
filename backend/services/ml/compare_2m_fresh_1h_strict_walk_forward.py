from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_2M = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_2m"
)

CONTEXT_1H_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_context_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "2m_fresh_1h_strict_walk_forward.csv"
)


VALIDATION_FOLDS = 4
ROUND_TRIP_COST = 0.0010

TWO_MIN_HORIZON_BARS = 3
TWO_MIN_TARGET = 0.0010
TWO_MIN_CONFIDENCE = 0.50

ONE_HOUR_HORIZON_BARS = 1
ONE_HOUR_TARGET = 0.0020
ONE_HOUR_CONFIDENCE = 0.55


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

NON_FEATURE_1H = {
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
    "future_return_1h",
}


def build_2m_model() -> XGBClassifier:
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


def build_1h_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
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


def train_predict(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    horizon_bars: int,
    target_threshold: float,
    encoder: LabelEncoder,
    model: XGBClassifier,
) -> pd.DataFrame:
    train_returns = future_return(
        train_df,
        horizon_bars,
    )

    train_mask = train_returns.notna()

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

    model.fit(
        X_train,
        encoder.transform(y_train),
    )

    probabilities = model.predict_proba(
        eval_df[feature_columns]
    )

    predictions = encoder.inverse_transform(
        model.predict(
            eval_df[feature_columns]
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
        row_index = int(row["row_index"])

        if row_index < next_available:
            continue

        accepted.append(row)

        next_available = (
            row_index
            + TWO_MIN_HORIZON_BARS
        )

    if not accepted:
        return trades.iloc[0:0].copy()

    return (
        pd.DataFrame(accepted)
        .reset_index(drop=True)
    )


def score_variant(
    rows: pd.DataFrame,
    strict: bool,
) -> dict:
    selected = rows[
        rows["two_signal"]
    ].copy()

    if strict:
        selected = selected[
            (
                selected["one_hour_signal"]
            )
            & (
                selected["one_hour_prediction"]
                == selected["two_prediction"]
            )
        ].copy()

    selected["gross_return"] = np.where(
        selected["two_prediction"] == "UP",
        selected["future_return_2m_horizon"],
        -selected["future_return_2m_horizon"],
    )

    selected = remove_overlaps(selected)

    selected["net_return"] = (
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
            selected["net_return"].mean()
            * 100
        )
    else:
        win_rate_pct = 0.0
        avg_net_trade_pct = 0.0

    return {
        "trades": trades,
        "wins": wins,
        "win_rate_pct": win_rate_pct,
        "net_return_pct": net_return_pct,
        "avg_net_trade_pct": avg_net_trade_pct,
    }


def main() -> None:
    print(
        "Running 2m + strict fresh 1h "
        "agreement walk-forward test..."
    )

    train_2m = pd.read_csv(
        SPLIT_2M / "train.csv"
    )

    val_2m = pd.read_csv(
        SPLIT_2M / "validation.csv"
    )

    one_hour = pd.read_csv(
        CONTEXT_1H_FILE
    )

    for frame in [
        train_2m,
        val_2m,
        one_hour,
    ]:
        frame["open_time"] = pd.to_datetime(
            frame["open_time"],
            utc=True,
        )
        frame["close_time"] = pd.to_datetime(
            frame["close_time"],
            utc=True,
        )

    features_2m = [
        column
        for column in train_2m.columns
        if column not in NON_FEATURE_2M
    ]

    features_1h = [
        column
        for column in one_hour.columns
        if column not in NON_FEATURE_1H
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
        "2m primary: "
        "6m horizon, ±0.10% target, "
        "confidence >= 0.50"
    )
    print(
        "Fresh 1h strict filter: "
        "1h horizon, ±0.20% target, "
        "confidence >= 0.55"
    )
    print(
        "Cost: 0.10% round trip"
    )
    print(
        "Alignment: only the latest fully "
        "closed 1h candle is available to "
        "each 2m decision."
    )

    boundaries = np.linspace(
        0,
        len(val_2m),
        VALIDATION_FOLDS + 1,
        dtype=int,
    )

    fold_baseline = []
    fold_strict = []

    for fold in range(
        VALIDATION_FOLDS
    ):
        start = boundaries[fold]
        end = boundaries[fold + 1]

        eval_2m = (
            val_2m.iloc[start:end]
            .copy()
            .reset_index(drop=True)
        )

        eval_start = (
            eval_2m["open_time"].iloc[0]
        )

        eval_end_close = (
            eval_2m["close_time"].iloc[-1]
        )

        if fold == 0:
            expanding_2m = (
                train_2m.copy()
                .reset_index(drop=True)
            )
        else:
            expanding_2m = (
                pd.concat(
                    [
                        train_2m,
                        val_2m.iloc[:start],
                    ],
                    ignore_index=True,
                )
            )

        train_1h = (
            one_hour[
                one_hour["close_time"]
                < eval_start
            ]
            .copy()
            .reset_index(drop=True)
        )

        prior_context = (
            one_hour[
                one_hour["close_time"]
                < eval_start
            ]
            .tail(1)
            .copy()
        )

        eval_1h = (
            one_hour[
                (
                    one_hour["close_time"]
                    >= eval_start
                )
                & (
                    one_hour["close_time"]
                    <= eval_end_close
                )
            ]
            .copy()
        )

        eval_1h = (
            pd.concat(
                [
                    prior_context,
                    eval_1h,
                ],
                ignore_index=True,
            )
            .sort_values("close_time")
            .drop_duplicates(
                subset=["close_time"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        two_eval_returns = future_return(
            eval_2m,
            TWO_MIN_HORIZON_BARS,
        )

        eval_mask = (
            two_eval_returns.notna()
        )

        eval_2m_scored = (
            eval_2m.loc[
                eval_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        two_eval_returns = (
            two_eval_returns.loc[
                eval_mask
            ]
            .reset_index(drop=True)
        )

        print()
        print(
            f"Fold {fold + 1}: "
            f"{len(expanding_2m):,} 2m train, "
            f"{len(train_1h):,} 1h train, "
            f"{len(eval_2m_scored):,} 2m eval"
        )

        two_pred = train_predict(
            expanding_2m,
            eval_2m_scored,
            features_2m,
            TWO_MIN_HORIZON_BARS,
            TWO_MIN_TARGET,
            encoder,
            build_2m_model(),
        )

        one_hour_pred = train_predict(
            train_1h,
            eval_1h,
            features_1h,
            ONE_HOUR_HORIZON_BARS,
            ONE_HOUR_TARGET,
            encoder,
            build_1h_model(),
        )

        one_hour_pred = (
            one_hour_pred[
                [
                    "close_time",
                    "prediction",
                    "confidence",
                ]
            ]
            .rename(
                columns={
                    "prediction":
                        "one_hour_prediction",
                    "confidence":
                        "one_hour_confidence",
                }
            )
            .sort_values("close_time")
        )

        rows = (
            two_pred.rename(
                columns={
                    "prediction":
                        "two_prediction",
                    "confidence":
                        "two_confidence",
                }
            )
            .sort_values("close_time")
            .reset_index(drop=True)
        )

        rows[
            "future_return_2m_horizon"
        ] = two_eval_returns.to_numpy()

        rows = pd.merge_asof(
            rows,
            one_hour_pred,
            on="close_time",
            direction="backward",
            allow_exact_matches=True,
        )

        if rows[
            "one_hour_prediction"
        ].isna().any():
            raise ValueError(
                "Missing aligned fresh 1h "
                "regime signal."
            )

        rows["row_index"] = np.arange(
            len(rows)
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

        rows["one_hour_signal"] = (
            rows[
                "one_hour_prediction"
            ].isin(
                [
                    "UP",
                    "DOWN",
                ]
            )
            & (
                rows[
                    "one_hour_confidence"
                ]
                >= ONE_HOUR_CONFIDENCE
            )
        )

        baseline = score_variant(
            rows,
            strict=False,
        )

        strict = score_variant(
            rows,
            strict=True,
        )

        fold_baseline.append(baseline)
        fold_strict.append(strict)

        print(
            f"  TWO_MIN_ONLY: "
            f"{baseline['trades']} trades, "
            f"net "
            f"{baseline['net_return_pct']:.2f}%"
        )

        print(
            f"  STRICT_1H_AGREEMENT: "
            f"{strict['trades']} trades, "
            f"net "
            f"{strict['net_return_pct']:.2f}%"
        )

    rows_out = []

    for name, folds in [
        ("TWO_MIN_ONLY", fold_baseline),
        (
            "STRICT_1H_AGREEMENT",
            fold_strict,
        ),
    ]:
        trades = sum(
            row["trades"]
            for row in folds
        )

        wins = sum(
            row["wins"]
            for row in folds
        )

        net = sum(
            row["net_return_pct"]
            for row in folds
        )

        rows_out.append(
            {
                "variant": name,
                "trades": trades,
                "win_rate_pct":
                    (
                        wins / trades * 100
                        if trades
                        else 0.0
                    ),
                "net_return_pct": net,
                "avg_net_trade_pct":
                    (
                        net / trades
                        if trades
                        else 0.0
                    ),
                "positive_folds": sum(
                    row["net_return_pct"] > 0
                    for row in folds
                ),
                "fold_1_net_pct":
                    folds[0]["net_return_pct"],
                "fold_2_net_pct":
                    folds[1]["net_return_pct"],
                "fold_3_net_pct":
                    folds[2]["net_return_pct"],
                "fold_4_net_pct":
                    folds[3]["net_return_pct"],
            }
        )

    results = pd.DataFrame(
        rows_out
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
        "2M + STRICT FRESH 1H "
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
        "2m TEST SPLIT WAS NOT USED."
    )

    print()
    print(
        "Strict 1h agreement is useful "
        "only if it improves trade quality "
        "without collapsing robustness."
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
