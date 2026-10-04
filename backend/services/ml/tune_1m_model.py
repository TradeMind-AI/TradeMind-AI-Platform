from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
)
from sklearn.preprocessing import (
    LabelEncoder,
)
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits_1m"
)

TRAIN_FILE = (
    SPLIT_DIR
    / "train.csv"
)

VALIDATION_FILE = (
    SPLIT_DIR
    / "validation.csv"
)


TARGET_THRESHOLDS = [
    0.0003,   # 0.03%
    0.0005,   # 0.05%
    0.00075,  # 0.075%
    0.0010,   # 0.10%
    0.0015,   # 0.15%
]

CONFIDENCE_THRESHOLDS = [
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

# Conservative research assumption:
# 0.05% entry + 0.05% exit.
ROUND_TRIP_COST = 0.0010


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


def create_target(
    future_return: pd.Series,
    threshold: float,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                future_return
                < -threshold,
                future_return
                > threshold,
            ],
            [
                "DOWN",
                "UP",
            ],
            default="NEUTRAL",
        ),
        index=future_return.index,
    )


def evaluate_trades(
    future_return: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    encoder: LabelEncoder,
    confidence_threshold: float,
) -> dict:
    result = pd.DataFrame(
        {
            "future_return":
                future_return
                .to_numpy(),
            "prediction":
                predictions,
        }
    )

    class_index = {
        name: index
        for index, name
        in enumerate(
            encoder.classes_
        )
    }

    result["probability_down"] = (
        probabilities[
            :,
            class_index["DOWN"]
        ]
    )

    result["probability_neutral"] = (
        probabilities[
            :,
            class_index["NEUTRAL"]
        ]
    )

    result["probability_up"] = (
        probabilities[
            :,
            class_index["UP"]
        ]
    )

    result["confidence"] = (
        probabilities.max(
            axis=1
        )
    )

    directional = (
        result["prediction"]
        .isin(
            ["UP", "DOWN"]
        )
    )

    confident = (
        result["confidence"]
        >= confidence_threshold
    )

    trades = result[
        directional
        & confident
    ].copy()

    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "gross_return_pct": 0.0,
            "cost_pct": 0.0,
            "net_return_pct": 0.0,
            "average_net_trade_pct": 0.0,
        }

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

    trades[
        "net_return"
    ] = (
        trades[
            "gross_return"
        ]
        - ROUND_TRIP_COST
    )

    winning = (
        trades[
            "net_return"
        ] > 0
    )

    return {
        "trades":
            len(trades),

        "win_rate_pct":
            winning.mean()
            * 100,

        "gross_return_pct":
            trades[
                "gross_return"
            ].sum()
            * 100,

        "cost_pct":
            len(trades)
            * ROUND_TRIP_COST
            * 100,

        "net_return_pct":
            trades[
                "net_return"
            ].sum()
            * 100,

        "average_net_trade_pct":
            trades[
                "net_return"
            ].mean()
            * 100,
    }


def main() -> None:
    print(
        "Loading 1-minute train "
        "and validation data..."
    )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    feature_columns = [
        column
        for column
        in train_df.columns
        if column
        not in NON_FEATURE_COLUMNS
    ]

    if (
        "future_return_1m"
        in feature_columns
    ):
        raise ValueError(
            "Target leakage detected."
        )

    X_train = (
        train_df[
            feature_columns
        ]
    )

    X_validation = (
        validation_df[
            feature_columns
        ]
    )

    print(
        f"Train rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_validation):,}"
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    results = []

    encoder = LabelEncoder()

    encoder.fit(
        [
            "DOWN",
            "NEUTRAL",
            "UP",
        ]
    )

    for target_threshold in (
        TARGET_THRESHOLDS
    ):
        print(
            "\n"
            + "=" * 80
        )

        print(
            "Training target threshold "
            f"±{target_threshold * 100:.3f}%"
        )

        y_train = create_target(
            train_df[
                "future_return_1m"
            ],
            target_threshold,
        )

        y_validation = create_target(
            validation_df[
                "future_return_1m"
            ],
            target_threshold,
        )

        print(
            "Train target distribution:"
        )

        print(
            (
                y_train
                .value_counts(
                    normalize=True
                )
                .sort_index()
                * 100
            )
            .round(2)
            .to_string()
        )

        y_train_encoded = (
            encoder.transform(
                y_train
            )
        )

        model = XGBClassifier(
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

        model.fit(
            X_train,
            y_train_encoded,
        )

        prediction_encoded = (
            model.predict(
                X_validation
            )
        )

        probabilities = (
            model.predict_proba(
                X_validation
            )
        )

        predictions = (
            encoder.inverse_transform(
                prediction_encoded
            )
        )

        accuracy = (
            accuracy_score(
                y_validation,
                predictions,
            )
        )

        macro_f1 = (
            f1_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        up_precision = (
            precision_score(
                y_validation == "UP",
                predictions == "UP",
                zero_division=0,
            )
        )

        down_precision = (
            precision_score(
                y_validation == "DOWN",
                predictions == "DOWN",
                zero_division=0,
            )
        )

        for confidence_threshold in (
            CONFIDENCE_THRESHOLDS
        ):
            trading = (
                evaluate_trades(
                    validation_df[
                        "future_return_1m"
                    ],
                    predictions,
                    probabilities,
                    encoder,
                    confidence_threshold,
                )
            )

            results.append(
                {
                    "target_threshold_pct":
                        target_threshold
                        * 100,

                    "confidence_threshold":
                        confidence_threshold,

                    "accuracy_pct":
                        accuracy
                        * 100,

                    "macro_f1_pct":
                        macro_f1
                        * 100,

                    "up_precision_pct":
                        up_precision
                        * 100,

                    "down_precision_pct":
                        down_precision
                        * 100,

                    **trading,
                }
            )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by=[
                "net_return_pct",
                "average_net_trade_pct",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "1m_tuning_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "TOP 1-MINUTE VALIDATION RESULTS"
    )

    print(
        "=" * 120
    )

    print(
        results_df
        .head(20)
        .to_string(
            index=False,
            formatters={
                "target_threshold_pct":
                    lambda value:
                        f"{value:.3f}%",

                "confidence_threshold":
                    lambda value:
                        f"{value:.2f}",

                "accuracy_pct":
                    lambda value:
                        f"{value:.2f}%",

                "macro_f1_pct":
                    lambda value:
                        f"{value:.2f}%",

                "up_precision_pct":
                    lambda value:
                        f"{value:.2f}%",

                "down_precision_pct":
                    lambda value:
                        f"{value:.2f}%",

                "win_rate_pct":
                    lambda value:
                        f"{value:.2f}%",

                "gross_return_pct":
                    lambda value:
                        f"{value:.2f}%",

                "cost_pct":
                    lambda value:
                        f"{value:.2f}%",

                "net_return_pct":
                    lambda value:
                        f"{value:.2f}%",

                "average_net_trade_pct":
                    lambda value:
                        f"{value:.4f}%",
            },
        )
    )

    print(
        "\nRound-trip cost assumption: "
        f"{ROUND_TRIP_COST * 100:.2f}%"
    )

    print(
        "\nTEST SPLIT WAS NOT USED."
    )

    print(
        f"\nFull results saved to:"
        f"\n{output_file}"
    )


if __name__ == "__main__":
    main()
