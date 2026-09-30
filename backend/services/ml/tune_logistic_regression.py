from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
    / "train.csv"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
    / "validation.csv"
)


NON_FEATURE_COLUMNS = [
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
    "target",
]


def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)

    return train_df, validation_df


def prepare_data(train_df, validation_df):

    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X_train = train_df[feature_columns]
    X_validation = validation_df[feature_columns]

    y_train = train_df["target"]
    y_validation = validation_df["target"]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    )


def main():

    print("Loading ML data...")

    train_df, validation_df = load_data()

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    ) = prepare_data(
        train_df,
        validation_df,
    )

    print(f"Number of features: {len(feature_columns)}")

    # ---------------------------------------------------------
    # Scale using training data ONLY
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    print("\nTraining-only scaling: PASSED")

    # ---------------------------------------------------------
    # Hyperparameter experiments
    # ---------------------------------------------------------

    C_VALUES = [
        0.01,
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
    ]

    results = []

    print("\nStarting Logistic Regression experiments...")

    for C in C_VALUES:

        print(f"\nTesting C = {C}")

        model = LogisticRegression(
            C=C,
            max_iter=2000,
            random_state=42,
        )

        model.fit(
            X_train_scaled,
            y_train,
        )

        predictions = model.predict(
            X_validation_scaled
        )

        accuracy = accuracy_score(
            y_validation,
            predictions,
        )

        results.append(
            {
                "C": C,
                "validation_accuracy": accuracy,
            }
        )

        print(
            f"Validation accuracy: {accuracy:.4f}"
        )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="validation_accuracy",
        ascending=False,
    )

    print("\n" + "=" * 55)
    print("LOGISTIC REGRESSION TUNING RESULTS")
    print("=" * 55)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Best model
    # ---------------------------------------------------------

    best_row = results_df.iloc[0]

    best_C = best_row["C"]
    best_accuracy = best_row[
        "validation_accuracy"
    ]

    print("\nBest configuration:")
    print(f"C: {best_C}")
    print(
        f"Validation accuracy: "
        f"{best_accuracy:.4f}"
    )

    # ---------------------------------------------------------
    # Detailed report for best model
    # ---------------------------------------------------------

    best_model = LogisticRegression(
        C=best_C,
        max_iter=2000,
        random_state=42,
    )

    best_model.fit(
        X_train_scaled,
        y_train,
    )

    best_predictions = best_model.predict(
        X_validation_scaled
    )

    print("\nClassification report for best model:")

    print(
        classification_report(
            y_validation,
            best_predictions,
            digits=4,
        )
    )

    print(
        "\nLogistic Regression tuning complete."
    )


if __name__ == "__main__":
    main()