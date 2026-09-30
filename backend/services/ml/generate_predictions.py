from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

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

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
    / "test.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_predictions.csv"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "logistic_regression"
MODEL_VERSION = "v1"
C_VALUE = 2.0
RANDOM_STATE = 42


# --------------------------------------------------
# Columns excluded from ML features
# --------------------------------------------------

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


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train rows: {len(train_df)}")
print(f"Validation rows: {len(validation_df)}")
print(f"Test rows: {len(test_df)}")


# --------------------------------------------------
# Identify feature columns
# --------------------------------------------------

feature_columns = [
    column
    for column in train_df.columns
    if column not in NON_FEATURE_COLUMNS
]

print(f"Number of features: {len(feature_columns)}")


# --------------------------------------------------
# Prepare training data
# --------------------------------------------------

X_train = train_df[feature_columns]
y_train = train_df["target"]

X_validation = validation_df[feature_columns]
y_validation = validation_df["target"]

X_test = test_df[feature_columns]


# --------------------------------------------------
# Validate feature consistency
# --------------------------------------------------

if list(X_train.columns) != list(X_validation.columns):
    raise ValueError(
        "Train and validation feature columns do not match."
    )

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Train and test feature columns do not match."
    )


# --------------------------------------------------
# Combine training + validation
# --------------------------------------------------

X_train_final = pd.concat(
    [X_train, X_validation],
    ignore_index=True,
)

y_train_final = pd.concat(
    [y_train, y_validation],
    ignore_index=True,
)

print(f"Final training rows: {len(X_train_final)}")


# --------------------------------------------------
# Scale features
# --------------------------------------------------

print("\nFitting scaler...")

scaler = StandardScaler()

X_train_final_scaled = scaler.fit_transform(
    X_train_final
)

X_test_scaled = scaler.transform(X_test)

print("Scaler fitted using train + validation data.")


# --------------------------------------------------
# Train final model
# --------------------------------------------------

print("\nTraining final Logistic Regression model...")

model = LogisticRegression(
    C=C_VALUE,
    max_iter=2000,
    random_state=RANDOM_STATE,
    class_weight=None,
)

model.fit(
    X_train_final_scaled,
    y_train_final,
)

print("Model training complete.")


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

print("\nGenerating predictions...")

predictions = model.predict(X_test_scaled)

probabilities = model.predict_proba(X_test_scaled)

class_order = list(model.classes_)

print(f"Model classes: {class_order}")


# --------------------------------------------------
# Extract class probabilities
# --------------------------------------------------

class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_order)
}

probability_down = probabilities[
    :, class_to_index["DOWN"]
]

probability_neutral = probabilities[
    :, class_to_index["NEUTRAL"]
]

probability_up = probabilities[
    :, class_to_index["UP"]
]


# --------------------------------------------------
# Build prediction dataset
# --------------------------------------------------

prediction_df = pd.DataFrame(
    {
        "timestamp": test_df["open_time"],
        "close_time": test_df["close_time"],
        "close": test_df["close"],
        "actual_target": test_df["target"],
        "future_return_1h": test_df["future_return_1h"],
        "prediction": predictions,
        "probability_down": probability_down,
        "probability_neutral": probability_neutral,
        "probability_up": probability_up,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }
)


# --------------------------------------------------
# Validate prediction output
# --------------------------------------------------

if len(prediction_df) != len(test_df):
    raise ValueError(
        "Prediction row count does not match test row count."
    )

if prediction_df[
    [
        "probability_down",
        "probability_neutral",
        "probability_up",
    ]
].isna().any().any():
    raise ValueError(
        "Prediction probabilities contain NaN values."
    )

probability_sum = (
    prediction_df["probability_down"]
    + prediction_df["probability_neutral"]
    + prediction_df["probability_up"]
)

if not (probability_sum.sub(1.0).abs() < 1e-6).all():
    raise ValueError(
        "Prediction probabilities do not sum to 1."
    )


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nPrediction generation complete.")

print(f"Rows generated: {len(prediction_df)}")
print(f"Output file: {OUTPUT_FILE}")

print("\nPrediction distribution:")
print(
    prediction_df["prediction"]
    .value_counts()
)

print("\nFirst 5 predictions:")
print(
    prediction_df.head().to_string(index=False)
)