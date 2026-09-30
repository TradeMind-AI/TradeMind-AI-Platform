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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_validation_predictions.csv"
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
# Load data
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print(f"Train rows: {len(train_df)}")
print(f"Validation rows: {len(validation_df)}")


# --------------------------------------------------
# Identify features
# --------------------------------------------------

feature_columns = [
    column
    for column in train_df.columns
    if column not in NON_FEATURE_COLUMNS
]

print(f"Number of features: {len(feature_columns)}")


# --------------------------------------------------
# Prepare training and validation data
# --------------------------------------------------

X_train = train_df[feature_columns]
y_train = train_df["target"]

X_validation = validation_df[feature_columns]


# --------------------------------------------------
# Validate feature columns
# --------------------------------------------------

if list(X_train.columns) != list(
    X_validation.columns
):
    raise ValueError(
        "Train and validation feature columns do not match."
    )


# --------------------------------------------------
# Scale features
# --------------------------------------------------

print("\nFitting scaler on training data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_validation_scaled = scaler.transform(
    X_validation
)


# --------------------------------------------------
# Train model
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    C=C_VALUE,
    max_iter=2000,
    random_state=RANDOM_STATE,
    class_weight=None,
)

model.fit(
    X_train_scaled,
    y_train,
)

print("Model training complete.")


# --------------------------------------------------
# Generate validation predictions
# --------------------------------------------------

print("\nGenerating validation predictions...")

predictions = model.predict(
    X_validation_scaled
)

probabilities = model.predict_proba(
    X_validation_scaled
)

class_order = list(model.classes_)

class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_order)
}


# --------------------------------------------------
# Extract probabilities
# --------------------------------------------------

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
# Build output
# --------------------------------------------------

prediction_df = pd.DataFrame(
    {
        "timestamp": validation_df["open_time"],
        "close_time": validation_df["close_time"],
        "close": validation_df["close"],
        "actual_target": validation_df["target"],
        "future_return_1h": validation_df[
            "future_return_1h"
        ],
        "prediction": predictions,
        "probability_down": probability_down,
        "probability_neutral": probability_neutral,
        "probability_up": probability_up,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }
)


# --------------------------------------------------
# Validate probabilities
# --------------------------------------------------

probability_sum = (
    prediction_df["probability_down"]
    + prediction_df["probability_neutral"]
    + prediction_df["probability_up"]
)

if not (
    probability_sum.sub(1.0).abs() < 1e-6
).all():
    raise ValueError(
        "Prediction probabilities do not sum to 1."
    )


# --------------------------------------------------
# Save
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

print("\nValidation prediction generation complete.")

print(
    f"Rows generated: "
    f"{len(prediction_df)}"
)

print(
    f"Output file:"
    f"\n{OUTPUT_FILE}"
)

print("\nPrediction distribution:")

print(
    prediction_df["prediction"]
    .value_counts()
)

print("\nFirst 5 predictions:")

print(
    prediction_df.head().to_string(
        index=False
    )
)