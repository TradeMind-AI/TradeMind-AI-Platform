from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


# --------------------------------------------------
# Columns that must NOT be used as ML features
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
test_df = pd.read_csv(TEST_FILE)

print(f"Train rows: {len(train_df)}")
print(f"Validation rows: {len(validation_df)}")
print(f"Test rows: {len(test_df)}")


# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

feature_columns = [
    column
    for column in train_df.columns
    if column not in NON_FEATURE_COLUMNS
]

X_train = train_df[feature_columns]
y_train = train_df["target"]

X_validation = validation_df[feature_columns]
y_validation = validation_df["target"]

X_test = test_df[feature_columns]
y_test = test_df["target"]


# --------------------------------------------------
# Validate feature consistency
# --------------------------------------------------

if list(X_train.columns) != list(X_validation.columns):
    raise ValueError("Train and validation feature columns do not match.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train and test feature columns do not match.")

print(f"Number of features: {len(feature_columns)}")


# --------------------------------------------------
# Combine train + validation
# --------------------------------------------------

X_train_final = pd.concat(
    [X_train, X_validation],
    axis=0,
    ignore_index=True,
)

y_train_final = pd.concat(
    [y_train, y_validation],
    axis=0,
    ignore_index=True,
)

print(f"Final training samples: {len(X_train_final)}")


# --------------------------------------------------
# Final preprocessing
# --------------------------------------------------

print("\nFitting StandardScaler on train + validation data...")

scaler = StandardScaler()

X_train_final_scaled = scaler.fit_transform(X_train_final)

# IMPORTANT:
# Test data is transformed using the scaler fitted above.
X_test_scaled = scaler.transform(X_test)

print("Test data transformed using training scaler.")


# --------------------------------------------------
# Final model
# --------------------------------------------------

print("\nTraining final Logistic Regression model...")

model = LogisticRegression(
    C=2.0,
    max_iter=2000,
    random_state=42,
    class_weight=None,
)

model.fit(X_train_final_scaled, y_train_final)

print("Final model training complete.")


# --------------------------------------------------
# Test prediction
# --------------------------------------------------

print("\nEvaluating on untouched test set...")

y_test_pred = model.predict(X_test_scaled)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_test_pred)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"\nTest accuracy: {accuracy:.4f}")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        labels=["DOWN", "NEUTRAL", "UP"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_test,
        y_test_pred,
        labels=["DOWN", "NEUTRAL", "UP"],
    )
)

print("\nClass distribution in test set:")
print(y_test.value_counts())

print("\nFinal evaluation complete.")