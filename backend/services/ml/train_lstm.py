from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# Configuration
# ============================================================

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

SEQUENCE_LENGTH = 24

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

RANDOM_SEED = 42


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


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# LSTM Model
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        num_classes,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Use the final time step
        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        return self.fc(last_output)


# ============================================================
# Data loading
# ============================================================

def load_data():

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    return train_df, validation_df


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(
    train_df,
    validation_df,
):

    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    validation_feature_columns = [
        column
        for column in validation_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    if feature_columns != validation_feature_columns:
        raise ValueError(
            "Training and validation feature columns do not match."
        )

    if "target" in feature_columns:
        raise ValueError(
            "Target leakage detected: target is in features."
        )

    if "future_return_1h" in feature_columns:
        raise ValueError(
            "Target leakage detected: future_return_1h is in features."
        )

    X_train = train_df[
        feature_columns
    ].values

    X_validation = validation_df[
        feature_columns
    ].values

    label_mapping = {
        "DOWN": 0,
        "NEUTRAL": 1,
        "UP": 2,
    }

    y_train = (
        train_df["target"]
        .map(label_mapping)
        .values
    )

    y_validation = (
        validation_df["target"]
        .map(label_mapping)
        .values
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    )


# ============================================================
# Scaling
# ============================================================

def scale_features(
    X_train,
    X_validation,
):

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit scaler ONLY on training data.
    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    return (
        X_train_scaled,
        X_validation_scaled,
        scaler,
    )


# ============================================================
# Sequence creation
# ============================================================

def create_sequences(
    X,
    y,
    sequence_length,
):

    sequences = []
    labels = []

    for i in range(
        sequence_length,
        len(X),
    ):

        sequence = X[
            i - sequence_length:i
        ]

        label = y[i]

        sequences.append(sequence)
        labels.append(label)

    return (
        np.asarray(
            sequences,
            dtype=np.float32,
        ),
        np.asarray(
            labels,
            dtype=np.int64,
        ),
    )


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
):

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    for epoch in range(
        EPOCHS
    ):

        total_loss = 0.0

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            outputs = model(
                X_batch
            )

            loss = criterion(
                outputs,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * len(X_batch)
            )

        average_loss = (
            total_loss
            / len(dataset)
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"- Loss: {average_loss:.6f}"
        )


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
):

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_validation),
        torch.tensor(y_validation),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model.eval()

    predictions = []

    with torch.no_grad():

        for X_batch, _ in loader:

            outputs = model(
                X_batch
            )

            batch_predictions = (
                torch.argmax(
                    outputs,
                    dim=1,
                )
                .cpu()
                .numpy()
            )

            predictions.extend(
                batch_predictions
            )

    predictions = np.asarray(
        predictions
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    print("\nValidation accuracy:")
    print(f"{accuracy:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_validation,
            predictions,
            labels=[0, 1, 2],
            target_names=[
                "DOWN",
                "NEUTRAL",
                "UP",
            ],
            digits=4,
        )
    )

    print("Confusion matrix:")

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1, 2],
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual DOWN",
            "Actual NEUTRAL",
            "Actual UP",
        ],
        columns=[
            "Pred DOWN",
            "Pred NEUTRAL",
            "Pred UP",
        ],
    )

    print(matrix_df)


# ============================================================
# Main
# ============================================================

def main():

    print("Loading ML data...")

    train_df, validation_df = load_data()

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_df)}"
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    ) = prepare_features(
        train_df,
        validation_df,
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    print(
        "\nTarget leakage check: PASSED"
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    (
        X_train_scaled,
        X_validation_scaled,
        _,
    ) = scale_features(
        X_train,
        X_validation,
    )

    print(
        "Training-only scaling: PASSED"
    )

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    (
        X_train_sequences,
        y_train_sequences,
    ) = create_sequences(
        X_train_scaled,
        y_train,
        SEQUENCE_LENGTH,
    )

    (
        X_validation_sequences,
        y_validation_sequences,
    ) = create_sequences(
        X_validation_scaled,
        y_validation,
        SEQUENCE_LENGTH,
    )

    print(
        f"\nSequence length: "
        f"{SEQUENCE_LENGTH} hours"
    )

    print(
        "Training sequences: "
        f"{len(X_train_sequences)}"
    )

    print(
        "Validation sequences: "
        f"{len(X_validation_sequences)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LSTMClassifier(
        input_size=len(feature_columns),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=3,
    )

    print(
        "\nLSTM model created."
    )

    print(
        f"Hidden size: {HIDDEN_SIZE}"
    )

    print(
        f"LSTM layers: {NUM_LAYERS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nTraining LSTM..."
    )

    train_model(
        model,
        X_train_sequences,
        y_train_sequences,
    )

    print(
        "\nLSTM training complete."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_validation_sequences,
        y_validation_sequences,
    )

    print(
        "\nLSTM experiment complete."
    )


if __name__ == "__main__":
    main()