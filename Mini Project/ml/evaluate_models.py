
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_FILE = Path("ml/data/processed/har_sequences.npz")
MODEL_DIR = Path("ml/models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITIES = [
    "Walking",
    "Walking upstairs",
    "Walking downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

data = np.load(DATA_FILE)
X_test = data["X_test"]
y_test = data["y_test"]

models = {
    "MLP": MODEL_DIR / "mlp_best.keras",
    "1D CNN": MODEL_DIR / "cnn_best.keras",
}

comparison = {}

for name, model_path in models.items():
    print(f"\n{'=' * 45}")
    print(f"Evaluating {name}")
    print("=" * 45)

    model = tf.keras.models.load_model(model_path)

    # MLP expects flattened inputs; CNN expects time-series inputs.
    if name == "MLP":
        X_input = X_test.reshape(len(X_test), -1)
    else:
        X_input = X_test

    probabilities = model.predict(X_input, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(6)),
            target_names=ACTIVITIES,
            digits=4,
            zero_division=0,
        )
    )

    comparison[name] = {
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
    }

    # Save a separate confusion matrix for each model.
    cm = confusion_matrix(
        y_test, predictions, labels=list(range(6))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=ACTIVITIES,
    )
    display.plot(
        xticks_rotation=35,
        values_format="d",
        colorbar=False,
    )
    plt.title(f"{name} — Official Test Set")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        RESULTS_DIR / f"{filename}_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

with open(RESULTS_DIR / "test_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("\nSaved comparison to results/test_comparison.json")
print("Saved confusion matrices to the results folder.")
