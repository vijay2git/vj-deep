
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

DATA_FILE = Path("ml/data/processed/har_sequences.npz")
MODEL_DIR = Path("ml/models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the preprocessed dataset
data = np.load(DATA_FILE)

X = data["X_train"]
y = data["y_train"]
subjects = data["subjects_train"]

X_test = data["X_test"]
y_test = data["y_test"]

# Split by participant, not by individual overlapping windows
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED,
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=subjects)
)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Validation participants:", np.unique(subjects[val_idx]))
print("Test shape:", X_test.shape)

# Flatten each 128 x 9 sensor sequence for the MLP
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)

# MLP baseline
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128 * 9,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "mlp_best.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
]

history = model.fit(
    X_train_flat,
    y_train,
    validation_data=(X_val_flat, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)

# Validation metrics select the model; the test set stays untouched
val_prob = model.predict(X_val_flat, verbose=0)
val_pred = np.argmax(val_prob, axis=1)

val_accuracy = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred, average="macro")

print("\nValidation accuracy:", round(val_accuracy, 4))
print("Validation macro F1:", round(val_f1, 4))
print("\nValidation classification report:")
print(classification_report(y_val, val_pred, digits=4))

# Save history and validation results
results = {
    "model": "MLP",
    "validation_accuracy": float(val_accuracy),
    "validation_macro_f1": float(val_f1),
    "epochs_trained": len(history.history["loss"]),
    "validation_participants": np.unique(
        subjects[val_idx]
    ).tolist(),
}

with open(RESULTS_DIR / "mlp_validation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved model to:", MODEL_DIR / "mlp_best.keras")
print("Saved results to:", RESULTS_DIR / "mlp_validation.json")
