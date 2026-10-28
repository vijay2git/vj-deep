
from pathlib import Path
import numpy as np

# Dataset locations
DATA_DIR = Path("ml/data/raw/UCI HAR Dataset")
OUTPUT_DIR = Path("ml/data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nine sensor channels used by our project
SIGNALS = [
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]


def load_partition(partition):
    """Load sensor sequences, labels and participant IDs."""

    signal_dir = DATA_DIR / partition / "Inertial Signals"
    channels = []

    for signal in SIGNALS:
        filename = f"{signal}_{partition}.txt"
        filepath = signal_dir / filename

        data = np.loadtxt(filepath, dtype=np.float32)
        channels.append(data)

        print(f"Loaded {filename}: {data.shape}")

    # Each channel: (samples, 128)
    # Combined sequence: (samples, 128, 9)
    X = np.stack(channels, axis=-1)

    y = np.loadtxt(
        DATA_DIR / partition / f"y_{partition}.txt",
        dtype=np.int64,
    )

    subjects = np.loadtxt(
        DATA_DIR / partition / f"subject_{partition}.txt",
        dtype=np.int64,
    )

    # Convert labels from 1-6 to 0-5 for model training
    y = y - 1

    return X, y, subjects


def main():
    X_train, y_train, subjects_train = load_partition("train")
    X_test, y_test, subjects_test = load_partition("test")

    print("\nDataset summary")
    print("-" * 40)
    print("Training sequences:", X_train.shape)
    print("Training labels:", y_train.shape)
    print("Test sequences:", X_test.shape)
    print("Test labels:", y_test.shape)
    print("Training label values:", np.unique(y_train))
    print("Test label values:", np.unique(y_test))

    # Save processed arrays for reuse by training scripts
    output_file = OUTPUT_DIR / "har_sequences.npz"

    np.savez_compressed(
        output_file,
        X_train=X_train,
        y_train=y_train,
        subjects_train=subjects_train,
        X_test=X_test,
        y_test=y_test,
        subjects_test=subjects_test,
    )

    print(f"\nProcessed dataset saved to: {output_file}")


if __name__ == "__main__":
    main()
