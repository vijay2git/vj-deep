
from pathlib import Path
import numpy as np

DATA_DIR = Path("ml/data/raw/UCI HAR Dataset")

required_files = [
    DATA_DIR / "train" / "y_train.txt",
    DATA_DIR / "test" / "y_test.txt",
    DATA_DIR / "train" / "subject_train.txt",
    DATA_DIR / "test" / "subject_test.txt",
]

for file in required_files:
    print(f"{file}: {'FOUND' if file.exists() else 'MISSING'}")

if not all(file.exists() for file in required_files):
    raise FileNotFoundError(
        "Dataset files are missing. Check the extracted folder structure."
    )

y_train = np.loadtxt(DATA_DIR / "train" / "y_train.txt", dtype=int)
y_test = np.loadtxt(DATA_DIR / "test" / "y_test.txt", dtype=int)

subject_train = np.loadtxt(
    DATA_DIR / "train" / "subject_train.txt", dtype=int
)
subject_test = np.loadtxt(
    DATA_DIR / "test" / "subject_test.txt", dtype=int
)

print("\nTraining samples:", len(y_train))
print("Test samples:", len(y_test))
print("Training participants:", len(np.unique(subject_train)))
print("Test participants:", len(np.unique(subject_test)))
print("Activity labels:", np.unique(np.concatenate([y_train, y_test])))
