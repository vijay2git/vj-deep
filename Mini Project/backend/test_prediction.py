
import numpy as np
import requests

# Load a real sample from the official test dataset
data = np.load("ml/data/processed/har_sequences.npz")

sequence = data["X_test"][0].tolist()
actual_label = int(data["y_test"][0])

activities = [
    "Walking",
    "Walking upstairs",
    "Walking downstairs",
    "Sitting",
    "Standing",
    "Laying",
]

# Send the sample to our running FastAPI server
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"sequence": sequence},
    timeout=30,
)

response.raise_for_status()
result = response.json()

print("\n--- ActivitySense AI Prediction Test ---")
print("Actual activity:    ", activities[actual_label])
print("Predicted activity: ", result["activity"])
print("Confidence:         ", f"{result['confidence']:.2%}")

print("\nClass probabilities:")
for activity, probability in result["probabilities"].items():
    print(f"{activity:22s} {probability:.2%}")
