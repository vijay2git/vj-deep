# ActivitySense AI

A deep learning-based Human Activity Recognition (HAR) system that classifies six human activities using smartphone sensor data from the UCI Human Activity Recognition dataset.

## Activities Classified

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

## Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Models:** Multilayer Perceptron (MLP), 1D Convolutional Neural Network (CNN)
- **Backend:** FastAPI
- **Frontend:** React, TypeScript, Vite
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Recharts

## Model Performance

| Model | Test Accuracy | Macro F1-Score |
|---|---:|---:|
| MLP | 88.53% | 88.48% |
| 1D CNN | 92.87% | 92.88% |

The 1D CNN achieved the best test performance in this experiment.

## Project Structure

```text
ActivitySense-AI/
├── backend/
│   ├── main.py
│   └── test_prediction.py
├── frontend/
├── ml/
│   ├── models/
│   ├── preprocessing.py
│   ├── train_cnn.py
│   ├── train_mlp.py
│   └── evaluate_models.py
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

This project uses the UCI Human Activity Recognition Using Smartphones dataset.

The raw and processed datasets are excluded from this repository to keep it lightweight. Obtain the dataset from the official UCI Machine Learning Repository and run the preprocessing script before training or evaluating the models.

## Setup

### 1. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the dataset

Download and extract the UCI HAR dataset into `ml/data/raw/` so the expected `UCI HAR Dataset/` directory is present.

Run the preprocessing script:

```bash
python ml/preprocessing.py
```

### 3. Start the backend

From the project root:

```bash
python -m uvicorn backend.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

API documentation: `http://127.0.0.1:8000/docs`

### 4. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, usually `http://localhost:5173/`.

## Results

The `results/` directory contains model evaluation metrics and confusion matrices for comparing the MLP and 1D CNN models.

## Future Improvements

- Test on live smartphone sensor data.
- Improve real-time activity prediction.
- Add model performance visualizations and additional evaluation metrics.
- Explore more deep learning architectures.

## License

Add an appropriate license before distributing or reusing this project.
