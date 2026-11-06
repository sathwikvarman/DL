# 🏃‍♂️ Human Activity Recognition (HAR) Using LSTM Deep Learning

A full-stack, end-to-end Machine Learning web application that classifies human physical activities from smartphone inertial sensors (3-axis accelerometer and 3-axis gyroscope) in real time using a stacked Long Short-Term Memory (LSTM) recurrent neural network.

Trained on the benchmark **UCI Human Activity Recognition Using Smartphones Dataset** (10,299 instances across 30 human subjects).

---

## 🌟 Key Features

1. **Stacked LSTM Deep Learning Architecture**:
   - **Input**: `(Batch, 128 Timesteps, 9 Inertial Channels)` representing 2.56-second sliding windows sampled at 50 Hz.
   - **Layer 1**: 128-unit LSTM with `return_sequences=True` for temporal dependency extraction.
   - **Regularization**: 30% Dropout & L2 weight regularization to prevent overfitting.
   - **Layer 2**: 64-unit LSTM for latent sequence condensation.
   - **Dense Layer**: 64 neurons with ReLU activation.
   - **Output Layer**: 6 neurons with Softmax activation predicting physical activities.

2. **6 Classified Human Activities**:
   - 🚶 **Walking** (Dynamic locomotion, ~1.8 Hz cadence, ~280 kcal/h)
   - 🧗 **Walking Upstairs** (High vertical acceleration peaks against gravity, ~580 kcal/h)
   - 📉 **Walking Downstairs** (Sharp heel impact decelerations, ~340 kcal/h)
   - 🪑 **Sitting** (Static upright resting posture, vertical gravity ~0.98g, ~95 kcal/h)
   - 🧍 **Standing** (Static posture with low-frequency sway oscillations <0.5 Hz, ~130 kcal/h)
   - 🛏️ **Laying** (Horizontal posture with 1g gravity vector reoriented along X/Z axis, ~70 kcal/h)

3. **9 Inertial Sensor Channels**:
   - `body_acc_x`, `body_acc_y`, `body_acc_z` (Triaxial body linear acceleration in $g$)
   - `body_gyro_x`, `body_gyro_y`, `body_gyro_z` (Triaxial angular velocity in $\text{rad/s}$)
   - `total_acc_x`, `total_acc_y`, `total_acc_z` (Total acceleration including gravitational component)

4. **Interactive Real-Time Sensor Stream Simulator**:
   - 60 FPS HTML5 Canvas oscilloscope rendering multi-axis accelerometer and gyroscope waveforms.
   - Dynamic frequency, cadence, and micro-tremor noise controls.
   - Live stream mode: sends 128-frame windows into the LSTM model and displays instantaneous activity classification, confidence meter, METs, and estimated caloric burn rate.

5. **Sample Benchmark Explorer**:
   - 1-click test cards for all 6 activities with actual test sequences extracted from the dataset.
   - Multi-channel sensor vector viewer and live prediction breakdown.

6. **File Upload & Sliding Window Segmentation**:
   - Drag & drop CSV or JSON time-series sensor logs.
   - Auto-detects column names, handles missing values, and segments continuous recordings into overlapping 128-timestep windows (50% overlap).
   - Generates an interactive temporal activity timeline (e.g. 0–10s Walking, 10–25s Upstairs, etc.).
   - Includes a built-in **"Download Sample Test CSV"** generator button for instant testing.

7. **Model Inspector & Evaluation Suite**:
   - Confusion Matrix heatmap with true positive vs predicted classifications.
   - Per-class Precision, Recall, and F1-Score table.
   - Layer-by-layer architectural parameter breakdown.
   - Live retraining trigger with customizable epochs, batch size, and learning rate.

8. **Audit Trail & SQLite History**:
   - Persistent SQLite database recording every prediction timestamp, source, recognized activity, confidence, and sensor kinematics (mean/peak acceleration).
   - Filter, search, and CSV export functionality.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Deep Learning / ML** | Python 3.12, TensorFlow 2.16+, Keras 3, Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn |
| **Backend API** | FastAPI, Uvicorn (ASGI), Pydantic v2, Python-Multipart |
| **Frontend UI** | React 18, Vite, Tailwind CSS, Lucide Icons, HTML5 Canvas Oscilloscope |
| **Database** | SQLite 3 (`backend/data/predictions.db`) |
| **Dataset** | UCI Human Activity Recognition Using Smartphones (50 Hz, 128 timesteps, 9 channels) |

---

## 📁 Project Architecture

```
har-lstm-app/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI endpoints, CORS, static frontend mount
│   │   ├── config.py              # Activity metadata, METs, channel definitions
│   │   ├── database.py            # SQLite schema and query functions
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── services/
│   │       ├── model_service.py   # LSTM inference engine & statistics
│   │       ├── preprocessor.py    # Standard scaling & sliding window generator
│   │       ├── simulator.py       # Biomechanical sensor generator
│   │       └── dataset_service.py # Sample retrieval service
│   ├── ml/
│   │   ├── config.py              # Hyperparameters & paths
│   │   ├── data_loader.py         # UCI HAR parser, scaler, train-test split
│   │   ├── model.py               # Stacked LSTM architecture definition
│   │   ├── train.py               # Training loop, EarlyStopping, metrics export
│   │   └── saved_models/
│   │       ├── har_lstm_model.keras   # Trained Keras model
│   │       ├── scaler_params.json     # Feature normalization parameters
│   │       ├── training_metrics.json  # Loss, accuracy, confusion matrix
│   │       ├── confusion_matrix.png   # Seaborn heatmap plot
│   │       └── loss_accuracy_curves.png # Matplotlib learning curves
│   ├── data/
│   │   ├── samples/               # Pre-extracted clean JSON & CSV activity samples
│   │   └── predictions.db         # SQLite database
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── LiveSensorStream.jsx      # Real-time oscilloscope & live classification
│   │   │   ├── SampleTester.jsx          # 1-click test cards for 6 activities
│   │   │   ├── FileUploadAnalyzer.jsx    # CSV upload & timeline segmentation
│   │   │   ├── ModelInspector.jsx        # Confusion matrix & architecture viewer
│   │   │   ├── PredictionHistory.jsx     # SQLite history table & export
│   │   │   └── ActivityProbabilityBar.jsx
│   │   ├── services/
│   │   │   └── api.js                    # Fetch API client
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── scripts/
│   └── start.sh                   # Unified launcher for Backend and Frontend
└── README.md
```

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.11 or 3.12
- Node.js (v18+) & npm

### 1. Automated Start (Recommended)
Run the unified start script from the project directory:

```bash
cd /Users/varma/.gemini/antigravity/scratch/har-lstm-app
./scripts/start.sh
```

This script:
1. Activates the Python virtual environment (`.venv`).
2. Checks for the trained `.keras` model (trains automatically if not present).
3. Launches the **FastAPI Backend** on `http://127.0.0.1:8000`.
4. Launches the **React/Vite Frontend** on `http://127.0.0.1:5173`.

---

### 2. Manual Step-by-Step Setup

#### Backend Setup
```bash
cd backend

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training pipeline (downloads UCI HAR dataset and trains LSTM)
python -m backend.ml.train

# Start FastAPI server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Frontend Setup
```bash
cd frontend

# Install npm dependencies
npm install

# Start Vite development server
npm run dev
```

Visit `http://localhost:5173` in your browser.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health, model status, sampling rate |
| `GET` | `/api/activities` | Metadata for all 6 physical activities, METs, colors |
| `GET` | `/api/model-info` | Model architecture, parameter counts, test metrics |
| `GET` | `/api/model/confusion-matrix-image` | Confusion matrix heatmap PNG |
| `GET` | `/api/model/curves-image` | Loss and accuracy curves PNG |
| `GET` | `/api/samples` | List of benchmark activity test sequences |
| `GET` | `/api/samples/{activity}` | Full 128×9 sensor array for specified activity |
| `POST` | `/api/predict` | Classifies 128×9 sensor matrix, returns probabilities |
| `POST` | `/api/predict/upload` | Uploads CSV/JSON file, segments timeline, returns breakdown |
| `POST` | `/api/simulate/window` | Simulates 128 frames for specified activity & predicts |
| `GET` | `/api/history` | Fetches SQLite prediction history |
| `DELETE` | `/api/history/{id}` | Deletes record from SQLite |
| `DELETE` | `/api/history` | Clears all prediction history |
| `GET` | `/api/stats` | Prediction count & confidence breakdown |
| `POST` | `/api/train` | Triggers background model training run |

Interactive Swagger API docs available at `http://127.0.0.1:8000/docs`.
#   D L  
 