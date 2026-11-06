#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$(dirname "$DIR")"

echo "=========================================================="
echo " Starting Human Activity Recognition (HAR) LSTM Application"
echo "=========================================================="

# Activate Python Virtual Environment
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "✔ Activated Python virtual environment (.venv)"
else
    echo "✘ Virtual environment .venv not found. Please create one with Python 3.12."
    exit 1
fi

# Check if model exists, if not run training
MODEL_PATH="$PROJECT_ROOT/backend/ml/saved_models/har_lstm_model.keras"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ℹ No trained model found at $MODEL_PATH."
    echo "▶ Training initial HAR LSTM model now..."
    python -m backend.ml.train
fi

# Start Backend API in background
echo "▶ Starting FastAPI Backend on http://127.0.0.1:8000..."
cd "$PROJECT_ROOT"
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# Start Frontend
echo "▶ Starting React/Vite Frontend on http://127.0.0.1:5173..."
cd "$PROJECT_ROOT/frontend"
npm run dev -- --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!

echo ""
echo "=========================================================="
echo " Application is up and running!"
echo " • Frontend: http://127.0.0.1:5173"
echo " • Backend API & Swagger Docs: http://127.0.0.1:8000/docs"
echo " Press Ctrl+C to terminate both servers."
echo "=========================================================="

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
