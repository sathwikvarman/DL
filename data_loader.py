"""
Dataset loading, downloading, preprocessing, and sequence preparation for UCI HAR.
"""
import os
import zipfile
import json
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

from .config import (
    DATA_DIR,
    UCI_RAW_DIR,
    UCI_DATASET_URL,
    MIRROR_DATASET_URL,
    WINDOW_SIZE,
    N_CHANNELS,
    N_CLASSES,
    ACTIVITY_NAMES,
    CHANNEL_NAMES,
    SAVED_MODELS_DIR
)

SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]

def download_uci_dataset(dest_dir: Path = DATA_DIR) -> Path:
    """
    Downloads and extracts the UCI HAR dataset.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "uci_har_dataset.zip"
    extracted_dir = dest_dir / "UCI HAR Dataset"
    
    if extracted_dir.exists() and (extracted_dir / "train").exists():
        print(f"Dataset already extracted at {extracted_dir}")
        return extracted_dir
    
    if not zip_path.exists():
        print(f"Downloading UCI HAR Dataset from {UCI_DATASET_URL}...")
        try:
            # Try primary UCI repository
            req = urllib.request.Request(
                UCI_DATASET_URL,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Downloaded UCI HAR zip ({zip_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"Primary download failed: {e}. Trying mirror: {MIRROR_DATASET_URL}")
            try:
                urllib.request.urlretrieve(MIRROR_DATASET_URL, zip_path)
            except Exception as e2:
                print(f"Mirror download failed: {e2}")
                raise RuntimeError(f"Could not download UCI HAR dataset: {e2}")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
        
    nested_zip = dest_dir / "UCI HAR Dataset.zip"
    if nested_zip.exists():
        print(f"Extracting nested zip {nested_zip}...")
        with zipfile.ZipFile(nested_zip, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        
    print(f"Successfully extracted dataset to {extracted_dir}")
    return extracted_dir

def load_signal_file(file_path: Path) -> np.ndarray:
    """
    Reads space-separated text file of shape (N, 128).
    Handles potential multiple whitespace and missing values.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Signal file not found: {file_path}")
    
    # Load using pandas for robust delimiter handling
    df = pd.read_csv(file_path, header=None, sep=r'\s+')
    arr = df.to_numpy(dtype=np.float32)
    
    # Missing-value handling (interpolate or fill mean)
    if np.isnan(arr).any():
        print(f"Warning: NaNs found in {file_path.name}, imputing...")
        df_imputed = pd.DataFrame(arr).interpolate(method='linear', axis=1).bfill(axis=1).ffill(axis=1)
        arr = df_imputed.to_numpy(dtype=np.float32)
        
    return arr

def load_inertial_signals(dataset_dir: Path, split: str = "train") -> np.ndarray:
    """
    Loads and stacks all 9 inertial signals for a given split ('train' or 'test').
    Returns 3D array of shape (N, 128, 9).
    """
    signals_dir = dataset_dir / split / "Inertial Signals"
    signals_data = []
    
    for sig in SIGNALS:
        sig_file = signals_dir / f"{sig}_{split}.txt"
        sig_arr = load_signal_file(sig_file)  # (N, 128)
        signals_data.append(sig_arr)
        
    # Stack along 3rd axis: (N, 128, 9)
    stacked = np.stack(signals_data, axis=-1)
    return stacked

def load_labels(dataset_dir: Path, split: str = "train") -> np.ndarray:
    """
    Loads y_{split}.txt and converts 1-indexed labels (1..6) to 0-indexed (0..5).
    """
    label_file = dataset_dir / split / f"y_{split}.txt"
    y = np.loadtxt(label_file, dtype=int) - 1
    return y

def generate_synthetic_har_data(n_samples: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-fidelity biomechanical time-series generator adhering to the exact 50Hz,
    128-timestep, 9-channel UCI HAR specification.
    Used for instant fallback, testing, or offline development.
    """
    np.random.seed(42)
    X = np.zeros((n_samples, WINDOW_SIZE, N_CHANNELS), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=int)
    
    samples_per_class = n_samples // N_CLASSES
    t = np.linspace(0, 2.56, WINDOW_SIZE)  # 2.56 seconds @ 50 Hz
    
    for c in range(N_CLASSES):
        start_idx = c * samples_per_class
        end_idx = start_idx + samples_per_class if c < N_CLASSES - 1 else n_samples
        count = end_idx - start_idx
        y[start_idx:end_idx] = c
        
        for i in range(count):
            idx = start_idx + i
            noise = np.random.normal(0, 0.05, (WINDOW_SIZE, N_CHANNELS))
            
            if c == 0:  # WALKING: ~1.8 Hz rhythmic vertical and lateral cadence
                freq = np.random.uniform(1.6, 2.0)
                phase = np.random.uniform(0, 2 * np.pi)
                X[idx, :, 0] = 0.3 * np.sin(2 * np.pi * freq * t + phase) + noise[:, 0]        # body_acc_x
                X[idx, :, 1] = 0.5 * np.cos(2 * np.pi * freq * 2 * t + phase) + noise[:, 1]    # body_acc_y
                X[idx, :, 2] = 0.4 * np.sin(2 * np.pi * freq * t + phase) + noise[:, 2]        # body_acc_z
                X[idx, :, 3] = 0.8 * np.sin(2 * np.pi * freq * t) + noise[:, 3]                # gyro_x
                X[idx, :, 4] = 0.4 * np.cos(2 * np.pi * freq * t) + noise[:, 4]                # gyro_y
                X[idx, :, 5] = 0.3 * np.sin(2 * np.pi * freq * 2 * t) + noise[:, 5]            # gyro_z
                X[idx, :, 6] = X[idx, :, 0] + 0.1                                              # total_acc_x
                X[idx, :, 7] = X[idx, :, 1] + 0.98                                             # total_acc_y (gravity)
                X[idx, :, 8] = X[idx, :, 2] + 0.05
                
            elif c == 1:  # WALKING_UPSTAIRS: Higher vertical impact, lower cadence ~1.4 Hz
                freq = np.random.uniform(1.2, 1.5)
                phase = np.random.uniform(0, 2 * np.pi)
                X[idx, :, 0] = 0.4 * np.sin(2 * np.pi * freq * t + phase) + noise[:, 0]
                X[idx, :, 1] = 0.8 * np.sin(2 * np.pi * freq * 2 * t + phase) + noise[:, 1]
                X[idx, :, 2] = 0.6 * np.cos(2 * np.pi * freq * t + phase) + noise[:, 2]
                X[idx, :, 3] = 1.2 * np.sin(2 * np.pi * freq * t) + noise[:, 3]
                X[idx, :, 4] = 0.6 * np.cos(2 * np.pi * freq * t) + noise[:, 4]
                X[idx, :, 5] = 0.5 * np.sin(2 * np.pi * freq * 2 * t) + noise[:, 5]
                X[idx, :, 6] = X[idx, :, 0] + 0.15
                X[idx, :, 7] = X[idx, :, 1] + 1.05
                X[idx, :, 8] = X[idx, :, 2] + 0.1
                
            elif c == 2:  # WALKING_DOWNSTAIRS: Sharp deceleration peaks, ~1.9 Hz
                freq = np.random.uniform(1.7, 2.2)
                phase = np.random.uniform(0, 2 * np.pi)
                X[idx, :, 0] = 0.5 * np.sin(2 * np.pi * freq * t + phase) + noise[:, 0]
                X[idx, :, 1] = 0.9 * np.cos(2 * np.pi * freq * 2 * t + phase) + noise[:, 1]
                X[idx, :, 2] = 0.5 * np.sin(2 * np.pi * freq * t + phase) + noise[:, 2]
                X[idx, :, 3] = 1.4 * np.cos(2 * np.pi * freq * t) + noise[:, 3]
                X[idx, :, 4] = 0.8 * np.sin(2 * np.pi * freq * t) + noise[:, 4]
                X[idx, :, 5] = 0.4 * np.cos(2 * np.pi * freq * t) + noise[:, 5]
                X[idx, :, 6] = X[idx, :, 0] + 0.2
                X[idx, :, 7] = X[idx, :, 1] + 0.95
                X[idx, :, 8] = X[idx, :, 2] + 0.15
                
            elif c == 3:  # SITTING: Quiet sensor readings, vertical gravity ~0.98g
                X[idx, :, 0] = np.random.normal(0, 0.02, WINDOW_SIZE)
                X[idx, :, 1] = np.random.normal(0, 0.02, WINDOW_SIZE)
                X[idx, :, 2] = np.random.normal(0, 0.02, WINDOW_SIZE)
                X[idx, :, 3] = np.random.normal(0, 0.03, WINDOW_SIZE)
                X[idx, :, 4] = np.random.normal(0, 0.03, WINDOW_SIZE)
                X[idx, :, 5] = np.random.normal(0, 0.02, WINDOW_SIZE)
                X[idx, :, 6] = 0.05 + noise[:, 0] * 0.2
                X[idx, :, 7] = 0.98 + noise[:, 1] * 0.2
                X[idx, :, 8] = 0.12 + noise[:, 2] * 0.2
                
            elif c == 4:  # STANDING: Upright postural sway (< 0.5 Hz)
                sway = 0.05 * np.sin(2 * np.pi * 0.3 * t)
                X[idx, :, 0] = sway + np.random.normal(0, 0.025, WINDOW_SIZE)
                X[idx, :, 1] = np.random.normal(0, 0.025, WINDOW_SIZE)
                X[idx, :, 2] = sway * 0.5 + np.random.normal(0, 0.025, WINDOW_SIZE)
                X[idx, :, 3] = np.random.normal(0, 0.04, WINDOW_SIZE)
                X[idx, :, 4] = np.random.normal(0, 0.04, WINDOW_SIZE)
                X[idx, :, 5] = np.random.normal(0, 0.03, WINDOW_SIZE)
                X[idx, :, 6] = -0.05 + noise[:, 0] * 0.2
                X[idx, :, 7] = 0.99 + noise[:, 1] * 0.2
                X[idx, :, 8] = 0.08 + noise[:, 2] * 0.2
                
            elif c == 5:  # LAYING: Horizontal orientation; gravity along X/Z axis
                X[idx, :, 0] = np.random.normal(0, 0.015, WINDOW_SIZE)
                X[idx, :, 1] = np.random.normal(0, 0.015, WINDOW_SIZE)
                X[idx, :, 2] = np.random.normal(0, 0.015, WINDOW_SIZE)
                X[idx, :, 3] = np.random.normal(0, 0.02, WINDOW_SIZE)
                X[idx, :, 4] = np.random.normal(0, 0.02, WINDOW_SIZE)
                X[idx, :, 5] = np.random.normal(0, 0.02, WINDOW_SIZE)
                X[idx, :, 6] = 0.95 + noise[:, 0] * 0.15  # gravity on X
                X[idx, :, 7] = 0.05 + noise[:, 1] * 0.15  # minimal vertical
                X[idx, :, 8] = 0.25 + noise[:, 2] * 0.15
                
    # Shuffle dataset
    shuffle_indices = np.random.permutation(n_samples)
    return X[shuffle_indices], y[shuffle_indices]

def scale_inertial_sequences(
    X_train: np.ndarray,
    X_test: np.ndarray = None,
    save_path: Path = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Normalizes time-series sequences channel-by-channel across samples and timesteps.
    Saves mean and std per channel for inference scaling.
    """
    n_channels = X_train.shape[2]
    means = []
    stds = []
    
    X_train_scaled = np.zeros_like(X_train)
    X_test_scaled = np.zeros_like(X_test) if X_test is not None else None
    
    for ch in range(n_channels):
        mean = float(np.mean(X_train[:, :, ch]))
        std = float(np.std(X_train[:, :, ch]))
        if std < 1e-6:
            std = 1.0
        means.append(mean)
        stds.append(std)
        
        X_train_scaled[:, :, ch] = (X_train[:, :, ch] - mean) / std
        if X_test is not None:
            X_test_scaled[:, :, ch] = (X_test[:, :, ch] - mean) / std
            
    scaler_params = {
        "channel_names": CHANNEL_NAMES,
        "means": means,
        "stds": stds
    }
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(scaler_params, f, indent=2)
        print(f"Saved scaler parameters to {save_path}")
        
    return X_train_scaled, X_test_scaled, scaler_params

def extract_and_save_sample_files(
    X_raw: np.ndarray,
    y: np.ndarray,
    samples_dir: Path
):
    """
    Extracts representative clean sequences for each of the 6 activities
    and writes them as JSON and CSV sample files for quick frontend testing.
    """
    samples_dir.mkdir(parents=True, exist_ok=True)
    samples_index = []
    
    for activity_id, activity_name in enumerate(ACTIVITY_NAMES):
        indices = np.where(y == activity_id)[0]
        if len(indices) == 0:
            continue
            
        chosen_idx = int(indices[0])
        sequence_data = X_raw[chosen_idx].tolist()  # (128, 9)
        
        sample_filename = f"sample_{activity_name.lower()}.json"
        sample_path = samples_dir / sample_filename
        
        sample_dict = {
            "activity": activity_name,
            "activity_id": activity_id + 1,
            "timesteps": WINDOW_SIZE,
            "channels": CHANNEL_NAMES,
            "sampling_rate_hz": 50,
            "data": sequence_data
        }
        
        with open(sample_path, "w") as f:
            json.dump(sample_dict, f)
            
        # Also write a CSV version
        csv_path = samples_dir / f"sample_{activity_name.lower()}.csv"
        df_sample = pd.DataFrame(sequence_data, columns=CHANNEL_NAMES)
        df_sample.to_csv(csv_path, index=False)
        
        samples_index.append({
            "activity": activity_name,
            "activity_id": activity_id + 1,
            "json_file": sample_filename,
            "csv_file": f"sample_{activity_name.lower()}.csv"
        })
        
    index_path = samples_dir / "samples_index.json"
    with open(index_path, "w") as f:
        json.dump(samples_index, f, indent=2)
    print(f"Saved sample test files to {samples_dir}")

def get_har_dataset(use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Full dataset preparation pipeline:
    1. Downloads or loads UCI HAR.
    2. Falls back to realistic synthetic if unavailable.
    3. Handles missing values.
    4. Scales features per channel.
    5. One-hot encodes labels.
    6. Extracts test samples.
    """
    scaler_path = SAVED_MODELS_DIR / "scaler_params.json"
    samples_dir = DATA_DIR / "samples"
    
    try:
        dataset_dir = download_uci_dataset(DATA_DIR)
        print("Loading UCI HAR train set...")
        X_train_raw = load_inertial_signals(dataset_dir, "train")
        y_train = load_labels(dataset_dir, "train")
        
        print("Loading UCI HAR test set...")
        X_test_raw = load_inertial_signals(dataset_dir, "test")
        y_test = load_labels(dataset_dir, "test")
        print(f"Loaded real UCI HAR: Train shape {X_train_raw.shape}, Test shape {X_test_raw.shape}")
    except Exception as e:
        print(f"Could not load real UCI HAR dataset ({e}). Generating realistic synthetic dataset...")
        X_synthetic, y_synthetic = generate_synthetic_har_data(n_samples=3000)
        split_idx = int(0.75 * len(X_synthetic))
        X_train_raw, X_test_raw = X_synthetic[:split_idx], X_synthetic[split_idx:]
        y_train, y_test = y_synthetic[:split_idx], y_synthetic[split_idx:]
        print(f"Synthetic dataset created: Train {X_train_raw.shape}, Test {X_test_raw.shape}")
        
    # Scale sequences
    X_train, X_test, scaler_params = scale_inertial_sequences(
        X_train_raw,
        X_test_raw,
        save_path=scaler_path
    )
    
    # Save test sample files for UI/API
    extract_and_save_sample_files(X_test_raw, y_test, samples_dir)
    
    # Convert labels to one-hot categorical vectors (N, 6)
    y_train_cat = to_categorical(y_train, num_classes=N_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=N_CLASSES)
    
    return X_train, y_train_cat, X_test, y_test_cat, scaler_params
