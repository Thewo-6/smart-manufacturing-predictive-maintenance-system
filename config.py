"""
Configuration file for the Predictive Maintenance System
Contains all hyperparameters, paths, and settings
"""

import torch
import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, CHECKPOINTS_DIR, FIGURES_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================
# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# DATA PARAMETERS
# ============================================================
DATASET_NAME = "FD001"  # Can be FD001, FD002, FD003, FD004
SENSOR_COLUMNS = [
    'Operational_Setting_1', 'Operational_Setting_2', 'Operational_Setting_3',
    'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_6', 'Sensor_7',
    'Sensor_8', 'Sensor_9', 'Sensor_10', 'Sensor_11', 'Sensor_12', 'Sensor_13',
    'Sensor_14', 'Sensor_15', 'Sensor_16', 'Sensor_17', 'Sensor_18', 'Sensor_19',
    'Sensor_20', 'Sensor_21'
]
RUL_CLIP_VALUE = 125  # Maximum RUL value to clip at
SEQUENCE_LENGTH = 30  # Number of timesteps in each window
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% validation

# ============================================================
# MODEL PARAMETERS
# ============================================================
# LSTM Model
LSTM_INPUT_SIZE = len(SENSOR_COLUMNS)  # Number of sensor features
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_OUTPUT_SIZE = 1  # Predict single RUL value

# Transformer Model
TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 256
TRANSFORMER_DROPOUT = 0.2
TRANSFORMER_OUTPUT_SIZE = 1

# ============================================================
# TRAINING PARAMETERS
# ============================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
GRADIENT_CLIP_VALUE = 1.0

# Loss function and optimizer
LOSS_FUNCTION = "mse"  # Options: 'mse', 'mae'
OPTIMIZER = "adam"  # Options: 'adam', 'sgd'

# ============================================================
# EVALUATION PARAMETERS
# ============================================================
# NASA RUL scoring function parameters
ALPHA1 = 10  # Score multiplier for negative errors
ALPHA2 = 13  # Score multiplier for positive errors

# ============================================================
# RANDOM SEED
# ============================================================
RANDOM_SEED = 42

# ============================================================
# LOGGING AND CHECKPOINT
# ============================================================
SAVE_CHECKPOINT_EVERY_N_EPOCHS = 10
SAVE_BEST_MODEL = True
RESUME_FROM_CHECKPOINT = False  # Set to True to resume training from last checkpoint
CHECKPOINT_NAME = "best_model.pt"

# ============================================================
# VISUALIZATION PARAMETERS
# ============================================================
FIGURE_DPI = 100
FIGURE_FIGSIZE = (12, 6)
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# ============================================================
# DOWNLOAD INSTRUCTIONS
# ============================================================
DATASET_DOWNLOAD_URL = "https://data.nasa.gov/mbaas/api/v1/metabase/card/573576/query/json"
DATASET_INSTRUCTIONS = """
Download the NASA C-MAPSS Turbofan Engine Dataset:
1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
2. Download the "Turbofan Engine Degradation Simulation Data Set"
3. Extract the files to: data/raw/
4. You should have files like: train_FD001.txt, test_FD001.txt, RUL_FD001.txt
"""

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_PROCESSED_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Number of Sensors: {len(SENSOR_COLUMNS)}")
