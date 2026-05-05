# 🚀 Quick Start Guide

Get the Predictive Maintenance System running in 10 minutes!

## 5-Minute Setup

### 1. Install (2 minutes)

```bash
cd predictive_maintenance_project
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Download Data (2 minutes)

```bash
# Visit and download:
# https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

# Extract to: data/raw/
# You'll have:
# - train_FD001.txt
# - test_FD001.txt  
# - RUL_FD001.txt
```

### 3. Preprocess (1 minute)

```bash
python -c "from src.data_preprocessing import preprocess_dataset; preprocess_dataset('FD001', reload=True)"
```

---

## 5-Minute First Run

### Option A: Interactive Dashboard

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

**Dashboard pages:**
- 🏠 Home - Overview
- 📊 Dataset - Explore data
- 🏋️ Training - Train models
- 🔮 Predictions - Make predictions
- 📈 Visualizations - View results
- ⚖️ Comparison - Compare models

### Option B: Command Line

#### Train models (5-10 minutes)

```bash
# Train both LSTM and Transformer
python run.py train --model both --dataset FD001 --epochs 50

# Or train individually
python run.py train --model lstm --epochs 50
python run.py train --model transformer --epochs 50
```

#### Evaluate models

```bash
python run.py evaluate --model both
```

#### Make predictions

```bash
python run.py predict --model lstm
```

#### Run entire pipeline

```bash
python run.py all --epochs 50
```

---

## Common Tasks

### Just explore data (no training)

```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
train_df, test_df, train_rul, test_rul = preprocessor.read_raw_files()
print(f"Train data: {train_df.shape}")
print(f"Test data: {test_df.shape}")
print(train_df.head())
```

### Train one model (fastest)

```bash
python run.py train --model lstm --epochs 30
```

Expected output:
```
==============================================================
PREDICTIVE MAINTENANCE SYSTEM
==============================================================
Command: train
Model: lstm
Device: cuda (or cpu)
==============================================================

[Training progress shown with tqdm bars]

✓ lstm training complete!
```

### Quick predictions

```python
from src.predict import RULPredictor, create_sample_engine_data
from src.data_preprocessing import preprocess_dataset
import torch
from config import DEVICE
from src.models import LSTMModel

# Load model
model = LSTMModel(21, 128, 2, 0.2, 1).to(DEVICE)
ckpt = torch.load('checkpoints/lstm_best.pt', map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])

# Load scaler
data = preprocess_dataset('FD001')
scaler = data['scaler']

# Predict
predictor = RULPredictor(model, scaler)
engine = create_sample_engine_data(100)
rul = predictor.predict_engine_trajectory(engine)
print(f"RUL: {rul['current_rul']:.2f} cycles")
```

---

## File Structure Reference

```
predictive_maintenance_project/
├── config.py                 # Settings (modify here!)
├── app.py                    # Dashboard
├── run.py                    # Main script
├── requirements.txt          # Dependencies
├── README.md                 # Full docs
├── INSTALL.md               # Installation help
├── src/
│   ├── data_preprocessing.py # Data loading
│   ├── models.py             # LSTM & Transformer
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Metrics & plots
│   ├── predict.py            # Make predictions
│   └── utils.py              # Helpers
├── data/
│   ├── raw/                  # Your downloaded files
│   └── processed/            # Cached preprocessed data
├── checkpoints/              # Trained models
└── outputs/
    ├── figures/              # Plot images
    └── results/              # CSV/JSON results
```

---

## Modifying Configuration

Edit `config.py` to change:

```python
# Number of sensors to use
NUM_SENSORS = 21

# Maximum RUL clipping
RUL_CLIP_VALUE = 125

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100

# Model parameters
LSTM_HIDDEN_SIZE = 128
TRANSFORMER_D_MODEL = 128
```

Then retrain for new results.

---

**Actual results depend on:**
- Dataset variant (FD001/FD002/FD003/FD004)
- Number of training epochs
- Hardware (GPU much faster than CPU)
- Hyperparameter choices

---

## Next Steps

1. **Try different datasets:** Change `FD001` to `FD002-4` in commands
2. **Tune hyperparameters:** Edit `config.py` and retrain
3. **Deploy:** Use `run.py` in production scripts
4. **Visualize:** Check `outputs/figures/` for plots
5. **Read full docs:** See `README.md` for complete guide

---

**Ready?** Run `streamlit run app.py` and start exploring! 🚀

For detailed documentation, see **README.md**
