# Installation Guide

## Quick Installation (macOS/Linux/Windows)

### Prerequisites
- Python 3.8+
- pip
- 4GB+ RAM

### Step 1: Navigate to Project

```bash
cd predictive_maintenance_project
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Download from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
2. Extract files to: `data/raw/`

### Step 5: Verify Setup

```bash
python config.py
```

You should see output confirming your device (GPU or CPU).

## Troubleshooting

### Issue: Python not found
```bash
# Try python3 instead
python3 -m venv venv
python3 -m pip install -r requirements.txt
```

### Issue: Permission denied
```bash
chmod +x venv/bin/activate
```

### Issue: CUDA/GPU not found
This is normal on CPU-only systems. The code will run on CPU but slower.

## Next Steps

After installation, follow the Quick Start guide in README.md:

1. **Preprocess data:**
   ```bash
   python -c "from src.data_preprocessing import preprocess_dataset; preprocess_dataset('FD001', reload=True)"
   ```

2. **Train models:**
   ```bash
   python src/train.py
   ```

3. **Evaluate:**
   ```bash
   python src/evaluate.py
   ```

4. **Launch dashboard:**
   ```bash
   streamlit run app.py
   ```

## Virtual Environment Management

```bash
# Activate
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Deactivate
deactivate

# Remove
rm -rf venv  # macOS/Linux
rmdir venv   # Windows
```

## GPU Setup (Optional)

If you have an NVIDIA GPU and want CUDA acceleration:

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

For detailed information, see README.md
