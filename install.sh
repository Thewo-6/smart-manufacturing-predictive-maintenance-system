#!/bin/bash

# Installation script for Predictive Maintenance System on Mac M4
# This script sets up the environment and installs all dependencies

echo "=================================================="
echo "Predictive Maintenance System - Installation"
echo "Mac M4 (Apple Silicon) Setup"
echo "=================================================="

# Ensure we're in the right directory
cd "$(dirname "$0")" || exit 1

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip, setuptools, wheel
echo ""
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel
echo "✓ Pip tools upgraded"

# Install PyTorch for Apple Silicon (if M4/M3/M2/M1 Mac)
echo ""
echo "Installing PyTorch for Apple Silicon..."
pip install --upgrade torch torchvision torchaudio
echo "✓ PyTorch installed"

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt
echo "✓ All dependencies installed"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ GPU available: {torch.cuda.is_available()}'); print(f'✓ Metal available: {torch.backends.mps.is_available()}')"

echo ""
echo "=================================================="
echo "✓ Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Download dataset: See QUICK_START.md"
echo "3. Preprocess data: python -c \"from src.data_preprocessing import preprocess_dataset; preprocess_dataset('FD001', reload=True)\""
echo "4. Run dashboard: streamlit run app.py"
echo ""
