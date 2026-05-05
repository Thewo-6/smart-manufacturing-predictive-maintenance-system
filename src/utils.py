"""
Utility functions for the Predictive Maintenance System
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from config import RANDOM_SEED, DEVICE


def set_random_seed(seed=RANDOM_SEED):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count total and trainable parameters in model
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        dict: Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def print_model_info(model, model_name):
    """
    Print comprehensive model information
    
    Args:
        model (nn.Module): PyTorch model
        model_name (str): Name of model
    """
    params = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Total Parameters:        {params['total_parameters']:,}")
    print(f"Trainable Parameters:    {params['trainable_parameters']:,}")
    print(f"Non-trainable Params:    {params['non_trainable_parameters']:,}")
    print(f"Device:                  {DEVICE}")
    print(f"{'='*60}\n")


def plot_training_history(history, model_name, save_dir=None):
    """
    Plot training history
    
    Args:
        history (dict): Training history dictionary
        model_name (str): Name of model
        save_dir (Path): Directory to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    lr = history['learning_rate']
    
    # Loss plot
    axes[0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name.upper()} - Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1].plot(epochs, lr, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title(f'{model_name.upper()} - Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / f"{model_name}_training_history.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")
    
    plt.show()


def compare_models(metrics_dict):
    """
    Create comparison table of models
    
    Args:
        metrics_dict (dict): Dictionary of model names to metrics
        
    Returns:
        pd.DataFrame: Comparison dataframe
    """
    comparison_df = pd.DataFrame(metrics_dict).T
    comparison_df = comparison_df.round(4)
    
    return comparison_df


def print_comparison_table(metrics_dict):
    """
    Print model comparison table
    
    Args:
        metrics_dict (dict): Dictionary of model names to metrics
    """
    comparison_df = compare_models(metrics_dict)
    
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON")
    print(f"{'='*80}")
    print(comparison_df.to_string())
    print(f"{'='*80}\n")


def save_metrics_json(metrics, model_name, save_dir):
    """
    Save metrics to JSON file
    
    Args:
        metrics (dict): Metrics dictionary
        model_name (str): Name of model
        save_dir (Path): Directory to save
    """
    save_path = Path(save_dir) / f"{model_name}_metrics.json"
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved: {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device=DEVICE):
    """
    Load checkpoint
    
    Args:
        checkpoint_path (Path): Path to checkpoint
        model (nn.Module): Model to load weights into
        optimizer (optim.Optimizer): Optimizer to load state into
        device (torch.device): Device
        
    Returns:
        dict: Checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    return checkpoint


def save_checkpoint(checkpoint_path, epoch, model, optimizer, best_val_loss, history, model_name):
    """
    Save checkpoint
    
    Args:
        checkpoint_path (Path): Path to save checkpoint
        epoch (int): Current epoch
        model (nn.Module): Model to save
        optimizer (optim.Optimizer): Optimizer state to save
        best_val_loss (float): Best validation loss
        history (dict): Training history
        model_name (str): Name of model
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'history': history,
        'model_name': model_name
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def get_device_info():
    """
    Get information about available device
    
    Returns:
        str: Device information string
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        return f"GPU: {gpu_name} (Count: {gpu_count})"
    else:
        return "CPU (No GPU available)"


def plot_predictions_comparison(predictions_dict, targets, save_dir=None):
    """
    Plot predictions from multiple models side by side
    
    Args:
        predictions_dict (dict): Dictionary of model names to predictions
        targets (np.array): True RUL values
        save_dir (Path): Directory to save plot
    """
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
        axes[idx].scatter(targets, predictions, alpha=0.5, s=20)
        axes[idx].plot([targets.min(), targets.max()], 
                      [targets.min(), targets.max()], 
                      'r--', lw=2, label='Perfect')
        axes[idx].set_xlabel('True RUL')
        axes[idx].set_ylabel('Predicted RUL')
        axes[idx].set_title(f'{model_name.upper()}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / "models_comparison.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
    
    plt.show()


def generate_report(results_dict, save_path=None):
    """
    Generate text report
    
    Args:
        results_dict (dict): Results dictionary
        save_path (Path): Path to save report
        
    Returns:
        str: Report text
    """
    report = f"""
{'='*80}
PREDICTIVE MAINTENANCE SYSTEM - EVALUATION REPORT
{'='*80}

Dataset Information:
{results_dict.get('dataset_info', 'N/A')}

Model Comparison:
{results_dict.get('model_comparison', 'N/A')}

Recommendations:
{results_dict.get('recommendations', 'N/A')}

{'='*80}
Generated at: {pd.Timestamp.now()}
{'='*80}
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved: {save_path}")
    
    return report


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    set_random_seed()
    print(f"Device: {get_device_info()}")
    
    # Test model parameter counting
    from src.models import LSTMModel
    from config import LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT, LSTM_OUTPUT_SIZE
    
    model = LSTMModel(
        input_size=LSTM_INPUT_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        output_size=LSTM_OUTPUT_SIZE
    )
    
    print_model_info(model, "LSTM")
