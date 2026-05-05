"""
Training script for Turbofan Engine RUL Prediction models
Handles training loop, validation, checkpointing, and resume functionality
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_VALUE, LOSS_FUNCTION,
    OPTIMIZER, SAVE_CHECKPOINT_EVERY_N_EPOCHS, SAVE_BEST_MODEL,
    CHECKPOINTS_DIR, RESULTS_DIR, RANDOM_SEED
)
from src.models import create_model
from src.dataset import create_data_loaders
from src.data_preprocessing import preprocess_dataset


class Trainer:
    """
    Trainer class for model training and validation
    Handles:
    - Training loop with validation
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving/loading
    - Training history
    """
    
    def __init__(self, model, train_loader, val_loader, model_name, config):
        """
        Initialize trainer
        
        Args:
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            model_name (str): Name of model ('lstm' or 'transformer')
            config (dict): Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.config = config
        self.device = DEVICE
        
        # Loss function
        if LOSS_FUNCTION.lower() == 'mse':
            self.criterion = nn.MSELoss()
        elif LOSS_FUNCTION.lower() == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {LOSS_FUNCTION}")
        
        # Optimizer
        if OPTIMIZER.lower() == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )
        elif OPTIMIZER.lower() == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {OPTIMIZER}")
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch': []
        }
        
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.checkpoint_dir = CHECKPOINTS_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self):
        """
        Train for one epoch
        
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            if len(batch) == 3:
                sequences, rul_values, _ = batch
            else:
                sequences, rul_values = batch
            
            sequences = sequences.to(self.device)
            rul_values = rul_values.to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(sequences)
            loss = self.criterion(predictions, rul_values)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_VALUE)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / num_batches
        return avg_train_loss
    
    def validate(self):
        """
        Validate model on validation set
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for batch in pbar:
                if len(batch) == 3:
                    sequences, rul_values, _ = batch
                else:
                    sequences, rul_values = batch
                
                sequences = sequences.to(self.device)
                rul_values = rul_values.to(self.device).unsqueeze(1)
                
                predictions = self.model(sequences)
                loss = self.criterion(predictions, rul_values)
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = total_loss / num_batches
        return avg_val_loss
    
    def train(self, num_epochs=NUM_EPOCHS, resume_from_checkpoint=False):
        """
        Main training loop
        
        Args:
            num_epochs (int): Number of epochs to train
            resume_from_checkpoint (bool): Whether to resume from saved checkpoint
        """
        start_epoch = 0
        
        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_latest.pt"
            if checkpoint_path.exists():
                print(f"\nResuming from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                self.history = checkpoint['history']
                self.best_val_loss = checkpoint['best_val_loss']
                self.early_stopping_counter = checkpoint['early_stopping_counter']
                print(f"Resumed from epoch {start_epoch}")
            else:
                print(f"No checkpoint found, starting from scratch")
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs - start_epoch} epochs")
        print(f"Model: {self.model_name.upper()}")
        print(f"Loss function: {LOSS_FUNCTION}")
        print(f"Optimizer: {OPTIMIZER}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Early stopping and checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                
                if SAVE_BEST_MODEL:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint periodically
            if (epoch + 1) % SAVE_CHECKPOINT_EVERY_N_EPOCHS == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Save latest checkpoint for resume
            self.save_checkpoint(epoch, is_latest=True)
            
            # Early stopping
            if self.early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {elapsed_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False, is_latest=False):
        """
        Save checkpoint
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model
            is_latest (bool): Whether this is the latest checkpoint
        """
        if is_best:
            checkpoint_name = f"{self.model_name}_best.pt"
        elif is_latest:
            checkpoint_name = f"{self.model_name}_latest.pt"
        else:
            checkpoint_name = f"{self.model_name}_epoch_{epoch + 1}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping_counter,
            'history': self.history,
            'model_name': self.model_name
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            print(f"✓ Best model saved: {checkpoint_path}")
        elif is_latest:
            pass  # Don't print for every latest save
        else:
            print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def save_training_history(self):
        """Save training history to JSON file"""
        history_file = RESULTS_DIR / f"{self.model_name}_training_history.json"
        
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"Training history saved to: {history_file}")


def train_model(model_name, dataset_name="FD001", resume_from_checkpoint=False, num_epochs=NUM_EPOCHS):
    """
    Complete training pipeline
    
    Args:
        model_name (str): 'lstm' or 'transformer'
        dataset_name (str): Dataset variant
        resume_from_checkpoint (bool): Resume training from checkpoint
        num_epochs (int): Number of epochs to train
    """
    
    print(f"\nPreparing data for {dataset_name}...")
    processed_data = preprocess_dataset(dataset_name=dataset_name, reload=False)
    
    print(f"Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        processed_data,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        shuffle=True
    )
    
    # Prepare config dict for model creation
    config_dict = {
        'LSTM_INPUT_SIZE': len(processed_data['train_sequences'][0][0]),
        'LSTM_HIDDEN_SIZE': 128,
        'LSTM_NUM_LAYERS': 2,
        'LSTM_DROPOUT': 0.2,
        'LSTM_OUTPUT_SIZE': 1,
        'TRANSFORMER_INPUT_SIZE': len(processed_data['train_sequences'][0][0]),
        'TRANSFORMER_D_MODEL': 128,
        'TRANSFORMER_NHEAD': 8,
        'TRANSFORMER_NUM_LAYERS': 2,
        'TRANSFORMER_DIM_FEEDFORWARD': 256,
        'TRANSFORMER_DROPOUT': 0.2,
        'TRANSFORMER_OUTPUT_SIZE': 1,
    }
    
    print(f"Creating {model_name.upper()} model...")
    model = create_model(model_name, config_dict, DEVICE)
    
    print(f"Initializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, model_name, config_dict)
    
    print(f"Starting training...")
    history = trainer.train(num_epochs=num_epochs, resume_from_checkpoint=resume_from_checkpoint)
    
    trainer.save_training_history()
    
    return trainer, history


if __name__ == "__main__":
    # Train LSTM model
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    trainer_lstm, history_lstm = train_model(
        model_name='lstm',
        dataset_name='FD001',
        resume_from_checkpoint=False,
        num_epochs=NUM_EPOCHS
    )
    
    # Train Transformer model
    print("\n" + "="*60)
    print("TRAINING TRANSFORMER MODEL")
    print("="*60)
    trainer_transformer, history_transformer = train_model(
        model_name='transformer',
        dataset_name='FD001',
        resume_from_checkpoint=False,
        num_epochs=NUM_EPOCHS
    )
