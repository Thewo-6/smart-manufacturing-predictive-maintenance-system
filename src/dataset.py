"""
PyTorch Dataset Class for Turbofan Engine RUL Prediction
Handles loading sequences and creating batches for training/testing
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from config import RANDOM_SEED, BATCH_SIZE


class TurbofanDataset(Dataset):
    """
    Custom PyTorch Dataset for Turbofan Engine sequences
    
    Attributes:
        sequences: Array of shape (n_samples, sequence_length, n_features)
        rul_values: Array of shape (n_samples,)
        engine_ids: Array of shape (n_samples,)
    """
    
    def __init__(self, sequences, rul_values, engine_ids=None):
        """
        Initialize dataset
        
        Args:
            sequences (np.array): Time-series sequences, shape (n_samples, seq_len, n_features)
            rul_values (np.array): RUL targets, shape (n_samples,)
            engine_ids (np.array, optional): Engine IDs for tracking
        """
        self.sequences = torch.from_numpy(sequences).float()
        self.rul_values = torch.from_numpy(rul_values).float()
        
        if engine_ids is not None:
            self.engine_ids = torch.from_numpy(engine_ids).long()
        else:
            self.engine_ids = None
    
    def __len__(self):
        """Return number of samples"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx (int): Index of sample
            
        Returns:
            Tuple of (sequence, rul_value) or (sequence, rul_value, engine_id)
        """
        sequence = self.sequences[idx]
        rul_value = self.rul_values[idx]
        
        if self.engine_ids is not None:
            engine_id = self.engine_ids[idx]
            return sequence, rul_value, engine_id
        
        return sequence, rul_value
    
    def get_data_info(self):
        """Print dataset information"""
        print(f"Dataset size: {len(self)}")
        print(f"Sequence shape: {self.sequences[0].shape}")
        print(f"RUL range: [{self.rul_values.min():.2f}, {self.rul_values.max():.2f}]")
        print(f"RUL mean: {self.rul_values.mean():.2f}, std: {self.rul_values.std():.2f}")


def create_data_loaders(processed_data, batch_size=BATCH_SIZE, val_split=0.2, shuffle=True):
    """
    Create PyTorch DataLoaders from processed data
    Splits training data into train and validation sets
    
    Args:
        processed_data (dict): Output from DataPreprocessor.preprocess_all()
        batch_size (int): Batch size for DataLoader
        val_split (float): Fraction of training data to use for validation
        shuffle (bool): Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = TurbofanDataset(
        sequences=processed_data['train_sequences'],
        rul_values=processed_data['train_rul'],
        engine_ids=processed_data['train_engine_ids']
    )
    
    test_dataset = TurbofanDataset(
        sequences=processed_data['test_sequences'],
        rul_values=processed_data['test_rul'],
        engine_ids=processed_data['test_engine_ids']
    )
    
    # Split training data into train and validation
    num_train = int(len(train_dataset) * (1 - val_split))
    num_val = len(train_dataset) - num_train
    
    train_dataset_split, val_dataset = random_split(
        train_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset_split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for macOS compatibility
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\nDataLoader Configuration:")
    print(f"Train set: {len(train_dataset_split)} samples, {len(train_loader)} batches")
    print(f"Validation set: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test set: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader


def create_test_loader(processed_data, batch_size=BATCH_SIZE):
    """
    Create test DataLoader for inference
    
    Args:
        processed_data (dict): Output from DataPreprocessor.preprocess_all()
        batch_size (int): Batch size for DataLoader
        
    Returns:
        DataLoader for test set
    """
    
    test_dataset = TurbofanDataset(
        sequences=processed_data['test_sequences'],
        rul_values=processed_data['test_rul'],
        engine_ids=processed_data['test_engine_ids']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return test_loader


def create_inference_dataset(sequences):
    """
    Create a dataset for inference (without RUL labels)
    
    Args:
        sequences (np.array): Sequences for inference, shape (n_samples, seq_len, n_features)
        
    Returns:
        TurbofanDataset with dummy RUL values
    """
    
    # Create dummy RUL values (will be ignored during inference)
    dummy_rul = np.zeros(len(sequences), dtype=np.float32)
    
    dataset = TurbofanDataset(sequences=sequences, rul_values=dummy_rul)
    
    return dataset


if __name__ == "__main__":
    # Example: Create dataloaders from processed data
    from data_preprocessing import preprocess_dataset
    
    # Load processed data
    processed_data = preprocess_dataset(dataset_name="FD001")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(
        processed_data,
        batch_size=32,
        val_split=0.2
    )
    
    # Display sample batch
    print("\nSample batch from training loader:")
    for sequences, rul_values, engine_ids in train_loader:
        print(f"Sequences shape: {sequences.shape}")
        print(f"RUL values shape: {rul_values.shape}")
        print(f"Engine IDs shape: {engine_ids.shape}")
        print(f"RUL values: {rul_values[:5]}")
        break
