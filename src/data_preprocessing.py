"""
Data Preprocessing Module
Handles reading, cleaning, and preprocessing the NASA C-MAPSS Turbofan Engine Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, SENSOR_COLUMNS, RUL_CLIP_VALUE,
    SEQUENCE_LENGTH, RANDOM_SEED
)


class DataPreprocessor:
    """
    Handles all data preprocessing operations:
    - Reading raw dataset files
    - Feature engineering
    - Data normalization
    - Creating sliding windows for sequence modeling
    """
    
    def __init__(self, dataset_name="FD001"):
        """
        Initialize the preprocessor
        
        Args:
            dataset_name (str): Dataset variant (FD001, FD002, FD003, FD004)
        """
        self.dataset_name = dataset_name
        self.raw_train_file = DATA_RAW_DIR / f"train_{dataset_name}.txt"
        self.raw_test_file = DATA_RAW_DIR / f"test_{dataset_name}.txt"
        self.raw_rul_file = DATA_RAW_DIR / f"RUL_{dataset_name}.txt"
        
        self.train_data = None
        self.test_data = None
        self.train_rul = None
        self.test_rul = None
        
        self.scaler = StandardScaler()
        
    def read_raw_files(self):
        """
        Read the raw dataset files from NASA C-MAPSS
        Files are space-separated, without headers
        
        Returns:
            Tuple of (train_df, test_df, train_rul, test_rul)
        """
        print(f"Reading raw files for {self.dataset_name}...")
        
        # Check if files exist
        if not self.raw_train_file.exists():
            raise FileNotFoundError(f"Train file not found: {self.raw_train_file}")
        if not self.raw_test_file.exists():
            raise FileNotFoundError(f"Test file not found: {self.raw_test_file}")
        if not self.raw_rul_file.exists():
            raise FileNotFoundError(f"RUL file not found: {self.raw_rul_file}")
        
        # Read files with space separator, no header
        self.train_data = pd.read_csv(
            self.raw_train_file,
            sep=r'\s+',
            header=None,
            engine='python'
        )
        
        self.test_data = pd.read_csv(
            self.raw_test_file,
            sep=r'\s+',
            header=None,
            engine='python'
        )
        
        # Read RUL values
        self.train_rul = pd.read_csv(
            self.raw_rul_file,
            sep=r'\s+',
            header=None,
            engine='python'
        ).values.flatten()
        
        # Read test RUL - last column is true RUL for test set
        self.test_rul = self.test_data.iloc[:, -1].values
        
        # Assign column names
        # Columns: Engine_ID, Cycle, 3 Operational Settings, 21 Sensors
        column_names = ['Engine_ID', 'Cycle'] + SENSOR_COLUMNS
        self.train_data.columns = column_names
        self.test_data.columns = column_names
        
        print(f"Train data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        return self.train_data, self.test_data, self.train_rul, self.test_rul
    
    def calculate_train_rul(self):
        """
        Calculate RUL for training data.
        RUL is calculated as: total_cycles - current_cycle
        This represents remaining cycles until engine failure.
        """
        print("Calculating RUL for training data...")
        
        # Get the maximum cycle for each engine
        max_cycles = self.train_data.groupby('Engine_ID')['Cycle'].max()
        
        # Create new RUL column
        self.train_data['RUL'] = self.train_data.groupby('Engine_ID')['Cycle'].transform(
            lambda x: max_cycles[x.name] - x
        )
        
        # Clip RUL to maximum value (prevents extreme values at beginning)
        self.train_data['RUL'] = self.train_data['RUL'].clip(upper=RUL_CLIP_VALUE)
        
        print(f"RUL range - Min: {self.train_data['RUL'].min()}, Max: {self.train_data['RUL'].max()}")
        
        return self.train_data
    
    def calculate_test_rul(self):
        """
        Calculate RUL for test data.
        For test set, we have the true RUL values, but we calculate RUL from cycles
        for consistency with training procedure.
        """
        print("Calculating RUL for test data...")
        
        # Get the last cycle for each engine in test set
        max_cycles = self.test_data.groupby('Engine_ID')['Cycle'].max()
        
        # RUL is: max_cycle - current_cycle + true_remaining_rul_for_that_engine
        # We need to look up the true RUL from test_rul file
        self.test_data['RUL'] = self.test_data.groupby('Engine_ID')['Cycle'].transform(
            lambda x: max_cycles[x.name] - x
        ) + self.test_rul[self.test_data['Engine_ID'].values - 1]
        
        # Clip RUL
        self.test_data['RUL'] = self.test_data['RUL'].clip(upper=RUL_CLIP_VALUE)
        
        print(f"Test RUL range - Min: {self.test_data['RUL'].min()}, Max: {self.test_data['RUL'].max()}")
        
        return self.test_data
    
    def normalize_features(self, fit_on_train=True):
        """
        Normalize sensor features using StandardScaler.
        
        Args:
            fit_on_train (bool): If True, fit scaler on training data and apply to both
        """
        print("Normalizing sensor features...")
        
        if fit_on_train:
            # Fit scaler on training data only
            self.train_data[SENSOR_COLUMNS] = self.scaler.fit_transform(
                self.train_data[SENSOR_COLUMNS]
            )
            
            # Apply the same scaler to test data
            self.test_data[SENSOR_COLUMNS] = self.scaler.transform(
                self.test_data[SENSOR_COLUMNS]
            )
        else:
            # Fit on both datasets combined (not recommended, but available)
            combined_data = pd.concat([
                self.train_data[SENSOR_COLUMNS],
                self.test_data[SENSOR_COLUMNS]
            ])
            self.scaler.fit(combined_data)
            
            self.train_data[SENSOR_COLUMNS] = self.scaler.transform(
                self.train_data[SENSOR_COLUMNS]
            )
            self.test_data[SENSOR_COLUMNS] = self.scaler.transform(
                self.test_data[SENSOR_COLUMNS]
            )
        
        print("Normalization complete")
        return self.train_data, self.test_data
    
    def create_sequences(self, data, sequence_length=SEQUENCE_LENGTH):
        """
        Create sliding windows (sequences) for time-series modeling.
        
        Args:
            data (pd.DataFrame): Input data with Engine_ID, Cycle, sensor columns, and RUL
            sequence_length (int): Length of each sequence/window
            
        Returns:
            Tuple of (sequences, rul_values, engine_ids)
            - sequences: (n_samples, sequence_length, n_features)
            - rul_values: (n_samples,) - RUL value at end of each sequence
            - engine_ids: (n_samples,) - Engine ID for tracking
        """
        sequences = []
        rul_values = []
        engine_ids = []
        
        # Group by engine
        for engine_id, engine_data in data.groupby('Engine_ID'):
            # Get sensor values (excluding Engine_ID, Cycle, RUL columns)
            engine_sensors = engine_data[SENSOR_COLUMNS].values
            engine_rul = engine_data['RUL'].values
            
            # Create sliding windows
            for i in range(len(engine_sensors) - sequence_length + 1):
                sequences.append(engine_sensors[i:i + sequence_length])
                rul_values.append(engine_rul[i + sequence_length - 1])
                engine_ids.append(engine_id)
        
        sequences = np.array(sequences, dtype=np.float32)
        rul_values = np.array(rul_values, dtype=np.float32)
        engine_ids = np.array(engine_ids, dtype=np.int32)
        
        print(f"Created {len(sequences)} sequences from {len(data['Engine_ID'].unique())} engines")
        print(f"Sequence shape: {sequences.shape}")
        
        return sequences, rul_values, engine_ids
    
    def preprocess_all(self):
        """
        Run complete preprocessing pipeline:
        1. Read raw files
        2. Calculate RUL
        3. Normalize features
        4. Create sequences
        """
        print(f"\n{'='*60}")
        print(f"Preprocessing {self.dataset_name}")
        print(f"{'='*60}\n")
        
        # Step 1: Read raw files
        self.read_raw_files()
        
        # Step 2: Calculate RUL
        self.calculate_train_rul()
        self.calculate_test_rul()
        
        # Step 3: Normalize features
        self.normalize_features(fit_on_train=True)
        
        # Step 4: Create sequences
        train_sequences, train_rul, train_engine_ids = self.create_sequences(
            self.train_data,
            sequence_length=SEQUENCE_LENGTH
        )
        
        test_sequences, test_rul, test_engine_ids = self.create_sequences(
            self.test_data,
            sequence_length=SEQUENCE_LENGTH
        )
        
        # Create output dictionary
        processed_data = {
            'train_sequences': train_sequences,
            'train_rul': train_rul,
            'train_engine_ids': train_engine_ids,
            'test_sequences': test_sequences,
            'test_rul': test_rul,
            'test_engine_ids': test_engine_ids,
            'scaler': self.scaler,
            'dataset_name': self.dataset_name
        }
        
        return processed_data
    
    def save_processed_data(self, processed_data):
        """
        Save processed data to disk for later use
        """
        output_file = DATA_PROCESSED_DIR / f"{self.dataset_name}_processed.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"\nProcessed data saved to: {output_file}")
        return output_file
    
    @staticmethod
    def load_processed_data(dataset_name="FD001"):
        """
        Load previously processed data from disk
        
        Args:
            dataset_name (str): Dataset variant
            
        Returns:
            dict: Processed data dictionary
        """
        input_file = DATA_PROCESSED_DIR / f"{dataset_name}_processed.pkl"
        
        if not input_file.exists():
            raise FileNotFoundError(f"Processed data file not found: {input_file}")
        
        with open(input_file, 'rb') as f:
            processed_data = pickle.load(f)
        
        print(f"Loaded processed data from: {input_file}")
        return processed_data


def preprocess_dataset(dataset_name="FD001", reload=False):
    """
    Preprocess dataset, using cached version if available
    
    Args:
        dataset_name (str): Dataset variant
        reload (bool): If True, force reprocessing even if cached version exists
        
    Returns:
        dict: Processed data dictionary
    """
    output_file = DATA_PROCESSED_DIR / f"{dataset_name}_processed.pkl"
    
    # Use cached version if available and reload not requested
    if output_file.exists() and not reload:
        print(f"Using cached processed data for {dataset_name}")
        return DataPreprocessor.load_processed_data(dataset_name)
    
    # Process fresh
    preprocessor = DataPreprocessor(dataset_name=dataset_name)
    processed_data = preprocessor.preprocess_all()
    preprocessor.save_processed_data(processed_data)
    
    return processed_data


if __name__ == "__main__":
    # Example usage
    data = preprocess_dataset(dataset_name="FD001", reload=True)
    
    print(f"\nProcessed Data Summary:")
    print(f"Train sequences shape: {data['train_sequences'].shape}")
    print(f"Train RUL range: [{data['train_rul'].min():.2f}, {data['train_rul'].max():.2f}]")
    print(f"Test sequences shape: {data['test_sequences'].shape}")
    print(f"Test RUL range: [{data['test_rul'].min():.2f}, {data['test_rul'].max():.2f}]")
