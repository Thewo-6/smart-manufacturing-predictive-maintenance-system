"""
Prediction module for making RUL predictions on new engine data
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from config import DEVICE, SEQUENCE_LENGTH, SENSOR_COLUMNS


class RULPredictor:
    """
    Predictor class for making RUL predictions
    """
    
    def __init__(self, model, scaler, sequence_length=SEQUENCE_LENGTH, device=DEVICE):
        """
        Initialize predictor
        
        Args:
            model (nn.Module): Trained model
            scaler (StandardScaler): Fitted scaler for normalization
            sequence_length (int): Length of input sequences
            device (torch.device): Device to use
        """
        self.model = model
        self.scaler = scaler
        self.sequence_length = sequence_length
        self.device = device
        self.model.eval()
    
    def preprocess_engine_data(self, engine_data):
        """
        Preprocess raw engine sensor data
        
        Args:
            engine_data (pd.DataFrame): Dataframe with columns: Cycle, Sensor_1, ..., Sensor_21
                                       Should have Operational_Setting columns if available
            
        Returns:
            np.array: Normalized sensor data
        """
        # Select only sensor columns
        sensor_data = engine_data[SENSOR_COLUMNS].values
        
        # Normalize using the fitted scaler
        normalized_data = self.scaler.transform(sensor_data)
        
        return normalized_data
    
    def predict_single_sample(self, sequence):
        """
        Predict RUL for a single sequence
        
        Args:
            sequence (np.array): Sequence of shape (sequence_length, num_features)
            
        Returns:
            float: Predicted RUL value
        """
        # Convert to tensor
        sequence_tensor = torch.from_numpy(sequence).float().unsqueeze(0)  # Add batch dimension
        sequence_tensor = sequence_tensor.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
            rul_pred = prediction.cpu().numpy().flatten()[0]
        
        # Ensure positive RUL
        rul_pred = max(0, rul_pred)
        
        return rul_pred
    
    def predict_from_sequences(self, sequences):
        """
        Predict RUL for multiple sequences
        
        Args:
            sequences (np.array): Array of shape (n_samples, sequence_length, num_features)
            
        Returns:
            np.array: Predicted RUL values
        """
        sequences_tensor = torch.from_numpy(sequences).float()
        sequences_tensor = sequences_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(sequences_tensor)
            rul_preds = predictions.cpu().numpy().flatten()
        
        # Ensure positive RUL
        rul_preds = np.maximum(0, rul_preds)
        
        return rul_preds
    
    def predict_engine_trajectory(self, engine_data):
        """
        Predict RUL trajectory for an engine's operational history
        
        Args:
            engine_data (pd.DataFrame): Dataframe with sensor readings over time
                                       Columns should include SENSOR_COLUMNS
            
        Returns:
            dict: Dictionary with prediction results
        """
        # Preprocess data
        normalized_data = self.preprocess_engine_data(engine_data)
        
        cycles = engine_data['Cycle'].values if 'Cycle' in engine_data.columns else np.arange(len(engine_data))
        
        predictions = []
        valid_cycles = []
        
        # Create sliding windows
        for i in range(len(normalized_data) - self.sequence_length + 1):
            sequence = normalized_data[i:i + self.sequence_length]
            rul_pred = self.predict_single_sample(sequence)
            predictions.append(rul_pred)
            valid_cycles.append(cycles[i + self.sequence_length - 1])
        
        # Get the last prediction as the current RUL estimate
        current_rul = predictions[-1] if predictions else None
        
        result = {
            'cycles': np.array(valid_cycles),
            'rul_predictions': np.array(predictions),
            'current_rul': current_rul,
            'num_observations': len(engine_data),
            'trajectory': list(predictions)
        }
        
        return result
    
    def get_health_status(self, rul_prediction, warning_threshold=50, critical_threshold=20):
        """
        Determine engine health status based on predicted RUL
        
        Args:
            rul_prediction (float): Predicted RUL value
            warning_threshold (int): RUL threshold for warning status
            critical_threshold (int): RUL threshold for critical status
            
        Returns:
            dict: Health status information
        """
        if rul_prediction < critical_threshold:
            status = "CRITICAL"
            color = "red"
            recommendation = "Immediate maintenance required. Engine may fail soon."
        elif rul_prediction < warning_threshold:
            status = "WARNING"
            color = "orange"
            recommendation = "Schedule maintenance within next operational cycle."
        else:
            status = "HEALTHY"
            color = "green"
            recommendation = "Engine operating normally. Continue monitoring."
        
        result = {
            'status': status,
            'color': color,
            'recommendation': recommendation,
            'rul': rul_prediction,
            'warning_threshold': warning_threshold,
            'critical_threshold': critical_threshold
        }
        
        return result
    
    def batch_predict(self, data_list):
        """
        Make predictions for a batch of engines
        
        Args:
            data_list (list): List of DataFrames, one per engine
            
        Returns:
            list: List of prediction results
        """
        results = []
        for engine_id, engine_data in enumerate(data_list, 1):
            trajectory = self.predict_engine_trajectory(engine_data)
            health = self.get_health_status(trajectory['current_rul'])
            
            result = {
                'engine_id': engine_id,
                'trajectory': trajectory,
                'health_status': health
            }
            results.append(result)
        
        return results


def load_model_and_scaler(model_path, model_class, device=DEVICE):
    """
    Load trained model and scaler
    
    Args:
        model_path (Path): Path to checkpoint file
        model_class: Model class to instantiate
        device (torch.device): Device to use
        
    Returns:
        Tuple of (model, scaler)
    """
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Try to extract scaler from checkpoint, otherwise return None
    scaler = checkpoint.get('scaler', None)
    
    print(f"Loaded model from {model_path}")
    
    return model, scaler


def create_sample_engine_data(num_cycles=100, num_sensors=21):
    """
    Create sample engine sensor data for demonstration
    
    Args:
        num_cycles (int): Number of operational cycles
        num_sensors (int): Number of sensors
        
    Returns:
        pd.DataFrame: Sample engine data
    """
    np.random.seed(42)
    
    data = pd.DataFrame()
    data['Cycle'] = np.arange(1, num_cycles + 1)
    
    # Add operational settings
    data['Operational_Setting_1'] = np.random.randn(num_cycles) * 10 + 42
    data['Operational_Setting_2'] = np.random.randn(num_cycles) * 2 + 0.84
    data['Operational_Setting_3'] = np.random.randn(num_cycles) * 100 + 100
    
    # Add sensor readings with degradation pattern
    for i in range(num_sensors):
        # Simulated degradation: linear trend + noise
        degradation = np.linspace(0, 5, num_cycles)
        noise = np.random.randn(num_cycles) * 0.5
        sensor_name = f'Sensor_{i + 2}'
        data[sensor_name] = 100 + degradation + noise
    
    return data


if __name__ == "__main__":
    # Example usage
    print("Example: RUL Prediction")
    print("="*60)
    
    # Create sample data
    print("\nCreating sample engine data...")
    sample_data = create_sample_engine_data(num_cycles=100)
    print(f"Sample data shape: {sample_data.shape}")
    print(sample_data.head())
    
    # Load scaler from checkpoint
    from src.data_preprocessing import preprocess_dataset
    processed_data = preprocess_dataset("FD001")
    scaler = processed_data['scaler']
    
    # Create dummy model for demonstration
    from src.models import LSTMModel
    from config import LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT, LSTM_OUTPUT_SIZE
    
    model = LSTMModel(
        input_size=LSTM_INPUT_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        output_size=LSTM_OUTPUT_SIZE
    ).to(DEVICE)
    
    # Create predictor
    predictor = RULPredictor(model, scaler)
    
    # Make predictions
    print("\nMaking predictions...")
    trajectory = predictor.predict_engine_trajectory(sample_data)
    print(f"Current RUL estimate: {trajectory['current_rul']:.2f} cycles")
    
    # Get health status
    health = predictor.get_health_status(trajectory['current_rul'])
    print(f"Health status: {health['status']}")
    print(f"Recommendation: {health['recommendation']}")
