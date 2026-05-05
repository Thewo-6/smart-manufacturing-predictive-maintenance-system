"""
LSTM and Transformer models for Turbofan Engine RUL Prediction
"""

import torch
import torch.nn as nn
import math


class LSTMModel(nn.Module):
    """
    LSTM-based model for Remaining Useful Life prediction
    
    Architecture:
    - Input: (batch_size, sequence_length, num_features)
    - LSTM layers: Learns temporal dependencies
    - Fully connected layers: Produces RUL prediction
    - Output: (batch_size, 1) - predicted RUL
    """
    
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        """
        Initialize LSTM model
        
        Args:
            input_size (int): Number of input features (sensors)
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            output_size (int): Output size (1 for RUL prediction)
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        # Flatten output from LSTM (hidden_size) -> intermediate -> output_size
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_size)
        """
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last timestep output
        # lstm_last_out shape: (batch_size, hidden_size)
        lstm_last_out = lstm_out[:, -1, :]
        
        # Apply dropout
        lstm_last_out = self.dropout(lstm_last_out)
        
        # Fully connected layers
        out = self.relu(self.fc1(lstm_last_out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TransformerEncoderModel(nn.Module):
    """
    Transformer Encoder-based model for Remaining Useful Life prediction
    
    Architecture:
    - Input: (batch_size, sequence_length, num_features)
    - Positional encoding: Adds position information
    - Transformer encoder: Multi-head self-attention + feed-forward
    - Output projection: Produces RUL prediction
    - Output: (batch_size, 1) - predicted RUL
    """
    
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, dropout, output_size):
        """
        Initialize Transformer Encoder model
        
        Args:
            input_size (int): Number of input features (sensors)
            d_model (int): Dimension of model (embedding size)
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimension of feed-forward network
            dropout (float): Dropout probability
            output_size (int): Output size (1 for RUL prediction)
        """
        super(TransformerEncoderModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        
        # Input projection: project sensor features to d_model dimension
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding: add position information
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_size)
        """
        # x shape: (batch_size, sequence_length, input_size)
        
        # Project input to d_model dimension
        # x_proj shape: (batch_size, sequence_length, d_model)
        x_proj = self.input_projection(x)
        
        # Add positional encoding
        x_pos = self.positional_encoding(x_proj)
        
        # Transformer encoder
        # transformer_out shape: (batch_size, sequence_length, d_model)
        transformer_out = self.transformer_encoder(x_pos)
        
        # Use the last timestep output (or could use mean/max pooling)
        # last_out shape: (batch_size, d_model)
        last_out = transformer_out[:, -1, :]
        
        # Output projection
        out = self.relu(self.fc1(last_out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    Adds information about the position of each token in the sequence
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding
        
        Args:
            d_model (int): Dimension of the model
            max_len (int): Maximum length of sequences
        """
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Create position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create dimension indices
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Add positional encoding to input
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            torch.Tensor: Input + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


def create_model(model_name, config_dict, device):
    """
    Factory function to create model
    
    Args:
        model_name (str): 'lstm' or 'transformer'
        config_dict (dict): Configuration dictionary from config.py
        device (torch.device): Device to place model on
        
    Returns:
        nn.Module: Model instance
    """
    
    if model_name.lower() == 'lstm':
        model = LSTMModel(
            input_size=config_dict['LSTM_INPUT_SIZE'],
            hidden_size=config_dict['LSTM_HIDDEN_SIZE'],
            num_layers=config_dict['LSTM_NUM_LAYERS'],
            dropout=config_dict['LSTM_DROPOUT'],
            output_size=config_dict['LSTM_OUTPUT_SIZE']
        )
        print(f"Created LSTM model")
        
    elif model_name.lower() == 'transformer':
        model = TransformerEncoderModel(
            input_size=config_dict['TRANSFORMER_INPUT_SIZE'],
            d_model=config_dict['TRANSFORMER_D_MODEL'],
            nhead=config_dict['TRANSFORMER_NHEAD'],
            num_layers=config_dict['TRANSFORMER_NUM_LAYERS'],
            dim_feedforward=config_dict['TRANSFORMER_DIM_FEEDFORWARD'],
            dropout=config_dict['TRANSFORMER_DROPOUT'],
            output_size=config_dict['TRANSFORMER_OUTPUT_SIZE']
        )
        print(f"Created Transformer model")
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def get_model_summary(model):
    """
    Print model architecture summary
    
    Args:
        model (nn.Module): PyTorch model
    """
    print(f"\nModel Architecture:")
    print(f"{'='*60}")
    print(model)
    print(f"{'='*60}")


if __name__ == "__main__":
    import config
    
    # Create sample input
    batch_size = 16
    sequence_length = 30
    num_sensors = len(config.SENSOR_COLUMNS)
    
    x = torch.randn(batch_size, sequence_length, num_sensors)
    
    # Test LSTM model
    print("\nTesting LSTM Model:")
    lstm_model = LSTMModel(
        input_size=config.LSTM_INPUT_SIZE,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        dropout=config.LSTM_DROPOUT,
        output_size=config.LSTM_OUTPUT_SIZE
    ).to(config.DEVICE)
    
    lstm_output = lstm_model(x.to(config.DEVICE))
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {lstm_output.shape}")
    get_model_summary(lstm_model)
    
    # Test Transformer model
    print("\nTesting Transformer Model:")
    transformer_model = TransformerEncoderModel(
        input_size=config.TRANSFORMER_INPUT_SIZE,
        d_model=config.TRANSFORMER_D_MODEL,
        nhead=config.TRANSFORMER_NHEAD,
        num_layers=config.TRANSFORMER_NUM_LAYERS,
        dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
        dropout=config.TRANSFORMER_DROPOUT,
        output_size=config.TRANSFORMER_OUTPUT_SIZE
    ).to(config.DEVICE)
    
    transformer_output = transformer_model(x.to(config.DEVICE))
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {transformer_output.shape}")
    get_model_summary(transformer_model)
