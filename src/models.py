import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging
import math

def setup_logger(name: str = 'models') -> logging.Logger:
    """Setup logger for models."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class LSTMPricePredictor(nn.Module):
    """
    LSTM-based price predictor for financial time series.
    
    Architecture:
    - 2-layer LSTM with dropout
    - Fully connected output layer
    - Designed for next-period price prediction
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize the LSTM price predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_size: Output size (1 for price prediction)
        """
        super(LSTMPricePredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        self.logger = setup_logger()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Only apply dropout if multiple layers
            batch_first=True,
            bias=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._init_weights()
        
        self.logger.info(f"Initialized LSTM model: input_size={input_size}, "
                        f"hidden_size={hidden_size}, num_layers={num_layers}, "
                        f"dropout={dropout}")
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases
                nn.init.constant_(param.data, 0)
                # Set forget gate bias to 1 (helps with gradient flow)
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional hidden state tuple (h_0, c_0)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Apply fully connected layer
        output = self.fc(last_output)  # Shape: (batch_size, output_size)
        
        return output
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state for LSTM.
        
        Args:
            batch_size: Batch size
            device: Device to place tensors on
            
        Returns:
            Tuple of (h_0, c_0) hidden state tensors
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        return h_0, c_0
    
    def get_model_size(self) -> Dict[str, int]:
        """Get information about model size and parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'lstm_parameters': sum(p.numel() for name, p in self.named_parameters() if 'lstm' in name),
            'fc_parameters': sum(p.numel() for name, p in self.named_parameters() if 'fc' in name)
        }

class EnhancedLSTMPredictor(nn.Module):
    """
    Enhanced LSTM predictor with additional features:
    - Residual connections
    - Layer normalization
    - Attention mechanism (optional)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        use_layer_norm: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize the enhanced LSTM predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_size: Output size
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
        """
        super(EnhancedLSTMPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        self.logger = setup_logger()
        
        # Input projection (if needed)
        self.input_projection = None
        if input_size != hidden_size:
            self.input_projection = nn.Linear(input_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size if self.input_projection else input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bias=True
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights
        self._init_weights()
        
        self.logger.info(f"Initialized Enhanced LSTM model with layer_norm={use_layer_norm}, "
                        f"residual={use_residual}")
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the enhanced LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Input projection if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        # Store input for residual connection
        residual_input = x[:, -1, :] if self.use_residual else None
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Layer normalization
        if self.use_layer_norm:
            last_output = self.layer_norm(last_output)
        
        # Residual connection
        if self.use_residual and residual_input is not None:
            last_output = last_output + residual_input
        
        # Dropout
        last_output = self.dropout_layer(last_output)
        
        # Output layers with activation
        out = F.relu(self.fc1(last_output))
        out = self.dropout_layer(out)
        output = self.fc2(out)
        
        return output

def create_model(config: Dict[str, Any], input_size: int) -> nn.Module:
    """
    Factory function to create LSTM model based on configuration.
    
    Args:
        config: Configuration dictionary containing model parameters
        input_size: Number of input features
        
    Returns:
        Initialized PyTorch model
    """
    logger = setup_logger()
    
    model_type = config.get('model_type', 'basic')
    hidden_size = config.get('hidden_size', 128)
    num_layers = config.get('num_layers', 2)
    dropout = config.get('dropout', 0.2)
    
    if model_type == 'enhanced':
        model = EnhancedLSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=config.get('use_layer_norm', True),
            use_residual=config.get('use_residual', True)
        )
    else:
        model = LSTMPricePredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    
    # Get model size info
    model_info = model.get_model_size() if hasattr(model, 'get_model_size') else {}
    logger.info(f"Created {model_type} model with {model_info.get('total_parameters', 'unknown')} parameters")
    
    return model

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count the number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def get_model_memory_usage(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    """
    Estimate memory usage of a model.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, seq_len, features)
        
    Returns:
        Dictionary with memory usage estimates in MB
    """
    # Parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Gradient memory (same as parameters)
    grad_memory = param_memory
    
    # Activation memory (rough estimate)
    batch_size, seq_len, features = input_shape
    activation_memory = batch_size * seq_len * features * 4  # Assuming float32
    
    total_memory = param_memory + grad_memory + activation_memory
    
    return {
        'parameters_mb': param_memory / (1024 * 1024),
        'gradients_mb': grad_memory / (1024 * 1024),
        'activations_mb': activation_memory / (1024 * 1024),
        'total_mb': total_memory / (1024 * 1024)
    }

if __name__ == "__main__":
    # Test the model implementations
    logger = setup_logger()
    
    try:
        # Test basic LSTM model
        input_size = 12
        batch_size = 32
        seq_len = 60
        
        model = LSTMPricePredictor(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        
        # Test forward pass
        x_dummy = torch.randn(batch_size, seq_len, input_size)
        output = model(x_dummy)
        
        logger.info(f"Basic LSTM test passed: input {x_dummy.shape} -> output {output.shape}")
        
        # Test enhanced model
        enhanced_model = EnhancedLSTMPredictor(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        
        output_enhanced = enhanced_model(x_dummy)
        logger.info(f"Enhanced LSTM test passed: input {x_dummy.shape} -> output {output_enhanced.shape}")
        
        # Test model factory
        config = {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'model_type': 'basic'
        }
        
        factory_model = create_model(config, input_size)
        logger.info("Model factory test passed")
        
        # Test utility functions
        param_count = count_parameters(model)
        memory_usage = get_model_memory_usage(model, (batch_size, seq_len, input_size))
        
        logger.info(f"Parameter count: {param_count}")
        logger.info(f"Memory usage: {memory_usage}")
        
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
        raise