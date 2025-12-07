import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

def setup_logger(name: str = 'datasets') -> logging.Logger:
    """Setup logger for datasets."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class LSTMPriceDataset(Dataset):
    """
    PyTorch Dataset for LSTM price prediction.
    
    This dataset handles sequences of price and technical indicator data
    for training LSTM models to predict next-period closing prices.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device: str = 'cpu'
    ):
        """
        Initialize the dataset.
        
        Args:
            X: Input sequences of shape (n_samples, seq_len, n_features)
            y: Target values of shape (n_samples,)
            device: Device to place tensors on ('cpu', 'cuda', 'mps')
        """
        self.logger = setup_logger()
        
        # Validate input shapes
        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D (n_samples, seq_len, n_features), got shape {X.shape}")
        
        if len(y.shape) != 1:
            raise ValueError(f"y must be 1D (n_samples,), got shape {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
        
        self.device = device
        
        # Convert to PyTorch tensors and move to device
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)
        
        self.n_samples, self.seq_len, self.n_features = self.X.shape
        
        self.logger.info(f"Dataset initialized: {self.n_samples} samples, "
                        f"seq_len={self.seq_len}, features={self.n_features}, device={device}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (sequence, target) tensors
        """
        return self.X[idx], self.y[idx]
    
    def get_feature_stats(self) -> Dict[str, float]:
        """Get statistics about the features in the dataset."""
        return {
            'mean_target': float(self.y.mean()),
            'std_target': float(self.y.std()),
            'min_target': float(self.y.min()),
            'max_target': float(self.y.max()),
            'mean_features': float(self.X.mean()),
            'std_features': float(self.X.std())
        }

def create_data_loaders(
    train_data: Dict[str, np.ndarray],
    val_data: Optional[Dict[str, np.ndarray]] = None,
    test_data: Optional[Dict[str, np.ndarray]] = None,
    batch_size: int = 256,
    device: str = 'cpu',
    num_workers: int = 0,
    pin_memory: bool = False
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and test sets.
    
    Args:
        train_data: Dictionary with 'X' and 'y' keys for training data
        val_data: Optional dictionary with 'X' and 'y' keys for validation data
        test_data: Optional dictionary with 'X' and 'y' keys for test data
        batch_size: Batch size for DataLoaders
        device: Device to place tensors on
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Dictionary containing DataLoaders for each split
    """
    logger = setup_logger()
    
    loaders = {}
    
    # Create training DataLoader
    train_dataset = LSTMPriceDataset(
        X=train_data['X'],
        y=train_data['y'],
        device=device
    )
    
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    logger.info(f"Training DataLoader: {len(train_dataset)} samples, "
               f"{len(loaders['train'])} batches of size {batch_size}")
    
    # Create validation DataLoader if data provided
    if val_data is not None:
        val_dataset = LSTMPriceDataset(
            X=val_data['X'],
            y=val_data['y'],
            device=device
        )
        
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        logger.info(f"Validation DataLoader: {len(val_dataset)} samples, "
                   f"{len(loaders['val'])} batches")
    
    # Create test DataLoader if data provided
    if test_data is not None:
        test_dataset = LSTMPriceDataset(
            X=test_data['X'],
            y=test_data['y'],
            device=device
        )
        
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for test
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        logger.info(f"Test DataLoader: {len(test_dataset)} samples, "
                   f"{len(loaders['test'])} batches")
    
    return loaders

def collate_fn(batch):
    """
    Custom collate function for handling variable-length sequences if needed.
    Currently just uses default behavior but can be extended.
    """
    return torch.utils.data.default_collate(batch)

class SequenceDataModule:
    """
    High-level data module for managing all data loading operations.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'cpu'
    ):
        """
        Initialize the data module.
        
        Args:
            config: Configuration dictionary containing data parameters
            device: Device to place tensors on
        """
        self.config = config
        self.device = device
        self.logger = setup_logger()
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def setup_loaders(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]] = None,
        test_data: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Setup all DataLoaders with the provided data.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary (optional)
            test_data: Test data dictionary (optional)
        """
        batch_size = self.config.get('batch_size', 256)
        num_workers = self.config.get('num_workers', 0)
        pin_memory = self.config.get('pin_memory', False)
        
        # Adjust num_workers and pin_memory based on device
        if self.device == 'mps':
            num_workers = 0  # MPS doesn't support multiprocessing
            pin_memory = False
        elif self.device == 'cuda':
            pin_memory = True
        
        loaders = create_data_loaders(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            batch_size=batch_size,
            device=self.device,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        self.train_loader = loaders['train']
        self.val_loader = loaders.get('val')
        self.test_loader = loaders.get('test')
        
        self.logger.info("Data loaders setup completed")
    
    def get_train_loader(self) -> DataLoader:
        """Get the training DataLoader."""
        if self.train_loader is None:
            raise ValueError("Training loader not initialized. Call setup_loaders() first.")
        return self.train_loader
    
    def get_val_loader(self) -> Optional[DataLoader]:
        """Get the validation DataLoader."""
        return self.val_loader
    
    def get_test_loader(self) -> Optional[DataLoader]:
        """Get the test DataLoader."""
        return self.test_loader
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        stats = {}
        
        if self.train_loader:
            # Get a sample batch to extract stats
            sample_batch = next(iter(self.train_loader))
            X_sample, y_sample = sample_batch
            
            stats['train'] = {
                'num_batches': len(self.train_loader),
                'batch_size': X_sample.shape[0],
                'seq_len': X_sample.shape[1],
                'n_features': X_sample.shape[2],
                'sample_target_mean': float(y_sample.mean()),
                'sample_target_std': float(y_sample.std())
            }
        
        if self.val_loader:
            stats['val'] = {
                'num_batches': len(self.val_loader)
            }
        
        if self.test_loader:
            stats['test'] = {
                'num_batches': len(self.test_loader)
            }
        
        return stats

if __name__ == "__main__":
    # Test the dataset and DataLoader functionality
    logger = setup_logger()
    
    try:
        # Create dummy data for testing
        n_samples, seq_len, n_features = 1000, 60, 10
        X_dummy = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        y_dummy = np.random.randn(n_samples).astype(np.float32)
        
        # Test dataset creation
        dataset = LSTMPriceDataset(X_dummy, y_dummy, device='cpu')
        logger.info(f"Dataset test passed: {len(dataset)} samples")
        
        # Test DataLoader creation
        train_data = {'X': X_dummy[:800], 'y': y_dummy[:800]}
        val_data = {'X': X_dummy[800:900], 'y': y_dummy[800:900]}
        test_data = {'X': X_dummy[900:], 'y': y_dummy[900:]}
        
        loaders = create_data_loaders(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            batch_size=32,
            device='cpu'
        )
        
        logger.info("DataLoader test passed")
        
        # Test data module
        config = {'batch_size': 32}
        data_module = SequenceDataModule(config, device='cpu')
        data_module.setup_loaders(train_data, val_data, test_data)
        
        stats = data_module.get_data_stats()
        logger.info(f"Data module test passed: {stats}")
        
    except Exception as e:
        logger.error(f"Dataset test failed: {str(e)}")
        raise