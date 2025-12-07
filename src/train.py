import os
import sys
import yaml
import json
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import our custom modules
from real_data_verification import verify_real_data_or_die
from data_ingestion import ingest_all_data
from feature_engineering import build_sequences, calculate_technical_indicators, add_cross_asset_features
from lstm_datasets import SequenceDataModule, create_data_loaders
from models import create_model, count_parameters
from artifact_manager import ArtifactManager

def setup_logger(name: str = 'train', log_file: str = None) -> logging.Logger:
    """Setup logger for training."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        # MPS uses the same random state as CPU
        pass

def setup_device(preferred_device: str = 'mps') -> torch.device:
    """Setup computation device with MPS support."""
    if preferred_device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        # Set high precision matmul if available
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
    elif preferred_device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return device

def _assert_int(value: Any, name: str) -> int:
    """Assert that a value is an integer."""
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}: {value}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value

def create_temporal_folds_with_months(
    df: pd.DataFrame,
    n_folds: int,
    train_min_years: int,
    val_months: int,
    test_holdout_months: int,
    step_months: int = 1
) -> List[Dict[str, str]]:
    """
    Create temporal cross-validation folds with expanding window using integer months.
    
    Args:
        df: DataFrame with Date column
        n_folds: Number of CV folds
        train_min_years: Minimum training period in years
        val_months: Validation period in months
        test_holdout_months: Test holdout period in months
        step_months: Step size between folds in months
        
    Returns:
        List of fold configurations
    """
    from dateutil.relativedelta import relativedelta
    
    logger = setup_logger()
    
    # Validate integer inputs
    n_folds = _assert_int(n_folds, "n_folds")
    train_min_years = _assert_int(train_min_years, "train_min_years")
    val_months = _assert_int(val_months, "val_months")
    test_holdout_months = _assert_int(test_holdout_months, "test_holdout_months")
    step_months = _assert_int(step_months, "step_months")
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    # Reserve test period using relativedelta for exact month arithmetic
    test_start_date = max_date - relativedelta(months=test_holdout_months)
    available_end_date = test_start_date
    
    logger.info(f"Data range: {min_date.date()} to {max_date.date()}")
    logger.info(f"Test holdout: {test_start_date.date()} to {max_date.date()}")
    logger.info(f"Available for train/val: {min_date.date()} to {available_end_date.date()}")
    
    # Calculate minimum training end date
    min_train_end = min_date + relativedelta(years=train_min_years)
    
    if min_train_end >= available_end_date:
        raise ValueError(f"Insufficient data: need {train_min_years} years for training, "
                        f"but only have {(available_end_date - min_date).days / 365.25:.1f} years available")
    
    # Calculate fold boundaries
    total_available_months = (available_end_date.year - min_date.year) * 12 + (available_end_date.month - min_date.month)
    min_train_months = train_min_years * 12
    max_train_months = total_available_months - val_months
    
    logger.info(f"Total available months: {total_available_months}")
    logger.info(f"Training range: {min_train_months} to {max_train_months} months")
    
    if max_train_months <= min_train_months:
        raise ValueError(f"Insufficient data for validation: max_train_months={max_train_months}, "
                        f"min_train_months={min_train_months}")
    
    # Create folds with expanding windows
    folds = []
    train_months_increment = (max_train_months - min_train_months) // max(1, n_folds - 1)
    
    for fold_idx in range(n_folds):
        # Expanding window: start with minimum, increase by increment
        train_months = min_train_months + (fold_idx * train_months_increment)
        train_months = min(train_months, max_train_months)  # Clamp to maximum
        
        # Calculate exact dates using relativedelta
        train_start_date = min_date
        train_end_date = min_date + relativedelta(months=train_months)
        val_start_date = train_end_date
        val_end_date = val_start_date + relativedelta(months=val_months)
        
        # Clamp validation end to available data
        if val_end_date > available_end_date:
            val_end_date = available_end_date
            val_start_date = val_end_date - relativedelta(months=val_months)
            # Adjust training end if needed
            if val_start_date <= train_end_date:
                train_end_date = val_start_date
        
        # Clamp dates to actual data index
        train_start_clamped = max(train_start_date, min_date)
        train_end_clamped = min(train_end_date, max_date)
        val_start_clamped = max(val_start_date, min_date)
        val_end_clamped = min(val_end_date, max_date)
        test_start_clamped = max(test_start_date, min_date)
        test_end_clamped = max_date
        
        # Create data slices and assert non-empty
        train_slice = df[(df['Date'] >= train_start_clamped) & (df['Date'] <= train_end_clamped)]
        val_slice = df[(df['Date'] > val_start_clamped) & (df['Date'] <= val_end_clamped)]
        test_slice = df[(df['Date'] > test_start_clamped) & (df['Date'] <= test_end_clamped)]
        
        if len(train_slice) == 0:
            raise ValueError(f"Fold {fold_idx}: Empty training slice "
                           f"{train_start_clamped.date()} to {train_end_clamped.date()}")
        
        if len(val_slice) == 0:
            logger.warning(f"Fold {fold_idx}: Empty validation slice "
                          f"{val_start_clamped.date()} to {val_end_clamped.date()}")
        
        if len(test_slice) == 0:
            logger.warning(f"Fold {fold_idx}: Empty test slice "
                          f"{test_start_clamped.date()} to {test_end_clamped.date()}")
        
        fold_config = {
            'fold_idx': fold_idx,
            'train_start': train_start_clamped.strftime('%Y-%m-%d'),
            'train_end': train_end_clamped.strftime('%Y-%m-%d'),
            'val_start': val_start_clamped.strftime('%Y-%m-%d'),
            'val_end': val_end_clamped.strftime('%Y-%m-%d'),
            'test_start': test_start_clamped.strftime('%Y-%m-%d'),
            'test_end': test_end_clamped.strftime('%Y-%m-%d'),
            'train_samples': len(train_slice),
            'val_samples': len(val_slice),
            'test_samples': len(test_slice),
            'train_months_actual': train_months,
            'val_months_actual': val_months
        }
        
        folds.append(fold_config)
        
        logger.info(f"Fold {fold_idx}: "
                   f"Train {fold_config['train_start']} to {fold_config['train_end']} ({len(train_slice)} samples), "
                   f"Val {fold_config['val_start']} to {fold_config['val_end']} ({len(val_slice)} samples), "
                   f"Test {fold_config['test_start']} to {fold_config['test_end']} ({len(test_slice)} samples)")
    
    return folds

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state_dict = None
        self.early_stop = False
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save state if best
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_state_dict = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best and self.best_state_dict is not None:
                model.load_state_dict(self.best_state_dict)
            return True
        
        return False

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    target_scaler
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x).squeeze()
        
        # Calculate loss on normalized values to maintain gradients
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return {'train_loss': total_loss / num_batches}

def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_scaler
) -> Dict[str, float]:
    """Validate model for one epoch."""
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x).squeeze()
            
            # Calculate loss on normalized values
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            # De-normalize for metrics calculation
            outputs_denorm = target_scaler.inverse_transform(
                outputs.cpu().numpy().reshape(-1, 1)
            ).flatten()
            targets_denorm = target_scaler.inverse_transform(
                batch_y.cpu().numpy().reshape(-1, 1)
            ).flatten()
            
            all_outputs.extend(outputs_denorm)
            all_targets.extend(targets_denorm)
            num_batches += 1
    
    # Calculate additional metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
    mae = mean_absolute_error(all_targets, all_outputs)
    
    # MAPE (handle division by zero)
    mape = np.mean(np.abs((np.array(all_targets) - np.array(all_outputs)) / 
                         np.maximum(np.abs(all_targets), 1e-8))) * 100
    
    return {
        'val_loss': total_loss / num_batches,
        'val_rmse': rmse,
        'val_mae': mae,
        'val_mape': mape
    }

def train_fold(
    asset_name: str,
    fold_config: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """Train model for a single fold."""
    logger = setup_logger()
    
    fold_idx = fold_config['fold_idx']
    logger.info(f"Training {asset_name} fold {fold_idx}")
    
    # Load processed data
    if asset_name == 'SP':
        data_path = config['paths']['sp_clean']
    else:
        raise ValueError(f"Unknown asset: {asset_name}")
    
    # Build sequences for this fold
    sequences_data = build_sequences(
        asset_csv_path=data_path,
        seq_len=config['training']['seq_len'],
        horizon=config['training']['horizon_days'],
        features=config['training']['features'],
        train_end_date=fold_config['train_end'],
        val_end_date=fold_config['val_end']
    )
    
    if sequences_data['train_samples'] < 100:
        raise ValueError(f"Insufficient training data for {asset_name} fold {fold_idx}")
    
    # Create data loaders
    train_data = {'X': sequences_data['X_train'], 'y': sequences_data['y_train']}
    val_data = {'X': sequences_data['X_val'], 'y': sequences_data['y_val']} if 'X_val' in sequences_data else None
    
    loaders = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        batch_size=config['training']['batch_size'],
        device=device
    )
    
    # Create model
    input_size = sequences_data['X_train'].shape[-1]
    model = create_model(config['training'], input_size)
    model = model.to(device)
    
    # Setup training components
    criterion = nn.MSELoss()
    
    if config['training']['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # Cosine annealing scheduler
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stop_patience'],
        restore_best=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    logger.info(f"Starting training for {config['training']['epochs']} epochs")
    
    for epoch in range(config['training']['epochs']):
        # Train epoch
        train_metrics = train_epoch(
            model, loaders['train'], criterion, optimizer, device,
            sequences_data['target_scaler']
        )
        
        train_losses.append(train_metrics['train_loss'])
        
        # Validate if validation data available
        if 'val' in loaders and loaders['val'] is not None:
            val_metrics = validate_epoch(
                model, loaders['val'], criterion, device,
                sequences_data['target_scaler']
            )
            val_losses.append(val_metrics['val_loss'])
            
            # Check for best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
            
            # Early stopping check
            if early_stopping(val_metrics['val_loss'], model):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}: Train Loss {train_metrics['train_loss']:.4f}, "
                           f"Val Loss {val_metrics['val_loss']:.4f}, "
                           f"Val RMSE {val_metrics['val_rmse']:.2f}")
        else:
            # No validation data
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}: Train Loss {train_metrics['train_loss']:.4f}")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
    
    # Save model and scaler
    fold_model_dir = os.path.join(config['paths']['models_dir'], asset_name, f'fold_{fold_idx}')
    os.makedirs(fold_model_dir, exist_ok=True)
    
    # Save legacy model checkpoint for backward compatibility
    model_path = os.path.join(fold_model_dir, 'best.ckpt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config['training'],
        'input_size': input_size,
        'fold_config': fold_config
    }, model_path)
    
    # Save legacy scaler for backward compatibility
    scaler_path = os.path.join(fold_model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'feature_scaler': sequences_data['feature_scaler'],
            'target_scaler': sequences_data['target_scaler'],
            'feature_columns': sequences_data['feature_columns']
        }, f)
    
    # Save versioned artifacts using ArtifactManager
    try:
        artifact_manager = ArtifactManager(config['paths']['models_dir'])
        
        # Prepare training metadata
        training_metadata = {
            'train_samples': sequences_data['train_samples'],
            'val_samples': sequences_data.get('val_samples', 0),
            'final_train_loss': train_losses[-1],
            'best_val_loss': best_val_loss if val_losses else None,
            'total_epochs': len(train_losses),
            'early_stopped': early_stopping.early_stop,
            'fold_config': fold_config,
            'data_splits': {
                'train_end': str(fold_config['train_end']),
                'val_end': str(fold_config['val_end'])
            }
        }
        
        # Save versioned artifacts
        artifact_dir = artifact_manager.save_fold_artifacts(
            asset=asset_name,
            fold=fold_idx,
            model=model,
            feature_names=sequences_data['feature_columns'],
            seq_len=config['training']['seq_len'],
            feature_scaler=sequences_data['feature_scaler'],
            target_scaler=sequences_data['target_scaler'],
            training_config=config,
            training_metadata=training_metadata
        )
        
        logger.info(f"✅ Saved versioned artifacts to {artifact_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save versioned artifacts: {str(e)}")
        # Don't fail the training if artifact saving fails
    
    # Prepare results
    results = {
        'asset_name': asset_name,
        'fold_idx': fold_idx,
        'train_samples': sequences_data['train_samples'],
        'val_samples': sequences_data.get('val_samples', 0),
        'final_train_loss': train_losses[-1],
        'best_val_loss': best_val_loss if val_losses else None,
        'total_epochs': len(train_losses),
        'model_path': model_path,
        'scaler_path': scaler_path,
        'model_parameters': count_parameters(model)
    }
    
    logger.info(f"Completed {asset_name} fold {fold_idx}: "
               f"Train samples {results['train_samples']}, "
               f"Final train loss {results['final_train_loss']:.4f}")
    
    return results

def main(config_path: str = 'configs/default.yaml'):
    """Main training function."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_file = os.path.join(config['paths']['logs_dir'], 'training.log')
    logger = setup_logger('main', log_file)
    
    logger.info("=== S&P LSTM PREDICTOR TRAINING ===")
    logger.info(f"Configuration: {config_path}")
    
    # Set random seeds
    set_random_seeds(config['training']['seed'])
    logger.info(f"Random seed set to {config['training']['seed']}")
    
    # Setup device
    device = setup_device(config['training']['device'])
    logger.info(f"Using device: {device}")
    
    try:
        # Step 1: Verify real data
        logger.info("Step 1: Verifying real data integrity...")
        verify_real_data_or_die(
            sp_glob=config['paths']['sp_raw_glob'],
            min_rows=config['cv']['min_rows_required'],
            min_years=config['cv']['min_years_required'],
            log_path=os.path.join(config['paths']['logs_dir'], 'data_integrity.log'),
            hashes_out=os.path.join(config['paths']['logs_dir'], 'data_hashes.json')
        )

        # Step 2: Data ingestion
        logger.info("Step 2: Ingesting and cleaning data...")
        ingestion_results = ingest_all_data(config['paths'])

        # Step 3: Create temporal folds
        logger.info("Step 3: Creating temporal cross-validation folds...")

        # Load the clean dataset to create folds
        sp_df = pd.read_csv(config['paths']['sp_clean'])
        folds = create_temporal_folds_with_months(
            sp_df,
            n_folds=config['cv']['n_folds'],
            train_min_years=config['cv']['train_min_years'],
            val_months=config['cv']['val_months'],
            test_holdout_months=config['cv']['test_holdout_months'],
            step_months=config['cv']['step_months']
        )
        
        # Step 4: Train models for each asset and fold
        logger.info("Step 4: Training LSTM models...")
        
        all_results = {}
        
        for asset_name in config['training']['asset_list']:
            logger.info(f"\n=== Training {asset_name} models ===")
            asset_results = []
            
            for fold_config in folds:
                try:
                    fold_result = train_fold(asset_name, fold_config, config, device)
                    asset_results.append(fold_result)
                except Exception as e:
                    logger.error(f"Failed to train {asset_name} fold {fold_config['fold_idx']}: {str(e)}")
                    continue
            
            all_results[asset_name] = asset_results
            
            # Log asset summary
            if asset_results:
                avg_train_loss = np.mean([r['final_train_loss'] for r in asset_results])
                avg_val_loss = np.mean([r['best_val_loss'] for r in asset_results if r['best_val_loss']])
                logger.info(f"{asset_name} summary: Avg train loss {avg_train_loss:.4f}, "
                           f"Avg val loss {avg_val_loss:.4f}")
        
        # Step 5: Save training summary
        summary_path = os.path.join(config['paths']['reports_dir'], 'training_summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        training_summary = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'device': str(device),
            'ingestion_results': ingestion_results,
            'fold_configs': folds,
            'training_results': all_results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        logger.info(f"Training completed successfully. Summary saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM price prediction models')
    parser.add_argument('--config', default='configs/default.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    main(args.config)