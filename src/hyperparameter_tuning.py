#!/usr/bin/env python3
"""
Hyperparameter Tuning for LSTM Price Predictor

Performs systematic hyperparameter search to optimize model performance.
Tests combinations of learning rates, hidden sizes, layers, dropout, etc.

Author: AI Assistant
Date: 2025
"""

import os
import sys
import yaml
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom modules
from train import (
    setup_logger, set_random_seeds, setup_device,
    create_temporal_folds_with_months
)
from data_ingestion import ingest_all_data
from feature_engineering import build_sequences, calculate_technical_indicators, add_cross_asset_features
from lstm_datasets import SequenceDataModule, create_data_loaders
from models import create_model, count_parameters
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def setup_tuning_logger(log_dir: str = 'logs') -> logging.Logger:
    """Setup logger for hyperparameter tuning."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('hyperparameter_tuning')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(log_dir, f'hyperparameter_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def get_hyperparameter_search_space() -> Dict[str, List]:
    """Define hyperparameter search space."""
    return {
        'hidden_size': [64, 128, 256],
        'num_layers': [2, 3],
        'dropout': [0.1, 0.2, 0.3],
        'lr': [0.001, 0.0005, 0.0001],
        'batch_size': [128, 256],
        'optimizer': ['adam', 'adamw'],
        'scheduler': ['cosine', 'plateau']
    }

def train_single_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int,
    early_stop_patience: int,
    logger: logging.Logger
) -> Tuple[float, float, float]:
    """
    Train a single fold and return validation metrics.

    Returns:
        Tuple of (best_val_loss, final_train_loss, best_epoch)
    """
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # Early stopping
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}, best epoch was {best_epoch}")
                break
        else:
            # No validation set
            best_val_loss = avg_train_loss
            best_epoch = epoch

    return best_val_loss, avg_train_loss, best_epoch

def evaluate_hyperparameters(
    hyperparams: Dict[str, Any],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    asset: str,
    config: Dict,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Evaluate a single hyperparameter configuration.

    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"Testing hyperparameters: {hyperparams}")

    # Build sequences
    seq_len = config['training']['seq_len']
    features = config['training']['features']

    # Add cross-asset features
    other_asset = 'GOLD' if asset == 'SENSEX' else 'SENSEX'
    other_asset_clean_path = config['paths'][f"{other_asset.lower()}_clean"]
    other_df = pd.read_csv(other_asset_clean_path, parse_dates=['Date'])

    train_data_with_corr = add_cross_asset_features(
        train_data.copy(), other_df, asset, other_asset
    )
    val_data_with_corr = add_cross_asset_features(
        val_data.copy(), other_df, asset, other_asset
    )

    # Build sequences
    train_sequences = build_sequences(
        train_data_with_corr, seq_len=seq_len,
        feature_cols=features, target_col='Close'
    )

    if len(val_data_with_corr) > seq_len:
        val_sequences = build_sequences(
            val_data_with_corr, seq_len=seq_len,
            feature_cols=features, target_col='Close'
        )
    else:
        val_sequences = None

    # Create data loaders
    train_loader, val_loader, feature_scaler, target_scaler = create_data_loaders(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        batch_size=hyperparams['batch_size'],
        device=device
    )

    # Create model
    input_size = train_sequences['X'].shape[2]
    model = create_model(
        input_size=input_size,
        hidden_size=hyperparams['hidden_size'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout'],
        device=device
    )

    # Setup optimizer
    if hyperparams['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=hyperparams['lr'], weight_decay=0.01)

    # Setup scheduler
    if hyperparams['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs']
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )

    # Train
    best_val_loss, final_train_loss, best_epoch = train_single_fold(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=config['training']['epochs'],
        early_stop_patience=config['training']['early_stop_patience'],
        logger=logger
    )

    # Evaluate on validation set
    if val_loader is not None and len(val_loader) > 0:
        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Denormalize
        predictions_denorm = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_denorm = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

        # Calculate metrics
        mse = mean_squared_error(actuals_denorm, predictions_denorm)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_denorm, predictions_denorm)
        mape = np.mean(np.abs((actuals_denorm - predictions_denorm) / actuals_denorm)) * 100
        r2 = r2_score(actuals_denorm, predictions_denorm)

        # Directional accuracy
        actual_direction = np.sign(np.diff(actuals_denorm, prepend=actuals_denorm[0]))
        pred_direction = np.sign(np.diff(predictions_denorm, prepend=predictions_denorm[0]))
        directional_accuracy = np.mean(actual_direction == pred_direction)
    else:
        rmse = best_val_loss
        mae = best_val_loss
        mape = 100.0
        r2 = -1.0
        directional_accuracy = 0.5

    results = {
        'val_loss': best_val_loss,
        'train_loss': final_train_loss,
        'best_epoch': best_epoch,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'directional_accuracy': directional_accuracy,
        'model_params': count_parameters(model)
    }

    logger.info(f"Results: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, R2={r2:.4f}, Dir Acc={directional_accuracy:.2%}")

    return results

def run_hyperparameter_search(
    config_path: str,
    search_space: Dict[str, List],
    max_trials: int = 20,
    output_dir: str = 'reports'
) -> Dict[str, Any]:
    """
    Run hyperparameter search using random search.

    Args:
        config_path: Path to configuration file
        search_space: Dictionary of hyperparameter ranges
        max_trials: Maximum number of trials to run
        output_dir: Directory to save results

    Returns:
        Dictionary with best hyperparameters and all results
    """
    logger = setup_tuning_logger()
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER TUNING - LSTM PRICE PREDICTOR")
    logger.info("=" * 80)

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    set_random_seeds(config['training']['seed'])
    device = setup_device(config['training']['device'])

    logger.info(f"Device: {device}")
    logger.info(f"Max trials: {max_trials}")
    logger.info(f"Search space: {search_space}")

    # Load data (data should already be processed)
    logger.info("Loading processed data...")
    # Data is already processed, we'll load it directly when needed

    # Results storage
    all_results = []

    # Test both assets
    for asset in config['training']['asset_list']:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TUNING HYPERPARAMETERS FOR {asset}")
        logger.info(f"{'=' * 80}\n")

        # Load asset data
        asset_clean_path = config['paths'][f"{asset.lower()}_clean"]
        df = pd.read_csv(asset_clean_path, parse_dates=['Date'])
        df = calculate_technical_indicators(df)
        df = df.dropna().reset_index(drop=True)

        # Create a single fold for tuning (use fold 4 - middle of data)
        folds = create_temporal_folds_with_months(
            df,
            n_folds=config['cv']['n_folds'],
            train_min_years=config['cv']['train_min_years'],
            val_months=config['cv']['val_months'],
            test_holdout_months=config['cv']['test_holdout_months'],
            step_months=config['cv']['step_months']
        )

        # Use middle fold for hyperparameter tuning
        tune_fold = folds[4]
        train_data = df[
            (df['Date'] >= tune_fold['train_start']) &
            (df['Date'] <= tune_fold['train_end'])
        ].copy()
        val_data = df[
            (df['Date'] > tune_fold['val_start']) &
            (df['Date'] <= tune_fold['val_end'])
        ].copy()

        logger.info(f"Tuning on fold 4:")
        logger.info(f"  Train: {tune_fold['train_start']} to {tune_fold['train_end']} ({len(train_data)} samples)")
        logger.info(f"  Val: {tune_fold['val_start']} to {tune_fold['val_end']} ({len(val_data)} samples)")

        # Generate random hyperparameter combinations
        keys = list(search_space.keys())
        all_combinations = list(itertools.product(*[search_space[k] for k in keys]))

        # Randomly sample combinations
        np.random.shuffle(all_combinations)
        sampled_combinations = all_combinations[:max_trials]

        logger.info(f"Testing {len(sampled_combinations)} hyperparameter combinations")

        # Test each combination
        for trial_idx, combination in enumerate(sampled_combinations):
            hyperparams = dict(zip(keys, combination))

            logger.info(f"\n--- Trial {trial_idx + 1}/{len(sampled_combinations)} for {asset} ---")

            try:
                results = evaluate_hyperparameters(
                    hyperparams=hyperparams,
                    train_data=train_data,
                    val_data=val_data,
                    asset=asset,
                    config=config,
                    device=device,
                    logger=logger
                )

                # Store results
                result_entry = {
                    'asset': asset,
                    'trial': trial_idx + 1,
                    'hyperparams': hyperparams,
                    'metrics': results
                }
                all_results.append(result_entry)

            except Exception as e:
                logger.error(f"Trial {trial_idx + 1} failed: {str(e)}")
                continue

    # Analyze results
    logger.info("\n" + "=" * 80)
    logger.info("HYPERPARAMETER TUNING COMPLETE")
    logger.info("=" * 80)

    # Find best hyperparameters for each asset
    best_configs = {}

    for asset in config['training']['asset_list']:
        asset_results = [r for r in all_results if r['asset'] == asset]

        if not asset_results:
            logger.warning(f"No successful trials for {asset}")
            continue

        # Sort by validation loss (lower is better)
        asset_results_sorted = sorted(asset_results, key=lambda x: x['metrics']['val_loss'])
        best_result = asset_results_sorted[0]

        best_configs[asset] = {
            'hyperparams': best_result['hyperparams'],
            'metrics': best_result['metrics']
        }

        logger.info(f"\n{asset} - Best Hyperparameters:")
        for key, value in best_result['hyperparams'].items():
            logger.info(f"  {key}: {value}")
        logger.info(f"  Validation Loss: {best_result['metrics']['val_loss']:.6f}")
        logger.info(f"  RMSE: {best_result['metrics']['rmse']:.2f}")
        logger.info(f"  MAE: {best_result['metrics']['mae']:.2f}")
        logger.info(f"  MAPE: {best_result['metrics']['mape']:.2f}%")
        logger.info(f"  R²: {best_result['metrics']['r2']:.4f}")
        logger.info(f"  Directional Accuracy: {best_result['metrics']['directional_accuracy']:.2%}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'hyperparameter_tuning_results.json')

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'search_space': search_space,
        'max_trials': max_trials,
        'best_configs': best_configs,
        'all_results': all_results
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    return output_data

def main(config_path: str = 'configs/default.yaml'):
    """Main entry point for hyperparameter tuning."""

    # Define search space
    search_space = get_hyperparameter_search_space()

    # Run hyperparameter search
    results = run_hyperparameter_search(
        config_path=config_path,
        search_space=search_space,
        max_trials=20,  # Test 20 random combinations
        output_dir='reports'
    )

    return results

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/default.yaml'
    main(config_path)
