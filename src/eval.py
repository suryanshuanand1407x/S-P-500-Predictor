import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

# Import our custom modules
from models import create_model
from lstm_datasets import create_data_loaders
from feature_engineering import build_sequences

def setup_logger(name: str = 'eval', log_file: str = None) -> logging.Logger:
    """Setup logger for evaluation."""
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

def load_model_and_scaler(model_path: str, scaler_path: str, device: torch.device) -> Tuple[nn.Module, Dict, Dict]:
    """Load trained model and scalers."""
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    input_size = checkpoint['input_size']
    
    # Create and load model
    model = create_model(model_config, input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load scalers
    with open(scaler_path, 'rb') as f:
        scaler_data = pickle.load(f)
    
    return model, scaler_data, checkpoint

def predict_sequences(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256
) -> np.ndarray:
    """Make predictions on sequences."""
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
            batch_pred = model(batch_X).cpu().numpy()
            predictions.extend(batch_pred.flatten())
    
    return np.array(predictions)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (handle division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    
    # Directional accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred) * 100 if len(direction_true) > 0 else 0
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Residual statistics
    residuals = y_true - y_pred
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'directional_accuracy': directional_accuracy,
        'residual_mean': residual_mean,
        'residual_std': residual_std,
        'min_error': np.min(residuals),
        'max_error': np.max(residuals)
    }

def evaluate_model_on_test_data(
    model_path: str,
    scaler_path: str,
    test_data_path: str,
    fold_config: Dict[str, str],
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate a single model on test data using SAVED model scalers."""
    logger = setup_logger()

    # Load model and SAVED scalers from training
    model, scaler_data, checkpoint = load_model_and_scaler(model_path, scaler_path, device)

    logger.info(f"Using saved model scalers: target_mean={scaler_data['target_scaler'].mean_[0]:.2f}")

    # Load test data manually to use SAVED scalers
    import pandas as pd
    from feature_engineering import calculate_technical_indicators, prepare_feature_columns, create_sequences, transform_sequences

    df = pd.read_csv(test_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Extract ONLY the designated test holdout period
    test_df = df[(df['Date'] >= fold_config['test_start']) &
                 (df['Date'] <= fold_config['test_end'])].copy()

    if len(test_df) < config['training']['seq_len'] + 10:
        logger.warning("Insufficient data in test holdout period")
        return None

    # Calculate technical indicators
    if 'ret_1d' not in test_df.columns:
        test_df = calculate_technical_indicators(test_df)

    # Prepare features
    asset_name = test_df['Asset'].iloc[0] if 'Asset' in test_df.columns else 'UNKNOWN'
    test_df = prepare_feature_columns(test_df, config['training']['features'], asset_name)

    # Create raw unscaled sequences - PREDICTING RETURNS
    X_test, y_test, feature_cols, test_current_prices = create_sequences(
        test_df,
        list(test_df.columns),
        target_column='Close',
        sequence_length=config['training']['seq_len'],
        horizon=config['training']['horizon_days'],
        predict_returns=True  # KEY: Predict returns for stationarity
    )

    if len(X_test) < 10:
        logger.warning("Insufficient test sequences created")
        return None

    # Apply SAVED model scalers to test data
    X_test_scaled, y_test_scaled = transform_sequences(
        X_test, y_test,
        scaler_data['feature_scaler'],
        scaler_data['target_scaler']
    )

    # Make predictions (model predicts RETURNS, not prices)
    y_pred_returns_scaled = predict_sequences(model, X_test_scaled, device)

    # De-normalize predictions and targets using SAVED scalers
    y_pred_returns = scaler_data['target_scaler'].inverse_transform(y_pred_returns_scaled.reshape(-1, 1)).flatten()
    y_true_returns = scaler_data['target_scaler'].inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    logger.info(f"Predicted returns: mean={y_pred_returns.mean():.4f}, std={y_pred_returns.std():.4f}")
    logger.info(f"Actual returns: mean={y_true_returns.mean():.4f}, std={y_true_returns.std():.4f}")

    # CONVERT RETURNS BACK TO PRICES for evaluation metrics
    # predicted_price = current_price * (1 + predicted_return)
    y_pred = test_current_prices * (1 + y_pred_returns)
    y_true = test_current_prices * (1 + y_true_returns)

    logger.info(f"Test predictions (prices): {len(y_pred)} samples, range {y_pred.min():.2f} - {y_pred.max():.2f}")
    logger.info(f"Test actuals (prices): {len(y_true)} samples, range {y_true.min():.2f} - {y_true.max():.2f}")

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)

    # Extract dates corresponding to the test sequences
    # Each sequence predicts horizon days ahead, so we need dates starting from seq_len onwards
    seq_len = config['training']['seq_len']
    horizon = config['training']['horizon_days']

    # The prediction dates are the target dates for each sequence
    # test_df already contains only the test holdout period
    test_dates = test_df.iloc[seq_len + horizon - 1:seq_len + horizon - 1 + len(y_true)]['Date'].values
    
    return {
        'metrics': metrics,
        'predictions': y_pred,
        'actual': y_true,
        'dates': test_dates,
        'fold_config': fold_config,
        'test_samples': len(y_true)
    }

def create_evaluation_plots(
    results: Dict[str, Any],
    asset_name: str,
    output_dir: str
) -> List[str]:
    """Create evaluation plots and save them."""
    
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Actual vs Predicted scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    actual = results['actual']
    predicted = results['predictions']
    
    ax.scatter(actual, predicted, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Price (INR)')
    ax.set_ylabel('Predicted Price (INR)')
    ax.set_title(f'{asset_name} - Actual vs Predicted Prices')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² to plot
    r2 = results['metrics']['r2']
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plot_file = os.path.join(output_dir, f'{asset_name.lower()}_actual_vs_predicted.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_file)
    
    # 2. Time series plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    dates = results['dates']
    
    # Price comparison
    ax1.plot(dates, actual, label='Actual', linewidth=1.5, alpha=0.8)
    ax1.plot(dates, predicted, label='Predicted', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Price (INR)')
    ax1.set_title(f'{asset_name} - Price Prediction Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    residuals = actual - predicted
    ax2.plot(dates, residuals, label='Residuals', color='red', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Residuals (INR)')
    ax2.set_xlabel('Date')
    ax2.set_title('Prediction Residuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f'{asset_name.lower()}_time_series.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_file)
    
    # 3. Residual analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residual histogram
    ax1.hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--')
    ax1.set_xlabel('Residuals (INR)')
    ax1.set_ylabel('Density')
    ax1.set_title('Residual Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal Distribution)')
    ax2.grid(True, alpha=0.3)
    
    # Rolling RMSE
    window_size = min(20, len(residuals) // 5)
    if window_size >= 2:
        rolling_rmse = pd.Series(residuals).rolling(window=window_size).apply(
            lambda x: np.sqrt(np.mean(x**2))
        )
        ax3.plot(dates, rolling_rmse, linewidth=2)
        ax3.set_ylabel('Rolling RMSE')
        ax3.set_xlabel('Date')
        ax3.set_title(f'Rolling RMSE (Window: {window_size} days)')
        ax3.grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    ax4.scatter(predicted, residuals, alpha=0.6, s=20)
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.set_xlabel('Predicted Price (INR)')
    ax4.set_ylabel('Residuals (INR)')
    ax4.set_title('Residuals vs Predicted')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f'{asset_name.lower()}_residual_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_file)
    
    return plot_files

def evaluate_all_models(config_path: str = 'configs/default.yaml'):
    """Main evaluation function for all trained models."""
    
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_file = os.path.join(config['paths']['logs_dir'], 'evaluation.log')
    logger = setup_logger('eval', log_file)
    
    logger.info("=== MODEL EVALUATION STARTING ===")
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load training summary to get fold configurations
    training_summary_path = os.path.join(config['paths']['reports_dir'], 'training_summary.json')
    
    if not os.path.exists(training_summary_path):
        logger.error(f"Training summary not found at {training_summary_path}")
        return
    
    with open(training_summary_path, 'r') as f:
        training_summary = json.load(f)
    
    fold_configs = training_summary['fold_configs']
    training_results = training_summary['training_results']
    
    # Evaluation results
    evaluation_results = {}
    
    for asset_name in config['training']['asset_list']:
        logger.info(f"\n=== Evaluating {asset_name} models ===")
        
        asset_results = []
        asset_path = config['paths'][f'{asset_name.lower()}_clean']
        
        if asset_name not in training_results:
            logger.warning(f"No training results found for {asset_name}")
            continue
        
        for fold_result in training_results[asset_name]:
            fold_idx = fold_result['fold_idx']
            model_path = fold_result['model_path']
            scaler_path = fold_result['scaler_path']
            
            logger.info(f"Evaluating {asset_name} fold {fold_idx}")
            
            try:
                # Find corresponding fold config
                fold_config = next(fc for fc in fold_configs if fc['fold_idx'] == fold_idx)
                
                # Evaluate model
                eval_result = evaluate_model_on_test_data(
                    model_path=model_path,
                    scaler_path=scaler_path,
                    test_data_path=asset_path,
                    fold_config=fold_config,
                    config=config,
                    device=device
                )
                
                if eval_result is not None:
                    eval_result['fold_idx'] = fold_idx
                    asset_results.append(eval_result)
                    
                    logger.info(f"Fold {fold_idx} - RMSE: {eval_result['metrics']['rmse']:.2f}, "
                               f"MAE: {eval_result['metrics']['mae']:.2f}, "
                               f"Directional Accuracy: {eval_result['metrics']['directional_accuracy']:.1f}%")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {asset_name} fold {fold_idx}: {str(e)}")
                continue
        
        if asset_results:
            evaluation_results[asset_name] = asset_results
            
            # Calculate average metrics across folds
            avg_metrics = {}
            for metric in asset_results[0]['metrics'].keys():
                avg_metrics[f'avg_{metric}'] = np.mean([r['metrics'][metric] for r in asset_results])
                avg_metrics[f'std_{metric}'] = np.std([r['metrics'][metric] for r in asset_results])
            
            logger.info(f"{asset_name} average metrics:")
            logger.info(f"  RMSE: {avg_metrics['avg_rmse']:.2f} ± {avg_metrics['std_rmse']:.2f}")
            logger.info(f"  MAE: {avg_metrics['avg_mae']:.2f} ± {avg_metrics['std_mae']:.2f}")
            logger.info(f"  Directional Accuracy: {avg_metrics['avg_directional_accuracy']:.1f}% ± {avg_metrics['std_directional_accuracy']:.1f}%")
            
            evaluation_results[f'{asset_name}_summary'] = avg_metrics
        
        # Create plots for best fold (lowest RMSE)
        if asset_results:
            best_fold = min(asset_results, key=lambda x: x['metrics']['rmse'])
            plot_dir = os.path.join(config['paths']['reports_dir'], 'plots')
            
            try:
                plot_files = create_evaluation_plots(best_fold, asset_name, plot_dir)
                logger.info(f"Created plots for {asset_name}: {len(plot_files)} files")
            except Exception as e:
                logger.error(f"Failed to create plots for {asset_name}: {str(e)}")
    
    # Save evaluation results
    eval_summary_path = os.path.join(config['paths']['reports_dir'], 'evaluation_summary.json')
    
    final_summary = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'device': str(device),
        'evaluation_results': evaluation_results
    }
    
    with open(eval_summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    # Create final metrics summary
    metrics_summary = {}
    for asset_name in config['training']['asset_list']:
        if f'{asset_name}_summary' in evaluation_results:
            summary = evaluation_results[f'{asset_name}_summary']
            metrics_summary[asset_name] = {
                'rmse_inr': f"{summary['avg_rmse']:.2f} ± {summary['std_rmse']:.2f}",
                'mae_inr': f"{summary['avg_mae']:.2f} ± {summary['std_mae']:.2f}",
                'mape_percent': f"{summary['avg_mape']:.1f}% ± {summary['std_mape']:.1f}%",
                'directional_accuracy_percent': f"{summary['avg_directional_accuracy']:.1f}% ± {summary['std_directional_accuracy']:.1f}%",
                'r2_score': f"{summary['avg_r2']:.3f} ± {summary['std_r2']:.3f}"
            }
    
    metrics_summary_path = os.path.join(config['paths']['reports_dir'], 'metrics_summary.json')
    with open(metrics_summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    logger.info(f"\n=== EVALUATION COMPLETED ===")
    logger.info(f"Results saved to: {eval_summary_path}")
    logger.info(f"Metrics summary saved to: {metrics_summary_path}")
    logger.info(f"Final metrics:")
    
    for asset, metrics in metrics_summary.items():
        logger.info(f"\n{asset}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained LSTM models')
    parser.add_argument('--config', default='configs/default.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    evaluate_all_models(args.config)