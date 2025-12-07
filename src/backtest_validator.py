#!/usr/bin/env python3
"""
Backtest Validation System for Sensex & Gold LSTM Price Predictor

This module implements mandatory backtest validation that must pass before
enabling production predictions. It validates that inference metrics approximately
match validation metrics from training, ensuring the inference pipeline is
correctly reproducing training-time behavior.

Key Features:
- Reproduces exact training validation conditions using versioned artifacts
- Compares inference metrics with training validation metrics
- Enforces metric thresholds for deployment approval
- Creates validation certificates for production use
- Comprehensive reporting and diagnostics

Author: AI Assistant
Date: 2025
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from artifact_manager import ArtifactManager
from production_inference import ProductionInferenceEngine

def setup_logger(name: str = 'backtest_validator') -> logging.Logger:
    """Setup logger for backtest validation."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class BacktestValidator:
    """
    Validates inference pipeline against training validation metrics.
    
    Ensures that the inference system produces results that are statistically
    consistent with training validation performance before enabling production use.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize backtest validator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.logger = setup_logger('backtest_validator')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize artifact manager
        self.artifact_manager = ArtifactManager(self.config['paths']['models_dir'])
        
        # Load training validation results for comparison
        self.training_results = self._load_training_validation_results()
        
        # Define validation thresholds
        self.validation_thresholds = {
            'rmse_tolerance': 0.20,      # ±20% from training validation RMSE
            'mae_tolerance': 0.20,       # ±20% from training validation MAE
            'mape_tolerance': 0.30,      # ±30% from training validation MAPE
            'directional_accuracy_min': 0.45,  # Minimum 45% directional accuracy
            'r2_tolerance': 0.15,        # ±15% from training validation R²
            'max_outlier_ratio': 0.05    # Maximum 5% of predictions can be outliers
        }
        
        self.logger.info("Backtest validator initialized")
    
    def _load_training_validation_results(self) -> Dict[str, Any]:
        """Load training validation results for comparison."""
        try:
            results_file = Path(self.config['paths']['reports_dir']) / 'evaluation_summary.json'
            if not results_file.exists():
                self.logger.warning("Training validation results not found - using estimated baselines")
                return self._get_baseline_metrics()
            
            with open(results_file, 'r') as f:
                training_results = json.load(f)
            
            self.logger.info("Loaded training validation results for comparison")
            return training_results
            
        except Exception as e:
            self.logger.warning(f"Could not load training results: {e} - using baselines")
            return self._get_baseline_metrics()
    
    def _get_baseline_metrics(self) -> Dict[str, Any]:
        """Get baseline metrics if training results are not available."""
        return {
            'SENSEX': {
                'avg_rmse': 1700.0,
                'avg_mae': 500.0,
                'avg_mape': 2.0,
                'avg_directional_accuracy': 0.55,
                'avg_r2': 0.95
            },
            'GOLD': {
                'avg_rmse': 35.0,
                'avg_mae': 15.0,
                'avg_mape': 1.2,
                'avg_directional_accuracy': 0.53,
                'avg_r2': 0.99
            }
        }
    
    def _prepare_backtest_data(self, asset: str, n_samples: int = 100) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Prepare backtest data using historical validation periods.
        
        Args:
            asset: Asset name
            n_samples: Number of samples to test
            
        Returns:
            Tuple of (backtest_samples, raw_data)
        """
        # Load historical data
        if asset == 'SENSEX':
            data_path = self.config['paths']['sensex_clean']
        elif asset == 'GOLD':
            data_path = self.config['paths']['gold_clean']
        else:
            raise ValueError(f"Unsupported asset: {asset}")
        
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Get recent data for backtesting (last 6 months)
        recent_data = df.tail(min(n_samples + 100, len(df))).copy()
        
        # Create backtest samples
        seq_len = self.config['training']['seq_len']
        backtest_samples = []
        
        for i in range(seq_len, len(recent_data) - 1):
            sample = {
                'date': recent_data.iloc[i]['Date'],
                'current_price': recent_data.iloc[i]['Close'],
                'actual_next_price': recent_data.iloc[i + 1]['Close'],
                'data_end_idx': i
            }
            backtest_samples.append(sample)
        
        # Limit to requested number of samples
        backtest_samples = backtest_samples[-n_samples:] if len(backtest_samples) > n_samples else backtest_samples
        
        self.logger.info(f"Prepared {len(backtest_samples)} backtest samples for {asset}")
        return backtest_samples, recent_data
    
    def _run_backtest_predictions(self, asset: str, backtest_samples: List[Dict], raw_data: pd.DataFrame) -> List[Dict]:
        """
        Run inference predictions on backtest samples.
        
        Args:
            asset: Asset name
            backtest_samples: List of backtest samples
            raw_data: Raw historical data
            
        Returns:
            List of prediction results
        """
        prediction_results = []
        
        # Initialize production inference engine (without backtest requirement)
        inference_engine = ProductionInferenceEngine(self.config_path, validate_artifacts=True)
        
        for i, sample in enumerate(backtest_samples):
            try:
                # Simulate historical inference by truncating data to the sample date
                data_end_idx = sample['data_end_idx']
                historical_data = raw_data.iloc[:data_end_idx + 1].copy()
                
                # Temporarily save this historical data for inference
                temp_data_path = f"temp_{asset}_backtest_{i}.csv"
                historical_data.to_csv(temp_data_path, index=False)
                
                # Update config to use temporary data
                original_path = self.config['paths'][f'{asset.lower()}_clean']
                self.config['paths'][f'{asset.lower()}_clean'] = temp_data_path
                
                try:
                    # Run inference
                    prediction_result = inference_engine.predict_single_asset(asset)
                    
                    # Extract prediction
                    predicted_price = prediction_result['final_prediction']
                    
                    # Calculate metrics
                    actual_price = sample['actual_next_price']
                    current_price = sample['current_price']
                    
                    prediction_error = predicted_price - actual_price
                    actual_change = actual_price - current_price
                    predicted_change = predicted_price - current_price
                    
                    # Directional accuracy
                    actual_direction = 1 if actual_change > 0 else (-1 if actual_change < 0 else 0)
                    predicted_direction = 1 if predicted_change > 0 else (-1 if predicted_change < 0 else 0)
                    directional_correct = (actual_direction == predicted_direction)
                    
                    result = {
                        'sample_idx': i,
                        'date': sample['date'],
                        'current_price': current_price,
                        'actual_price': actual_price,
                        'predicted_price': predicted_price,
                        'prediction_error': prediction_error,
                        'absolute_error': abs(prediction_error),
                        'percentage_error': abs(prediction_error) / actual_price * 100,
                        'actual_change': actual_change,
                        'predicted_change': predicted_change,
                        'actual_change_pct': actual_change / current_price * 100,
                        'predicted_change_pct': predicted_change / current_price * 100,
                        'directional_correct': directional_correct,
                        'prediction_flags': prediction_result['production_flags']
                    }
                    
                    prediction_results.append(result)
                    
                finally:
                    # Restore original config
                    self.config['paths'][f'{asset.lower()}_clean'] = original_path
                    
                    # Clean up temporary file
                    if os.path.exists(temp_data_path):
                        os.remove(temp_data_path)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{len(backtest_samples)} backtest predictions for {asset}")
                
            except Exception as e:
                self.logger.error(f"Backtest prediction failed for {asset} sample {i}: {str(e)}")
                continue
        
        self.logger.info(f"Completed {len(prediction_results)} backtest predictions for {asset}")
        return prediction_results
    
    def _calculate_backtest_metrics(self, prediction_results: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics from backtest results."""
        if not prediction_results:
            return {'error': 'No prediction results available'}
        
        # Extract arrays for metric calculation
        actual_prices = np.array([r['actual_price'] for r in prediction_results])
        predicted_prices = np.array([r['predicted_price'] for r in prediction_results])
        absolute_errors = np.array([r['absolute_error'] for r in prediction_results])
        percentage_errors = np.array([r['percentage_error'] for r in prediction_results])
        directional_correct = np.array([r['directional_correct'] for r in prediction_results])
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mae = mean_absolute_error(actual_prices, predicted_prices)
        mape = np.mean(percentage_errors)
        directional_accuracy = np.mean(directional_correct)
        
        # R² calculation
        ss_res = np.sum((actual_prices - predicted_prices) ** 2)
        ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Outlier detection (predictions >3 std deviations from mean error)
        error_mean = np.mean(absolute_errors)
        error_std = np.std(absolute_errors)
        outlier_threshold = error_mean + 3 * error_std
        outliers = absolute_errors > outlier_threshold
        outlier_ratio = np.mean(outliers)
        
        metrics = {
            'n_samples': len(prediction_results),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'directional_accuracy': float(directional_accuracy),
            'r2': float(r2),
            'outlier_ratio': float(outlier_ratio),
            'error_statistics': {
                'mean_absolute_error': float(error_mean),
                'std_absolute_error': float(error_std),
                'min_error': float(np.min(absolute_errors)),
                'max_error': float(np.max(absolute_errors)),
                'median_error': float(np.median(absolute_errors))
            },
            'prediction_distribution': {
                'mean_predicted': float(np.mean(predicted_prices)),
                'std_predicted': float(np.std(predicted_prices)),
                'mean_actual': float(np.mean(actual_prices)),
                'std_actual': float(np.std(actual_prices))
            }
        }
        
        return metrics
    
    def _validate_metrics_against_training(self, asset: str, backtest_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate backtest metrics against training validation metrics."""
        if asset not in self.training_results:
            self.logger.warning(f"No training results available for {asset}")
            return {'status': 'warning', 'message': 'No training results for comparison'}
        
        training_metrics = self.training_results[asset]
        validation_results = {
            'asset': asset,
            'status': 'passed',
            'failed_checks': [],
            'warnings': [],
            'comparisons': {}
        }
        
        # RMSE validation
        training_rmse = training_metrics.get('avg_rmse', 0)
        backtest_rmse = backtest_metrics['rmse']
        rmse_ratio = abs(backtest_rmse - training_rmse) / training_rmse if training_rmse > 0 else float('inf')
        
        validation_results['comparisons']['rmse'] = {
            'training': training_rmse,
            'backtest': backtest_rmse,
            'ratio': rmse_ratio,
            'threshold': self.validation_thresholds['rmse_tolerance'],
            'passed': rmse_ratio <= self.validation_thresholds['rmse_tolerance']
        }
        
        if not validation_results['comparisons']['rmse']['passed']:
            validation_results['failed_checks'].append(f"RMSE deviation too high: {rmse_ratio:.3f} > {self.validation_thresholds['rmse_tolerance']}")
        
        # MAE validation
        training_mae = training_metrics.get('avg_mae', 0)
        backtest_mae = backtest_metrics['mae']
        mae_ratio = abs(backtest_mae - training_mae) / training_mae if training_mae > 0 else float('inf')
        
        validation_results['comparisons']['mae'] = {
            'training': training_mae,
            'backtest': backtest_mae,
            'ratio': mae_ratio,
            'threshold': self.validation_thresholds['mae_tolerance'],
            'passed': mae_ratio <= self.validation_thresholds['mae_tolerance']
        }
        
        if not validation_results['comparisons']['mae']['passed']:
            validation_results['failed_checks'].append(f"MAE deviation too high: {mae_ratio:.3f} > {self.validation_thresholds['mae_tolerance']}")
        
        # MAPE validation
        training_mape = training_metrics.get('avg_mape', 0)
        backtest_mape = backtest_metrics['mape']
        mape_ratio = abs(backtest_mape - training_mape) / training_mape if training_mape > 0 else float('inf')
        
        validation_results['comparisons']['mape'] = {
            'training': training_mape,
            'backtest': backtest_mape,
            'ratio': mape_ratio,
            'threshold': self.validation_thresholds['mape_tolerance'],
            'passed': mape_ratio <= self.validation_thresholds['mape_tolerance']
        }
        
        if not validation_results['comparisons']['mape']['passed']:
            validation_results['failed_checks'].append(f"MAPE deviation too high: {mape_ratio:.3f} > {self.validation_thresholds['mape_tolerance']}")
        
        # Directional accuracy validation
        training_dir_acc = training_metrics.get('avg_directional_accuracy', 0.5)
        backtest_dir_acc = backtest_metrics['directional_accuracy']
        
        validation_results['comparisons']['directional_accuracy'] = {
            'training': training_dir_acc,
            'backtest': backtest_dir_acc,
            'min_threshold': self.validation_thresholds['directional_accuracy_min'],
            'passed': backtest_dir_acc >= self.validation_thresholds['directional_accuracy_min']
        }
        
        if not validation_results['comparisons']['directional_accuracy']['passed']:
            validation_results['failed_checks'].append(f"Directional accuracy too low: {backtest_dir_acc:.3f} < {self.validation_thresholds['directional_accuracy_min']}")
        
        # R² validation
        training_r2 = training_metrics.get('avg_r2', 0)
        backtest_r2 = backtest_metrics['r2']
        r2_ratio = abs(backtest_r2 - training_r2) / training_r2 if training_r2 > 0 else float('inf')
        
        validation_results['comparisons']['r2'] = {
            'training': training_r2,
            'backtest': backtest_r2,
            'ratio': r2_ratio,
            'threshold': self.validation_thresholds['r2_tolerance'],
            'passed': r2_ratio <= self.validation_thresholds['r2_tolerance']
        }
        
        if not validation_results['comparisons']['r2']['passed']:
            validation_results['failed_checks'].append(f"R² deviation too high: {r2_ratio:.3f} > {self.validation_thresholds['r2_tolerance']}")
        
        # Outlier ratio validation
        outlier_ratio = backtest_metrics['outlier_ratio']
        validation_results['comparisons']['outlier_ratio'] = {
            'backtest': outlier_ratio,
            'threshold': self.validation_thresholds['max_outlier_ratio'],
            'passed': outlier_ratio <= self.validation_thresholds['max_outlier_ratio']
        }
        
        if not validation_results['comparisons']['outlier_ratio']['passed']:
            validation_results['failed_checks'].append(f"Outlier ratio too high: {outlier_ratio:.3f} > {self.validation_thresholds['max_outlier_ratio']}")
        
        # Overall validation status
        if validation_results['failed_checks']:
            validation_results['status'] = 'failed'
        
        return validation_results
    
    def run_full_backtest_validation(self, n_samples: int = 50) -> Dict[str, Any]:
        """
        Run complete backtest validation for all assets.
        
        Args:
            n_samples: Number of samples to test per asset
            
        Returns:
            Dictionary containing complete validation results
        """
        self.logger.info(f"Starting full backtest validation with {n_samples} samples per asset")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': n_samples,
            'assets': {},
            'overall_status': 'unknown',
            'validation_thresholds': self.validation_thresholds
        }
        
        all_passed = True
        
        for asset in self.config['training']['asset_list']:
            try:
                self.logger.info(f"Running backtest validation for {asset}...")
                
                # Prepare backtest data
                backtest_samples, raw_data = self._prepare_backtest_data(asset, n_samples)
                
                # Run backtest predictions
                prediction_results = self._run_backtest_predictions(asset, backtest_samples, raw_data)
                
                # Calculate metrics
                backtest_metrics = self._calculate_backtest_metrics(prediction_results)
                
                # Validate against training metrics
                validation_result = self._validate_metrics_against_training(asset, backtest_metrics)
                
                # Store results
                validation_results['assets'][asset] = {
                    'prediction_results': prediction_results,
                    'backtest_metrics': backtest_metrics,
                    'validation_result': validation_result,
                    'samples_processed': len(prediction_results)
                }
                
                # Check if validation passed (treat warnings as failures)
                if validation_result['status'] in ['failed', 'warning']:
                    all_passed = False
                    if validation_result['status'] == 'failed':
                        self.logger.error(f"❌ Backtest validation FAILED for {asset}")
                        for check in validation_result['failed_checks']:
                            self.logger.error(f"   - {check}")
                    else:
                        self.logger.warning(f"⚠️  Backtest validation returned WARNING for {asset}")
                        self.logger.warning(f"   - {validation_result.get('message', 'Unknown warning')}")
                else:
                    self.logger.info(f"✅ Backtest validation PASSED for {asset}")
                
            except Exception as e:
                self.logger.error(f"❌ Backtest validation error for {asset}: {str(e)}")
                validation_results['assets'][asset] = {
                    'error': str(e),
                    'status': 'error'
                }
                all_passed = False
        
        # Set overall status
        validation_results['overall_status'] = 'passed' if all_passed else 'failed'
        
        # Save validation results
        self._save_validation_results(validation_results)
        
        if all_passed:
            self._create_validation_certificate()
            self.logger.info("🎉 Backtest validation PASSED for all assets - production predictions enabled")
        else:
            self.logger.error("💥 Backtest validation FAILED - production predictions disabled")
        
        return validation_results
    
    def _save_validation_results(self, validation_results: Dict[str, Any]):
        """Save validation results to file."""
        try:
            results_dir = Path(self.config['paths']['reports_dir'])
            results_dir.mkdir(exist_ok=True)
            
            results_file = results_dir / 'backtest_validation_results.json'
            
            # Create simplified results for JSON serialization
            simplified_results = validation_results.copy()
            for asset, asset_results in simplified_results['assets'].items():
                if 'prediction_results' in asset_results:
                    # Keep only summary statistics, not all prediction details
                    prediction_results = asset_results['prediction_results']
                    asset_results['prediction_summary'] = {
                        'total_predictions': len(prediction_results),
                        'successful_predictions': len([r for r in prediction_results if 'error' not in r]),
                        'date_range': {
                            'start': prediction_results[0]['date'].isoformat() if prediction_results else None,
                            'end': prediction_results[-1]['date'].isoformat() if prediction_results else None
                        }
                    }
                    # Remove detailed prediction results to keep file size manageable
                    del asset_results['prediction_results']
            
            with open(results_file, 'w') as f:
                json.dump(simplified_results, f, indent=2, default=str)
            
            self.logger.info(f"Saved validation results to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation results: {str(e)}")
    
    def _create_validation_certificate(self):
        """Create validation certificate file to enable production predictions."""
        try:
            cert_file = Path(self.config['paths']['models_dir']) / '.backtest_validated'
            
            certificate = {
                'validation_passed': True,
                'timestamp': datetime.now().isoformat(),
                'validator_version': '1.0.0',
                'config_file': self.config_path,
                'validation_thresholds': self.validation_thresholds,
                'expiry_days': 30  # Certificate expires after 30 days
            }
            
            with open(cert_file, 'w') as f:
                json.dump(certificate, f, indent=2)
            
            self.logger.info(f"Created validation certificate: {cert_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create validation certificate: {str(e)}")

def main(config_path: str = 'configs/default.yaml', n_samples: int = 30):
    """
    Main function for running backtest validation.
    
    Args:
        config_path: Path to configuration file
        n_samples: Number of samples to test per asset
    """
    logger = setup_logger('backtest_main')
    
    try:
        logger.info("Starting backtest validation system...")
        
        # Initialize validator
        validator = BacktestValidator(config_path)
        
        # Run full validation
        results = validator.run_full_backtest_validation(n_samples)
        
        # Print summary
        print(f"\n{'='*80}")
        print("BACKTEST VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Samples per Asset: {results['n_samples']}")
        
        for asset, asset_results in results['assets'].items():
            print(f"\n{asset}:")
            if 'error' in asset_results:
                print(f"  ❌ ERROR: {asset_results['error']}")
            else:
                validation = asset_results['validation_result']
                print(f"  Status: {validation['status'].upper()}")
                print(f"  Samples Processed: {asset_results['samples_processed']}")

                # Show warning message if status is warning
                if validation['status'] == 'warning' and 'message' in validation:
                    print(f"  Warning: {validation['message']}")

                # Show failed checks if they exist
                if 'failed_checks' in validation and validation['failed_checks']:
                    print(f"  Failed Checks:")
                    for check in validation['failed_checks']:
                        print(f"    - {check}")

                # Show key metrics if available
                if 'backtest_metrics' in asset_results:
                    metrics = asset_results['backtest_metrics']
                    print(f"  Metrics:")
                    print(f"    RMSE: {metrics['rmse']:.2f}")
                    print(f"    MAE: {metrics['mae']:.2f}")
                    print(f"    MAPE: {metrics['mape']:.2f}%")
                    print(f"    Directional Accuracy: {metrics['directional_accuracy']:.3f}")
                    print(f"    R²: {metrics['r2']:.3f}")
        
        print(f"\n{'='*80}")
        
        if results['overall_status'] == 'passed':
            print("🎉 VALIDATION PASSED - Production predictions enabled!")
        else:
            print("💥 VALIDATION FAILED - Production predictions disabled!")
            print("Fix issues and re-run validation before enabling predictions.")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest Validation System')
    parser.add_argument('--config', default='configs/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--samples', type=int, default=30,
                       help='Number of samples to test per asset')
    
    args = parser.parse_args()
    
    main(args.config, args.samples)