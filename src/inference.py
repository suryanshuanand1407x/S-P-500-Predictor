#!/usr/bin/env python3
"""
Inference Pipeline for Sensex & Gold LSTM Price Predictor

This module provides the complete inference system for making predictions
using the trained LSTM models with ensemble averaging and uncertainty quantification.

Key Components:
1. ModelEnsemble - Manages loading and prediction with multiple trained models
2. InferenceEngine - Orchestrates the complete prediction pipeline
3. Data preprocessing pipeline for inference
4. Uncertainty quantification and confidence intervals

Usage:
    from inference import InferenceEngine
    
    engine = InferenceEngine('configs/default.yaml')
    predictions = engine.predict(['SENSEX', 'GOLD'])
    
Author: AI Assistant
Date: 2025
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import yaml
import warnings

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from models import LSTMPricePredictor, create_model
from feature_engineering import calculate_technical_indicators
# from data_ingestion import DataLoader  # Not using DataLoader class, just direct CSV loading

warnings.filterwarnings('ignore')

def setup_logger(name: str = 'inference') -> logging.Logger:
    """Setup logger for inference pipeline."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class ModelEnsemble:
    """
    Manages ensemble of trained LSTM models for a single asset.
    
    Handles loading multiple fold models and generating ensemble predictions
    with uncertainty quantification.
    """
    
    def __init__(self, asset: str, models_dir: str, config: Dict[str, Any]):
        """
        Initialize model ensemble for a specific asset.
        
        Args:
            asset: Asset name (SENSEX or GOLD)
            models_dir: Directory containing trained models
            config: Configuration dictionary
        """
        self.asset = asset
        self.models_dir = Path(models_dir)
        self.config = config
        self.logger = setup_logger(f'ensemble_{asset.lower()}')
        
        self.models = {}
        self.scalers = {}
        self.device = torch.device(config['training']['device'])
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models and scalers for the asset."""
        asset_dir = self.models_dir / self.asset
        
        if not asset_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {asset_dir}")
        
        n_folds = self.config['cv']['n_folds']
        loaded_folds = []
        
        for fold in range(n_folds):
            fold_dir = asset_dir / f'fold_{fold}'
            model_path = fold_dir / 'best.ckpt'
            scaler_path = fold_dir / 'scaler.pkl'
            
            if model_path.exists() and scaler_path.exists():
                try:
                    # Load model with correct input size (13 features: 12 config + 1 Close)
                    input_size = len(self.config['training']['features']) + 1  # +1 for Close as target feature
                    model = create_model(self.config['training'], input_size)
                    
                    checkpoint = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.device)
                    model.eval()
                    
                    # Load scaler
                    with open(scaler_path, 'rb') as f:
                        scaler_data = pickle.load(f)
                    
                    self.models[fold] = model
                    self.scalers[fold] = scaler_data
                    loaded_folds.append(fold)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load fold {fold}: {str(e)}")
                    continue
        
        if not loaded_folds:
            raise RuntimeError(f"No valid models found for {self.asset}")
        
        self.logger.info(f"Loaded {len(loaded_folds)} models for {self.asset}: folds {loaded_folds}")
    
    def predict(self, input_data: np.ndarray) -> Dict[str, float]:
        """
        Generate ensemble prediction for input data.
        
        Args:
            input_data: Preprocessed input features of shape (seq_len, n_features)
            
        Returns:
            Dictionary containing prediction statistics
        """
        if len(self.models) == 0:
            raise RuntimeError(f"No models loaded for {self.asset}")
        
        predictions = []
        
        # Get predictions from all models
        with torch.no_grad():
            for fold, model in self.models.items():
                scaler_data = self.scalers[fold]
                
                # Scale input data
                input_scaled = scaler_data['feature_scaler'].transform(input_data)
                
                # Convert to tensor and add batch dimension
                input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(self.device)
                
                # Get prediction
                pred_scaled = model(input_tensor).cpu().numpy().flatten()[0]
                
                # Inverse scale prediction
                pred_unscaled = scaler_data['target_scaler'].inverse_transform([[pred_scaled]])[0][0]
                
                predictions.append(pred_unscaled)
        
        predictions = np.array(predictions)
        
        # Calculate ensemble statistics
        ensemble_stats = {
            'prediction': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'confidence_95_lower': float(np.percentile(predictions, 2.5)),
            'confidence_95_upper': float(np.percentile(predictions, 97.5)),
            'model_count': len(predictions),
            'individual_predictions': predictions.tolist()
        }
        
        return ensemble_stats

class InferenceEngine:
    """
    Main inference engine that orchestrates the complete prediction pipeline.
    
    Handles data fetching, preprocessing, model ensemble predictions, and
    result formatting for both SENSEX and GOLD assets.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the inference engine.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.logger = setup_logger('inference_engine')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # No need for DataLoader class, we'll load data directly
        
        # Initialize model ensembles
        self.ensembles = {}
        self._initialize_ensembles()
        
        self.logger.info("Inference engine initialized successfully")
    
    def _initialize_ensembles(self):
        """Initialize model ensembles for all assets."""
        for asset in self.config['training']['asset_list']:
            try:
                self.ensembles[asset] = ModelEnsemble(
                    asset=asset,
                    models_dir=self.config['paths']['models_dir'],
                    config=self.config
                )
                self.logger.info(f"Initialized ensemble for {asset}")
            except Exception as e:
                self.logger.error(f"Failed to initialize ensemble for {asset}: {str(e)}")
                raise
    
    def _get_latest_data(self, asset: str, lookback_days: int = 100) -> pd.DataFrame:
        """
        Get the latest market data for an asset.
        
        Args:
            asset: Asset name (SENSEX or GOLD)
            lookback_days: Number of days to look back for data
            
        Returns:
            DataFrame with latest market data
        """
        if asset == 'SENSEX':
            data_path = self.config['paths']['sensex_clean']
        elif asset == 'GOLD':
            data_path = self.config['paths']['gold_clean']
        else:
            raise ValueError(f"Unsupported asset: {asset}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Processed data not found: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Get recent data
        latest_data = df.tail(lookback_days).copy()
        
        return latest_data
    
    def _prepare_inference_data(self, asset: str) -> Tuple[np.ndarray, pd.Series]:
        """
        Prepare data for inference by applying the same preprocessing as training.
        
        Args:
            asset: Asset name
            
        Returns:
            Tuple of (features_array, latest_row_info)
        """
        # Get latest data
        data = self._get_latest_data(asset)
        
        if len(data) < self.config['training']['seq_len']:
            raise ValueError(f"Insufficient data for {asset}. Need at least {self.config['training']['seq_len']} days")
        
        # Apply feature engineering (same as training)
        engineered_data = calculate_technical_indicators(data)
        
        # Select required features (same as training: config features + Close as target feature)
        feature_cols = self.config['training']['features'].copy()
        
        # Add cross-asset correlation if needed (same as training)
        if asset == 'SENSEX' and 'corr_gold_20d' not in engineered_data.columns:
            # For inference, we can't compute cross-correlation easily, so use a default value
            engineered_data['corr_gold_20d'] = 0.0
        elif asset == 'GOLD' and 'corr_sensex_20d' not in engineered_data.columns:
            engineered_data['corr_sensex_20d'] = 0.0
        
        # Add cross-asset feature to feature columns (same as training)
        if asset == 'SENSEX':
            feature_cols.append('corr_gold_20d')
        elif asset == 'GOLD':
            feature_cols.append('corr_sensex_20d')
        
        # Remove 'Close' from feature_cols if present (to avoid duplication)
        if 'Close' in feature_cols:
            feature_cols.remove('Close')
        
        # Add Close as the final feature (same as training: X_cols = feature_cols + [target_column])
        final_feature_cols = feature_cols + ['Close']
        
        # Ensure all required features exist
        missing_features = [col for col in final_feature_cols if col not in engineered_data.columns]
        if missing_features:
            raise ValueError(f"Missing features for {asset}: {missing_features}")
        
        # Get the sequence for prediction
        seq_len = self.config['training']['seq_len']
        features_data = engineered_data[final_feature_cols].tail(seq_len)
        
        # Check for missing values
        if features_data.isnull().any().any():
            self.logger.warning(f"Found missing values in {asset} features, forward filling...")
            features_data = features_data.fillna(method='ffill').fillna(method='bfill')
        
        latest_info = engineered_data.iloc[-1]
        
        return features_data.values, latest_info
    
    def predict_single_asset(self, asset: str) -> Dict[str, Any]:
        """
        Generate prediction for a single asset.
        
        Args:
            asset: Asset name (SENSEX or GOLD)
            
        Returns:
            Dictionary containing prediction results
        """
        if asset not in self.ensembles:
            raise ValueError(f"No ensemble available for asset: {asset}")
        
        try:
            # Prepare input data
            input_data, latest_info = self._prepare_inference_data(asset)
            
            # Generate ensemble prediction
            ensemble_prediction = self.ensembles[asset].predict(input_data)
            
            # Calculate additional metrics
            latest_close = float(latest_info['Close'])
            predicted_close = ensemble_prediction['prediction']
            
            price_change = predicted_close - latest_close
            price_change_pct = (price_change / latest_close) * 100
            
            # Determine direction
            direction = 'UP' if price_change > 0 else 'DOWN' if price_change < 0 else 'FLAT'
            direction_confidence = abs(price_change_pct)
            
            # Create result dictionary
            result = {
                'asset': asset,
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'target_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'current_price': latest_close,
                'predicted_price': predicted_close,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'direction': direction,
                'direction_confidence': direction_confidence,
                'ensemble_stats': ensemble_prediction,
                'data_as_of': latest_info['Date'].strftime('%Y-%m-%d') if pd.notnull(latest_info['Date']) else 'Unknown',
                'sequence_length': self.config['training']['seq_len'],
                'model_count': ensemble_prediction['model_count']
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {asset}: {str(e)}")
            raise
    
    def predict(self, assets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate predictions for specified assets.
        
        Args:
            assets: List of asset names. If None, predict all available assets.
            
        Returns:
            Dictionary containing predictions for all requested assets
        """
        if assets is None:
            assets = list(self.ensembles.keys())
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'summary': {}
        }
        
        successful_predictions = 0
        total_confidence = 0
        
        for asset in assets:
            try:
                self.logger.info(f"Generating prediction for {asset}...")
                prediction = self.predict_single_asset(asset)
                results['predictions'][asset] = prediction
                
                successful_predictions += 1
                total_confidence += prediction['direction_confidence']
                
                self.logger.info(f"✅ {asset}: {prediction['direction']} "
                               f"{prediction['predicted_price']:.2f} "
                               f"({prediction['price_change_pct']:+.2f}%)")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to predict {asset}: {str(e)}")
                results['predictions'][asset] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Calculate summary statistics
        if successful_predictions > 0:
            avg_confidence = total_confidence / successful_predictions
            results['summary'] = {
                'successful_predictions': successful_predictions,
                'failed_predictions': len(assets) - successful_predictions,
                'average_confidence': avg_confidence,
                'status': 'completed'
            }
        else:
            results['summary'] = {
                'successful_predictions': 0,
                'failed_predictions': len(assets),
                'status': 'all_failed'
            }
        
        return results
    
    def predict_with_scenarios(self, asset: str, scenarios: List[float]) -> Dict[str, Any]:
        """
        Generate predictions under different market scenarios.
        
        Args:
            asset: Asset name
            scenarios: List of percentage changes to apply to latest price
            
        Returns:
            Dictionary containing scenario-based predictions
        """
        base_prediction = self.predict_single_asset(asset)
        
        scenario_results = []
        
        for scenario_pct in scenarios:
            # Modify the latest price by the scenario percentage
            original_data, latest_info = self._prepare_inference_data(asset)
            modified_data = original_data.copy()
            
            # Apply scenario to the last price data point
            close_idx = self.config['training']['features'].index('Close')
            scenario_multiplier = 1 + (scenario_pct / 100)
            modified_data[-1, close_idx] *= scenario_multiplier
            
            # Generate prediction with modified data
            ensemble_prediction = self.ensembles[asset].predict(modified_data)
            
            scenario_results.append({
                'scenario_pct': scenario_pct,
                'modified_price': modified_data[-1, close_idx],
                'predicted_price': ensemble_prediction['prediction'],
                'ensemble_stats': ensemble_prediction
            })
        
        return {
            'asset': asset,
            'base_prediction': base_prediction,
            'scenario_analysis': scenario_results,
            'timestamp': datetime.now().isoformat()
        }

def main(config_path: str = 'configs/default.yaml'):
    """
    Main function for running inference pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    logger = setup_logger('main_inference')
    
    try:
        logger.info("Starting inference pipeline...")
        
        # Initialize inference engine
        engine = InferenceEngine(config_path)
        
        # Generate predictions for all assets
        results = engine.predict()
        
        # Print results
        print("\n" + "="*80)
        print("🔮 SENSEX & GOLD PRICE PREDICTIONS")
        print("="*80)
        print(f"⏰ Generated: {results['timestamp']}")
        print(f"📊 Success Rate: {results['summary']['successful_predictions']}/{len(results['predictions'])}")
        
        for asset, prediction in results['predictions'].items():
            if 'error' not in prediction:
                print(f"\n📈 {asset}:")
                print(f"   Current Price: {prediction['current_price']:.2f}")
                print(f"   Predicted Price: {prediction['predicted_price']:.2f}")
                print(f"   Change: {prediction['price_change']:+.2f} ({prediction['price_change_pct']:+.2f}%)")
                print(f"   Direction: {prediction['direction']} (confidence: {prediction['direction_confidence']:.2f}%)")
                print(f"   95% CI: [{prediction['ensemble_stats']['confidence_95_lower']:.2f}, {prediction['ensemble_stats']['confidence_95_upper']:.2f}]")
                print(f"   Models Used: {prediction['model_count']}")
                print(f"   Data As Of: {prediction['data_as_of']}")
            else:
                print(f"\n❌ {asset}: {prediction['error']}")
        
        print("\n" + "="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"Inference pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM Price Prediction Inference')
    parser.add_argument('--config', default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--asset', choices=['SENSEX', 'GOLD', 'ALL'], default='ALL',
                       help='Asset to predict')
    parser.add_argument('--scenarios', nargs='*', type=float,
                       help='Scenario analysis with percentage changes')
    
    args = parser.parse_args()
    
    if args.scenarios:
        # Run scenario analysis
        engine = InferenceEngine(args.config)
        asset = args.asset if args.asset != 'ALL' else 'SENSEX'
        scenario_results = engine.predict_with_scenarios(asset, args.scenarios)
        
        print(f"\n🎯 SCENARIO ANALYSIS FOR {asset}")
        print("="*50)
        print(f"Base Prediction: {scenario_results['base_prediction']['predicted_price']:.2f}")
        
        for scenario in scenario_results['scenario_analysis']:
            print(f"Scenario {scenario['scenario_pct']:+.1f}%: {scenario['predicted_price']:.2f}")
    else:
        # Run standard inference
        assets = None if args.asset == 'ALL' else [args.asset]
        engine = InferenceEngine(args.config)
        
        if assets:
            results = engine.predict(assets)
        else:
            results = main(args.config)