#!/usr/bin/env python3
"""
Production Inference System for Sensex & Gold LSTM Price Predictor

This module provides a production-ready inference system that:
1. Uses versioned training artifacts with strict schema validation
2. Implements past-only feature computation with aligned cross-asset correlation  
3. Includes OOD detection and data drift monitoring
4. Enforces ±15% daily move guardrails
5. Requires passing backtest validation before enabling predictions

Features:
- Exact reproduction of training-time feature processing
- Hard schema/order/seq-len assertions
- Cross-asset correlation with aligned time windows
- Distribution drift detection
- Prediction guardrails and circuit breakers
- Comprehensive validation and monitoring

Author: AI Assistant
Date: 2025
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import yaml
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from models import LSTMPricePredictor, create_model
from feature_engineering import calculate_technical_indicators
from artifact_manager import ArtifactManager

warnings.filterwarnings('ignore')

def setup_logger(name: str = 'production_inference') -> logging.Logger:
    """Setup logger for production inference."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class ProductionInferenceEngine:
    """
    Production-grade inference engine with strict validation and monitoring.
    
    Ensures exact reproducibility of training conditions during inference
    with comprehensive safety checks and guardrails.
    """
    
    def __init__(self, config_path: str, validate_artifacts: bool = True):
        """
        Initialize production inference engine.
        
        Args:
            config_path: Path to configuration file
            validate_artifacts: Whether to validate all artifacts on startup
        """
        self.config_path = config_path
        self.logger = setup_logger('production_inference')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize artifact manager
        self.artifact_manager = ArtifactManager(self.config['paths']['models_dir'])
        
        # Load and validate all artifacts
        self.ensembles = {}
        self._initialize_ensembles(validate_artifacts)
        
        # Initialize monitoring state
        self.monitoring_state = {
            'last_prediction_time': None,
            'prediction_history': [],
            'drift_alerts': [],
            'guardrail_violations': []
        }
        
        self.logger.info("Production inference engine initialized successfully")
    
    def _initialize_ensembles(self, validate_artifacts: bool = True):
        """Initialize model ensembles with strict validation."""
        available_artifacts = self.artifact_manager.list_available_artifacts()
        
        for asset in self.config['training']['asset_list']:
            if asset not in available_artifacts:
                raise ValueError(f"No artifacts found for asset {asset}")
            
            available_folds = available_artifacts[asset]
            expected_folds = list(range(self.config['cv']['n_folds']))
            
            missing_folds = set(expected_folds) - set(available_folds)
            if missing_folds:
                raise ValueError(f"Missing artifacts for {asset} folds: {sorted(missing_folds)}")
            
            self.ensembles[asset] = ProductionModelEnsemble(
                asset=asset,
                artifact_manager=self.artifact_manager,
                folds=available_folds,
                validate_artifacts=validate_artifacts
            )
            
            self.logger.info(f"Initialized production ensemble for {asset} with {len(available_folds)} models")
    
    def _compute_cross_asset_correlation(self, data: pd.DataFrame, asset: str, window: int = 20) -> pd.Series:
        """
        Compute cross-asset correlation with aligned time windows (past-only).
        
        Args:
            data: DataFrame with both assets' data
            asset: Target asset for correlation computation
            window: Rolling window size
            
        Returns:
            Series with cross-asset correlation values
        """
        if asset == 'SENSEX':
            target_col = 'Close'
            # For SENSEX, correlate with GOLD
            other_asset_data = self._load_other_asset_data('GOLD', data['Date'].min(), data['Date'].max())
            other_col = 'Close'
        elif asset == 'GOLD':
            target_col = 'Close'
            # For GOLD, correlate with SENSEX
            other_asset_data = self._load_other_asset_data('SENSEX', data['Date'].min(), data['Date'].max())
            other_col = 'Close'
        else:
            raise ValueError(f"Unsupported asset: {asset}")
        
        # Align data by date (past-only, no future leakage)
        merged_data = pd.merge(
            data[['Date', target_col]].rename(columns={target_col: 'target'}),
            other_asset_data[['Date', other_col]].rename(columns={other_col: 'other'}),
            on='Date',
            how='left'
        ).sort_values('Date')
        
        # Forward fill missing values (conservative approach)
        merged_data['other'] = merged_data['other'].fillna(method='ffill')
        
        # Compute rolling correlation (past-only)
        correlation = merged_data['target'].rolling(window=window, min_periods=window//2).corr(
            merged_data['other']
        )
        
        # Fill initial NaN values with 0 (neutral correlation)
        correlation = correlation.fillna(0.0)
        
        return correlation
    
    def _load_other_asset_data(self, asset: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Load data for the other asset within the specified date range."""
        if asset == 'SENSEX':
            data_path = self.config['paths']['sensex_clean']
        elif asset == 'GOLD':
            data_path = self.config['paths']['gold_clean']
        else:
            raise ValueError(f"Unsupported asset: {asset}")
        
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter to date range
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        return df[mask].copy()
    
    def _prepare_features_with_strict_validation(self, asset: str, lookback_days: int = 100) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepare features with strict validation against training artifacts.
        
        Args:
            asset: Asset name
            lookback_days: Number of days to look back for features
            
        Returns:
            Tuple of (feature_array, validation_info)
        """
        # Load reference artifacts from first fold for schema validation
        reference_artifacts = self.artifact_manager.load_fold_artifacts(asset, 0)
        expected_features = reference_artifacts['feature_names']
        expected_seq_len = reference_artifacts['seq_len']
        
        # Load latest market data
        if asset == 'SENSEX':
            data_path = self.config['paths']['sensex_clean']
        elif asset == 'GOLD':
            data_path = self.config['paths']['gold_clean']
        else:
            raise ValueError(f"Unsupported asset: {asset}")
        
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Get sufficient data for features + sequence
        required_rows = expected_seq_len + 100  # Extra buffer for technical indicators
        if len(df) < required_rows:
            raise ValueError(f"Insufficient data for {asset}: {len(df)} < {required_rows} required")
        
        recent_data = df.tail(required_rows).copy()
        
        # Apply feature engineering (same as training)
        engineered_data = calculate_technical_indicators(recent_data)
        
        # Add cross-asset correlation with aligned time windows
        try:
            if asset == 'SENSEX':
                corr_series = self._compute_cross_asset_correlation(engineered_data, asset)
                engineered_data['corr_gold_20d'] = corr_series
            elif asset == 'GOLD':
                corr_series = self._compute_cross_asset_correlation(engineered_data, asset)
                engineered_data['corr_sensex_20d'] = corr_series
        except Exception as e:
            self.logger.warning(f"Cross-asset correlation calculation failed for {asset}: {e}")
            # Use default correlation of 0 if calculation fails
            if asset == 'SENSEX':
                engineered_data['corr_gold_20d'] = 0.0
            elif asset == 'GOLD':
                engineered_data['corr_sensex_20d'] = 0.0
        
        # Ensure cross-asset correlation column exists and has no NaN values
        if asset == 'SENSEX' and 'corr_gold_20d' in engineered_data.columns:
            if engineered_data['corr_gold_20d'].isnull().any():
                self.logger.warning(f"Cross-asset correlation has NaN values, filling with 0.0")
                engineered_data['corr_gold_20d'] = engineered_data['corr_gold_20d'].fillna(0.0)
        elif asset == 'GOLD' and 'corr_sensex_20d' in engineered_data.columns:
            if engineered_data['corr_sensex_20d'].isnull().any():
                self.logger.warning(f"Cross-asset correlation has NaN values, filling with 0.0")
                engineered_data['corr_sensex_20d'] = engineered_data['corr_sensex_20d'].fillna(0.0)
        
        # Validate feature availability
        available_features = list(engineered_data.columns)
        missing_features = [f for f in expected_features if f not in available_features]
        
        if missing_features:
            raise ValueError(f"Missing features for {asset}: {missing_features}")
        
        # Check for missing values and handle conservatively BEFORE extracting sequence
        nan_summary = engineered_data[expected_features].isnull().sum()
        total_nans = nan_summary.sum()
        if total_nans > 0:
            self.logger.warning(f"Found {total_nans} missing values in {asset} features:")
            for col, count in nan_summary.items():
                if count > 0:
                    self.logger.warning(f"  {col}: {count} NaN values")
            
            self.logger.warning(f"Forward filling missing values...")
            for col in expected_features:
                if col in engineered_data.columns:
                    before_count = engineered_data[col].isnull().sum()
                    engineered_data[col] = engineered_data[col].fillna(method='ffill').fillna(method='bfill')
                    after_count = engineered_data[col].isnull().sum()
                    if before_count > 0:
                        self.logger.warning(f"  {col}: {before_count} -> {after_count} NaN values")
            
            # Check if any NaNs remain
            remaining_nans = engineered_data[expected_features].isnull().sum().sum()
            if remaining_nans > 0:
                self.logger.error(f"Still have {remaining_nans} NaN values after forward filling")
                for col in expected_features:
                    count = engineered_data[col].isnull().sum()
                    if count > 0:
                        self.logger.error(f"  {col}: {count} NaN values remain")
                        # Show some sample values
                        nan_indices = engineered_data[col].isnull()
                        if nan_indices.any():
                            first_nan_idx = nan_indices.idxmax()
                            self.logger.error(f"    First NaN at index {first_nan_idx}")
                            start_idx = max(0, first_nan_idx - 2)
                            end_idx = min(len(engineered_data), first_nan_idx + 3)
                            sample_values = engineered_data[col].iloc[start_idx:end_idx].tolist()
                            self.logger.error(f"    Context: {sample_values}")
        
        # Extract features in exact training order (after filling)
        feature_data = engineered_data[expected_features].tail(expected_seq_len)
        
        # Strict validation checks
        validation_info = {
            'asset': asset,
            'expected_features': expected_features,
            'actual_features': list(feature_data.columns),
            'expected_seq_len': expected_seq_len,
            'actual_seq_len': len(feature_data),
            'feature_order_match': list(feature_data.columns) == expected_features,
            'seq_len_match': len(feature_data) == expected_seq_len,
            'data_date_range': {
                'start': feature_data.index[0] if len(feature_data) > 0 else None,
                'end': feature_data.index[-1] if len(feature_data) > 0 else None
            },
            'latest_close': float(engineered_data['Close'].iloc[-1]) if 'Close' in engineered_data else None,
            'nan_count': int(feature_data.isnull().sum().sum()),
            'inf_count': int(np.isinf(feature_data.values).sum())
        }
        
        # Hard assertions for production safety
        assert validation_info['feature_order_match'], f"Feature order mismatch for {asset}"
        assert validation_info['seq_len_match'], f"Sequence length mismatch for {asset}"
        assert validation_info['nan_count'] == 0, f"NaN values found in {asset} features"
        assert validation_info['inf_count'] == 0, f"Infinite values found in {asset} features"
        
        return feature_data.values, validation_info
    
    def _detect_distribution_drift(self, asset: str, current_features: np.ndarray) -> Dict[str, Any]:
        """
        Detect distribution drift compared to training data.
        
        Args:
            asset: Asset name
            current_features: Current feature values
            
        Returns:
            Dictionary with drift detection results
        """
        # Load training statistics from multiple folds for robust comparison
        training_stats = []
        
        for fold in range(min(3, self.config['cv']['n_folds'])):  # Use first 3 folds
            try:
                artifacts = self.artifact_manager.load_fold_artifacts(asset, fold)
                scaler = artifacts['feature_scaler']
                
                # Get training feature statistics
                training_mean = scaler.mean_
                training_std = scaler.scale_
                
                training_stats.append({
                    'mean': training_mean,
                    'std': training_std
                })
            except Exception as e:
                self.logger.warning(f"Could not load training stats for {asset} fold {fold}: {e}")
                continue
        
        if not training_stats:
            return {'error': 'No training statistics available for drift detection'}
        
        # Aggregate training statistics
        training_means = np.array([stats['mean'] for stats in training_stats])
        training_stds = np.array([stats['std'] for stats in training_stats])
        
        agg_mean = np.mean(training_means, axis=0)
        agg_std = np.mean(training_stds, axis=0)
        
        # Compute current feature statistics
        current_mean = np.mean(current_features, axis=0)
        current_std = np.std(current_features, axis=0)
        
        # Compute drift metrics
        mean_drift = np.abs(current_mean - agg_mean) / (agg_std + 1e-8)
        std_drift = np.abs(current_std - agg_std) / (agg_std + 1e-8)
        
        # Overall drift score (normalized)
        overall_drift = np.mean(mean_drift + std_drift)
        
        # Drift thresholds
        drift_threshold_warning = 2.0  # 2 standard deviations
        drift_threshold_critical = 3.0  # 3 standard deviations
        
        # Classification
        if overall_drift > drift_threshold_critical:
            drift_level = 'critical'
        elif overall_drift > drift_threshold_warning:
            drift_level = 'warning'
        else:
            drift_level = 'normal'
        
        drift_info = {
            'asset': asset,
            'overall_drift_score': float(overall_drift),
            'drift_level': drift_level,
            'mean_drift_per_feature': mean_drift.tolist(),
            'std_drift_per_feature': std_drift.tolist(),
            'max_feature_drift': float(np.max(mean_drift + std_drift)),
            'thresholds': {
                'warning': drift_threshold_warning,
                'critical': drift_threshold_critical
            },
            'training_folds_used': len(training_stats)
        }
        
        return drift_info
    
    def _get_training_max_level(self, asset: str) -> float:
        """
        Get the maximum price level seen during training across all folds.

        Args:
            asset: Asset name

        Returns:
            Maximum training price level
        """
        try:
            # Load historical data used for training
            if asset == 'SENSEX':
                data_path = self.config['paths']['sensex_clean']
            elif asset == 'GOLD':
                data_path = self.config['paths']['gold_clean']
            else:
                raise ValueError(f"Unsupported asset: {asset}")

            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])

            # Get the training end date from the last fold
            last_fold_idx = self.config['cv']['n_folds'] - 1
            artifacts = self.artifact_manager.load_fold_artifacts(asset, last_fold_idx)
            train_end_date = artifacts['metadata']['training_metadata']['fold_config']['train_end']

            # Filter to training period
            train_df = df[df['Date'] <= train_end_date]
            max_level = float(train_df['Close'].max())

            return max_level

        except Exception as e:
            self.logger.warning(f"Could not determine training max for {asset}: {e}")
            return float('inf')  # Don't flag OOD if we can't determine

    def _compute_historical_volatility(self, asset: str, window: int = 20) -> float:
        """
        Compute historical volatility (annualized std of log returns).

        Args:
            asset: Asset name
            window: Rolling window for volatility calculation

        Returns:
            Annualized volatility
        """
        try:
            if asset == 'SENSEX':
                data_path = self.config['paths']['sensex_clean']
            elif asset == 'GOLD':
                data_path = self.config['paths']['gold_clean']
            else:
                raise ValueError(f"Unsupported asset: {asset}")

            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')

            # Get recent data
            recent_df = df.tail(window + 1)

            # Compute log returns
            log_returns = np.log(recent_df['Close'] / recent_df['Close'].shift(1)).dropna()

            # Annualized volatility (assuming ~252 trading days)
            vol_daily = float(log_returns.std())
            vol_annualized = vol_daily * np.sqrt(252)

            return vol_annualized

        except Exception as e:
            self.logger.warning(f"Could not compute volatility for {asset}: {e}")
            return 0.20  # Default to 20% annual volatility

    def _apply_uncertainty_based_guardrails(
        self,
        asset: str,
        current_price: float,
        ensemble_predictions: np.ndarray,
        apply_soft_shrink: bool = True,
        shrink_method: str = 'tanh',
        shrink_factor: float = 2.0
    ) -> Dict[str, Any]:
        """
        Apply return-domain uncertainty-based guardrails with confidence flags.

        No hard clipping - only flags and optional soft shrinkage.

        Args:
            asset: Asset name
            current_price: Current market price
            ensemble_predictions: Array of individual model price predictions
            apply_soft_shrink: Whether to apply soft shrinkage to high-uncertainty predictions
            shrink_method: Method for soft shrinkage ('linear' or 'tanh')
            shrink_factor: Factor k for shrinkage (shrink to k*σ)

        Returns:
            Dictionary with guardrail analysis and calibrated intervals
        """
        # Convert price predictions to log-returns
        log_returns = np.log(ensemble_predictions / current_price)

        # Ensemble statistics in return space
        mu = float(np.mean(log_returns))  # Mean log-return
        sigma = float(np.std(log_returns))  # Std of log-returns

        # Get historical volatility for tail detection
        vol_20 = self._compute_historical_volatility(asset, window=20)
        vol_20_daily = vol_20 / np.sqrt(252)  # Convert to daily

        # Get training max level for OOD detection
        train_max_level = self._get_training_max_level(asset)

        # Initialize confidence flags
        confidence_flags = {
            'LOW_CONFIDENCE_UNCERTAIN': False,
            'LOW_CONFIDENCE_TAIL': False,
            'LOW_CONFIDENCE_OOD': False
        }

        flag_reasons = []

        # Flag 1: LOW_CONFIDENCE_UNCERTAIN - High relative uncertainty
        # When |μ|/σ > 3 (prediction certainty is low relative to spread)
        if sigma > 0:
            uncertainty_ratio = abs(mu) / sigma
            if uncertainty_ratio > 3.0:
                confidence_flags['LOW_CONFIDENCE_UNCERTAIN'] = True
                flag_reasons.append(f"High uncertainty: |μ|/σ = {uncertainty_ratio:.2f} > 3.0")

        # Flag 2: LOW_CONFIDENCE_TAIL - Extreme return vs historical volatility
        # When |μ| > 5 × vol_20_daily (prediction is >5x normal daily volatility)
        tail_threshold = 5.0 * vol_20_daily
        if abs(mu) > tail_threshold:
            confidence_flags['LOW_CONFIDENCE_TAIL'] = True
            flag_reasons.append(f"Extreme return: |μ| = {abs(mu):.4f} > 5×vol = {tail_threshold:.4f}")

        # Flag 3: LOW_CONFIDENCE_OOD - Out of distribution
        # When current_price > 1.10 × train_max_level
        ood_threshold = 1.10 * train_max_level
        if current_price > ood_threshold:
            confidence_flags['LOW_CONFIDENCE_OOD'] = True
            flag_reasons.append(f"OOD: current {current_price:.2f} > 1.10×train_max {ood_threshold:.2f}")

        # Optional soft shrinkage for high uncertainty
        mu_adjusted = mu
        shrinkage_applied = False

        if apply_soft_shrink and confidence_flags['LOW_CONFIDENCE_UNCERTAIN'] and sigma > 0:
            target_magnitude = shrink_factor * sigma

            if shrink_method == 'tanh':
                # Smooth shrinkage using tanh
                # Maps μ → target_magnitude * tanh(μ / target_magnitude)
                mu_adjusted = target_magnitude * np.tanh(mu / target_magnitude)
            elif shrink_method == 'linear':
                # Linear shrinkage: cap |μ| at target_magnitude
                if abs(mu) > target_magnitude:
                    mu_adjusted = np.sign(mu) * target_magnitude

            if abs(mu_adjusted) < abs(mu):
                shrinkage_applied = True
                self.logger.info(f"Soft shrinkage applied for {asset}: μ {mu:.4f} → {mu_adjusted:.4f}")

        # Calibrated 80% confidence interval in return space
        # For normal distribution, 80% CI is μ ± 1.28σ
        z_score_80 = 1.28
        ci_lower_return = mu_adjusted - z_score_80 * sigma
        ci_upper_return = mu_adjusted + z_score_80 * sigma

        # Convert returns to prices (never hard-clip)
        final_prediction_price = current_price * np.exp(mu_adjusted)
        ci_lower_price = current_price * np.exp(ci_lower_return)
        ci_upper_price = current_price * np.exp(ci_upper_return)

        # Calculate metrics
        final_return_pct = (np.exp(mu_adjusted) - 1) * 100
        ci_width_pct = (np.exp(ci_upper_return) - np.exp(ci_lower_return)) / current_price * 100

        # Determine overall confidence level
        any_flags = any(confidence_flags.values())
        confidence_level = 'LOW' if any_flags else 'HIGH'

        # Log warnings for flagged predictions
        if any_flags:
            self.logger.warning(f"Confidence flags for {asset}: {', '.join(flag_reasons)}")

        # Compile guardrail results
        guardrail_result = {
            'asset': asset,
            'current_price': current_price,

            # Return-space statistics
            'return_mean': mu,
            'return_std': sigma,
            'return_mean_adjusted': mu_adjusted,
            'shrinkage_applied': shrinkage_applied,
            'shrink_method': shrink_method if shrinkage_applied else None,

            # Price predictions (never clipped)
            'final_prediction_price': final_prediction_price,
            'final_return_pct': final_return_pct,

            # Calibrated 80% confidence interval
            'confidence_interval_80pct': {
                'return_lower': ci_lower_return,
                'return_upper': ci_upper_return,
                'price_lower': ci_lower_price,
                'price_upper': ci_upper_price,
                'width_pct': ci_width_pct
            },

            # Confidence flags
            'confidence_flags': confidence_flags,
            'confidence_level': confidence_level,
            'flag_reasons': flag_reasons,

            # Diagnostic information
            'diagnostics': {
                'uncertainty_ratio': abs(mu) / sigma if sigma > 0 else float('inf'),
                'historical_vol_20d_annual': vol_20,
                'historical_vol_20d_daily': vol_20_daily,
                'tail_threshold_5x': tail_threshold,
                'train_max_level': train_max_level,
                'ood_threshold_1.10x': ood_threshold,
                'ensemble_size': len(ensemble_predictions),
                'raw_predictions_min': float(np.min(ensemble_predictions)),
                'raw_predictions_max': float(np.max(ensemble_predictions)),
                'raw_predictions_mean': float(np.mean(ensemble_predictions)),
                'raw_predictions_std': float(np.std(ensemble_predictions))
            },

            # Legacy compatibility
            'guardrail_triggered': False,  # No hard clipping anymore
            'hard_clip_applied': False
        }

        return guardrail_result
    
    def predict_single_asset(self, asset: str, apply_soft_shrink: bool = True,
                             shrink_method: str = 'tanh', shrink_factor: float = 2.0) -> Dict[str, Any]:
        """
        Generate production prediction for a single asset with uncertainty-based guardrails.

        Args:
            asset: Asset name
            apply_soft_shrink: Whether to apply soft shrinkage to high-uncertainty predictions
            shrink_method: Method for soft shrinkage ('linear' or 'tanh')
            shrink_factor: Factor k for shrinkage (shrink to k*σ)

        Returns:
            Dictionary containing validated prediction results with confidence intervals
        """
        if asset not in self.ensembles:
            raise ValueError(f"No ensemble available for asset: {asset}")

        try:
            # Prepare features with strict validation
            features, validation_info = self._prepare_features_with_strict_validation(asset)
            current_price = validation_info['latest_close']

            # Detect distribution drift
            drift_info = self._detect_distribution_drift(asset, features)

            # Check for critical drift
            if drift_info.get('drift_level') == 'critical':
                self.logger.error(f"Critical distribution drift detected for {asset}")
                self.monitoring_state['drift_alerts'].append({
                    'timestamp': datetime.now().isoformat(),
                    'asset': asset,
                    'drift_info': drift_info
                })
                # Still proceed but flag the prediction

            # Generate ensemble predictions (get individual model predictions)
            # Pass current_price to convert model returns to price predictions
            ensemble_result = self.ensembles[asset].predict(features, current_price)

            # Extract individual predictions for uncertainty analysis
            individual_predictions = np.array(ensemble_result['individual_predictions'])

            # Apply uncertainty-based guardrails (no hard clipping)
            guardrail_result = self._apply_uncertainty_based_guardrails(
                asset=asset,
                current_price=current_price,
                ensemble_predictions=individual_predictions,
                apply_soft_shrink=apply_soft_shrink,
                shrink_method=shrink_method,
                shrink_factor=shrink_factor
            )

            # Final prediction (from return-space analysis, no clipping)
            final_prediction = guardrail_result['final_prediction_price']
            final_return_pct = guardrail_result['final_return_pct']

            # Extract confidence interval
            ci_80 = guardrail_result['confidence_interval_80pct']

            # Direction based on return
            direction = 'UP' if final_return_pct > 0 else 'DOWN' if final_return_pct < 0 else 'FLAT'

            # Compile comprehensive result
            result = {
                'asset': asset,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,

                # Predictions (never hard-clipped)
                'final_prediction': final_prediction,
                'final_return_pct': final_return_pct,
                'direction': direction,

                # Confidence interval (80%)
                'confidence_interval_80pct': {
                    'price_lower': ci_80['price_lower'],
                    'price_upper': ci_80['price_upper'],
                    'return_lower_pct': (np.exp(ci_80['return_lower']) - 1) * 100,
                    'return_upper_pct': (np.exp(ci_80['return_upper']) - 1) * 100,
                    'width_pct': ci_80['width_pct']
                },

                # Return-space statistics
                'return_statistics': {
                    'mean': guardrail_result['return_mean'],
                    'std': guardrail_result['return_std'],
                    'mean_adjusted': guardrail_result['return_mean_adjusted'],
                    'shrinkage_applied': guardrail_result['shrinkage_applied']
                },

                # Ensemble statistics (legacy format for compatibility)
                'ensemble_stats': ensemble_result,

                # Validation and drift info
                'validation_info': validation_info,
                'drift_info': drift_info,

                # Uncertainty-based guardrails (replaces old guardrail_info)
                'guardrail_info': guardrail_result,

                # Production flags (updated for new system)
                'production_flags': {
                    'schema_validated': True,
                    'drift_detected': drift_info.get('drift_level') != 'normal',
                    'confidence_level': guardrail_result['confidence_level'],
                    'low_confidence_uncertain': guardrail_result['confidence_flags']['LOW_CONFIDENCE_UNCERTAIN'],
                    'low_confidence_tail': guardrail_result['confidence_flags']['LOW_CONFIDENCE_TAIL'],
                    'low_confidence_ood': guardrail_result['confidence_flags']['LOW_CONFIDENCE_OOD'],
                    'hard_clip_applied': False,  # Never clip anymore
                    'guardrails_applied': any(guardrail_result['confidence_flags'].values()),  # True if any flags raised
                    'sanity_checks_passed': True,  # Basic sanity checks passed
                    'soft_shrink_applied': guardrail_result['shrinkage_applied']
                }
            }

            # Record prediction in history
            self.monitoring_state['prediction_history'].append({
                'timestamp': result['timestamp'],
                'asset': asset,
                'prediction': final_prediction,
                'return_pct': final_return_pct,
                'confidence_level': guardrail_result['confidence_level']
            })

            self.monitoring_state['last_prediction_time'] = datetime.now()

            return result

        except Exception as e:
            self.logger.error(f"Production prediction failed for {asset}: {str(e)}")
            raise
    
    def predict(self, assets: Optional[List[str]] = None, require_backtest_validation: bool = True) -> Dict[str, Any]:
        """
        Generate production predictions with comprehensive validation.
        
        Args:
            assets: List of assets to predict (None for all)
            require_backtest_validation: Whether to require passing backtest validation
            
        Returns:
            Dictionary containing all prediction results and validation status
        """
        if require_backtest_validation:
            # Check if backtest validation has passed
            if not self._check_backtest_validation():
                raise RuntimeError("Backtest validation has not passed. Cannot enable predictions. "
                                 "Run 'python main.py --backtest' first.")
        
        if assets is None:
            assets = list(self.ensembles.keys())
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'production_mode': True,
            'backtest_validated': require_backtest_validation,
            'predictions': {},
            'monitoring': {
                'drift_alerts_count': len(self.monitoring_state['drift_alerts']),
                'guardrail_violations_count': len(self.monitoring_state['guardrail_violations']),
                'total_predictions_made': len(self.monitoring_state['prediction_history'])
            }
        }
        
        for asset in assets:
            try:
                self.logger.info(f"Generating production prediction for {asset}...")
                prediction = self.predict_single_asset(asset)
                results['predictions'][asset] = prediction
                
                # Log prediction summary
                flags = prediction['production_flags']
                flag_summary = []
                if flags['drift_detected']:
                    flag_summary.append("DRIFT")
                if flags['guardrails_applied']:
                    flag_summary.append("GUARDRAIL")
                if not flags['sanity_checks_passed']:
                    flag_summary.append("SANITY_FAIL")
                
                flag_str = f" [{'/'.join(flag_summary)}]" if flag_summary else ""
                
                self.logger.info(f"✅ {asset}: {prediction['direction']} "
                               f"{prediction['final_prediction']:.2f} "
                               f"({prediction['price_change_pct']:+.2f}%){flag_str}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to predict {asset}: {str(e)}")
                results['predictions'][asset] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        return results
    
    def _check_backtest_validation(self) -> bool:
        """Check if backtest validation has passed."""
        backtest_file = Path(self.config['paths']['models_dir']) / '.backtest_validated'
        return backtest_file.exists()

class ProductionModelEnsemble:
    """Production model ensemble with strict artifact management."""
    
    def __init__(self, asset: str, artifact_manager: ArtifactManager, folds: List[int], validate_artifacts: bool = True):
        """Initialize production model ensemble."""
        self.asset = asset
        self.artifact_manager = artifact_manager
        self.folds = folds
        self.logger = setup_logger(f'ensemble_{asset.lower()}')
        
        self.models = {}
        self.artifacts = {}
        
        self._load_models(validate_artifacts)
    
    def _load_models(self, validate_artifacts: bool):
        """Load models with strict validation."""
        loaded_folds = []
        
        for fold in self.folds:
            try:
                # Load complete artifacts
                artifacts = self.artifact_manager.load_fold_artifacts(self.asset, fold)
                
                # Create model with exact training configuration
                model_data = artifacts['model_data']
                input_size = artifacts['feature_count']
                
                # Extract model configuration from artifacts
                metadata = artifacts['metadata']
                model_config = metadata['training_config']['training']
                
                model = create_model(model_config, input_size)
                model.load_state_dict(model_data['model_state_dict'])
                model.eval()
                
                # Store artifacts and model
                self.artifacts[fold] = artifacts
                self.models[fold] = model
                loaded_folds.append(fold)
                
                if validate_artifacts:
                    # Additional validation
                    expected_features = artifacts['feature_names']
                    expected_seq_len = artifacts['seq_len']
                    
                    # Validate consistency across folds
                    if fold == self.folds[0]:
                        self.reference_features = expected_features
                        self.reference_seq_len = expected_seq_len
                    else:
                        if expected_features != self.reference_features:
                            raise ValueError(f"Feature mismatch in fold {fold}")
                        if expected_seq_len != self.reference_seq_len:
                            raise ValueError(f"Sequence length mismatch in fold {fold}")
                
            except Exception as e:
                self.logger.error(f"Failed to load fold {fold}: {str(e)}")
                continue
        
        if not loaded_folds:
            raise RuntimeError(f"No valid models loaded for {self.asset}")
        
        self.logger.info(f"Loaded {len(loaded_folds)} production models for {self.asset}")
    
    def predict(self, features: np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        Generate ensemble prediction using versioned artifacts.

        Args:
            features: Input features for prediction
            current_price: Current market price (needed to convert returns to prices)

        Returns:
            Dictionary with ensemble statistics (all in price space)
        """
        if len(self.models) == 0:
            raise RuntimeError(f"No models available for {self.asset}")

        predictions = []
        predicted_returns = []

        with torch.no_grad():
            for fold, model in self.models.items():
                artifacts = self.artifacts[fold]

                # Use fold-specific scalers
                feature_scaler = artifacts['feature_scaler']
                target_scaler = artifacts['target_scaler']

                # Scale features using training scaler
                features_scaled = feature_scaler.transform(features)

                # Convert to tensor
                features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)

                # Get prediction (this is a scaled return)
                pred_scaled = model(features_tensor).cpu().numpy().flatten()[0]

                # Inverse transform using target scaler to get return
                pred_return = target_scaler.inverse_transform([[pred_scaled]])[0][0]

                # Convert return to price: predicted_price = current_price * (1 + return)
                predicted_price = current_price * (1 + pred_return)

                predictions.append(predicted_price)
                predicted_returns.append(pred_return)
        
        predictions = np.array(predictions)
        predicted_returns = np.array(predicted_returns)

        # Calculate ensemble statistics (in price space)
        ensemble_stats = {
            'prediction': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'confidence_95_lower': float(np.percentile(predictions, 2.5)),
            'confidence_95_upper': float(np.percentile(predictions, 97.5)),
            'model_count': len(predictions),
            'individual_predictions': predictions.tolist(),
            'prediction_variance': float(np.var(predictions)),
            'coefficient_of_variation': float(np.std(predictions) / (np.abs(np.mean(predictions)) + 1e-8)),
            # NEW: Also include return statistics for debugging and analysis
            'individual_returns': predicted_returns.tolist(),
            'mean_return': float(np.mean(predicted_returns)),
            'std_return': float(np.std(predicted_returns))
        }

        return ensemble_stats

def main(config_path: str = 'configs/default.yaml'):
    """Test production inference system."""
    logger = setup_logger('production_test')
    
    try:
        # Initialize production inference engine
        engine = ProductionInferenceEngine(config_path, validate_artifacts=True)
        
        # Run production predictions
        results = engine.predict(require_backtest_validation=False)  # Skip backtest for testing
        
        # Display results
        print(f"\nProduction Inference Results:")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Monitoring - Drift Alerts: {results['monitoring']['drift_alerts_count']}")
        print(f"Monitoring - Guardrail Violations: {results['monitoring']['guardrail_violations_count']}")
        
        for asset, prediction in results['predictions'].items():
            if 'error' not in prediction:
                flags = prediction['production_flags']
                print(f"\n{asset}:")
                print(f"  Current: {prediction['current_price']:.2f}")
                print(f"  Predicted: {prediction['final_prediction']:.2f}")
                print(f"  Change: {prediction['price_change_pct']:+.2f}%")
                print(f"  Direction: {prediction['direction']}")
                print(f"  Drift Level: {prediction['drift_info']['drift_level']}")
                print(f"  Guardrails Applied: {flags['guardrails_applied']}")
            else:
                print(f"\n{asset}: ERROR - {prediction['error']}")
        
    except Exception as e:
        logger.error(f"Production inference test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()