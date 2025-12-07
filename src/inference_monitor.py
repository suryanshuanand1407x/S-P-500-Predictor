#!/usr/bin/env python3
"""
Inference Monitoring and Diagnostics for Sensex & Gold LSTM Price Predictor

This module provides monitoring, diagnostics, and validation tools for the inference pipeline.
It helps debug prediction issues, validate data quality, and ensure model outputs are reasonable.

Usage:
    from inference_monitor import InferenceMonitor
    
    monitor = InferenceMonitor('configs/default.yaml')
    monitor.validate_inference_data('SENSEX')
    monitor.analyze_prediction_anomalies(predictions)
    
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
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from feature_engineering import calculate_technical_indicators

def setup_logger(name: str = 'inference_monitor') -> logging.Logger:
    """Setup logger for inference monitoring."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class InferenceMonitor:
    """
    Monitoring and diagnostic tools for inference pipeline.
    
    Provides validation, quality checks, and anomaly detection for
    inference data and predictions.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the inference monitor.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.logger = setup_logger('inference_monitor')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def validate_inference_data(self, asset: str) -> Dict[str, Any]:
        """
        Validate and analyze inference data quality.
        
        Args:
            asset: Asset name (SENSEX or GOLD)
            
        Returns:
            Dictionary containing validation results
        """
        self.logger.info(f"Validating inference data for {asset}...")
        
        validation_results = {
            'asset': asset,
            'timestamp': datetime.now().isoformat(),
            'status': 'unknown',
            'issues': [],
            'warnings': [],
            'statistics': {},
            'feature_analysis': {}
        }
        
        try:
            # Load latest data
            if asset == 'SENSEX':
                data_path = self.config['paths']['sensex_clean']
            elif asset == 'GOLD':
                data_path = self.config['paths']['gold_clean']
            else:
                raise ValueError(f"Unsupported asset: {asset}")
            
            if not os.path.exists(data_path):
                validation_results['status'] = 'failed'
                validation_results['issues'].append(f"Data file not found: {data_path}")
                return validation_results
            
            # Load and process data
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Check data freshness
            latest_date = df['Date'].max()
            days_old = (datetime.now() - latest_date).days
            
            if days_old > 5:
                validation_results['warnings'].append(f"Data is {days_old} days old (latest: {latest_date.date()})")
            
            # Check data completeness
            seq_len = self.config['training']['seq_len']
            if len(df) < seq_len:
                validation_results['issues'].append(f"Insufficient data: {len(df)} < {seq_len} required")
            
            # Apply feature engineering
            recent_data = df.tail(seq_len + 20)  # Extra buffer for technical indicators
            engineered_data = calculate_technical_indicators(recent_data)
            
            # Analyze features
            feature_cols = self.config['training']['features'].copy()
            
            # Add cross-asset correlation feature
            if asset == 'SENSEX':
                feature_cols.append('corr_gold_20d')
            elif asset == 'GOLD':
                feature_cols.append('corr_sensex_20d')
            
            # Remove 'Close' if present and add it back (matching training)
            if 'Close' in feature_cols:
                feature_cols.remove('Close')
            final_feature_cols = feature_cols + ['Close']
            
            # Check feature availability
            missing_features = [col for col in final_feature_cols if col not in engineered_data.columns]
            if missing_features:
                validation_results['issues'].append(f"Missing features: {missing_features}")
            
            # Analyze feature statistics
            feature_data = engineered_data[final_feature_cols].tail(seq_len)
            
            for col in final_feature_cols:
                if col in feature_data.columns:
                    col_stats = {
                        'mean': float(feature_data[col].mean()),
                        'std': float(feature_data[col].std()),
                        'min': float(feature_data[col].min()),
                        'max': float(feature_data[col].max()),
                        'null_count': int(feature_data[col].isnull().sum()),
                        'inf_count': int(np.isinf(feature_data[col]).sum()),
                        'range': float(feature_data[col].max() - feature_data[col].min())
                    }
                    
                    # Check for anomalies
                    if col_stats['null_count'] > 0:
                        validation_results['warnings'].append(f"Feature {col} has {col_stats['null_count']} null values")
                    
                    if col_stats['inf_count'] > 0:
                        validation_results['issues'].append(f"Feature {col} has {col_stats['inf_count']} infinite values")
                    
                    if col_stats['std'] == 0:
                        validation_results['warnings'].append(f"Feature {col} has zero variance")
                    
                    # Check for extreme values (specific to financial data)
                    if col == 'ret_1d' and abs(col_stats['max']) > 0.2:
                        validation_results['warnings'].append(f"Extreme daily return detected: {col_stats['max']:.4f}")
                    
                    if col in ['rsi_14'] and (col_stats['min'] < 0 or col_stats['max'] > 100):
                        validation_results['warnings'].append(f"RSI out of valid range [0,100]: [{col_stats['min']:.2f}, {col_stats['max']:.2f}]")
                    
                    validation_results['feature_analysis'][col] = col_stats
            
            # Overall statistics
            validation_results['statistics'] = {
                'data_length': len(df),
                'latest_date': latest_date.strftime('%Y-%m-%d'),
                'days_old': days_old,
                'sequence_length': len(feature_data),
                'feature_count': len(final_feature_cols),
                'current_price': float(engineered_data['Close'].iloc[-1]),
                'price_change_1d': float(engineered_data['ret_1d'].iloc[-1]) if 'ret_1d' in engineered_data else None,
                'volatility_10d': float(engineered_data['vol_10d'].iloc[-1]) if 'vol_10d' in engineered_data else None
            }
            
            # Determine overall status
            if validation_results['issues']:
                validation_results['status'] = 'failed'
            elif validation_results['warnings']:
                validation_results['status'] = 'warning'
            else:
                validation_results['status'] = 'passed'
            
            self.logger.info(f"Data validation for {asset}: {validation_results['status']}")
            
        except Exception as e:
            validation_results['status'] = 'error'
            validation_results['issues'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Data validation failed for {asset}: {str(e)}")
        
        return validation_results
    
    def analyze_prediction_anomalies(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze predictions for anomalies and provide insights.
        
        Args:
            predictions: Prediction results from inference engine
            
        Returns:
            Dictionary containing anomaly analysis
        """
        self.logger.info("Analyzing prediction anomalies...")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'anomalies': [],
            'warnings': [],
            'insights': [],
            'asset_analysis': {}
        }
        
        try:
            for asset, prediction in predictions.get('predictions', {}).items():
                if 'error' in prediction:
                    continue
                
                asset_analysis = {
                    'asset': asset,
                    'anomaly_score': 0.0,
                    'flags': [],
                    'metrics': {}
                }
                
                current_price = prediction['current_price']
                predicted_price = prediction['predicted_price']
                price_change_pct = prediction['price_change_pct']
                ensemble_stats = prediction['ensemble_stats']
                
                # Check for extreme price changes
                if abs(price_change_pct) > 20:  # More than 20% change
                    asset_analysis['flags'].append(f"Extreme price change: {price_change_pct:+.1f}%")
                    asset_analysis['anomaly_score'] += 0.5
                
                if abs(price_change_pct) > 50:  # More than 50% change
                    analysis['anomalies'].append(f"{asset}: Unrealistic price change {price_change_pct:+.1f}%")
                    asset_analysis['anomaly_score'] += 1.0
                
                # Check prediction uncertainty
                prediction_std = ensemble_stats['std']
                prediction_cv = prediction_std / abs(predicted_price) if predicted_price != 0 else float('inf')
                
                if prediction_cv > 0.1:  # High coefficient of variation
                    asset_analysis['flags'].append(f"High prediction uncertainty: CV={prediction_cv:.3f}")
                    asset_analysis['anomaly_score'] += 0.3
                
                # Check confidence interval width
                ci_width = ensemble_stats['confidence_95_upper'] - ensemble_stats['confidence_95_lower']
                ci_width_pct = (ci_width / current_price) * 100
                
                if ci_width_pct > 30:  # Wide confidence interval
                    asset_analysis['flags'].append(f"Wide confidence interval: {ci_width_pct:.1f}% of current price")
                    asset_analysis['anomaly_score'] += 0.2
                
                # Check model consensus
                individual_preds = np.array(ensemble_stats['individual_predictions'])
                pred_range = np.max(individual_preds) - np.min(individual_preds)
                pred_range_pct = (pred_range / current_price) * 100
                
                if pred_range_pct > 20:  # Large disagreement between models
                    asset_analysis['flags'].append(f"Low model consensus: {pred_range_pct:.1f}% range")
                    asset_analysis['anomaly_score'] += 0.3
                
                # Calculate metrics
                asset_analysis['metrics'] = {
                    'prediction_cv': prediction_cv,
                    'ci_width_pct': ci_width_pct,
                    'model_range_pct': pred_range_pct,
                    'abs_change_pct': abs(price_change_pct)
                }
                
                analysis['asset_analysis'][asset] = asset_analysis
                
                # Generate insights
                if asset_analysis['anomaly_score'] > 1.0:
                    analysis['insights'].append(f"{asset}: High anomaly score ({asset_analysis['anomaly_score']:.1f}) - review prediction carefully")
                elif asset_analysis['anomaly_score'] > 0.5:
                    analysis['insights'].append(f"{asset}: Moderate anomaly score ({asset_analysis['anomaly_score']:.1f}) - prediction may be uncertain")
                else:
                    analysis['insights'].append(f"{asset}: Normal prediction (anomaly score: {asset_analysis['anomaly_score']:.1f})")
            
            # Overall assessment
            total_anomaly_score = sum(asset['anomaly_score'] for asset in analysis['asset_analysis'].values())
            avg_anomaly_score = total_anomaly_score / len(analysis['asset_analysis']) if analysis['asset_analysis'] else 0
            
            if avg_anomaly_score > 1.0:
                analysis['overall_status'] = 'high_risk'
                analysis['recommendation'] = 'Review predictions carefully - multiple anomalies detected'
            elif avg_anomaly_score > 0.5:
                analysis['overall_status'] = 'medium_risk'
                analysis['recommendation'] = 'Predictions show some uncertainty - use with caution'
            else:
                analysis['overall_status'] = 'normal'
                analysis['recommendation'] = 'Predictions appear normal and reliable'
            
        except Exception as e:
            analysis['error'] = f"Anomaly analysis failed: {str(e)}"
            self.logger.error(f"Anomaly analysis failed: {str(e)}")
        
        return analysis
    
    def compare_with_historical(self, asset: str, prediction: float) -> Dict[str, Any]:
        """
        Compare prediction with historical patterns.
        
        Args:
            asset: Asset name
            prediction: Predicted price
            
        Returns:
            Dictionary containing historical comparison
        """
        try:
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
            
            # Calculate historical statistics
            current_price = df['Close'].iloc[-1]
            historical_prices = df['Close'].values
            
            # Calculate percentiles
            prediction_percentile = (historical_prices < prediction).mean() * 100
            
            # Calculate recent volatility
            recent_returns = df['Close'].pct_change().tail(252)  # Last year
            recent_volatility = recent_returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate z-score of prediction
            price_mean = historical_prices.mean()
            price_std = historical_prices.std()
            prediction_zscore = (prediction - price_mean) / price_std
            
            comparison = {
                'asset': asset,
                'prediction': prediction,
                'current_price': current_price,
                'historical_mean': price_mean,
                'historical_std': price_std,
                'prediction_percentile': prediction_percentile,
                'prediction_zscore': prediction_zscore,
                'recent_volatility_annualized': recent_volatility,
                'data_span_years': (df['Date'].max() - df['Date'].min()).days / 365.25,
                'interpretation': self._interpret_historical_comparison(prediction_percentile, prediction_zscore)
            }
            
            return comparison
            
        except Exception as e:
            return {'error': f"Historical comparison failed: {str(e)}"}
    
    def _interpret_historical_comparison(self, percentile: float, zscore: float) -> str:
        """Interpret historical comparison results."""
        if percentile > 95:
            return "Prediction is in the top 5% of historical prices - unusually high"
        elif percentile < 5:
            return "Prediction is in the bottom 5% of historical prices - unusually low"
        elif abs(zscore) > 2:
            return f"Prediction is {abs(zscore):.1f} standard deviations from historical mean - unusual"
        elif abs(zscore) > 1:
            return f"Prediction is {abs(zscore):.1f} standard deviations from historical mean - moderately unusual"
        else:
            return "Prediction is within normal historical range"
    
    def generate_monitoring_report(self, asset: str, predictions: Dict[str, Any]) -> str:
        """
        Generate a comprehensive monitoring report.
        
        Args:
            asset: Asset name
            predictions: Prediction results
            
        Returns:
            Formatted monitoring report string
        """
        try:
            # Validate data
            validation = self.validate_inference_data(asset)
            
            # Analyze anomalies
            anomaly_analysis = self.analyze_prediction_anomalies(predictions)
            
            # Compare with historical
            if asset in predictions.get('predictions', {}):
                pred_price = predictions['predictions'][asset]['predicted_price']
                historical_comp = self.compare_with_historical(asset, pred_price)
            else:
                historical_comp = {'error': 'No prediction available'}
            
            # Generate report
            report = f"""
INFERENCE MONITORING REPORT - {asset}
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA VALIDATION
--------------
Status: {validation['status'].upper()}
Issues: {len(validation['issues'])}
Warnings: {len(validation['warnings'])}

Latest Data: {validation['statistics'].get('latest_date', 'Unknown')} ({validation['statistics'].get('days_old', 'Unknown')} days old)
Sequence Length: {validation['statistics'].get('sequence_length', 'Unknown')}
Current Price: {validation['statistics'].get('current_price', 'Unknown')}

Issues:
{chr(10).join(f"  - {issue}" for issue in validation['issues']) if validation['issues'] else "  None"}

Warnings:
{chr(10).join(f"  - {warning}" for warning in validation['warnings']) if validation['warnings'] else "  None"}

PREDICTION ANALYSIS
------------------
Anomaly Score: {anomaly_analysis['asset_analysis'].get(asset, {}).get('anomaly_score', 'Unknown')}
Flags: {len(anomaly_analysis['asset_analysis'].get(asset, {}).get('flags', []))}

Flags:
{chr(10).join(f"  - {flag}" for flag in anomaly_analysis['asset_analysis'].get(asset, {}).get('flags', [])) if anomaly_analysis['asset_analysis'].get(asset, {}).get('flags') else "  None"}

HISTORICAL COMPARISON
--------------------
{'Error: ' + historical_comp.get('error', '') if 'error' in historical_comp else f'''Prediction Percentile: {historical_comp.get('prediction_percentile', 'Unknown'):.1f}%
Z-Score: {historical_comp.get('prediction_zscore', 'Unknown'):.2f}
Interpretation: {historical_comp.get('interpretation', 'Unknown')}'''}

RECOMMENDATION
-------------
Overall Status: {anomaly_analysis.get('overall_status', 'Unknown').upper()}
{anomaly_analysis.get('recommendation', 'No recommendation available')}
"""
            
            return report
            
        except Exception as e:
            return f"ERROR: Failed to generate monitoring report: {str(e)}"

def main():
    """Main function for testing monitoring tools."""
    monitor = InferenceMonitor('configs/default.yaml')
    
    # Test data validation
    for asset in ['SENSEX', 'GOLD']:
        print(f"\nValidating {asset} data...")
        validation = monitor.validate_inference_data(asset)
        print(f"Status: {validation['status']}")
        print(f"Issues: {len(validation['issues'])}")
        print(f"Warnings: {len(validation['warnings'])}")

if __name__ == "__main__":
    main()