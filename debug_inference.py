#!/usr/bin/env python3
"""
Debug script to investigate inference issues.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from production_inference import ProductionInferenceEngine
from artifact_manager import ArtifactManager
from feature_engineering import calculate_technical_indicators

# Load config
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize artifact manager
artifact_mgr = ArtifactManager(config['paths']['models_dir'])

# Load a single model for SENSEX fold 0
print("="*80)
print("DEBUGGING SENSEX INFERENCE")
print("="*80)

artifacts = artifact_mgr.load_fold_artifacts('SENSEX', 0)
print(f"\nArtifacts loaded for SENSEX fold 0:")
print(f"  Feature names: {artifacts['feature_names']}")
print(f"  Feature count: {artifacts['feature_count']}")
print(f"  Sequence length: {artifacts['seq_len']}")

# Load recent data
sensex_path = config['paths']['sensex_clean']
df = pd.read_csv(sensex_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\nData loaded: {len(df)} rows")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Latest Close: {df['Close'].iloc[-1]}")

# Apply feature engineering
print("\n" + "="*80)
print("APPLYING FEATURE ENGINEERING")
print("="*80)

recent_data = df.tail(200).copy()
engineered_data = calculate_technical_indicators(recent_data)

print(f"\nEngineered data columns: {list(engineered_data.columns)}")
print(f"Shape: {engineered_data.shape}")

# Add cross-asset correlation (dummy for now)
engineered_data['corr_gold_20d'] = 0.0

# Check for NaN values BEFORE extracting sequence
print("\n" + "="*80)
print("NaN CHECK BEFORE SEQUENCE EXTRACTION")
print("="*80)

expected_features = artifacts['feature_names']
print(f"\nExpected features: {expected_features}")

for feat in expected_features:
    if feat in engineered_data.columns:
        nan_count = engineered_data[feat].isnull().sum()
        if nan_count > 0:
            print(f"  {feat}: {nan_count} NaN values")
    else:
        print(f"  {feat}: MISSING FROM DATA")

# Forward fill
print("\nForward filling NaN values...")
for col in expected_features:
    if col in engineered_data.columns:
        before = engineered_data[col].isnull().sum()
        engineered_data[col] = engineered_data[col].fillna(method='ffill').fillna(method='bfill')
        after = engineered_data[col].isnull().sum()
        if before > 0:
            print(f"  {col}: {before} -> {after} NaN")

# Extract sequence
seq_len = artifacts['seq_len']
feature_data = engineered_data[expected_features].tail(seq_len)

print("\n" + "="*80)
print(f"EXTRACTED SEQUENCE (last {seq_len} rows)")
print("="*80)

print(f"\nSequence shape: {feature_data.shape}")
print(f"NaN count in sequence: {feature_data.isnull().sum().sum()}")
print(f"Inf count in sequence: {np.isinf(feature_data.values).sum()}")

# Show statistics for each feature
print("\nFeature statistics in sequence:")
for i, feat in enumerate(expected_features):
    vals = feature_data[feat].values
    print(f"  {i}. {feat:15s}: mean={np.mean(vals):10.2f}, std={np.std(vals):10.2f}, "
          f"min={np.min(vals):10.2f}, max={np.max(vals):10.2f}")

# Now scale using the scaler
feature_scaler = artifacts['feature_scaler']
target_scaler = artifacts['target_scaler']

print("\n" + "="*80)
print("SCALER INFORMATION")
print("="*80)

print(f"\nFeature scaler mean (first 5): {feature_scaler.mean_[:5]}")
print(f"Feature scaler std (first 5): {feature_scaler.scale_[:5]}")
print(f"\nTarget scaler mean: {target_scaler.mean_[0]:.2f}")
print(f"Target scaler std: {target_scaler.scale_[0]:.2f}")

# Scale the features
features_scaled = feature_scaler.transform(feature_data.values)

print("\n" + "="*80)
print("SCALED FEATURES")
print("="*80)

print(f"\nScaled features shape: {features_scaled.shape}")
print(f"Scaled features statistics:")
for i, feat in enumerate(expected_features):
    vals = features_scaled[:, i]
    print(f"  {i}. {feat:15s}: mean={np.mean(vals):8.2f}, std={np.std(vals):8.2f}, "
          f"min={np.min(vals):8.2f}, max={np.max(vals):8.2f}")

# Convert to tensor and run through model
features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)  # Add batch dimension

from models import create_model

model_data = artifacts['model_data']
model_config = artifacts['metadata']['training_config']['training']

model = create_model(model_config, artifacts['feature_count'])
model.load_state_dict(model_data['model_state_dict'])
model.eval()

print("\n" + "="*80)
print("MODEL PREDICTION")
print("="*80)

with torch.no_grad():
    pred_scaled = model(features_tensor).cpu().numpy().flatten()[0]

print(f"\nScaled prediction: {pred_scaled:.4f}")

# Inverse transform
pred_unscaled = target_scaler.inverse_transform([[pred_scaled]])[0][0]

print(f"Unscaled prediction: {pred_unscaled:.2f}")

current_price = df['Close'].iloc[-1]
change_pct = ((pred_unscaled - current_price) / current_price) * 100

print(f"\nCurrent price: {current_price:.2f}")
print(f"Predicted price: {pred_unscaled:.2f}")
print(f"Change: {change_pct:+.2f}%")

# Now check if the issue is with how Close is used
print("\n" + "="*80)
print("CLOSE VALUE ANALYSIS")
print("="*80)

close_values_in_sequence = feature_data['Close'].values
print(f"\nClose values in sequence (last 10):")
print(close_values_in_sequence[-10:])

print(f"\nCurrent Close (last): {close_values_in_sequence[-1]:.2f}")
print(f"Scaled Close values (last 10):")
close_idx = expected_features.index('Close')
print(features_scaled[-10:, close_idx])

# Check if scaler is appropriate
print("\n" + "="*80)
print("SCALER APPROPRIATENESS CHECK")
print("="*80)

# What Close values was the scaler trained on?
print(f"\nTarget scaler was trained on Close values with:")
print(f"  Mean: {target_scaler.mean_[0]:.2f}")
print(f"  Std:  {target_scaler.scale_[0]:.2f}")

print(f"\nCurrent Close value: {current_price:.2f}")
print(f"Z-score of current Close: {(current_price - target_scaler.mean_[0]) / target_scaler.scale_[0]:.2f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if pred_unscaled < current_price * 0.5:
    print("\n⚠️ ISSUE DETECTED: Prediction is less than 50% of current price")
    print("\nPossible causes:")
    print("1. Scaler mismatch - target scaler trained on different price range")
    print("2. Feature scaling issue - features not matching training distribution")
    print("3. Model trained on different feature order")
    print("4. Close value not properly included in features")
