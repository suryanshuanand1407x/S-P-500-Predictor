#!/usr/bin/env python3
"""
Artifact Management System for Sensex & Gold LSTM Price Predictor

This module manages versioned training artifacts to ensure exact reproducibility
during inference. Each asset/fold combination has a complete set of artifacts
that must match exactly during inference.

Artifacts per asset/fold:
- model.pt: PyTorch model state dict
- feature_names.json: Ordered list of feature names
- seq_len.json: Sequence length metadata
- feature_scaler.pkl: Fitted feature scaler
- target_scaler.pkl: Fitted target scaler
- metadata.json: Complete training metadata and versioning info

Author: AI Assistant
Date: 2025
"""

import os
import json
import pickle
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def setup_logger(name: str = 'artifact_manager') -> logging.Logger:
    """Setup logger for artifact management."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class ArtifactManager:
    """
    Manages versioned training artifacts for exact inference reproducibility.
    
    Ensures that inference uses exactly the same feature processing, scaling,
    and model configuration as was used during training.
    """
    
    ARTIFACT_VERSION = "1.0.0"
    REQUIRED_ARTIFACTS = [
        "model.pt",
        "feature_names.json", 
        "seq_len.json",
        "feature_scaler.pkl",
        "target_scaler.pkl",
        "metadata.json"
    ]
    
    def __init__(self, base_dir: str = "models"):
        """
        Initialize artifact manager.
        
        Args:
            base_dir: Base directory for storing artifacts
        """
        self.base_dir = Path(base_dir)
        self.logger = setup_logger('artifact_manager')
    
    def save_fold_artifacts(
        self,
        asset: str,
        fold: int,
        model: torch.nn.Module,
        feature_names: List[str],
        seq_len: int,
        feature_scaler: StandardScaler,
        target_scaler: StandardScaler,
        training_config: Dict[str, Any],
        training_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save complete set of artifacts for a specific asset/fold.
        
        Args:
            asset: Asset name (SENSEX, GOLD)
            fold: Fold number
            model: Trained PyTorch model
            feature_names: Ordered list of feature names
            seq_len: Sequence length used for training
            feature_scaler: Fitted feature scaler
            target_scaler: Fitted target scaler
            training_config: Complete training configuration
            training_metadata: Additional training metadata
            
        Returns:
            Path to saved artifacts directory
        """
        fold_dir = self.base_dir / asset / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            # 1. Save model state dict
            model_path = fold_dir / "model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'input_size': getattr(model, 'input_size', len(feature_names)),
                'hidden_size': getattr(model, 'hidden_size', training_config.get('hidden_size', 128)),
                'num_layers': getattr(model, 'num_layers', training_config.get('num_layers', 2)),
                'dropout': getattr(model, 'dropout', training_config.get('dropout', 0.2)),
                'timestamp': timestamp,
                'version': self.ARTIFACT_VERSION
            }, model_path)
            
            # 2. Save feature names with strict ordering
            feature_names_path = fold_dir / "feature_names.json"
            with open(feature_names_path, 'w') as f:
                json.dump({
                    'feature_names': feature_names,
                    'feature_count': len(feature_names),
                    'timestamp': timestamp,
                    'version': self.ARTIFACT_VERSION
                }, f, indent=2)
            
            # 3. Save sequence length metadata
            seq_len_path = fold_dir / "seq_len.json"
            with open(seq_len_path, 'w') as f:
                json.dump({
                    'seq_len': seq_len,
                    'timestamp': timestamp,
                    'version': self.ARTIFACT_VERSION
                }, f, indent=2)
            
            # 4. Save feature scaler
            feature_scaler_path = fold_dir / "feature_scaler.pkl"
            with open(feature_scaler_path, 'wb') as f:
                pickle.dump({
                    'scaler': feature_scaler,
                    'feature_names': feature_names,
                    'n_features': len(feature_names),
                    'timestamp': timestamp,
                    'version': self.ARTIFACT_VERSION
                }, f)
            
            # 5. Save target scaler
            target_scaler_path = fold_dir / "target_scaler.pkl"
            with open(target_scaler_path, 'wb') as f:
                pickle.dump({
                    'scaler': target_scaler,
                    'timestamp': timestamp,
                    'version': self.ARTIFACT_VERSION
                }, f)
            
            # 6. Create comprehensive metadata
            metadata = {
                'artifact_version': self.ARTIFACT_VERSION,
                'timestamp': timestamp,
                'asset': asset,
                'fold': fold,
                'training_config': training_config,
                'feature_schema': {
                    'feature_names': feature_names,
                    'feature_count': len(feature_names),
                    'seq_len': seq_len
                },
                'model_info': {
                    'class_name': model.__class__.__name__,
                    'input_size': getattr(model, 'input_size', len(feature_names)),
                    'hidden_size': getattr(model, 'hidden_size', training_config.get('hidden_size', 128)),
                    'num_layers': getattr(model, 'num_layers', training_config.get('num_layers', 2)),
                    'dropout': getattr(model, 'dropout', training_config.get('dropout', 0.2)),
                    'total_parameters': sum(p.numel() for p in model.parameters())
                },
                'checksums': {}
            }
            
            # Add training metadata if provided
            if training_metadata:
                metadata['training_metadata'] = training_metadata
            
            # Calculate checksums for all artifacts
            for artifact_name in self.REQUIRED_ARTIFACTS[:-1]:  # Exclude metadata.json itself
                artifact_path = fold_dir / artifact_name
                if artifact_path.exists():
                    metadata['checksums'][artifact_name] = self._calculate_file_checksum(artifact_path)
            
            # 7. Save metadata
            metadata_path = fold_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Calculate metadata checksum
            metadata['checksums']['metadata.json'] = self._calculate_file_checksum(metadata_path)
            
            # Update metadata with its own checksum
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"✅ Saved artifacts for {asset} fold {fold} to {fold_dir}")
            self.logger.info(f"   - Features: {len(feature_names)}")
            self.logger.info(f"   - Sequence length: {seq_len}")
            self.logger.info(f"   - Model parameters: {metadata['model_info']['total_parameters']:,}")
            
            return str(fold_dir)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save artifacts for {asset} fold {fold}: {str(e)}")
            raise
    
    def load_fold_artifacts(self, asset: str, fold: int) -> Dict[str, Any]:
        """
        Load complete set of artifacts for a specific asset/fold with validation.
        
        Args:
            asset: Asset name
            fold: Fold number
            
        Returns:
            Dictionary containing all loaded artifacts
            
        Raises:
            ValueError: If artifacts are missing, corrupted, or version mismatch
        """
        fold_dir = self.base_dir / asset / f"fold_{fold}"
        
        if not fold_dir.exists():
            raise ValueError(f"Artifacts directory not found: {fold_dir}")
        
        # Check all required artifacts exist
        missing_artifacts = []
        for artifact_name in self.REQUIRED_ARTIFACTS:
            artifact_path = fold_dir / artifact_name
            if not artifact_path.exists():
                missing_artifacts.append(artifact_name)
        
        if missing_artifacts:
            raise ValueError(f"Missing artifacts for {asset} fold {fold}: {missing_artifacts}")
        
        try:
            # Load metadata first for validation
            metadata_path = fold_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Validate artifact version
            if metadata.get('artifact_version') != self.ARTIFACT_VERSION:
                raise ValueError(f"Artifact version mismatch. Expected {self.ARTIFACT_VERSION}, "
                               f"got {metadata.get('artifact_version')}")
            
            # Validate asset/fold match
            if metadata.get('asset') != asset or metadata.get('fold') != fold:
                raise ValueError(f"Artifact asset/fold mismatch. Expected {asset}/{fold}, "
                               f"got {metadata.get('asset')}/{metadata.get('fold')}")
            
            # Validate checksums
            stored_checksums = metadata.get('checksums', {})
            for artifact_name in self.REQUIRED_ARTIFACTS[:-1]:  # Exclude metadata.json
                artifact_path = fold_dir / artifact_name
                if artifact_path.exists():
                    current_checksum = self._calculate_file_checksum(artifact_path)
                    stored_checksum = stored_checksums.get(artifact_name)
                    
                    if stored_checksum and current_checksum != stored_checksum:
                        raise ValueError(f"Checksum mismatch for {artifact_name}. "
                                       f"Expected {stored_checksum}, got {current_checksum}")
            
            # Load all artifacts
            artifacts = {
                'metadata': metadata,
                'asset': asset,
                'fold': fold,
                'artifact_dir': str(fold_dir)
            }
            
            # Load model info
            model_path = fold_dir / "model.pt"
            model_data = torch.load(model_path, map_location='cpu')
            artifacts['model_data'] = model_data
            
            # Load feature names
            feature_names_path = fold_dir / "feature_names.json"
            with open(feature_names_path, 'r') as f:
                feature_data = json.load(f)
            artifacts['feature_names'] = feature_data['feature_names']
            artifacts['feature_count'] = feature_data['feature_count']
            
            # Load sequence length
            seq_len_path = fold_dir / "seq_len.json"
            with open(seq_len_path, 'r') as f:
                seq_data = json.load(f)
            artifacts['seq_len'] = seq_data['seq_len']
            
            # Load feature scaler
            feature_scaler_path = fold_dir / "feature_scaler.pkl"
            with open(feature_scaler_path, 'rb') as f:
                feature_scaler_data = pickle.load(f)
            artifacts['feature_scaler'] = feature_scaler_data['scaler']
            
            # Load target scaler
            target_scaler_path = fold_dir / "target_scaler.pkl"
            with open(target_scaler_path, 'rb') as f:
                target_scaler_data = pickle.load(f)
            artifacts['target_scaler'] = target_scaler_data['scaler']
            
            # Validate feature schema consistency
            metadata_features = metadata['feature_schema']['feature_names']
            if artifacts['feature_names'] != metadata_features:
                raise ValueError(f"Feature name mismatch between metadata and feature_names.json")
            
            if artifacts['feature_count'] != len(metadata_features):
                raise ValueError(f"Feature count mismatch. Expected {len(metadata_features)}, "
                               f"got {artifacts['feature_count']}")
            
            self.logger.info(f"✅ Loaded artifacts for {asset} fold {fold}")
            self.logger.info(f"   - Version: {metadata['artifact_version']}")
            self.logger.info(f"   - Features: {artifacts['feature_count']}")
            self.logger.info(f"   - Sequence length: {artifacts['seq_len']}")
            self.logger.info(f"   - Timestamp: {metadata['timestamp']}")
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load artifacts for {asset} fold {fold}: {str(e)}")
            raise
    
    def validate_inference_compatibility(
        self,
        asset: str,
        fold: int,
        input_feature_names: List[str],
        input_seq_len: int
    ) -> Tuple[bool, List[str]]:
        """
        Validate that inference inputs are compatible with training artifacts.
        
        Args:
            asset: Asset name
            fold: Fold number
            input_feature_names: Feature names from inference
            input_seq_len: Sequence length from inference
            
        Returns:
            Tuple of (is_compatible, error_messages)
        """
        try:
            artifacts = self.load_fold_artifacts(asset, fold)
            
            errors = []
            
            # Check sequence length
            if input_seq_len != artifacts['seq_len']:
                errors.append(f"Sequence length mismatch: expected {artifacts['seq_len']}, "
                             f"got {input_seq_len}")
            
            # Check feature names and order
            expected_features = artifacts['feature_names']
            if input_feature_names != expected_features:
                errors.append(f"Feature schema mismatch:")
                errors.append(f"  Expected: {expected_features}")
                errors.append(f"  Got:      {input_feature_names}")
                
                # Detailed analysis
                missing = set(expected_features) - set(input_feature_names)
                extra = set(input_feature_names) - set(expected_features)
                
                if missing:
                    errors.append(f"  Missing features: {list(missing)}")
                if extra:
                    errors.append(f"  Extra features: {list(extra)}")
                
                # Check if just order is wrong
                if set(input_feature_names) == set(expected_features):
                    errors.append("  Note: Same features but different order!")
            
            # Check feature count
            if len(input_feature_names) != artifacts['feature_count']:
                errors.append(f"Feature count mismatch: expected {artifacts['feature_count']}, "
                             f"got {len(input_feature_names)}")
            
            is_compatible = len(errors) == 0
            
            if is_compatible:
                self.logger.info(f"✅ Inference inputs compatible with {asset} fold {fold}")
            else:
                self.logger.error(f"❌ Inference inputs incompatible with {asset} fold {fold}")
                for error in errors:
                    self.logger.error(f"   {error}")
            
            return is_compatible, errors
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
    
    def list_available_artifacts(self) -> Dict[str, List[int]]:
        """
        List all available artifacts by asset and fold.
        
        Returns:
            Dictionary mapping asset names to lists of available folds
        """
        available = {}
        
        if not self.base_dir.exists():
            return available
        
        for asset_dir in self.base_dir.iterdir():
            if asset_dir.is_dir():
                asset = asset_dir.name
                available[asset] = []
                
                for fold_dir in asset_dir.iterdir():
                    if fold_dir.is_dir() and fold_dir.name.startswith('fold_'):
                        try:
                            fold_num = int(fold_dir.name.split('_')[1])
                            
                            # Check if all required artifacts exist
                            all_exist = all(
                                (fold_dir / artifact).exists() 
                                for artifact in self.REQUIRED_ARTIFACTS
                            )
                            
                            if all_exist:
                                available[asset].append(fold_num)
                        except (ValueError, IndexError):
                            continue
                
                available[asset].sort()
        
        return available
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def cleanup_old_artifacts(self, keep_latest: int = 5) -> None:
        """
        Clean up old artifact versions, keeping only the most recent ones.
        
        Args:
            keep_latest: Number of latest versions to keep per asset/fold
        """
        # This is a placeholder for future implementation
        # Currently we only keep one version per fold
        self.logger.info("Artifact cleanup not implemented yet (single version per fold)")
    
    def export_artifact_summary(self) -> Dict[str, Any]:
        """
        Export a summary of all available artifacts.
        
        Returns:
            Dictionary containing artifact summary
        """
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'artifact_version': self.ARTIFACT_VERSION,
            'base_dir': str(self.base_dir),
            'assets': {}
        }
        
        available = self.list_available_artifacts()
        
        for asset, folds in available.items():
            summary['assets'][asset] = {
                'available_folds': folds,
                'fold_count': len(folds),
                'folds': {}
            }
            
            for fold in folds:
                try:
                    artifacts = self.load_fold_artifacts(asset, fold)
                    metadata = artifacts['metadata']
                    
                    summary['assets'][asset]['folds'][fold] = {
                        'timestamp': metadata['timestamp'],
                        'feature_count': artifacts['feature_count'],
                        'seq_len': artifacts['seq_len'],
                        'model_parameters': metadata['model_info']['total_parameters'],
                        'artifact_dir': artifacts['artifact_dir']
                    }
                except Exception as e:
                    summary['assets'][asset]['folds'][fold] = {
                        'error': str(e)
                    }
        
        return summary

def main():
    """Test artifact manager functionality."""
    logger = setup_logger('artifact_test')
    
    # Test artifact manager
    manager = ArtifactManager()
    
    # List available artifacts
    available = manager.list_available_artifacts()
    logger.info(f"Available artifacts: {available}")
    
    # Export summary
    summary = manager.export_artifact_summary()
    logger.info(f"Artifact summary exported with {len(summary['assets'])} assets")
    
    # Test loading artifacts if available
    for asset, folds in available.items():
        if folds:
            fold = folds[0]
            try:
                artifacts = manager.load_fold_artifacts(asset, fold)
                logger.info(f"Successfully loaded {asset} fold {fold}")
                
                # Test validation
                is_compatible, errors = manager.validate_inference_compatibility(
                    asset, fold, artifacts['feature_names'], artifacts['seq_len']
                )
                logger.info(f"Compatibility check: {is_compatible}")
                
            except Exception as e:
                logger.error(f"Failed to load {asset} fold {fold}: {str(e)}")

if __name__ == "__main__":
    main()