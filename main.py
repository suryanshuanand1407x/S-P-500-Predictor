#!/usr/bin/env python3
"""
Main script for S&P LSTM Price Predictor

This script runs the complete pipeline:
1. Data verification and integrity checks
2. Data ingestion and cleaning  
3. Feature engineering and sequence building
4. LSTM model training with temporal cross-validation
5. Model evaluation and reporting

Usage:
    python main.py --train          # Run training pipeline
    python main.py --eval           # Run evaluation only
    python main.py --full           # Run complete pipeline (default)
    python main.py --config custom.yaml  # Use custom config

Author: AI Assistant
Date: 2025
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import training and evaluation functions
def train_main(config_path):
    """Import and run training main function."""
    import train
    return train.main(config_path)

def evaluate_all_models(config_path):
    """Import and run evaluation function."""
    import eval
    return eval.evaluate_all_models(config_path)

def run_inference(config_path):
    """Import and run production inference function."""
    import production_inference
    return production_inference.main(config_path)

def run_backtest_validation(config_path):
    """Import and run backtest validation."""
    import backtest_validator
    return backtest_validator.main(config_path)

def setup_main_logger(log_dir: str = 'logs') -> logging.Logger:
    """Setup main pipeline logger."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('main_pipeline')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(log_dir, f'main_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    
    return logger

def validate_config(config_path: str) -> dict:
    """Validate and load configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['paths', 'training', 'cv']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section missing: {section}")
    
    # Validate paths
    required_paths = ['sp_raw_glob']
    for path_key in required_paths:
        if path_key not in config['paths']:
            raise ValueError(f"Required path missing in config: {path_key}")
    
    return config

def check_system_requirements():
    """Check if system meets requirements for running the pipeline."""
    logger = logging.getLogger('main_pipeline')
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required")
    
    # Check required packages
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn', 'matplotlib', 
        'seaborn', 'yaml', 'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise RuntimeError(f"Missing required packages: {missing_packages}")
    
    # Check PyTorch device availability
    import torch
    if torch.backends.mps.is_available():
        logger.info("✅ MPS (Apple Silicon) acceleration available")
    elif torch.cuda.is_available():
        logger.info("✅ CUDA acceleration available")
    else:
        logger.info("⚠️  Using CPU only (slower training)")
    
    logger.info("✅ System requirements check passed")

def run_training_pipeline(config_path: str):
    """Run the complete training pipeline."""
    logger = logging.getLogger('main_pipeline')
    
    logger.info("🚀 Starting LSTM training pipeline...")
    logger.info(f"📋 Configuration: {config_path}")
    
    try:
        # Run training
        train_main(config_path)
        logger.info("✅ Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        raise

def run_evaluation_pipeline(config_path: str):
    """Run the evaluation pipeline."""
    logger = logging.getLogger('main_pipeline')
    
    logger.info("📊 Starting model evaluation...")
    
    try:
        # Run evaluation
        evaluate_all_models(config_path)
        logger.info("✅ Evaluation pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Evaluation pipeline failed: {str(e)}")
        raise

def run_inference_pipeline(config_path: str):
    """Run the production inference pipeline."""
    logger = logging.getLogger('main_pipeline')
    
    logger.info("🔮 Starting production inference pipeline...")
    
    try:
        # Run production inference
        run_inference(config_path)
        logger.info("✅ Production inference pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Production inference pipeline failed: {str(e)}")
        raise

def run_backtest_pipeline(config_path: str):
    """Run the backtest validation pipeline."""
    logger = logging.getLogger('main_pipeline')
    
    logger.info("🧪 Starting backtest validation pipeline...")
    
    try:
        # Run backtest validation
        run_backtest_validation(config_path)
        logger.info("✅ Backtest validation pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Backtest validation pipeline failed: {str(e)}")
        raise

def run_full_pipeline(config_path: str):
    """Run the complete pipeline: training + evaluation."""
    logger = logging.getLogger('main_pipeline')
    
    logger.info("🔄 Starting complete LSTM prediction pipeline...")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Training
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: MODEL TRAINING")
        logger.info("="*60)
        run_training_pipeline(config_path)
        
        # Step 2: Evaluation
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: MODEL EVALUATION")
        logger.info("="*60)
        run_evaluation_pipeline(config_path)
        
        # Success summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"⏱️  Total execution time: {duration}")
        logger.info(f"📅 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📅 Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load and display final results
        try:
            with open('reports/metrics_summary.json', 'r') as f:
                import json
                metrics = json.load(f)
                
            logger.info("\n📈 FINAL RESULTS SUMMARY:")
            for asset, asset_metrics in metrics.items():
                logger.info(f"\n{asset}:")
                for metric, value in asset_metrics.items():
                    logger.info(f"  {metric}: {value}")
                    
        except Exception as e:
            logger.warning(f"Could not load final metrics: {str(e)}")
        
        logger.info(f"\n📁 Reports and plots saved in: reports/")
        logger.info(f"📁 Trained models saved in: models/")
        logger.info(f"📁 Logs saved in: logs/")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise

def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                   S&P LSTM PRICE PREDICTOR                       ║
    ║                                                                  ║
    ║  📈 Advanced LSTM models for financial time series prediction   ║
    ║  🔮 Next-day price forecasting with temporal cross-validation   ║
    ║  🍎 Optimized for Apple Silicon (MPS) and CUDA acceleration     ║
    ║  📊 Real data verification and comprehensive evaluation          ║
    ║                                                                  ║
    ║  Asset: S&P 500                                                  ║
    ║  Horizon: T+1 (next business day)                               ║
    ║  Features: Technical indicators + price history                  ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='S&P LSTM Price Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full                    # Run complete pipeline (default)
  python main.py --train                   # Training only
  python main.py --eval                    # Evaluation only
  python main.py --config custom.yaml      # Use custom configuration
  
The pipeline will:
1. Verify data integrity and authenticity
2. Clean and process historical price data
3. Engineer technical indicator features
4. Train LSTM models with temporal cross-validation
5. Evaluate models and generate comprehensive reports
        """
    )
    
    # Pipeline mode arguments (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--full', action='store_true', default=True,
                           help='Run complete pipeline: training + evaluation (default)')
    mode_group.add_argument('--train', action='store_true',
                           help='Run training pipeline only')
    mode_group.add_argument('--eval', action='store_true', 
                           help='Run evaluation pipeline only')
    mode_group.add_argument('--predict', action='store_true',
                           help='Run production inference/prediction pipeline only')
    mode_group.add_argument('--backtest', action='store_true',
                           help='Run backtest validation pipeline only')
    
    # Configuration
    parser.add_argument('--config', default='configs/default.yaml',
                       help='Path to configuration file (default: configs/default.yaml)')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--check-only', action='store_true',
                       help='Only run system checks and exit')
    
    args = parser.parse_args()
    
    # If other modes are specified, disable default full mode
    if args.train or args.eval or args.predict or args.backtest:
        args.full = False
    
    # Print banner
    print_banner()
    
    # Setup logging
    logger = setup_main_logger()
    logger.info("S&P LSTM Price Predictor starting...")
    
    try:
        # Validate configuration
        logger.info(f"📋 Loading configuration from: {args.config}")
        config = validate_config(args.config)
        logger.info("✅ Configuration validated successfully")
        
        # Check system requirements
        logger.info("🔍 Checking system requirements...")
        check_system_requirements()
        
        if args.check_only:
            logger.info("✅ System check completed. Exiting.")
            return
        
        # Create required directories
        required_dirs = [
            config['paths']['logs_dir'],
            config['paths']['reports_dir'], 
            config['paths']['models_dir'],
            'data/processed'
        ]
        
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        logger.info("📁 Directory structure verified")
        
        # Run selected pipeline
        if args.train:
            run_training_pipeline(args.config)
        elif args.eval:
            run_evaluation_pipeline(args.config)
        elif args.predict:
            run_inference_pipeline(args.config)
        elif args.backtest:
            run_backtest_pipeline(args.config)
        else:  # args.full (default)
            run_full_pipeline(args.config)
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Pipeline failed with error: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()