import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

def setup_logger(name: str = 'feature_engineering') -> logging.Logger:
    """Setup logger for feature engineering."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for LSTM features."""
    logger = setup_logger()
    
    df = df.copy()
    
    logger.info("Calculating technical indicators...")
    
    # Basic return features
    df['ret_1d'] = df['Close'].pct_change()
    df['ret_1d'] = df['ret_1d'].clip(-0.2, 0.2)  # Clip extreme returns as per PRD
    
    # Rolling volatility (10-day)
    df['vol_10d'] = df['ret_1d'].rolling(window=10, min_periods=5).std()
    
    # Z-score (20-day)
    sma_20 = df['Close'].rolling(window=20, min_periods=10).mean()
    std_20 = df['Close'].rolling(window=20, min_periods=10).std()
    df['zscore_20d'] = (df['Close'] - sma_20) / std_20
    
    # RSI (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=7).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Exponential Moving Averages
    df['ema_10'] = df['Close'].ewm(span=10, min_periods=5).mean()
    df['ema_20'] = df['Close'].ewm(span=20, min_periods=10).mean()
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr_14'] = true_range.rolling(window=14, min_periods=7).mean()
    
    logger.info("Technical indicators calculated successfully")
    
    return df

def add_cross_asset_features(sp_df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for cross-asset correlation features (removed)."""
    logger = setup_logger()
    logger.info("Cross-asset features not applicable for single asset")
    return sp_df

def prepare_feature_columns(df: pd.DataFrame, feature_list: List[str], asset_name: str) -> pd.DataFrame:
    """Prepare and validate feature columns."""
    logger = setup_logger()
    
    df = df.copy()
    
    # Check which features are available
    available_features = []
    for feature in feature_list:
        if feature in df.columns:
            available_features.append(feature)
        else:
            logger.warning(f"Feature '{feature}' not available for {asset_name}, skipping")
    
    # Cross-asset features removed (single asset only)
    
    # Ensure we have the essential features
    essential_features = ['Close', 'Open', 'High', 'Low']
    for feature in essential_features:
        if feature not in available_features and feature in df.columns:
            available_features.append(feature)
    
    logger.info(f"Available features for {asset_name}: {available_features}")
    
    # Select only available features
    feature_df = df[['Date'] + available_features].copy()
    
    # Handle any remaining NaN values
    # For technical indicators, forward fill then backward fill
    for col in available_features:
        if col not in ['Date']:
            feature_df[col] = feature_df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Final check - remove any rows that still have NaN
    initial_rows = len(feature_df)
    feature_df = feature_df.dropna()
    
    if len(feature_df) != initial_rows:
        logger.warning(f"Dropped {initial_rows - len(feature_df)} rows with NaN values for {asset_name}")
    
    return feature_df

def create_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'Close',
    sequence_length: int = 60,
    horizon: int = 1,
    stride: int = 1,
    predict_returns: bool = True  # NEW: Predict returns instead of prices
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    """
    Create sequences for LSTM training.

    Returns returns instead of absolute prices for stationarity.
    Returns: (X_sequences, y_sequences, feature_columns, current_prices)
    """
    logger = setup_logger()

    # Prepare feature matrix (exclude Date and target)
    feature_cols = [col for col in feature_columns if col not in ['Date', target_column]]
    X_cols = feature_cols + [target_column]  # Include target in features for sequence

    X_data = df[X_cols].values.astype(np.float32)

    if predict_returns:
        # CRITICAL CHANGE: Predict percentage returns instead of absolute prices
        # This makes the problem stationary and generalizable across price regimes
        current_prices = df[target_column].values.astype(np.float32)
        future_prices = df[target_column].shift(-horizon).values.astype(np.float32)

        # Compute percentage returns: (future - current) / current
        y_data = (future_prices - current_prices) / current_prices

        logger.info("🎯 PREDICTING RETURNS (stationary target) instead of prices")
    else:
        # Old behavior: predict absolute prices (non-stationary)
        y_data = df[target_column].shift(-horizon).values.astype(np.float32)
        current_prices = None
        logger.warning("⚠️  PREDICTING ABSOLUTE PRICES (non-stationary)")

    # Create sequences
    X_sequences = []
    y_sequences = []
    current_price_sequences = [] if predict_returns else None

    for i in range(0, len(X_data) - sequence_length - horizon + 1, stride):
        # Input sequence
        X_seq = X_data[i:i + sequence_length]

        # Target value (return or price)
        y_val = y_data[i + sequence_length - 1]

        if not np.isnan(y_val):  # Only add if target is valid
            X_sequences.append(X_seq)
            y_sequences.append(y_val)

            if predict_returns:
                # Store current price for converting predictions back to prices
                current_price_sequences.append(current_prices[i + sequence_length - 1])

    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.float32)

    if predict_returns:
        current_price_sequences = np.array(current_price_sequences, dtype=np.float32)
        logger.info(f"Return statistics: mean={y_sequences.mean():.4f}, std={y_sequences.std():.4f}")
        logger.info(f"Return range: [{y_sequences.min():.4f}, {y_sequences.max():.4f}]")

    logger.info(f"Created {len(X_sequences)} sequences of length {sequence_length}")
    logger.info(f"Feature shape: {X_sequences.shape}, Target shape: {y_sequences.shape}")

    return X_sequences, y_sequences, X_cols, current_price_sequences

def fit_scaler_on_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_columns: List[str]
) -> Tuple[StandardScaler, StandardScaler]:
    """Fit scalers on training data only."""
    logger = setup_logger()
    
    # Reshape X for scaler (combine all sequences and timesteps)
    X_reshaped = X_train.reshape(-1, X_train.shape[-1])
    
    # Fit feature scaler
    feature_scaler = StandardScaler()
    feature_scaler.fit(X_reshaped)
    
    # Fit target scaler (reshape y to be 2D)
    target_scaler = StandardScaler()
    target_scaler.fit(y_train.reshape(-1, 1))
    
    logger.info("Scalers fitted on training data")
    logger.info(f"Feature scaler mean: {feature_scaler.mean_[:3]}...")  # Show first 3
    logger.info(f"Target scaler mean: {target_scaler.mean_[0]:.2f}, std: {target_scaler.scale_[0]:.2f}")
    
    return feature_scaler, target_scaler

def transform_sequences(
    X: np.ndarray,
    y: np.ndarray,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform sequences using fitted scalers."""
    
    # Transform features
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled_reshaped = feature_scaler.transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(original_shape)
    
    # Transform targets
    y_scaled = target_scaler.transform(y.reshape(-1, 1)).flatten()
    
    return X_scaled.astype(np.float32), y_scaled.astype(np.float32)

def build_sequences(
    asset_csv_path: str,
    seq_len: int = 60,
    horizon: int = 1,
    features: List[str] = None,
    train_end_date: str = None,
    val_end_date: str = None
) -> Dict[str, Any]:
    """
    Main function to build sequences for LSTM training.
    
    Args:
        asset_csv_path: Path to cleaned asset CSV
        seq_len: Sequence length for LSTM
        horizon: Prediction horizon (days ahead)
        features: List of feature columns to use
        train_end_date: End date for training data (for temporal splits)
        val_end_date: End date for validation data
    
    Returns:
        Dictionary containing sequences, scalers, and metadata
    """
    logger = setup_logger()
    
    # Default features if not provided
    if features is None:
        features = ["Close", "Open", "High", "Low", "Volume", "ret_1d", "vol_10d", "zscore_20d", "rsi_14", "ema_10", "ema_20", "atr_14"]
    
    # Load processed data
    logger.info(f"Loading data from {asset_csv_path}")
    df = pd.read_csv(asset_csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract asset name
    asset_name = df['Asset'].iloc[0] if 'Asset' in df.columns else 'UNKNOWN'
    
    # Calculate technical indicators if not already present
    if 'ret_1d' not in df.columns:
        logger.info("Calculating technical indicators...")
        df = calculate_technical_indicators(df)
    
    # Prepare feature columns
    df = prepare_feature_columns(df, features, asset_name)
    
    # Split data temporally if dates provided
    if train_end_date:
        train_df = df[df['Date'] <= train_end_date].copy()
        if val_end_date:
            val_df = df[(df['Date'] > train_end_date) & (df['Date'] <= val_end_date)].copy()
            test_df = df[df['Date'] > val_end_date].copy()
        else:
            val_df = df[df['Date'] > train_end_date].copy()
            test_df = None
    else:
        # Use full dataset for sequence creation
        train_df = df.copy()
        val_df = None
        test_df = None
    
    # Create sequences for training data - NOW PREDICTING RETURNS
    X_train, y_train, feature_cols, train_current_prices = create_sequences(
        train_df,
        list(df.columns),
        target_column='Close',
        sequence_length=seq_len,
        horizon=horizon,
        predict_returns=True  # KEY CHANGE: Predict returns for stationarity
    )

    if len(X_train) == 0:
        raise ValueError(f"No valid sequences created for {asset_name}")

    # Fit scalers on training data only
    feature_scaler, target_scaler = fit_scaler_on_training_data(X_train, y_train, feature_cols)

    # Transform training data
    X_train_scaled, y_train_scaled = transform_sequences(X_train, y_train, feature_scaler, target_scaler)

    result = {
        'asset_name': asset_name,
        'X_train': X_train_scaled,
        'y_train': y_train_scaled,
        'train_current_prices': train_current_prices,  # NEW: Store for price conversion
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_columns': feature_cols,
        'sequence_length': seq_len,
        'horizon': horizon,
        'train_samples': len(X_train_scaled),
        'predict_returns': True  # NEW: Flag to indicate return-based prediction
    }
    
    # Process validation data if available
    if val_df is not None and len(val_df) > seq_len:
        X_val, y_val, _, val_current_prices = create_sequences(
            val_df, list(val_df.columns), 'Close', seq_len, horizon, predict_returns=True
        )
        if len(X_val) > 0:
            X_val_scaled, y_val_scaled = transform_sequences(X_val, y_val, feature_scaler, target_scaler)
            result['X_val'] = X_val_scaled
            result['y_val'] = y_val_scaled
            result['val_current_prices'] = val_current_prices  # NEW: For price conversion
            result['val_samples'] = len(X_val_scaled)

    # Process test data if available
    if test_df is not None and len(test_df) > seq_len:
        X_test, y_test, _, test_current_prices = create_sequences(
            test_df, list(test_df.columns), 'Close', seq_len, horizon, predict_returns=True
        )
        if len(X_test) > 0:
            X_test_scaled, y_test_scaled = transform_sequences(X_test, y_test, feature_scaler, target_scaler)
            result['X_test'] = X_test_scaled
            result['y_test'] = y_test_scaled
            result['test_current_prices'] = test_current_prices  # NEW: For price conversion
            result['test_samples'] = len(X_test_scaled)
    
    logger.info(f"Sequence building completed for {asset_name}")
    logger.info(f"Training samples: {result['train_samples']}")
    
    return result

if __name__ == "__main__":
    # Test the feature engineering pipeline
    logger = setup_logger()
    
    # This would typically be called after data ingestion
    try:
        # Example usage
        result = build_sequences(
            asset_csv_path='data/processed/sp_clean.csv',
            seq_len=60,
            horizon=1
        )
        logger.info("Feature engineering test completed successfully")
    except Exception as e:
        logger.error(f"Feature engineering test failed: {str(e)}")