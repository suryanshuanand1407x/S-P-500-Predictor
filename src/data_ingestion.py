import os
import glob
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

def setup_logger(name: str = 'data_ingestion') -> logging.Logger:
    """Setup logger for data ingestion."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def load_csv_with_multiple_formats(filepath: str) -> pd.DataFrame:
    """Load CSV handling different separators and date formats."""
    filename = os.path.basename(filepath)
    
    # First, try to detect the separator by reading the first line
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        if ';' in first_line and first_line.count(';') >= 5:  # Should have at least 5 semicolons for 6 columns
            separator = ';'
        else:
            separator = ','
    
    # Load with detected separator
    df = pd.read_csv(filepath, sep=separator)
    
    # Check if we got the columns correctly
    if len(df.columns) == 1 and ';' in df.columns[0]:
        # The separator detection failed, force semicolon
        df = pd.read_csv(filepath, sep=';')
    
    # Handle different date formats
    if 'Date' in df.columns:
        # SP format: d-Month-Year
        if 'SP' in filename.upper() or 'SENSEX' in filename.upper():
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%B-%Y')
            except:
                df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        else:
            try:
                df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
            except:
                df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def clean_and_validate_data(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """Clean and validate price data."""
    logger = setup_logger()
    
    # Ensure required columns exist
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns for {asset_name}: {missing_cols}")
    
    # Add Volume column if missing (set to 0 instead of NaN)
    if 'Volume' not in df.columns:
        df['Volume'] = 0.0
    
    # Convert to standard column set
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Convert price columns to numeric
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing dates or close prices
    initial_rows = len(df)
    df = df.dropna(subset=['Date', 'Close'])
    
    if len(df) != initial_rows:
        logger.info(f"Removed {initial_rows - len(df)} rows with missing Date/Close for {asset_name}")
    
    # Remove duplicates by date
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['Date'], keep='last')
    
    if len(df) != initial_rows:
        logger.info(f"Removed {initial_rows - len(df)} duplicate dates for {asset_name}")
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Validate price relationships (High >= Low, Close/Open between High/Low)
    invalid_hloc = (df['High'] < df['Low']) | (df['High'] < df['Close']) | (df['Low'] > df['Close'])
    if invalid_hloc.any():
        logger.warning(f"Found {invalid_hloc.sum()} rows with invalid HLOC relationships for {asset_name}")
        # Cap invalid values
        df.loc[df['High'] < df['Low'], 'High'] = df.loc[df['High'] < df['Low'], 'Low']
        df.loc[df['High'] < df['Close'], 'High'] = df.loc[df['High'] < df['Close'], 'Close']
        df.loc[df['Low'] > df['Close'], 'Low'] = df.loc[df['Low'] > df['Close'], 'Close']
    
    # Check for negative prices
    negative_prices = (df[['Open', 'High', 'Low', 'Close']] < 0).any(axis=1)
    if negative_prices.any():
        logger.warning(f"Found {negative_prices.sum()} rows with negative prices for {asset_name}")
        df = df[~negative_prices].reset_index(drop=True)
    
    return df

def detect_and_cap_outliers(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """Detect and cap extreme outliers using IQR method."""
    logger = setup_logger()
    
    df = df.copy()
    
    # Calculate daily returns
    df['daily_return'] = df['Close'].pct_change()
    
    # IQR method for outlier detection
    Q1 = df['daily_return'].quantile(0.25)
    Q3 = df['daily_return'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds (15x IQR as per PRD)
    lower_bound = Q1 - 15 * IQR
    upper_bound = Q3 + 15 * IQR
    
    # Also check for >50% daily moves (clearly erroneous data)
    extreme_moves = (df['daily_return'].abs() > 0.50) | \
                   (df['daily_return'] < lower_bound) | \
                   (df['daily_return'] > upper_bound)
    
    if extreme_moves.any():
        logger.warning(f"Found {extreme_moves.sum()} extreme price moves for {asset_name}")
        logger.warning(f"Extreme moves detected (review required):")
        logger.warning(f"\n{df.loc[extreme_moves, ['Date', 'Close', 'daily_return']].to_string()}")

        # For clearly erroneous data (>50% moves), log extensively and interpolate
        very_extreme = df['daily_return'].abs() > 0.50
        if very_extreme.any():
            logger.critical(f"⚠️  CRITICAL DATA QUALITY ALERT ⚠️")
            logger.critical(f"{very_extreme.sum()} data points with >50% daily moves detected in {asset_name}!")
            logger.critical(f"These extreme values will be INTERPOLATED - manual verification recommended:")

            # Log each extreme point individually for visibility
            extreme_points = df.loc[very_extreme, ['Date', 'Open', 'High', 'Low', 'Close', 'daily_return']].copy()
            for idx, row in extreme_points.iterrows():
                logger.critical(f"  Date: {row['Date']}, Close: {row['Close']:.2f}, Return: {row['daily_return']*100:.1f}%")

            # Save extreme points report to file for audit trail
            os.makedirs('logs', exist_ok=True)
            extreme_report_path = f'logs/{asset_name}_extreme_moves_interpolated.csv'
            extreme_points.to_csv(extreme_report_path, index=False)
            logger.critical(f"📊 Extreme moves report saved to: {extreme_report_path}")

            # Replace extreme values with NaN and interpolate
            logger.warning(f"Interpolating {very_extreme.sum()} extreme data points...")
            df.loc[very_extreme, 'Close'] = np.nan
            df.loc[very_extreme, 'Open'] = np.nan
            df.loc[very_extreme, 'High'] = np.nan
            df.loc[very_extreme, 'Low'] = np.nan

            # Interpolate the missing values
            df['Close'] = df['Close'].interpolate(method='linear', limit=3)
            df['Open'] = df['Open'].interpolate(method='linear', limit=3)
            df['High'] = df['High'].interpolate(method='linear', limit=3)
            df['Low'] = df['Low'].interpolate(method='linear', limit=3)

            # Fix HLOC relationships after interpolation
            df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
            df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))

            logger.warning(f"✓ Interpolation complete. Original extreme values preserved in report.")
    
    # Remove the temporary column
    df = df.drop('daily_return', axis=1)
    
    return df

def resample_to_business_days(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """Resample data to business days with forward fill."""
    logger = setup_logger()
    
    df = df.copy()
    
    # Sort by date first
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Check for any NaN values before processing
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values before resampling for {asset_name}")
        print(f"NaN locations: {df.isnull().sum()}")
        
        # Drop any rows that are completely NaN
        df = df.dropna(how='all')
        
        # For columns with NaN, fill with appropriate methods
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                # Forward fill, then backward fill, then use 0 for volume or median for prices
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                if df[col].isnull().any():
                    if col == 'Volume':
                        df[col] = df[col].fillna(0.0)
                    else:
                        df[col] = df[col].fillna(df[col].median())
    
    # Final check - ensure no NaN values remain
    final_nan_count = df.isnull().sum().sum()
    if final_nan_count > 0:
        logger.error(f"Still have {final_nan_count} NaN values after cleaning for {asset_name}")
        df = df.dropna()
    
    logger.info(f"Processed {asset_name} to {len(df)} days (keeping original trading calendar)")
    
    return df

def add_timezone_and_asset_info(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """Add timezone info and asset column."""
    df = df.copy()
    
    # Add timezone (Asia/Kolkata) - only if not already timezone-aware
    if df['Date'].dt.tz is None:
        kolkata_tz = pytz.timezone('Asia/Kolkata')
        df['Date'] = df['Date'].dt.tz_localize(kolkata_tz, ambiguous='infer')
    
    # Add asset column
    df['Asset'] = asset_name
    
    # Reorder columns to match canonical schema
    column_order = ['Date', 'Asset', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[column_order]
    
    return df

def process_asset_data(glob_pattern: str, asset_name: str, output_path: str) -> Dict:
    """Process all CSV files for a single asset."""
    logger = setup_logger()
    
    logger.info(f"Processing {asset_name} data from {glob_pattern}")
    
    # Find all CSV files
    csv_files = glob.glob(glob_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for {asset_name} at {glob_pattern}")
    
    logger.info(f"Found {len(csv_files)} CSV files for {asset_name}")
    
    # Load and combine all files
    all_dfs = []
    
    for csv_file in csv_files:
        logger.info(f"Loading {csv_file}")
        try:
            df = load_csv_with_multiple_formats(csv_file)
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {str(e)}")
            raise
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined data: {len(combined_df)} rows")
    
    # Clean and validate
    clean_df = clean_and_validate_data(combined_df, asset_name)
    logger.info(f"After cleaning: {len(clean_df)} rows")
    
    # Detect outliers
    outlier_checked_df = detect_and_cap_outliers(clean_df, asset_name)
    
    # Resample to business days
    resampled_df = resample_to_business_days(outlier_checked_df, asset_name)
    logger.info(f"After resampling: {len(resampled_df)} rows")
    
    # Add timezone and asset info
    final_df = add_timezone_and_asset_info(resampled_df, asset_name)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Saved processed {asset_name} data to {output_path}")
    
    # Return summary statistics
    summary = {
        'asset': asset_name,
        'input_files': len(csv_files),
        'final_rows': len(final_df),
        'date_range': {
            'start': final_df['Date'].min().strftime('%Y-%m-%d'),
            'end': final_df['Date'].max().strftime('%Y-%m-%d')
        },
        'output_path': output_path
    }
    
    return summary

def ingest_all_data(config_paths: Dict[str, str]) -> Dict[str, any]:
    """Main function to ingest all asset data according to config."""
    logger = setup_logger()
    
    logger.info("Starting data ingestion pipeline")
    
    results = {}

    # Process SP data
    try:
        sp_summary = process_asset_data(
            glob_pattern=config_paths['sp_raw_glob'],
            asset_name='SP',
            output_path=config_paths['sp_clean']
        )
        results['SP'] = sp_summary
        logger.info(f"✅ SP processing completed")
    except Exception as e:
        logger.error(f"❌ SP processing failed: {str(e)}")
        raise
    
    logger.info("Data ingestion pipeline completed successfully")
    
    return results

if __name__ == "__main__":
    # Test with default paths
    config_paths = {
        'sp_raw_glob': 'data/sp/*.csv',
        'sp_clean': 'data/processed/sp_clean.csv'
    }
    
    results = ingest_all_data(config_paths)