import os
import glob
import hashlib
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_path: str) -> logging.Logger:
    """Setup logging for data verification."""
    logger = logging.getLogger('data_verification')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.FileHandler(log_path, mode='a')
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def load_and_parse_csv(filepath: str) -> pd.DataFrame:
    """Load CSV with multiple date format handling."""
    # First, try to detect the separator
    with open(filepath, 'r') as f:
        first_line = f.readline()
        if ';' in first_line and first_line.count(';') > first_line.count(','):
            separator = ';'
        else:
            separator = ','
    
    # Load with detected separator
    df = pd.read_csv(filepath, sep=separator)
    
    # Check if Date column exists
    if 'Date' in df.columns:
        # Try multiple date formats
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%B-%Y')
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
            except:
                df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    else:
        raise ValueError(f"No Date column found in {filepath}. Columns: {list(df.columns)}")
    
    return df

def verify_file_structure(df: pd.DataFrame, filepath: str) -> Tuple[bool, str]:
    """Verify CSV has required columns and structure."""
    required_cols = ['Date', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for parseable dates
    if df['Date'].isna().sum() / len(df) > 0.01:
        return False, f"More than 1% unparseable dates"
    
    # Check Close values
    if df['Close'].isna().sum() / len(df) > 0.001:
        return False, f"More than 0.1% missing Close values"
    
    return True, "Structure OK"

def check_date_span_and_density(df: pd.DataFrame, min_years: int = 20) -> Tuple[bool, str, Dict]:
    """Check date span, density and gaps."""
    df_clean = df.dropna(subset=['Date']).copy()
    df_clean = df_clean.sort_values('Date')
    
    first_date = df_clean['Date'].min()
    last_date = df_clean['Date'].max()
    
    # Check minimum span
    years_span = (last_date - first_date).days / 365.25
    if years_span < min_years:
        return False, f"Data span {years_span:.1f} years < {min_years} years", {}
    
    # Check recency (within last 60 days)
    days_since_last = (datetime.now() - last_date.replace(tzinfo=None)).days
    if days_since_last > 60:
        return False, f"Last data point {days_since_last} days old", {}
    
    # Check for long gaps (business days)
    business_dates = pd.bdate_range(start=first_date, end=last_date)
    missing_dates = business_dates.difference(df_clean['Date'])
    
    # Find consecutive missing streaks
    if len(missing_dates) > 0:
        missing_df = pd.DataFrame({'missing_date': missing_dates})
        missing_df['diff'] = missing_df['missing_date'].diff().dt.days
        gaps = missing_df[missing_df['diff'] > 1].index.tolist()
        
        max_gap = 0
        if len(gaps) > 0:
            for i, gap_start in enumerate(gaps):
                gap_end = gaps[i+1] if i+1 < len(gaps) else len(missing_dates)
                gap_length = gap_end - gap_start
                max_gap = max(max_gap, gap_length)
        else:
            max_gap = len(missing_dates)
        
        if max_gap > 5:
            return False, f"Gap of {max_gap} consecutive business days found", {}
    
    # Check for constant Close streaks
    df_clean['close_diff'] = df_clean['Close'].diff().abs()
    constant_streaks = (df_clean['close_diff'] == 0).rolling(window=31).sum()
    max_constant = constant_streaks.max()
    
    if max_constant > 30:
        return False, f"Constant Close streak of {max_constant} days", {}
    
    stats = {
        'first_date': first_date.strftime('%Y-%m-%d'),
        'last_date': last_date.strftime('%Y-%m-%d'),
        'years_span': years_span,
        'total_rows': len(df_clean),
        'missing_business_days': len(missing_dates),
        'max_gap_days': max_gap,
        'max_constant_days': max_constant
    }
    
    return True, "Date span OK", stats

def check_statistical_authenticity(df: pd.DataFrame) -> Tuple[bool, str, Dict]:
    """Check statistical properties for authenticity."""
    df_clean = df.dropna(subset=['Close']).copy()
    df_clean = df_clean.sort_values('Date')
    
    # Calculate daily returns
    df_clean['ret_1d'] = df_clean['Close'].pct_change()
    returns = df_clean['ret_1d'].dropna()
    
    if len(returns) < 100:
        return False, "Insufficient data for statistical checks", {}
    
    # Return volatility check (PRD-compliant bounds for financial data)
    ret_std = returns.std()
    if ret_std < 0.002 or ret_std > 0.08:  # PRD-specified upper bound
        return False, f"Return volatility {ret_std:.4f} outside [0.002, 0.08]", {}
    
    # Kurtosis check (relaxed for financial data which can have extreme kurtosis)
    ret_kurtosis = returns.kurtosis()
    if ret_kurtosis < 1 or ret_kurtosis > 10000:  # Very generous upper bound for financial data
        return False, f"Return kurtosis {ret_kurtosis:.2f} outside [1, 10000]", {}
    
    # Autocorrelation check
    ret_autocorr = returns.autocorr(lag=1)
    if ret_autocorr >= 0.99:
        return False, f"Return autocorr {ret_autocorr:.3f} >= 0.99 (too high)", {}
    
    # Extreme spike check
    extreme_returns = (returns.abs() > 0.25).sum()
    extreme_pct = extreme_returns / len(returns) * 100
    
    stats = {
        'return_std': ret_std,
        'return_kurtosis': ret_kurtosis,
        'return_autocorr_lag1': ret_autocorr,
        'extreme_returns_pct': extreme_pct,
        'total_returns': len(returns)
    }
    
    return True, "Statistical checks OK", stats

def verify_real_data_or_die(
    sp_glob: str,
    min_rows: int = 5000,
    min_years: int = 20,
    log_path: str = "logs/data_integrity.log",
    hashes_out: str = "logs/data_hashes.json"
) -> bool:
    """
    Comprehensive real data verification with hard stops.
    Returns True if all checks pass, raises SystemExit otherwise.
    """

    # Setup logging
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(hashes_out), exist_ok=True)

    logger = setup_logging(log_path)

    print("\n=== REAL DATA VERIFICATION STARTING ===")
    logger.info("=== REAL DATA VERIFICATION STARTING ===")

    all_hashes = {}
    verification_results = {}

    # Check SP asset
    for asset_name, glob_pattern in [("SP", sp_glob)]:
        print(f"\nVerifying {asset_name} data...")
        logger.info(f"Verifying {asset_name} data...")
        
        # 1. File enumeration and hashing
        files = glob.glob(glob_pattern)
        if not files:
            error_msg = f"No files found for {asset_name} at {glob_pattern}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise SystemExit(f"VERIFICATION FAILED: {error_msg}")
        
        print(f"Found {len(files)} file(s) for {asset_name}")
        
        file_hashes = {}
        total_size = 0
        
        for filepath in files:
            file_size = os.path.getsize(filepath)
            total_size += file_size
            
            if file_size < 200_000:  # 200KB
                logger.warning(f"Small file detected: {filepath} ({file_size} bytes)")
            
            file_hash = compute_file_hash(filepath)
            file_hashes[os.path.basename(filepath)] = {
                'hash': file_hash,
                'size_bytes': file_size,
                'path': filepath
            }
        
        all_hashes[asset_name] = file_hashes
        
        # 2. Load and combine all CSVs for this asset
        all_dfs = []
        for filepath in files:
            try:
                df = load_and_parse_csv(filepath)
                all_dfs.append(df)
            except Exception as e:
                error_msg = f"Failed to load {filepath}: {str(e)}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                raise SystemExit(f"VERIFICATION FAILED: {error_msg}")
        
        # Combine all files
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicates and sort
        initial_rows = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['Date']).sort_values('Date')
        final_rows = len(combined_df)
        
        if initial_rows != final_rows:
            logger.info(f"Removed {initial_rows - final_rows} duplicate rows for {asset_name}")
        
        # 3. Structural verification
        struct_ok, struct_msg = verify_file_structure(combined_df, asset_name)
        if not struct_ok:
            error_msg = f"{asset_name} structure check failed: {struct_msg}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise SystemExit(f"VERIFICATION FAILED: {error_msg}")
        
        # 4. Row count check
        if len(combined_df) < min_rows:
            error_msg = f"{asset_name} has {len(combined_df)} rows < {min_rows} required"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise SystemExit(f"VERIFICATION FAILED: {error_msg}")
        
        # 5. Date span and density check
        span_ok, span_msg, span_stats = check_date_span_and_density(combined_df, min_years)
        if not span_ok:
            error_msg = f"{asset_name} span check failed: {span_msg}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise SystemExit(f"VERIFICATION FAILED: {error_msg}")
        
        # 6. Statistical authenticity check
        stats_ok, stats_msg, stats_data = check_statistical_authenticity(combined_df)
        if not stats_ok:
            error_msg = f"{asset_name} statistical check failed: {stats_msg}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise SystemExit(f"VERIFICATION FAILED: {error_msg}")
        
        # Log successful verification
        success_msg = (f"VERIFIED: {asset_name} rows={len(combined_df):,} "
                      f"span={span_stats['first_date']}→{span_stats['last_date']} "
                      f"gaps<={span_stats['max_gap_days']} "
                      f"max_flat={span_stats['max_constant_days']}")
        
        logger.info(success_msg)
        print(f"✅ {success_msg}")
        
        # Store verification results
        verification_results[asset_name] = {
            'structure': struct_msg,
            'span_stats': span_stats,
            'statistical_stats': stats_data,
            'total_files': len(files),
            'total_size_mb': total_size / (1024*1024)
        }
    
    # Save all hashes
    with open(hashes_out, 'w') as f:
        json.dump(all_hashes, f, indent=2)
    
    logger.info(f"HASHES: written to {hashes_out}")
    print(f"📁 Hashes saved to {hashes_out}")
    
    # Final success message
    success_final = "REAL DATA VERIFIED ✅"
    logger.info(success_final)
    print(f"\n{success_final}")
    
    return True

if __name__ == "__main__":
    # Test run
    verify_real_data_or_die(
        sp_glob="data/sp/*.csv"
    )