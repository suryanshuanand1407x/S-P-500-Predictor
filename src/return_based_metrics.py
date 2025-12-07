"""
Return-Based Evaluation Metrics for LSTM Price Predictor

This module provides evaluation metrics computed on percentage returns rather than
absolute prices, making them comparable across different price regimes and time periods.

Metrics:
- R² on returns (primary accuracy metric)
- MAE on returns (mean absolute error in percentage points)
- sMAPE (symmetric mean absolute percentage error)
- MASE (mean absolute scaled error)
- Directional Accuracy (percentage of correct up/down predictions)
- RMSE/MAPE on prices (secondary, for reference only)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate percentage returns from prices.

    Args:
        prices: Array of prices

    Returns:
        Array of returns (same length, first element is NaN)
    """
    returns = np.zeros_like(prices, dtype=float)
    returns[0] = np.nan
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    return returns


def prices_to_returns(current_prices: np.ndarray, predicted_prices: np.ndarray,
                      actual_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert price predictions to return predictions.

    Args:
        current_prices: Current/base prices (t)
        predicted_prices: Predicted future prices (t+1)
        actual_prices: Actual future prices (t+1)

    Returns:
        Tuple of (predicted_returns, actual_returns)
    """
    predicted_returns = (predicted_prices - current_prices) / current_prices
    actual_returns = (actual_prices - current_prices) / current_prices
    return predicted_returns, actual_returns


def r2_on_returns(actual_returns: np.ndarray, predicted_returns: np.ndarray) -> float:
    """
    Calculate R² score on returns.

    This is the primary accuracy metric. Even R² > 0.01 (1% of variance explained)
    is valuable in financial prediction.

    Args:
        actual_returns: Actual percentage returns
        predicted_returns: Predicted percentage returns

    Returns:
        R² score (-inf to 1.0, higher is better)
    """
    # Remove any NaN values
    mask = ~(np.isnan(actual_returns) | np.isnan(predicted_returns))
    if mask.sum() < 2:
        return np.nan

    return r2_score(actual_returns[mask], predicted_returns[mask])


def mae_on_returns(actual_returns: np.ndarray, predicted_returns: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error on returns (in percentage points).

    Example: MAE = 0.015 means average error of 1.5 percentage points.

    Args:
        actual_returns: Actual percentage returns
        predicted_returns: Predicted percentage returns

    Returns:
        MAE in percentage points (0 to infinity, lower is better)
    """
    mask = ~(np.isnan(actual_returns) | np.isnan(predicted_returns))
    if mask.sum() == 0:
        return np.nan

    return mean_absolute_error(actual_returns[mask], predicted_returns[mask])


def smape(actual_returns: np.ndarray, predicted_returns: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    sMAPE is symmetric (no bias) and bounded [0, 200%].
    Better than standard MAPE for financial returns.

    Formula: 100 * mean(|predicted - actual| / (|predicted| + |actual|))

    Args:
        actual_returns: Actual percentage returns
        predicted_returns: Predicted percentage returns

    Returns:
        sMAPE percentage (0 to 200, lower is better)
    """
    mask = ~(np.isnan(actual_returns) | np.isnan(predicted_returns))
    if mask.sum() == 0:
        return np.nan

    actual = actual_returns[mask]
    predicted = predicted_returns[mask]

    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted))

    # Handle cases where both are zero
    denominator = np.maximum(denominator, 1e-10)

    return 100 * np.mean(numerator / denominator)


def mase(actual_returns: np.ndarray, predicted_returns: np.ndarray,
         naive_mae: Optional[float] = None) -> float:
    """
    Calculate Mean Absolute Scaled Error.

    MASE compares prediction error to a naive baseline (typically random walk).
    MASE < 1.0 means better than naive baseline.

    Args:
        actual_returns: Actual percentage returns
        predicted_returns: Predicted percentage returns
        naive_mae: MAE of naive forecast (if None, uses random walk = mean(|actual|))

    Returns:
        MASE ratio (0 to infinity, <1.0 is good, lower is better)
    """
    mask = ~(np.isnan(actual_returns) | np.isnan(predicted_returns))
    if mask.sum() < 2:
        return np.nan

    actual = actual_returns[mask]
    predicted = predicted_returns[mask]

    # Model MAE
    model_mae = np.mean(np.abs(predicted - actual))

    # Naive baseline: random walk (predict zero return)
    if naive_mae is None:
        naive_mae = np.mean(np.abs(actual))

    if naive_mae == 0:
        return np.nan

    return model_mae / naive_mae


def directional_accuracy(actual_returns: np.ndarray, predicted_returns: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct up/down predictions).

    This is crucial for trading - being right about direction matters more
    than exact magnitude.

    Args:
        actual_returns: Actual percentage returns
        predicted_returns: Predicted percentage returns

    Returns:
        Directional accuracy percentage (0 to 100, higher is better)
    """
    mask = ~(np.isnan(actual_returns) | np.isnan(predicted_returns))
    if mask.sum() == 0:
        return np.nan

    actual = actual_returns[mask]
    predicted = predicted_returns[mask]

    # Sign agreement
    correct_direction = np.sign(actual) == np.sign(predicted)

    return 100 * np.mean(correct_direction)


def price_metrics_secondary(current_prices: np.ndarray, predicted_prices: np.ndarray,
                            actual_prices: np.ndarray) -> Dict[str, float]:
    """
    Calculate secondary price-based metrics for reference only.

    These are less meaningful across different price regimes but included
    for backward compatibility and intuition.

    Args:
        current_prices: Current/base prices
        predicted_prices: Predicted future prices
        actual_prices: Actual future prices

    Returns:
        Dictionary with RMSE and MAPE on prices
    """
    mask = ~(np.isnan(actual_prices) | np.isnan(predicted_prices))
    if mask.sum() == 0:
        return {'price_rmse': np.nan, 'price_mape': np.nan}

    actual = actual_prices[mask]
    predicted = predicted_prices[mask]

    # RMSE on prices
    price_rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE on prices
    price_mape = 100 * np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8)))

    return {
        'price_rmse': price_rmse,
        'price_mape': price_mape
    }


def comprehensive_evaluation(current_prices: np.ndarray,
                            predicted_prices: np.ndarray,
                            actual_prices: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive return-based evaluation metrics.

    Args:
        current_prices: Current/base prices (t)
        predicted_prices: Predicted future prices (t+1)
        actual_prices: Actual future prices (t+1)

    Returns:
        Dictionary with all metrics
    """
    # Convert to returns
    predicted_returns, actual_returns = prices_to_returns(
        current_prices, predicted_prices, actual_prices
    )

    # Primary metrics (on returns)
    metrics = {
        'return_r2': r2_on_returns(actual_returns, predicted_returns),
        'return_mae': mae_on_returns(actual_returns, predicted_returns),
        'return_smape': smape(actual_returns, predicted_returns),
        'return_mase': mase(actual_returns, predicted_returns),
        'directional_accuracy': directional_accuracy(actual_returns, predicted_returns),
    }

    # Secondary metrics (on prices, for reference)
    price_metrics = price_metrics_secondary(current_prices, predicted_prices, actual_prices)
    metrics.update(price_metrics)

    # Add return statistics for interpretation
    mask = ~np.isnan(actual_returns)
    if mask.sum() > 0:
        metrics['return_mean'] = np.mean(actual_returns[mask])
        metrics['return_std'] = np.std(actual_returns[mask])
        metrics['return_min'] = np.min(actual_returns[mask])
        metrics['return_max'] = np.max(actual_returns[mask])

    return metrics


def interpret_metrics(metrics: Dict[str, float]) -> str:
    """
    Provide human-readable interpretation of metrics.

    Args:
        metrics: Dictionary from comprehensive_evaluation()

    Returns:
        Formatted string with interpretation
    """
    interpretation = []

    # R² interpretation
    r2 = metrics.get('return_r2', np.nan)
    if not np.isnan(r2):
        if r2 > 0.05:
            r2_interp = "EXCELLENT (explains >5% of return variance)"
        elif r2 > 0.01:
            r2_interp = "GOOD (explains 1-5% of return variance)"
        elif r2 > 0:
            r2_interp = "WEAK (explains <1% of return variance)"
        else:
            r2_interp = "POOR (worse than predicting mean)"
        interpretation.append(f"R² on returns: {r2:.4f} - {r2_interp}")

    # MAE interpretation
    mae = metrics.get('return_mae', np.nan)
    if not np.isnan(mae):
        interpretation.append(f"MAE on returns: {mae:.4f} ({mae*100:.2f} percentage points average error)")

    # sMAPE interpretation
    smape_val = metrics.get('return_smape', np.nan)
    if not np.isnan(smape_val):
        interpretation.append(f"sMAPE: {smape_val:.2f}% (symmetric error)")

    # MASE interpretation
    mase_val = metrics.get('return_mase', np.nan)
    if not np.isnan(mase_val):
        if mase_val < 1.0:
            mase_interp = "GOOD (better than random walk)"
        else:
            mase_interp = "POOR (worse than random walk)"
        interpretation.append(f"MASE: {mase_val:.4f} - {mase_interp}")

    # Directional accuracy interpretation
    dir_acc = metrics.get('directional_accuracy', np.nan)
    if not np.isnan(dir_acc):
        if dir_acc > 55:
            dir_interp = "EXCELLENT (trading signal quality)"
        elif dir_acc > 52:
            dir_interp = "GOOD (useful for trading)"
        elif dir_acc > 48:
            dir_interp = "WEAK (near random)"
        else:
            dir_interp = "POOR (anti-predictive)"
        interpretation.append(f"Directional Accuracy: {dir_acc:.2f}% - {dir_interp}")

    return "\n".join(interpretation)


if __name__ == "__main__":
    # Example usage
    print("Return-Based Metrics Module")
    print("===========================")
    print("\nExample: Evaluate a simple prediction")

    # Simulated data
    current_prices = np.array([100, 100, 100, 100, 100])
    actual_prices = np.array([101, 99, 102, 98, 103])  # Returns: +1%, -1%, +2%, -2%, +3%
    predicted_prices = np.array([100.5, 99.5, 101.5, 98.5, 102.5])  # Close but not perfect

    metrics = comprehensive_evaluation(current_prices, predicted_prices, actual_prices)

    print("\nMetrics:")
    for key, value in metrics.items():
        if not np.isnan(value):
            print(f"  {key}: {value:.4f}")

    print("\nInterpretation:")
    print(interpret_metrics(metrics))
