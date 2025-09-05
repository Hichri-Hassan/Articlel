import numpy as np
import pandas as pd

def macd_crossover_strategy(df: pd.DataFrame) -> np.ndarray:
    """
    Classifies Buy/Sell/Hold based on MACD crossover strategy.
    Assumes df contains 'MACD' and 'MACD_signal' columns.

    Returns:
        np.ndarray of predictions ['BUY', 'SELL', 'HOLD']
    """
    if 'MACD' not in df.columns or 'MACD_signal' not in df.columns:
        raise ValueError("DataFrame must contain 'MACD' and 'MACD_signal' columns.")

    conditions = [
        df['MACD'] > df['MACD_signal'],     # Buy signal
        df['MACD'] < df['MACD_signal'],     # Sell signal
    ]
    choices = ['BUY', 'SELL']
    return np.select(conditions, choices, default='HOLD')


def rsi_threshold_strategy(df: pd.DataFrame) -> np.ndarray:
    """
    Classifies Buy/Sell/Hold based on RSI thresholds.
    Assumes df contains 'RSI' column.

    Returns:
        np.ndarray of predictions ['BUY', 'SELL', 'HOLD']
    """
    if 'RSI' not in df.columns:
        raise ValueError("DataFrame must contain 'RSI' column.")

    conditions = [
        df['RSI'] < 30,     # Oversold → Buy
        df['RSI'] > 70,     # Overbought → Sell
    ]
    choices = ['BUY', 'SELL']
    return np.select(conditions, choices, default='HOLD')


def random_strategy(df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """
    Randomly assigns Buy/Sell/Hold to each row with equal probability.

    Args:
        df: DataFrame with any number of rows
        seed: Random seed for reproducibility

    Returns:
        np.ndarray of predictions ['BUY', 'SELL', 'HOLD']
    """
    np.random.seed(seed)
    return np.random.choice(['BUY', 'SELL', 'HOLD'], size=len(df))
