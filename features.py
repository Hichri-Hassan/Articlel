"""
Feature Engineering Module for Trading Predictor
==============================================

This module contains all feature creation functions for the trading prediction system.
Implements comprehensive technical analysis and statistical feature engineering.

Features include:
- Basic price and volume features
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Statistical features and market-relative metrics
- Advanced volume-based indicators
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def comprehensive_feature_engineering(df):
    """Enhanced feature engineering with advanced technical indicators and statistical features"""
    print("🔧 Starting comprehensive feature engineering...")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    features_df = df.copy()
    
    print("📊 Creating basic features...")
    # Basic return features with more periods
    for period in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        features_df[f'Return_{period}D'] = features_df.groupby('Ticker')['Close'].pct_change(periods=period)
        features_df[f'Volume_Change_{period}D'] = features_df.groupby('Ticker')['Volume'].pct_change(periods=period)
        
        # Log returns for better distribution
        features_df[f'Log_Return_{period}D'] = features_df.groupby('Ticker')['Close'].transform(
            lambda x: np.log(x / x.shift(period))
        )
    
    print("📈 Creating advanced technical indicators...")
    # Enhanced moving averages
    for window in [3, 5, 8, 10, 13, 20, 21, 34, 50, 89]:
        features_df[f'SMA_{window}'] = features_df.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        features_df[f'EMA_{window}'] = features_df.groupby('Ticker')['Close'].transform(
            lambda x: x.ewm(span=window).mean()
        )
        features_df[f'Volatility_{window}'] = features_df.groupby('Ticker')['Return_1D'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        
        # Volume-weighted moving averages
        features_df[f'VWMA_{window}'] = features_df.groupby('Ticker').apply(
            lambda group: (group['Close'] * group['Volume']).rolling(window, min_periods=1).sum() /
                         group['Volume'].rolling(window, min_periods=1).sum()
        ).reset_index(level=0, drop=True)
    
    # Price ratios and momentum indicators
    for period in [5, 10, 20, 50]:
        features_df[f'Price_SMA_Ratio_{period}'] = features_df['Close'] / (features_df[f'SMA_{period}'] + 1e-10)
        features_df[f'Price_EMA_Ratio_{period}'] = features_df['Close'] / (features_df[f'EMA_{period}'] + 1e-10)
        
        # Momentum oscillator
        features_df[f'Momentum_{period}'] = features_df.groupby('Ticker')['Close'].transform(
            lambda x: (x / x.shift(period) - 1) * 100
        )
    
    # Multiple RSI periods
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    for rsi_period in [9, 14, 21]:
        features_df[f'RSI_{rsi_period}'] = features_df.groupby('Ticker')['Close'].transform(
            lambda x: calculate_rsi(x, rsi_period)
        )
    
    # Stochastic Oscillator
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(k_period, min_periods=1).min()
        highest_high = high.rolling(k_period, min_periods=1).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
        d_percent = k_percent.rolling(d_period, min_periods=1).mean()
        return k_percent, d_percent
    
    if 'High' in features_df.columns and 'Low' in features_df.columns:
        stoch_results = features_df.groupby('Ticker').apply(
            lambda group: calculate_stochastic(group['High'], group['Low'], group['Close'])
        )
        k_values, d_values = [], []
        for ticker in features_df['Ticker'].unique():
            ticker_data = stoch_results[ticker]
            k_values.extend(ticker_data[0].values)
            d_values.extend(ticker_data[1].values)
        features_df['Stoch_K'] = k_values
        features_df['Stoch_D'] = d_values
    
    # Enhanced MACD with multiple timeframes
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    # Standard MACD
    macd_results = features_df.groupby('Ticker')['Close'].apply(lambda x: calculate_macd(x, 12, 26, 9))
    macd_values, signal_values, hist_values = [], [], []
    
    for ticker in features_df['Ticker'].unique():
        ticker_data = macd_results[ticker]
        macd_values.extend(ticker_data[0].values)
        signal_values.extend(ticker_data[1].values)
        hist_values.extend(ticker_data[2].values)
    
    features_df['MACD'] = macd_values
    features_df['MACD_Signal'] = signal_values
    features_df['MACD_Hist'] = hist_values
    
    # Fast MACD
    macd_fast_results = features_df.groupby('Ticker')['Close'].apply(lambda x: calculate_macd(x, 5, 13, 5))
    macd_fast_values, signal_fast_values, hist_fast_values = [], [], []
    
    for ticker in features_df['Ticker'].unique():
        ticker_data = macd_fast_results[ticker]
        macd_fast_values.extend(ticker_data[0].values)
        signal_fast_values.extend(ticker_data[1].values)
        hist_fast_values.extend(ticker_data[2].values)
    
    features_df['MACD_Fast'] = macd_fast_values
    features_df['MACD_Signal_Fast'] = signal_fast_values
    features_df['MACD_Hist_Fast'] = hist_fast_values
    
    # Enhanced Bollinger Bands
    def calculate_bb(prices, period=20, std_dev=2):
        sma = prices.rolling(period, min_periods=1).mean()
        std = prices.rolling(period, min_periods=1).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    # Multiple Bollinger Band periods
    for bb_period in [10, 20, 50]:
        bb_results = features_df.groupby('Ticker')['Close'].apply(lambda x: calculate_bb(x, bb_period, 2))
        bb_upper_values, bb_middle_values, bb_lower_values = [], [], []
        
        for ticker in features_df['Ticker'].unique():
            ticker_data = bb_results[ticker]
            bb_upper_values.extend(ticker_data[0].values)
            bb_middle_values.extend(ticker_data[1].values)
            bb_lower_values.extend(ticker_data[2].values)
        
        features_df[f'BB_Upper_{bb_period}'] = bb_upper_values
        features_df[f'BB_Middle_{bb_period}'] = bb_middle_values
        features_df[f'BB_Lower_{bb_period}'] = bb_lower_values
        features_df[f'BB_Position_{bb_period}'] = (features_df['Close'] - features_df[f'BB_Lower_{bb_period}']) / (features_df[f'BB_Upper_{bb_period}'] - features_df[f'BB_Lower_{bb_period}'] + 1e-10)
        features_df[f'BB_Width_{bb_period}'] = (features_df[f'BB_Upper_{bb_period}'] - features_df[f'BB_Lower_{bb_period}']) / features_df[f'BB_Middle_{bb_period}']
    
    # Williams %R
    def calculate_williams_r(high, low, close, period=14):
        highest_high = high.rolling(period, min_periods=1).max()
        lowest_low = low.rolling(period, min_periods=1).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
        return williams_r
    
    if 'High' in features_df.columns and 'Low' in features_df.columns:
        features_df['Williams_R'] = features_df.groupby('Ticker').apply(
            lambda group: calculate_williams_r(group['High'], group['Low'], group['Close'])
        ).reset_index(level=0, drop=True)
    
    # Commodity Channel Index (CCI)
    def calculate_cci(high, low, close, period=20):
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(period, min_periods=1).mean()
        mad = tp.rolling(period, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
        return cci
    
    if 'High' in features_df.columns and 'Low' in features_df.columns:
        features_df['CCI'] = features_df.groupby('Ticker').apply(
            lambda group: calculate_cci(group['High'], group['Low'], group['Close'])
        ).reset_index(level=0, drop=True)
    
    print("📊 Creating statistical features...")
    # Statistical features for returns
    for window in [5, 10, 20]:
        features_df[f'Return_Skew_{window}'] = features_df.groupby('Ticker')['Return_1D'].transform(
            lambda x: x.rolling(window, min_periods=3).apply(lambda y: skew(y) if len(y) >= 3 else 0)
        )
        features_df[f'Return_Kurt_{window}'] = features_df.groupby('Ticker')['Return_1D'].transform(
            lambda x: x.rolling(window, min_periods=3).apply(lambda y: kurtosis(y) if len(y) >= 3 else 0)
        )
    
    # Volume features
    print("📊 Creating volume-based features...")
    for period in [5, 10, 20]:
        features_df[f'Volume_SMA_{period}'] = features_df.groupby('Ticker')['Volume'].transform(
            lambda x: x.rolling(period, min_periods=1).mean()
        )
        features_df[f'Volume_Ratio_{period}'] = features_df['Volume'] / (features_df[f'Volume_SMA_{period}'] + 1e-10)
        
        # On-Balance Volume
        features_df[f'OBV_{period}'] = features_df.groupby('Ticker').apply(
            lambda group: (group['Volume'] * np.sign(group['Return_1D'])).rolling(period, min_periods=1).sum()
        ).reset_index(level=0, drop=True)
    
    # Price and volume interaction features
    features_df['Price_Volume_Trend'] = features_df['Return_1D'] * features_df['Volume_Change_1D']
    features_df['Volume_Price_Trend'] = features_df['Volume_Change_1D'] * features_df['Return_1D']
    
    # Trend strength indicators
    features_df['SMA_5_20_Ratio'] = features_df['SMA_5'] / (features_df['SMA_20'] + 1e-10)
    features_df['SMA_10_50_Ratio'] = features_df['SMA_10'] / (features_df['SMA_50'] + 1e-10)
    features_df['EMA_5_20_Ratio'] = features_df['EMA_5'] / (features_df['EMA_20'] + 1e-10)
    
    # Market features
    print("📊 Creating market-relative features...")
    features_df['Market_Return_1D'] = features_df.groupby('Date')['Return_1D'].transform('mean')
    features_df['Market_Volume_1D'] = features_df.groupby('Date')['Volume_Change_1D'].transform('mean')
    features_df['Relative_Return_1D'] = features_df['Return_1D'] - features_df['Market_Return_1D']
    features_df['Relative_Volume_1D'] = features_df['Volume_Change_1D'] - features_df['Market_Volume_1D']
    
    # Market correlation features
    print("📊 Creating market correlation features...")
    for window in [5, 10, 20]:
        features_df[f'Market_Corr_{window}'] = features_df.groupby('Ticker').apply(
            lambda group: group['Return_1D'].rolling(window, min_periods=5).corr(group['Market_Return_1D'])
        ).reset_index(level=0, drop=True)
    
    # Beta calculation (rolling)
    for window in [20, 50]:
        features_df[f'Beta_{window}'] = features_df.groupby('Ticker').apply(
            lambda group: group['Return_1D'].rolling(window, min_periods=10).cov(group['Market_Return_1D']) /
                         (group['Market_Return_1D'].rolling(window, min_periods=10).var() + 1e-10)
        ).reset_index(level=0, drop=True)
    
    # Volatility features
    print("📊 Creating volatility features...")
    for window in [5, 10, 20]:
        # Realized volatility
        features_df[f'Realized_Vol_{window}'] = features_df.groupby('Ticker')['Return_1D'].transform(
            lambda x: x.rolling(window, min_periods=1).std() * np.sqrt(252)
        )
        
        # GARCH-like volatility
        features_df[f'Vol_of_Vol_{window}'] = features_df.groupby('Ticker')[f'Realized_Vol_{window}'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    # Clean data
    print("🧹 Cleaning and finalizing features...")
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    numeric_columns = features_df.select_dtypes(include=[np.number]).columns
    
    # Handle columns with all null values
    all_null_columns = []
    for col in numeric_columns:
        if features_df[col].isna().all():
            all_null_columns.append(col)
            # Drop columns that are completely null
            features_df = features_df.drop(col, axis=1)
            continue
        
        if features_df[col].isna().any():
            # Forward fill, then backward fill within each ticker
            features_df[col] = features_df.groupby('Ticker')[col].fillna(method='ffill').fillna(method='bfill')
            # If still NaN, use median
            if features_df[col].isna().any():
                median_val = features_df[col].median()
                if pd.notna(median_val):
                    features_df[col] = features_df[col].fillna(median_val)
                else:
                    # If median is also NaN, use 0
                    features_df[col] = features_df[col].fillna(0)
    
    if all_null_columns:
        logger.warning(f"Dropped {len(all_null_columns)} columns with all null values: {all_null_columns[:5]}{'...' if len(all_null_columns) > 5 else ''}")
    
    print(f"✅ Enhanced feature engineering completed. Shape: {features_df.shape}")
    if all_null_columns:
        print(f"   📝 Note: Dropped {len(all_null_columns)} columns with all null values")
    
    return features_df