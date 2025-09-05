"""
Adaptive Labeling Module for Trading Predictor
=============================================

This module contains the adaptive labeling logic for creating actionable trading targets.
Implements sophisticated target creation with risk-adjusted returns and market volatility adaptation.

Features include:
- Multi-horizon future return calculation
- Risk-adjusted return metrics
- Enhanced comprehensive scoring system
- Dynamic threshold adjustment based on market conditions
- Adaptive target percentages for different market regimes
"""

import pandas as pd
import numpy as np
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def create_actionable_targets(df, target_buy_pct=20, target_sell_pct=20):
    """Create targets with enhanced scoring system and risk-adjusted returns"""
    print(f"🎯 Creating actionable targets with {target_buy_pct}% BUY and {target_sell_pct}% SELL...")
    
    # Calculate multiple future return horizons
    for horizon in [1, 2, 3, 5]:
        df[f'Future_Return_{horizon}D'] = df.groupby('Ticker')['Close'].pct_change(periods=horizon).shift(-horizon)
    
    # Calculate risk-adjusted returns (Sharpe-like ratio)
    for horizon in [1, 2, 3, 5]:
        df[f'Risk_Adj_Return_{horizon}D'] = df.groupby('Ticker').apply(
            lambda group: group[f'Future_Return_{horizon}D'] / (group[f'Volatility_20'] + 1e-10)
        ).reset_index(level=0, drop=True)
    
    # Enhanced comprehensive scoring system
    def calculate_action_score(row):
        score = 0
        
        # Multi-horizon future returns (35% weight)
        future_weights = {1: 0.4, 2: 0.3, 3: 0.2, 5: 0.1}
        for horizon, weight in future_weights.items():
            future_return = row.get(f'Future_Return_{horizon}D', 0)
            if pd.notna(future_return):
                score += future_return * 35 * weight
        
        # Risk-adjusted returns (15% weight)
        risk_adj_return = row.get('Risk_Adj_Return_1D', 0)
        if pd.notna(risk_adj_return):
            score += np.clip(risk_adj_return, -0.1, 0.1) * 15
        
        # Multiple RSI signals (15% weight)
        rsi_signals = 0
        rsi_count = 0
        for period in [9, 14, 21]:
            rsi_val = row.get(f'RSI_{period}', 50)
            if pd.notna(rsi_val):
                if rsi_val < 30:
                    rsi_signals += 0.15
                elif rsi_val < 40:
                    rsi_signals += 0.05
                elif rsi_val > 70:
                    rsi_signals -= 0.15
                elif rsi_val > 60:
                    rsi_signals -= 0.05
                rsi_count += 1
        if rsi_count > 0:
            score += (rsi_signals / rsi_count) * 15
        
        # Enhanced MACD signals (10% weight)
        macd_score = 0
        macd = row.get('MACD', 0)
        macd_signal = row.get('MACD_Signal', 0)
        macd_fast = row.get('MACD_Fast', 0)
        macd_signal_fast = row.get('MACD_Signal_Fast', 0)
        
        if pd.notna(macd) and pd.notna(macd_signal):
            macd_diff = macd - macd_signal
            macd_score += np.clip(macd_diff * 50, -0.1, 0.1)
        
        if pd.notna(macd_fast) and pd.notna(macd_signal_fast):
            macd_fast_diff = macd_fast - macd_signal_fast
            macd_score += np.clip(macd_fast_diff * 30, -0.05, 0.05)
        
        score += macd_score * 10
        
        # Multiple Bollinger Band signals (8% weight)
        bb_score = 0
        bb_count = 0
        for period in [10, 20, 50]:
            bb_position = row.get(f'BB_Position_{period}', 0.5)
            if pd.notna(bb_position):
                if bb_position < 0.1:
                    bb_score += 0.2
                elif bb_position < 0.3:
                    bb_score += 0.1
                elif bb_position > 0.9:
                    bb_score -= 0.2
                elif bb_position > 0.7:
                    bb_score -= 0.1
                bb_count += 1
        if bb_count > 0:
            score += (bb_score / bb_count) * 8
        
        # Moving average trend signals (7% weight)
        ma_score = 0
        sma_5_20 = row.get('SMA_5_20_Ratio', 1)
        sma_10_50 = row.get('SMA_10_50_Ratio', 1)
        ema_5_20 = row.get('EMA_5_20_Ratio', 1)
        
        if pd.notna(sma_5_20):
            if sma_5_20 > 1.02:
                ma_score += 0.1
            elif sma_5_20 < 0.98:
                ma_score -= 0.1
        
        if pd.notna(sma_10_50):
            if sma_10_50 > 1.01:
                ma_score += 0.05
            elif sma_10_50 < 0.99:
                ma_score -= 0.05
        
        if pd.notna(ema_5_20):
            if ema_5_20 > 1.015:
                ma_score += 0.05
            elif ema_5_20 < 0.985:
                ma_score -= 0.05
        
        score += ma_score * 7
        
        # Volume signals (5% weight)
        volume_score = 0
        for period in [5, 10, 20]:
            vol_ratio = row.get(f'Volume_Ratio_{period}', 1)
            if pd.notna(vol_ratio):
                if vol_ratio > 1.5:
                    volume_score += 0.05
                elif vol_ratio > 1.2:
                    volume_score += 0.02
        score += volume_score * 5
        
        # Momentum signals (5% weight)
        momentum_score = 0
        for period in [5, 10, 20]:
            momentum = row.get(f'Momentum_{period}', 0)
            if pd.notna(momentum):
                momentum_score += np.clip(momentum / 100, -0.05, 0.05)
        score += momentum_score * 5
        
        # Market relative performance (5% weight)
        rel_return = row.get('Relative_Return_1D', 0)
        if pd.notna(rel_return):
            score += np.clip(rel_return * 100, -0.1, 0.1) * 5
        
        # Volatility consideration (3% weight) - prefer lower volatility for same returns
        volatility = row.get('Volatility_20', 0.02)
        if pd.notna(volatility) and volatility > 0:
            vol_penalty = min(volatility * 10, 0.1)  # Cap penalty
            score -= vol_penalty * 3
        
        # Statistical features (2% weight)
        skew_5 = row.get('Return_Skew_5', 0)
        if pd.notna(skew_5):
            # Positive skew is good for long positions
            score += np.clip(skew_5 / 2, -0.05, 0.05) * 2
        
        return score
    
    print("📊 Calculating enhanced action scores...")
    df['Action_Score'] = df.apply(calculate_action_score, axis=1)
    
    # Dynamic threshold adjustment based on market conditions
    print("🎯 Creating adaptive targets...")
    
    # Calculate market regime (trending vs sideways)
    market_vol = df.groupby('Date')['Return_1D'].std().rolling(20, min_periods=5).mean()
    high_vol_threshold = market_vol.quantile(0.7)
    
    # Adjust target percentages based on market volatility
    df['Market_Vol_Regime'] = df['Date'].map(
        lambda x: market_vol.loc[market_vol.index <= x].iloc[-1] if len(market_vol.loc[market_vol.index <= x]) > 0 else market_vol.mean()
    )
    
    # In high volatility periods, be more selective (reduce action percentages)
    df['Adjusted_Buy_Pct'] = target_buy_pct
    df['Adjusted_Sell_Pct'] = target_sell_pct
    
    high_vol_mask = df['Market_Vol_Regime'] > high_vol_threshold
    df.loc[high_vol_mask, 'Adjusted_Buy_Pct'] = target_buy_pct * 0.8
    df.loc[high_vol_mask, 'Adjusted_Sell_Pct'] = target_sell_pct * 0.8
    
    # Create targets with adaptive thresholds
    valid_scores = df.dropna(subset=['Action_Score'])
    
    # Use time-based rolling thresholds for better adaptation
    df['Target'] = 1  # Default HOLD
    
    # Calculate rolling percentile thresholds
    window_size = min(1000, len(valid_scores) // 5)  # Adaptive window size
    
    for i in range(len(df)):
        if pd.isna(df.iloc[i]['Action_Score']):
            continue
            
        # Get recent data for threshold calculation
        start_idx = max(0, i - window_size)
        recent_scores = df.iloc[start_idx:i+1]['Action_Score'].dropna()
        
        if len(recent_scores) < 50:  # Need minimum data
            continue
        
        buy_pct = df.iloc[i]['Adjusted_Buy_Pct']
        sell_pct = df.iloc[i]['Adjusted_Sell_Pct']
        
        buy_threshold = recent_scores.quantile(1 - buy_pct / 100)
        sell_threshold = recent_scores.quantile(sell_pct / 100)
        
        current_score = df.iloc[i]['Action_Score']
        
        if current_score >= buy_threshold:
            df.iloc[i, df.columns.get_loc('Target')] = 2  # BUY
        elif current_score <= sell_threshold:
            df.iloc[i, df.columns.get_loc('Target')] = 0  # SELL
    
    # Final distribution
    final_counts = df['Target'].value_counts().sort_index()
    final_total = df['Target'].count()
    print(f"📊 Final target distribution:")
    for target, count in final_counts.items():
        if pd.notna(target):
            target_name = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}.get(int(target), f'Class_{int(target)}')
            pct = count / final_total * 100
            print(f"   {target_name}: {count} ({pct:.1f}%)")
    
    return df