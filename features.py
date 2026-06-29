"""
Feature Engineering Module for Trading Bot
Computes causal technical indicators (features) and forward-looking target labels
from clean OHLCV data. Binds list of allowed model features as FEATURE_COLS.
"""

import pandas as pd
import numpy as np

# The exact list of columns the model is allowed to see (features only, never the label)
FEATURE_COLS = [
    'ret_1',
    'ret_5',
    'sma_10',
    'sma_20',
    'sma_ratio',
    'price_vs_sma20',
    'macd',
    'macd_signal',
    'macd_hist',
    'rsi_14',
    'atr_14',
    'atr_pct',
    'volatility_20',
    'volume_vs_sma20'
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators from OHLCV data.
    All calculations are strictly CAUSAL (using only past/current data).
    
    Expected columns: open, high, low, close, volume.
    Returns a copy of the DataFrame with added indicator columns.
    """
    # Create a copy to prevent modifying the original DataFrame in-place
    df = df.copy()
    
    # 1. Returns
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    
    # 2. Simple Moving Averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # SMA Ratio (10 vs 20) and Price vs SMA20
    df['sma_ratio'] = df['sma_10'] / df['sma_20']
    df['price_vs_sma20'] = df['close'] / df['sma_20']
    
    # 3. MACD (Moving Average Convergence Divergence)
    # Standard: EMA(12) - EMA(26), with Signal Line = EMA(9) of MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 4. RSI (Relative Strength Index) - 14 bars
    # Using Wilder's smoothing (exponential moving average with alpha = 1/14)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    
    # Handle division by zero where avg_loss is 0
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # If loss is 0 but there is gain, RSI is 100. If both are 0, RSI is 50.
    df['rsi_14'] = rsi.fillna(100.0)
    
    # 5. Average True Range (ATR) - 14 bars
    # True Range (TR) = max(high - low, |high - prev_close|, |low - prev_close|)
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder's smoothing for ATR
    df['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # ATR as a percentage of price
    df['atr_pct'] = (df['atr_14'] / df['close']) * 100.0
    
    # 6. Rolling Volatility (20-bar standard deviation of returns)
    df['volatility_20'] = df['ret_1'].rolling(window=20).std()
    
    # 7. Volume vs its 20-bar average
    volume_sma_20 = df['volume'].rolling(window=20).mean()
    df['volume_vs_sma20'] = df['volume'] / volume_sma_20
    
    return df


def build_dataset(df: pd.DataFrame, horizon: int = 4, threshold: float = 0.01) -> pd.DataFrame:
    """
    Computes causal technical indicators, adds the forward-looking classification label,
    and drops warmup/NaN rows.
    
    Label logic:
      label = 1 if close price 'horizon' bars ahead is > current close * (1 + threshold)
      label = 0 otherwise
      
    Returns a clean DataFrame with no NaNs in FEATURE_COLS or 'label'.
    """
    # 1. Compute causal features
    df = compute_features(df)
    
    # 2. Compute label looking forward
    # shift(-horizon) looks forward into the future by horizon bars
    future_close = df['close'].shift(-horizon)
    
    # Calculate binary label, but preserve NaNs for the last horizon rows
    label_cond = future_close > df['close'] * (1 + threshold)
    df['label'] = np.where(future_close.isna(), np.nan, label_cond.astype(float))
    
    # 3. Clean dataset
    # Drop rows that contain NaNs in our features (due to warmups) or in the label (due to future shifts)
    clean_cols = FEATURE_COLS + ['label']
    df_clean = df.dropna(subset=clean_cols).copy()
    
    # Convert label back to integer type
    df_clean['label'] = df_clean['label'].astype(int)
    
    return df_clean
