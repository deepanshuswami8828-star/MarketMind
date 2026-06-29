"""
Strategy and Backtesting Module for Trading Bot
Provides chronological train/test splits, GradientBoosting model training,
probabilistic signal generation, walk-forward validation, and an event-driven,
long-only backtester with transaction costs and ATR-based stops.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from features import FEATURE_COLS


def train_test_split_time(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset chronologically without shuffling to prevent look-ahead bias.
    """
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def train_model(train_df: pd.DataFrame) -> GradientBoostingClassifier:
    """
    Trains a Gradient Boosting Classifier on FEATURE_COLS to predict label.
    """
    X = train_df[FEATURE_COLS]
    y = train_df['label']
    
    # Train Gradient Boosting Classifier with default hyperparameters for stability
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X, y)
    return model


def add_signals(model: GradientBoostingClassifier, df: pd.DataFrame, prob_threshold: float = 0.55) -> pd.DataFrame:
    """
    Generates signals by predicting probabilities on the feature columns.
    A long signal (1) is generated only when the model's predicted probability
    of an upward move is >= prob_threshold.
    """
    df = df.copy()
    X = df[FEATURE_COLS]
    
    # Predict probabilities for class 1 (upward price move)
    probs = model.predict_proba(X)[:, 1]
    
    df['prob'] = probs
    df['signal'] = (probs >= prob_threshold).astype(int)
    
    return df


def backtest(
    df: pd.DataFrame,
    cost_per_trade: float = 0.0006,
    atr_stop_mult: float = 1.5,
    atr_target_mult: float = 2.0,
    max_hold: int = 8
) -> tuple[pd.DataFrame, dict]:
    """
    Performs an event-driven, long-only backtest.
    
    Rules:
      - Only one position is active at any time.
      - Enter on the NEXT bar's open after a signal is generated at bar t.
      - Stop-loss = entry_price - atr_stop_mult * ATR(t)
      - Profit-target = entry_price + atr_target_mult * ATR(t)
      - If neither is hit, exit at close of entry_idx + max_hold - 1 (i.e. hold at most max_hold bars).
      - Subtracts transaction costs (cost_per_trade) on both entry and exit.
      
    Returns:
      trades_df (pd.DataFrame): Log of completed trades.
      summary (dict): Net metrics of the backtest.
    """
    # Verify signal and atr exist
    if 'signal' not in df.columns or 'atr_14' not in df.columns:
        raise ValueError("DataFrame must contain 'signal' and 'atr_14' columns.")
        
    dates = df.index
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['atr_14'].values
    signals = df['signal'].values
    
    trades = []
    position = None
    
    n_bars = len(df)
    i = 0
    
    while i < n_bars:
        if position is None:
            # Check for a signal at bar i to enter on bar i+1
            if signals[i] == 1 and i + 1 < n_bars:
                entry_idx = i + 1
                entry_price = opens[entry_idx]
                entry_time = dates[entry_idx]
                
                # Use ATR from the signal bar (bar i)
                entry_atr = atrs[i]
                
                # Levels
                stop_loss = entry_price - atr_stop_mult * entry_atr
                profit_target = entry_price + atr_target_mult * entry_atr
                
                position = {
                    'entry_idx': entry_idx,
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'entry_atr': entry_atr,
                    'stop_loss': stop_loss,
                    'profit_target': profit_target
                }
                
                # Evaluate the entry bar (bar i+1) immediately
                i = entry_idx
                low_bar = lows[i]
                high_bar = highs[i]
                open_bar = opens[i]
                close_bar = closes[i]
                
                hit_stop = low_bar <= stop_loss
                hit_target = high_bar >= profit_target
                
                if hit_stop and hit_target:
                    # Conservatively assume stop loss is hit first
                    exit_price = open_bar if open_bar <= stop_loss else stop_loss
                    trades.append(
                        _create_trade_record(position, i, exit_price, dates[i], 'stop_loss', cost_per_trade)
                    )
                    position = None
                elif hit_stop:
                    exit_price = open_bar if open_bar <= stop_loss else stop_loss
                    trades.append(
                        _create_trade_record(position, i, exit_price, dates[i], 'stop_loss', cost_per_trade)
                    )
                    position = None
                elif hit_target:
                    exit_price = open_bar if open_bar >= profit_target else profit_target
                    trades.append(
                        _create_trade_record(position, i, exit_price, dates[i], 'profit_target', cost_per_trade)
                    )
                    position = None
            else:
                i += 1
        else:
            # Check if active position exits on bar i
            low_bar = lows[i]
            high_bar = highs[i]
            open_bar = opens[i]
            close_bar = closes[i]
            
            entry_idx = position['entry_idx']
            stop_loss = position['stop_loss']
            profit_target = position['profit_target']
            
            bars_held = i - entry_idx + 1
            
            hit_stop = low_bar <= stop_loss
            hit_target = high_bar >= profit_target
            
            if hit_stop and hit_target:
                # Conservatively assume stop loss is hit first
                exit_price = open_bar if open_bar <= stop_loss else stop_loss
                trades.append(
                    _create_trade_record(position, i, exit_price, dates[i], 'stop_loss', cost_per_trade)
                )
                position = None
                i += 1
            elif hit_stop:
                exit_price = open_bar if open_bar <= stop_loss else stop_loss
                trades.append(
                    _create_trade_record(position, i, exit_price, dates[i], 'stop_loss', cost_per_trade)
                )
                position = None
                i += 1
            elif hit_target:
                exit_price = open_bar if open_bar >= profit_target else profit_target
                trades.append(
                    _create_trade_record(position, i, exit_price, dates[i], 'profit_target', cost_per_trade)
                )
                position = None
                i += 1
            elif bars_held >= max_hold:
                # Max hold reached, exit at the close of bar i
                trades.append(
                    _create_trade_record(position, i, close_bar, dates[i], 'max_hold', cost_per_trade)
                )
                position = None
                i += 1
            else:
                i += 1
                
    # Close out open positions at the close of the final bar to resolve trade logs
    if position is not None:
        last_idx = n_bars - 1
        trades.append(
            _create_trade_record(position, last_idx, closes[last_idx], dates[last_idx], 'end_of_data', cost_per_trade)
        )
        
    # Build trade log DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculate summary metrics
    summary = {}
    if len(trades) > 0:
        summary['n_trades'] = len(trades)
        summary['win_rate'] = sum(1 for t in trades if t['net_return'] > 0) / len(trades)
        summary['avg_net_return'] = sum(t['net_return'] for t in trades) / len(trades)
        
        # Compound returns sequentially
        comp_return = 1.0
        for t in trades:
            comp_return *= (1.0 + t['net_return'])
        summary['total_return'] = comp_return - 1.0
    else:
        summary['n_trades'] = 0
        summary['win_rate'] = 0.0
        summary['avg_net_return'] = 0.0
        summary['total_return'] = 0.0
        
    return trades_df, summary


def _create_trade_record(pos: dict, exit_idx: int, exit_price: float, exit_time, exit_reason: str, cost: float) -> dict:
    """
    Helper function to calculate net trade return adjusted for costs and format log.
    """
    entry_price = pos['entry_price']
    
    # Calculate costs: multiply entry by (1 + cost) and exit by (1 - cost)
    adj_entry = entry_price * (1 + cost)
    adj_exit = exit_price * (1 - cost)
    
    net_return = (adj_exit / adj_entry) - 1.0
    bars_held = exit_idx - pos['entry_idx'] + 1
    
    return {
        'entry_time': pos['entry_time'],
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'net_return': net_return,
        'bars_held': bars_held,
        'exit_reason': exit_reason
    }


def walk_forward(
    df: pd.DataFrame,
    init_train_frac: float = 0.5,
    step_frac: float = 0.1,
    prob_threshold: float = 0.55
) -> pd.DataFrame:
    """
    Repeatedly trains on expanding window chunks and tests on the next chronological slice.
    Combines the out-of-sample predictions to evaluate performance.
    """
    df_clean = df.copy()
    n = len(df_clean)
    
    init_size = int(n * init_train_frac)
    step_size = int(n * step_frac)
    
    if step_size <= 0:
        step_size = 1
        
    oos_chunks = []
    train_end = init_size
    
    while train_end < n:
        test_end = min(train_end + step_size, n)
        
        train_chunk = df_clean.iloc[:train_end]
        test_chunk = df_clean.iloc[train_end:test_end]
        
        if len(test_chunk) == 0:
            break
            
        # Train model on expanding historical set
        model = train_model(train_chunk)
        
        # Predict on out-of-sample slice
        test_with_signals = add_signals(model, test_chunk, prob_threshold=prob_threshold)
        oos_chunks.append(test_with_signals)
        
        # Roll forward
        train_end = test_end
        
    if not oos_chunks:
        raise ValueError("Walk-forward failed: no out-of-sample periods generated.")
        
    # Concatenate the out-of-sample segments
    oos_df = pd.concat(oos_chunks)
    return oos_df
