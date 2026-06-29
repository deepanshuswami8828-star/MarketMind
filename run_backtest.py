"""
Run Backtest Command-Line Script
Loads data (CSV or synthetic), builds features, runs a single-split backtest
and a walk-forward validation backtest, and compares the strategy's return
to a Buy-and-Hold baseline (after transaction costs).
"""

import argparse
import os
import pandas as pd
import numpy as np

import data_loader
import features
import strategy


def run(csv_path: str = None):
    # 1. Load data
    if csv_path:
        print(f"Loading data from CSV: {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        df = data_loader.load_csv(csv_path)
    else:
        print("No CSV specified. Generating synthetic random-walk dataset (n=4000)...")
        df = data_loader.make_synthetic(n=4000)
        
    print(f"Loaded {len(df)} bars of raw data.")
    
    # 2. Build dataset (computes technical indicators + classification labels)
    print("Building dataset (horizon=4, threshold=0.01)...")
    dataset = features.build_dataset(df, horizon=4, threshold=0.01)
    print(f"Dataset ready. Warmup and border rows removed. Remaining: {len(dataset)} rows.")
    
    # Define parameters
    cost_per_trade = 0.0006
    prob_threshold = 0.55
    
    print("\n" + "="*50)
    print(" METHOD 1: SINGLE CHRONOLOGICAL SPLIT (70% Train, 30% Test)")
    print("="*50)
    
    train_df, test_df = strategy.train_test_split_time(dataset, train_frac=0.7)
    print(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows.")
    
    # Train model
    model = strategy.train_model(train_df)
    
    # Generate signals on test set
    test_df = strategy.add_signals(model, test_df, prob_threshold=prob_threshold)
    
    # Execute backtest on test set
    trades_df, summary = strategy.backtest(test_df, cost_per_trade=cost_per_trade)
    
    # Calculate Buy-and-Hold baseline for test set (after costs)
    bh_entry = test_df['open'].iloc[0]
    bh_exit = test_df['close'].iloc[-1]
    bh_return = (bh_exit * (1 - cost_per_trade)) / (bh_entry * (1 + cost_per_trade)) - 1.0
    
    print("\n--- RESULTS ON SINGLE CHRONOLOGICAL TEST SET ---")
    print(f"Number of trades executed: {summary['n_trades']}")
    if summary['n_trades'] > 0:
        print(f"Win Rate:                  {summary['win_rate'] * 100:.2f}%")
        print(f"Average Net Return/Trade:  {summary['avg_net_return'] * 100:.4f}%")
    print(f"Strategy Total Return:     {summary['total_return'] * 100:.2f}%")
    print(f"Buy-and-Hold Return:       {bh_return * 100:.2f}%")
    
    print("\n" + "="*50)
    print(" METHOD 2: EXPANDING-WINDOW WALK-FORWARD VALIDATION")
    print("="*50)
    print("Why this is more honest than a single train/test split:")
    print("  1. Simulates realistic model updates as new data becomes available.")
    print("  2. Tests performance across different market regimes (expanding test period).")
    print("  3. Reduces overfitting/cherry-picking of a single arbitrary split point.")
    
    # Run walk-forward validation (starts at 50% historical, steps by 10%)
    print("\nRunning walk-forward validation...")
    oos_df = strategy.walk_forward(
        dataset,
        init_train_frac=0.5,
        step_frac=0.1,
        prob_threshold=prob_threshold
    )
    print(f"Walk-forward out-of-sample data generated: {len(oos_df)} rows.")
    
    # Backtest on the combined out-of-sample signals
    wf_trades_df, wf_summary = strategy.backtest(oos_df, cost_per_trade=cost_per_trade)
    
    # Calculate Buy-and-Hold baseline for walk-forward period (after costs)
    wf_entry = oos_df['open'].iloc[0]
    wf_exit = oos_df['close'].iloc[-1]
    wf_bh_return = (wf_exit * (1 - cost_per_trade)) / (wf_entry * (1 + cost_per_trade)) - 1.0
    
    print("\n--- RESULTS ON WALK-FORWARD OUT-OF-SAMPLE PERIOD ---")
    print(f"Number of trades executed: {wf_summary['n_trades']}")
    if wf_summary['n_trades'] > 0:
        print(f"Win Rate:                  {wf_summary['win_rate'] * 100:.2f}%")
        print(f"Average Net Return/Trade:  {wf_summary['avg_net_return'] * 100:.4f}%")
    print(f"Strategy Total Return:     {wf_summary['total_return'] * 100:.2f}%")
    print(f"Buy-and-Hold Return:       {wf_bh_return * 100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trading bot strategy backtests.")
    parser.add_argument("--csv", type=str, default=None, help="Path to local Kaggle CSV file.")
    args = parser.parse_args()
    
    run(args.csv)
