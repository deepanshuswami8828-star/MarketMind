"""
Data Loader Module for Trading Bot
Provides interfaces to load historical stock data from local CSV, yfinance, 
synthetic random-walk generator, and Zerodha Kite Connect.
"""

import os
import pandas as pd
import numpy as np

# Config flag for Zerodha Kite.
# When False (default), the rest of the application runs using yfinance/CSV/synthetic data.
# Set this to True once you have valid Kite API credentials.
USE_KITE = False

# Try importing Kite Connect. We wrap it in a try-except block so that the app
# runs fully even if the kiteconnect library is not installed (when USE_KITE is False).
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False


def load_csv(path: str) -> pd.DataFrame:
    """
    Loads a local Kaggle CSV file.
    
    Expected format:
      Headers: Date,Symbol,Series,Prev Close,Open,High,Low,Last,Close,VWAP,Volume,Turnover,...
      Uses 'Close' rather than 'Last' for price.
      Parses 'Date' (YYYY-MM-DD) as the DatetimeIndex.
      All other columns are ignored.
      
    Returns a clean DataFrame with columns: open, high, low, close, volume.
    """
    # Read the raw CSV file
    df = pd.read_csv(path)
    
    # We only want the Date and OHLCV columns. Map them exactly as requested.
    target_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check if all required columns exist in the CSV
    missing_cols = [col for col in target_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")
        
    # Extract only our desired columns
    df = df[target_cols].copy()
    
    # Parse Date and set as DatetimeIndex
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date', inplace=True)
    
    # Chronologically sort the data (no future leakage, oldest to newest)
    df.sort_index(inplace=True)
    
    # Rename columns to lowercase exactly: open, high, low, close, volume
    df.columns = [col.lower() for col in df.columns]
    
    return df


def load_yfinance(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Downloads historical stock data from Yahoo Finance.
    
    Parameters:
      symbol (str): The ticker symbol (e.g., 'AAPL', 'TCS.NS')
      period (str): Time range to fetch (e.g., '1y', '2y', 'max')
      interval (str): Frequency (e.g., '1d', '1h')
      
    Returns a clean DataFrame with columns: open, high, low, close, volume.
    """
    import yfinance as yf
    
    # Download data using yfinance Ticker interface (returns cleaner single-ticker index)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}' from yfinance.")
        
    # Flatten multi-index columns if present (sometimes returned by yfinance downloads)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Standardise column names
    df.columns = [str(col).strip() for col in df.columns]
    
    # Map columns we need
    rename_dict = {}
    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in needed:
        # Check if the column is present, or try case-insensitive match
        if col not in df.columns:
            matches = [c for c in df.columns if c.lower() == col.lower()]
            if matches:
                rename_dict[matches[0]] = col
            else:
                raise ValueError(f"Required column '{col}' not found in yfinance data.")
        else:
            rename_dict[col] = col
            
    df = df.rename(columns=rename_dict)
    df = df[needed].copy()
    
    # Ensure DatetimeIndex and sort ascending
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    # Convert column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Remove timezone information to keep indices clean
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    return df


def make_synthetic(n: int = 4000) -> pd.DataFrame:
    """
    Generates a synthetic random-walk stock dataset for offline training/testing.
    
    Ensures mathematical consistency:
      - high >= open, high >= close
      - low <= open, low <= close
      - volume is always positive
      
    Returns a clean DataFrame with columns: open, high, low, close, volume.
    """
    # Use a fixed seed for reproducible test results
    np.random.seed(42)
    
    # Create daily business dates (excluding weekends) starting from 2010-01-01
    dates = pd.date_range(start="2010-01-01", periods=n, freq="B")
    
    # Simulate closing prices starting from 100 with 1.5% daily volatility
    initial_price = 100.0
    daily_returns = np.random.normal(0.0001, 0.015, n - 1)
    
    close_prices = [initial_price]
    for ret in daily_returns:
        close_prices.append(close_prices[-1] * (1 + ret))
    close_prices = np.array(close_prices)
    
    # Generate open, high, low deviations from the close price
    # Devs are bounded to keep OHLC relationships realistic
    open_deviation = np.random.normal(0, 0.005, n)
    high_deviation = np.abs(np.random.normal(0.006, 0.004, n))
    low_deviation = np.abs(np.random.normal(0.006, 0.004, n))
    
    opens = close_prices * (1 + open_deviation)
    
    # Ensure high is strictly greater than or equal to both open and close
    highs = np.maximum(opens, close_prices) * (1 + high_deviation)
    
    # Ensure low is strictly less than or equal to both open and close
    lows = np.minimum(opens, close_prices) * (1 - low_deviation)
    
    # Simulate volume as a positive log-normal distribution
    volumes = np.random.lognormal(12, 1, n).astype(int)
    
    # Build DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    df.index.name = 'Date'
    return df


def load_kite_historical(instrument_token: int, from_date: str, to_date: str, interval: str = "day") -> pd.DataFrame:
    """
    Fetches historical candles from Zerodha Kite.
    Disabled by default when USE_KITE = False.
    
    Credentials (API Key, Access Token) must be set in environment variables:
      KITE_API_KEY, KITE_ACCESS_TOKEN
    """
    if not USE_KITE:
        raise ValueError(
            "Kite integration is disabled (USE_KITE = False). "
            "Set USE_KITE = True at the top of data_loader.py to enable."
        )
        
    if not KITE_AVAILABLE:
        raise ImportError(
            "The 'kiteconnect' package is not installed. "
            "Please install it or set USE_KITE = False."
        )
        
    api_key = os.environ.get("KITE_API_KEY")
    access_token = os.environ.get("KITE_ACCESS_TOKEN")
    
    if not api_key or not access_token:
        raise ValueError("Kite Connect requires KITE_API_KEY and KITE_ACCESS_TOKEN environment variables.")
        
    # Initialise KiteConnect API client
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    
    # Fetch historical data
    # Zerodha API returns a list of dictionaries with keys: date, open, high, low, close, volume, etc.
    records = kite.historical_data(instrument_token, from_date, to_date, interval)
    
    if not records:
        raise ValueError("No historical data returned from Kite API.")
        
    df = pd.DataFrame(records)
    
    # Parse date and set as DatetimeIndex
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    # Ensure timezone naive
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    # Select and return only OHLCV columns (already in lowercase)
    cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[cols].copy()
    
    return df


def start_kite_ticker_websocket(on_ticks_callback, on_connect_callback=None):
    """
    Initialises and runs the live KiteTicker websocket.
    Disabled by default when USE_KITE = False.
    
    Reads credentials from KITE_API_KEY and KITE_ACCESS_TOKEN environment variables.
    
    ===========================================================================
    DAILY 2FA / LOGIN FLOW (HOW TO RETRIEVE ACCESS_TOKEN):
    ===========================================================================
    Kite API requires a fresh session daily. The process is:
    1. Navigate in a browser to the Zerodha login page using your API key:
       https://kite.zerodha.com/connect/login?api_key=YOUR_API_KEY&v=3
    2. Enter your password and standard daily 2FA credentials (mobile App TOTP/OTP).
    3. Once successful, Zerodha redirects you to your registered redirect URL.
    4. Copy the 'request_token' parameter appended to the redirect URL.
    5. Exchange this 'request_token' for a persistent 'access_token' via the API:
       
       >>> from kiteconnect import KiteConnect
       >>> kite = KiteConnect(api_key="YOUR_API_KEY")
       >>> session = kite.generate_session("COPIED_REQUEST_TOKEN", api_secret="YOUR_API_SECRET")
       >>> print(session["access_token"])
       
    6. Paste this daily access token into the KITE_ACCESS_TOKEN environment variable.
    ===========================================================================
    """
    if not USE_KITE:
        raise ValueError(
            "Kite integration is disabled (USE_KITE = False). "
            "Set USE_KITE = True at the top of data_loader.py to enable."
        )
        
    if not KITE_AVAILABLE:
        raise ImportError(
            "The 'kiteconnect' package is not installed. "
            "Please install it or set USE_KITE = False."
        )
        
    api_key = os.environ.get("KITE_API_KEY")
    access_token = os.environ.get("KITE_ACCESS_TOKEN")
    
    if not api_key or not access_token:
        raise ValueError("Kite Ticker requires KITE_API_KEY and KITE_ACCESS_TOKEN environment variables.")
        
    # Initialise KiteTicker client
    ticker = KiteTicker(api_key, access_token)
    
    # Define internal callbacks to route the events
    def on_ticks(ws, ticks):
        on_ticks_callback(ticks)
        
    def on_connect(ws, response):
        print("KiteTicker Websocket connected successfully.")
        if on_connect_callback:
            on_connect_callback(ws, response)
            
    def on_close(ws, code, reason):
        print(f"KiteTicker Websocket closed. Code: {code}, Reason: {reason}")
        
    def on_error(ws, code, reason):
        print(f"KiteTicker Websocket error. Code: {code}, Reason: {reason}")
        
    # Bind callbacks
    ticker.on_ticks = on_ticks
    ticker.on_connect = on_connect
    ticker.on_close = on_close
    ticker.on_error = on_error
    
    # Connect asynchronously
    print("Starting KiteTicker live websocket client...")
    ticker.connect(threaded=True)
    return ticker
