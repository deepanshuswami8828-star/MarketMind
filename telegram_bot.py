"""
===============================================================================
SEBI COMPLIANCE & REGULATORY INFORMATION (INDIA)
===============================================================================
This software is designed as a personal, learning-only algorithmic stock trading
SIGNAL bot. By using this software, you agree to comply with the following:

1. PERSONAL & FAMILY USE ONLY:
   This bot is strictly for educational, personal, or family trading simulation.
   You must NOT use this tool to trade on behalf of other individuals.
   
2. NO PUBLIC SIGNAL DISTRIBUTION:
   Under SEBI (Research Analyst) Regulations, 2014, sharing, selling, or 
   publishing trading signals, buy/sell recommendations, or performance claims
   to the public or clients requires formal registration with SEBI as a 
   Research Analyst (RA) or Investment Adviser (IA).
   
3. INFRASTRUCTURE & SECURITY:
   Zerodha Kite Connect terms require running your bot on a secure infrastructure
   (such as a static IP address) and performing mandatory manual 2FA logins daily.
   Fully automated login bypasses (e.g., scraping passwords) violate broker policies.
===============================================================================
"""

import os
import logging
import pandas as pd
import numpy as np

# python-telegram-bot v20+ uses async/await syntax
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes
)

import data_loader
import features
import strategy

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Reading Telegram Bot Token from environment variable
# If not set, you can paste your token directly here for local testing:
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "place_your_telegram_bot_token_here")

# Global cache to hold trained models so we don't retrain on every request
# Format: { "SYMBOL": trained_model_object }
MODEL_CACHE = {}

# Global dictionary to store the latest ticks from Zerodha Kite websocket (if enabled)
LATEST_KITE_TICKS = {}


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Sends a welcome message and basic usage instructions.
    """
    welcome_text = (
        "👋 Welcome to the Personal Stock Trading Signal Bot!\n\n"
        "This bot is a learning-only signal generator. It does not place real orders.\n\n"
        "Commands:\n"
        "  /signal <SYMBOL> - Get current trading signal for a symbol (e.g., /signal TCS.NS or /signal AAPL)\n"
        "  /help - Display help instructions"
    )
    await update.message.reply_text(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Sends detailed help instructions.
    """
    help_text = (
        "🤖 *Trading Bot Help*\n\n"
        "Usage:\n"
        "Use `/signal <SYMBOL>` to check for a signal.\n"
        "Example: `/signal TCS.NS` (for NSE ticker) or `/signal AAPL` (for US ticker)\n\n"
        "How it works:\n"
        "1. Loads historical daily data for training.\n"
        "2. Fits a Gradient Boosting classifier on causal indicators.\n"
        "3. Computes features for the latest closed day.\n"
        "4. Generates a signal:\n"
        "   - *LONG*: If model prediction probability >= 0.55.\n"
        "   - *WAIT*: Otherwise.\n\n"
        "⚠️ *Disclaimer*: For educational and simulated use only. No real orders are ever placed."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /signal command.
    Loads data, trains (if not cached), calculates features, and returns prediction.
    """
    # Parse the symbol argument
    if not context.args:
        await update.message.reply_text("Please specify a symbol. Example: `/signal TCS.NS` or `/signal AAPL`")
        return
        
    symbol = context.args[0].upper()
    await update.message.reply_text(f"🔍 Analyzing {symbol}... Please wait.")
    
    try:
        # 1. Train the model if not cached
        if symbol not in MODEL_CACHE:
            await update.message.reply_text(f"⚙️ No cached model found for {symbol}. Training model on historical data...")
            
            # Fetch 2 years of daily data for training
            if data_loader.USE_KITE:
                # In Kite, we look up the instrument token. For a educational placeholder, we assume symbol is a token
                # or write a mapping. In real use, map "TCS" to the numeric token.
                # Here we convert symbol to integer if possible, or print instructions.
                try:
                    instrument_token = int(symbol)
                except ValueError:
                    await update.message.reply_text(
                        f"❌ Kite mode requires a numeric Instrument Token. "
                        f"Please provide an integer (e.g., /signal 11536001)."
                    )
                    return
                # Retrieve daily data for training (approx 2 years)
                to_date = pd.Timestamp.now().strftime("%Y-%m-%d")
                from_date = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
                df_train = data_loader.load_kite_historical(
                    instrument_token=instrument_token,
                    from_date=from_date,
                    to_date=to_date,
                    interval="day"
                )
            else:
                # Use yfinance for free daily data
                df_train = data_loader.load_yfinance(symbol, period="2y", interval="1d")
                
            # Build training dataset (computes features, labels, and drops NaNs)
            dataset = features.build_dataset(df_train, horizon=4, threshold=0.01)
            
            if len(dataset) < 50:
                await update.message.reply_text("❌ Not enough historical data to train the model.")
                return
                
            # Train the GradientBoosting classifier
            model = strategy.train_model(dataset)
            MODEL_CACHE[symbol] = model
            await update.message.reply_text(f"✅ Model trained and cached successfully for {symbol}!")
            
        # Retrieve the cached model
        model = MODEL_CACHE[symbol]
        
        # 2. Fetch recent data to generate the live signal
        if data_loader.USE_KITE:
            instrument_token = int(symbol)
            to_date = pd.Timestamp.now().strftime("%Y-%m-%d")
            from_date = (pd.Timestamp.now() - pd.DateOffset(months=2)).strftime("%Y-%m-%d")
            df_recent = data_loader.load_kite_historical(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            
            # If the background websocket has a newer real-time tick, append it
            if instrument_token in LATEST_KITE_TICKS:
                tick = LATEST_KITE_TICKS[instrument_token]
                last_dt = pd.to_datetime(tick['timestamp'])
                # If this tick is newer than our historical data, append/update it
                if last_dt > df_recent.index[-1]:
                    new_row = pd.DataFrame([{
                        'open': tick['last_trade_price'],
                        'high': tick['high_price'],
                        'low': tick['low_price'],
                        'close': tick['last_trade_price'],
                        'volume': tick['volume_traded']
                    }], index=[last_dt])
                    df_recent = pd.concat([df_recent, new_row])
        else:
            # Use yfinance
            df_recent = data_loader.load_yfinance(symbol, period="2mo", interval="1d")
            
        # 3. Compute indicators (causal features) on recent data
        # Note: We call compute_features() instead of build_dataset() so we do NOT drop 
        # the very latest row (whose future label is unknown but features are fully valid).
        df_features = features.compute_features(df_recent)
        
        # Check if we have enough warmup bars to compute features on the last row
        latest_row = df_features.dropna(subset=features.FEATURE_COLS).tail(1)
        if latest_row.empty:
            await update.message.reply_text("❌ Error: Warmup indicators failed on recent data.")
            return
            
        # Extract features for the latest bar
        X_latest = latest_row[features.FEATURE_COLS]
        latest_time = latest_row.index[0].strftime("%Y-%m-%d")
        current_close = latest_row['close'].iloc[0]
        current_atr = latest_row['atr_14'].iloc[0]
        
        # Predict signal probability
        prob = model.predict_proba(X_latest)[0, 1]
        
        # Check signal threshold
        # signal = 1 (LONG) if probability of an up-move is >= 0.55
        is_long = prob >= 0.55
        
        if is_long:
            # Entry on the next bar's open (approximated by current close)
            suggested_entry = current_close
            stop_loss = suggested_entry - 1.5 * current_atr
            target = suggested_entry + 2.0 * current_atr
            
            response = (
                f"🟢 *SIGNAL: LONG* for `{symbol}`\n"
                f"📅 *As of*: {latest_time}\n"
                f"💰 *Current Close*: {current_close:.2f}\n"
                f"📈 *Probability*: {prob * 100:.1f}%\n\n"
                f"📋 *Trade Blueprint (Next Bar Open)*:\n"
                f"  • *Suggested Entry*: {suggested_entry:.2f}\n"
                f"  • *Stop-Loss (1.5x ATR)*: {stop_loss:.2f}\n"
                f"  • *Profit Target (2.0x ATR)*: {target:.2f}"
            )
        else:
            response = (
                f"🟡 *SIGNAL: WAIT* for `{symbol}`\n"
                f"📅 *As of*: {latest_time}\n"
                f"💰 *Current Close*: {current_close:.2f}\n"
                f"📈 *Probability*: {prob * 100:.1f}% (Threshold is 55.0%)"
            )
            
        await update.message.reply_text(response, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Error processing signal for {symbol}: {str(e)}", exc_info=True)
        await update.message.reply_text(f"❌ Error occurred: {str(e)}")


def run_bot():
    """
    Main entry point to start the Telegram bot.
    """
    if TELEGRAM_TOKEN == "PASTE_YOUR_TELEGRAM_BOT_TOKEN_HERE" or not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_TOKEN environment variable is not set.")
        print("Please set it in your system or paste it in telegram_bot.py.")
        return
        
    # Setup optional Zerodha Kite Ticker connection in background if USE_KITE is enabled
    if data_loader.USE_KITE:
        print("USE_KITE is True. Attempting to start background KiteTicker websocket...")
        try:
            def on_ticks_callback(ticks):
                for tick in ticks:
                    token = tick.get('instrument_token')
                    if token:
                        LATEST_KITE_TICKS[token] = tick
                        
            # Connect the ticker (threaded = True makes it run in a background thread)
            data_loader.start_kite_ticker_websocket(on_ticks_callback)
            print("Background KiteTicker started successfully.")
        except Exception as e:
            print(f"Warning: Could not start KiteTicker websocket: {e}")
            print("Kite operations will fall back to historical load commands.")
            
    # Build the Application
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("signal", signal_command))
    
    # Run the bot polling loop
    print("Telegram bot is running. Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == "__main__":
    run_bot()
