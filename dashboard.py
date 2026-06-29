"""
Streamlit Web Dashboard for Trading Bot
Displays interactive price charts, causal indicator values, current signal status,
and backtest metrics comparison against buy-and-hold.
Supports both Windows and macOS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

import data_loader
import features
import strategy

# Set Streamlit Page Configuration for a premium layout
st.set_page_config(
    layout="wide",
    page_title="Personal Algorithmic Trading Signal Bot",
    page_icon="📈"
)

# Sleek CSS styling to inject custom visual accents
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        border-radius: 12px;
        padding: 20px;
        background-color: #1e222b;
        border: 1px solid #2e323b;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 14px;
        color: #8a99ad;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card p {
        margin: 10px 0 0 0;
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    .signal-long {
        background: linear-gradient(135deg, #113824 0%, #1e222b 100%);
        border: 1.5px solid #28a745 !important;
    }
    .signal-wait {
        background: linear-gradient(135deg, #3d331d 0%, #1e222b 100%);
        border: 1.5px solid #ffc107 !important;
    }
</style>
""", unsafe_allow_html=True)


# Streamlit Caching for smooth, blazingly fast interface interactions
@st.cache_data(ttl=600)  # Cache loaded data for 10 minutes
def get_historical_data(source_type, identifier):
    """
    Loads historical daily OHLCV data based on source type.
    Includes robust error-handling to prevent dashboard crashes.
    """
    try:
        if source_type == "CSV File":
            if not identifier or not os.path.exists(identifier):
                return None
            return data_loader.load_csv(identifier)
        else:
            # yfinance
            if not identifier:
                return None
            return data_loader.load_yfinance(identifier, period="2y", interval="1d")
    except Exception as e:
        # Gracefully catch all errors (like connection loss or invalid symbols)
        # print error details to stderr for diagnostics and return None
        print(f"Error loading historical data for '{identifier}': {e}", file=sys.stderr)
        return None


@st.cache_resource(ttl=3600)  # Cache trained models for 1 hour
def get_trained_model(model_key, _df):
    """
    Trains and returns the GradientBoosting model for a given key and dataframe.
    """
    dataset = features.build_dataset(_df, horizon=4, threshold=0.01)
    if len(dataset) < 50:
        return None
    return strategy.train_model(dataset)


def explain_signal_logic(row_data):
    """
    Generates a plain-language textual explanation for a signal based on technical indicators.
    """
    rsi = row_data['rsi_14']
    close = row_data['close']
    sma_10 = row_data['sma_10']
    sma_20 = row_data['sma_20']
    macd_hist = row_data['macd_hist']
    
    # RSI interpretation
    if rsi <= 30:
        rsi_desc = f"Oversold ({rsi:.2f})"
        rsi_interp = "indicating the asset is potentially undervalued and due for a bounce."
    elif rsi >= 70:
        rsi_desc = f"Overbought ({rsi:.2f})"
        rsi_interp = "indicating the asset is potentially overvalued and overextended."
    else:
        rsi_desc = f"Neutral ({rsi:.2f})"
        rsi_interp = "indicating steady, non-extreme price action."
        
    # Price vs SMA20
    price_rel = "above" if close > sma_20 else "below"
    price_trend = "uptrend" if close > sma_20 else "downtrend"
    price_interp = f"The price is {price_rel} its 20-day simple moving average ({sma_20:.2f}), suggesting a short-term {price_trend}."
    
    # MACD Histogram
    macd_rel = "positive" if macd_hist > 0 else "negative"
    macd_trend = "upward momentum" if macd_hist > 0 else "downward momentum"
    macd_interp = f"The MACD histogram is {macd_rel} ({macd_hist:.4f}), indicating active {macd_trend}."
    
    # Trend (SMA10 vs SMA20)
    trend_rel = "bullish" if sma_10 > sma_20 else "bearish"
    trend_desc = "bullish alignment (SMA10 > SMA20)" if sma_10 > sma_20 else "bearish alignment (SMA10 < SMA20)"
    trend_interp = f"The moving averages show a {trend_desc}, showing a {trend_rel} trend structure."
    
    explanation = f"""
    * **RSI(14)**: {rsi_desc} — {rsi_interp}
    * **Price vs SMA20**: {price_interp}
    * **MACD Histogram**: {macd_interp}
    * **Trend (SMA10 vs SMA20)**: {trend_interp}
    
    💡 *Note*: This is an ML model combining these indicators (RSI, MACD, moving-average trend, momentum, volatility, volume). It is not a single classic strategy.
    """
    return explanation


# --- DASHBOARD HEADER ---
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🤖 Personal Stock Trading Signal Dashboard</h1>", unsafe_allow_html=True)

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("⚙️ Configuration Panel")

data_source = st.sidebar.selectbox(
    "Data Source",
    ["Yahoo Finance (yfinance)", "CSV File"]
)

if data_source == "Yahoo Finance (yfinance)":
    # Text input accepting any symbol
    symbol = st.sidebar.text_input(
        "Enter Symbol",
        value="TCS.NS",
        help="Type any symbol, e.g. RELIANCE.NS, SBIN.NS, AAPL."
    ).upper().strip()
    st.sidebar.markdown("<small style='color: #8a99ad;'>Type any symbol, e.g. RELIANCE.NS, SBIN.NS, AAPL.</small>", unsafe_allow_html=True)
    data_key = symbol
else:
    # CSV file path input
    csv_path = st.sidebar.text_input("Enter CSV File Path", "TCS.csv")
    symbol = os.path.basename(csv_path)
    data_key = csv_path

# Strategy parameters
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Strategy Parameters")
prob_threshold = st.sidebar.slider("Signal Probability Threshold", 0.50, 0.70, 0.55, 0.01)
atr_stop_mult = st.sidebar.slider("Stop-Loss ATR Multiplier", 1.0, 3.0, 1.5, 0.1)
atr_target_mult = st.sidebar.slider("Profit Target ATR Multiplier", 1.0, 5.0, 2.0, 0.1)
cost_per_trade = st.sidebar.number_input("Transaction Cost / Trade (Fraction)", 0.0, 0.01, 0.0006, 0.0001, format="%.5f")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "⚠️ **SEBI Compliance Reminder**:\n"
    "Personal & family educational use only. Public signal sharing requires formal SEBI RA/IA registration."
)

# --- NIFTY 100 LIST (Nifty 50 + Nifty Next 50) ---
NIFTY_100 = [
    # Nifty 50 (48 symbols)
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "SBIN.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS", "AXISBANK.NS", 
    "KOTAKBANK.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", 
    "MARUTI.NS", "TATAMOTORS.NS", "WIPRO.NS", "SUNPHARMA.NS", 
    "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "POWERGRID.NS", 
    "NTPC.NS", "TATASTEEL.NS", "HCLTECH.NS", "TECHM.NS", "ADANIENT.NS", 
    "ONGC.NS", "COALINDIA.NS", "JSWSTEEL.NS", "BAJAJFINSV.NS", 
    "GRASIM.NS", "DRREDDY.NS", "CIPLA.NS", "BPCL.NS", "EICHERMOT.NS", 
    "HEROMOTOCO.NS", "DIVISLAB.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", 
    "HINDALCO.NS", "TATACONSUM.NS", "INDUSINDBK.NS", "SBILIFE.NS", 
    "HDFCLIFE.NS", "ADANIPORTS.NS", "UPL.NS", "LTIM.NS",
    # Nifty Next 50 (52 symbols)
    "ACC.NS", "ADANIENSOL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS", 
    "BEL.NS", "BANKBARODA.NS", "BHEL.NS", "BOSCHLTD.NS", "CANBK.NS", 
    "CGPOWER.NS", "CHOLAFIN.NS", "COLPAL.NS", "DLF.NS", "FEDBANK.NS", 
    "GAIL.NS", "GMRINFRA.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS", 
    "ICICIGI.NS", "ICICIPRULI.NS", "IOC.NS", "IRCTC.NS", "IRFC.NS", 
    "INDIGO.NS", "JINDALSTEL.NS", "JIOFIN.NS", "LICI.NS", "MRF.NS", 
    "MAXHEALTH.NS", "MPHASIS.NS", "NHPC.NS", "NMDC.NS", "OBEROIRLTY.NS", 
    "OFSS.NS", "PIDILITIND.NS", "PFC.NS", "PNB.NS", "RECLTD.NS", 
    "SAIL.NS", "SIEMENS.NS", "SRF.NS", "TATACOMM.NS", "TATAPOWER.NS", 
    "TRENT.NS", "TVSMOTOR.NS", "UNIONBANK.NS", "VBL.NS", "ZOMATO.NS", 
    "ZYDUSLIFE.NS", "MUTHOOTFIN.NS"
]


def scan_symbol(sym, p_thresh, stop_mult, target_mult):
    """
    Helper function to process a single symbol for the market scanner.
    Catches all internal data/logic issues and reports them as skip reasons.
    """
    try:
        # 1. Fetch data safely
        df_sym = get_historical_data("Yahoo Finance (yfinance)", sym)
        if df_sym is None or df_sym.empty:
            return None, "No data returned (invalid symbol or offline)"
            
        # 2. Train model (or fetch cached version)
        model_sym = get_trained_model(sym, df_sym)
        if model_sym is None:
            return None, "Model training failed (insufficient data)"
            
        # 3. Compute indicators
        df_feat = features.compute_features(df_sym)
        df_val = df_feat.dropna(subset=features.FEATURE_COLS)
        if df_val.empty:
            return None, "Feature warmup calculation failed"
            
        # 4. Generate prediction on the latest bar
        latest_bar_sym = df_val.iloc[-1]
        X_lat = latest_bar_sym[features.FEATURE_COLS].to_frame().T
        prob_val = model_sym.predict_proba(X_lat)[0, 1]
        
        is_long_sym = prob_val >= p_thresh
        close_val = latest_bar_sym['close']
        atr_val = latest_bar_sym['atr_14']
        
        if is_long_sym:
            entry = close_val
            stop = entry - stop_mult * atr_val
            tgt = entry + target_mult * atr_val
            risk_rew = (tgt - entry) / (entry - stop) if (entry - stop) != 0 else 0.0
            
            return {
                'Symbol': sym,
                'Signal': 'LONG',
                'Probability %': round(prob_val * 100, 2),
                'Close': round(close_val, 2),
                'Suggested Entry': f"{entry:.2f}",
                'Stop-Loss': f"{stop:.2f}",
                'Profit Target': f"{tgt:.2f}",
                'Risk:Reward': f"{risk_rew:.2f}",
                # Stored raw metrics for explanation expander
                'rsi_14': float(latest_bar_sym['rsi_14']),
                'sma_10': float(latest_bar_sym['sma_10']),
                'sma_20': float(latest_bar_sym['sma_20']),
                'macd_hist': float(latest_bar_sym['macd_hist']),
                'close_raw': float(close_val)
            }, None
        else:
            return {
                'Symbol': sym,
                'Signal': 'WAIT',
                'Probability %': round(prob_val * 100, 2),
                'Close': round(close_val, 2),
                'Suggested Entry': '-',
                'Stop-Loss': '-',
                'Profit Target': '-',
                'Risk:Reward': '-',
                'rsi_14': float(latest_bar_sym['rsi_14']),
                'sma_10': float(latest_bar_sym['sma_10']),
                'sma_20': float(latest_bar_sym['sma_20']),
                'macd_hist': float(latest_bar_sym['macd_hist']),
                'close_raw': float(close_val)
            }, None
            
    except Exception as e:
        return None, f"Runtime error: {str(e)}"


# --- MARKET SCANNER GRAPHICAL UI ---
st.subheader("🔍 Nifty 100 Market Scanner")
col_scan1, col_scan2 = st.columns([2, 8])
with col_scan1:
    scan_clicked = st.button("Scan Nifty 100 Market", type="primary", use_container_width=True)

if scan_clicked:
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    results = []
    skipped = {}
    
    total_symbols = len(NIFTY_100)
    
    for idx, sym in enumerate(NIFTY_100):
        status_text.text(f"Scanning {sym} ({idx+1}/{total_symbols})...")
        res, err = scan_symbol(sym, prob_threshold, atr_stop_mult, atr_target_mult)
        if res:
            results.append(res)
        if err:
            skipped[sym] = err
        progress_bar.progress((idx + 1) / total_symbols)
        
    status_text.text("✅ Scan completed successfully!")
    progress_bar.empty()
    
    # Store results in Streamlit session state
    st.session_state['scan_results'] = results
    st.session_state['skipped_symbols'] = skipped
    st.session_state['has_scanned'] = True

# Display scan table if results are cached in session state
if st.session_state.get('has_scanned', False):
    results_list = st.session_state.get('scan_results', [])
    skipped_dict = st.session_state.get('skipped_symbols', {})
    
    if results_list:
        df_results = pd.DataFrame(results_list)
        
        # Display filtering controls
        col_ctrl1, col_ctrl2 = st.columns([3, 7])
        with col_ctrl1:
            show_only_long = st.checkbox("Show only LONG signals", value=False)
            
        if show_only_long:
            df_display = df_results[df_results['Signal'] == 'LONG'].copy()
        else:
            df_display = df_results.copy()
            
        # Default sort: Probability % descending
        df_display = df_display.sort_values(by='Probability %', ascending=False)
        
        # Styler function to highlight LONG rows in green
        def style_rows(row):
            if row['Signal'] == 'LONG':
                return ['background-color: #113824; color: #ffffff'] * len(row)
            return [''] * len(row)
            
        # Select columns to display in the table (omit raw metrics used for explanation)
        display_cols = [
            'Symbol', 'Signal', 'Probability %', 'Close', 
            'Suggested Entry', 'Stop-Loss', 'Profit Target', 'Risk:Reward'
        ]
        styled_df = df_display[display_cols].style.apply(style_rows, axis=1)
        
        st.markdown("**Nifty 100 Signal Scanner Results:**")
        st.dataframe(
            styled_df, 
            use_container_width=True, 
            height=350,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol"),
                "Signal": st.column_config.TextColumn("Signal"),
                "Probability %": st.column_config.NumberColumn("Probability %", format="%.2f%%"),
                "Close": st.column_config.NumberColumn("Close", format="%.2f"),
                "Suggested Entry": st.column_config.TextColumn("Suggested Entry"),
                "Stop-Loss": st.column_config.TextColumn("Stop-Loss"),
                "Profit Target": st.column_config.TextColumn("Profit Target"),
                "Risk:Reward": st.column_config.TextColumn("Risk:Reward")
            }
        )
        
        # Explain LONG signals expanders
        long_results = [r for r in results_list if r['Signal'] == 'LONG']
        if long_results:
            st.markdown("### 💡 Scan Signal Interpretations (Why LONG?)")
            for r in long_results:
                bar_data = {
                    'rsi_14': r['rsi_14'],
                    'close': r['close_raw'],
                    'sma_10': r['sma_10'],
                    'sma_20': r['sma_20'],
                    'macd_hist': r['macd_hist']
                }
                with st.expander(f"Why is {r['Symbol']} LONG?"):
                    explanation = explain_signal_logic(bar_data)
                    st.markdown(explanation)
    else:
        st.info("No results were generated in the scan.")
        
    if skipped_dict:
        with st.expander("⚠️ View Skipped Symbols"):
            st.markdown("The following symbols were skipped (due to missing data or training errors):")
            for sym, reason in skipped_dict.items():
                st.write(f"- **{sym}**: {reason}")

st.markdown("---")

# --- DETAILED SINGLE SYMBOL VIEW ---
st.subheader("🔎 Single-Symbol Detailed View & Backtest")

# Fetch and wrap data loading in try/except (Rule 1)
df_raw = None
try:
    df_raw = get_historical_data(data_source, data_key)
except Exception as e:
    st.error(f"Error fetching historical data: {e}")

if df_raw is None or df_raw.empty:
    if data_source == "CSV File":
        st.error(f"Could not load data from path: '{data_key}'. Check if file exists in project folder.")
    else:
        st.warning("⚠️ No data, try another symbol.")
else:
    # 1. Train Model
    model = get_trained_model(data_key, df_raw)
    
    if model is None:
        st.error("Failed to train model: insufficient historical data.")
    else:
        # 2. Compute live signals
        # Use compute_features so we don't drop the latest row (future label is unknown but features are valid)
        df_features = features.compute_features(df_raw)
        
        # Drop rows where features are NaN due to warmup
        df_valid_features = df_features.dropna(subset=features.FEATURE_COLS)
        
        if df_valid_features.empty:
            st.error("Warmup indicators failed: not enough recent bars to compute features.")
        else:
            latest_bar = df_valid_features.iloc[-1]
            latest_time = df_valid_features.index[-1].strftime("%Y-%m-%d")
            
            # Predict probability on latest bar
            X_latest = latest_bar[features.FEATURE_COLS].to_frame().T
            prob = model.predict_proba(X_latest)[0, 1]
            is_long = prob >= prob_threshold
            
            # Compute trade blueprint levels
            current_close = latest_bar['close']
            current_atr = latest_bar['atr_14']
            
            # Formulate levels and strings based on signal
            if is_long:
                suggested_entry = current_close
                stop_loss = suggested_entry - atr_stop_mult * current_atr
                target = suggested_entry + atr_target_mult * current_atr
                
                entry_str = f"{suggested_entry:.2f}"
                stop_str = f"{stop_loss:.2f}"
                target_str = f"{target:.2f}"
            else:
                entry_str = "-"
                stop_str = "-"
                target_str = "-"
            
            # --- MAIN AREA SECTION 1: SIGNAL KEYCARDS ---
            st.markdown(f"**Detailed Analysis for {symbol}** (As of {latest_time})")
            
            card_class = "signal-long" if is_long else "signal-wait"
            signal_text = "LONG 🟢" if is_long else "WAIT 🟡"
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <h3>Signal</h3>
                    <p style="color: {'#28a745' if is_long else '#ffc107'}">{signal_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Price</h3>
                    <p>{current_close:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Suggested Entry</h3>
                    <p>{entry_str}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Stop-Loss (ATR)</h3>
                    <p>{stop_str}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Profit Target</h3>
                    <p>{target_str}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"**Model Long Probability**: `{prob * 100:.2f}%` (Signal triggers at `{prob_threshold * 100:.1f}%`) | **ATR (14)**: `{current_atr:.4f}`")
            
            # Expandable "Why this signal?" section for the single symbol view
            current_bar_data = {
                'rsi_14': float(latest_bar['rsi_14']),
                'close': float(current_close),
                'sma_10': float(latest_bar['sma_10']),
                'sma_20': float(latest_bar['sma_20']),
                'macd_hist': float(latest_bar['macd_hist'])
            }
            with st.expander("❓ Why this signal?"):
                explanation = explain_signal_logic(current_bar_data)
                st.markdown(explanation)
                
            st.markdown("---")
            
            # --- MAIN AREA SECTION 2: INTERACTIVE CHART (PLOTLY) ---
            st.subheader("📈 Interactive Financial Price Chart")
            
            # Show last 150 bars for better visual spacing
            chart_df = df_features.tail(150)
            
            # Create subplots: Row 1 = Price + SMAs, Row 2 = Volume, Row 3 = RSI
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Add Candlesticks to Row 1
            fig.add_trace(
                go.Candlestick(
                    x=chart_df.index,
                    open=chart_df['open'],
                    high=chart_df['high'],
                    low=chart_df['low'],
                    close=chart_df['close'],
                    name="Candlestick"
                ),
                row=1, col=1
            )
            
            # Add SMA 10 to Row 1
            fig.add_trace(
                go.Scatter(
                    x=chart_df.index, y=chart_df['sma_10'],
                    line=dict(color='orange', width=1.5),
                    name='SMA 10'
                ),
                row=1, col=1
            )
            
            # Add SMA 20 to Row 1
            fig.add_trace(
                go.Scatter(
                    x=chart_df.index, y=chart_df['sma_20'],
                    line=dict(color='deepskyblue', width=1.5),
                    name='SMA 20'
                ),
                row=1, col=1
            )
            
            # Add Volume Bars to Row 2
            # Dynamic colors for volume based on close vs open
            vol_colors = ['green' if c >= o else 'red' for o, c in zip(chart_df['open'], chart_df['close'])]
            fig.add_trace(
                go.Bar(
                    x=chart_df.index, y=chart_df['volume'],
                    marker_color=vol_colors,
                    name='Volume'
                ),
                row=2, col=1
            )
            
            # Add RSI to Row 3
            fig.add_trace(
                go.Scatter(
                    x=chart_df.index, y=chart_df['rsi_14'],
                    line=dict(color='magenta', width=1.5),
                    name='RSI(14)'
                ),
                row=3, col=1
            )
            
            # Add RSI levels lines to Row 3
            fig.add_shape(
                type="line", x0=chart_df.index[0], x1=chart_df.index[-1], y0=70, y1=70,
                line=dict(color="red", width=1, dash="dash"), row=3, col=1
            )
            fig.add_shape(
                type="line", x0=chart_df.index[0], x1=chart_df.index[-1], y0=30, y1=30,
                line=dict(color="green", width=1, dash="dash"), row=3, col=1
            )
            
            # Layout customization for premium aesthetic (dark theme grid layout)
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=700,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,34,43,0.3)',
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2e323b', tickformat="%Y-%m-%d")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2e323b')
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            
            # --- MAIN AREA SECTION 3: BACKTEST COMPARISON ---
            st.subheader("🧪 Historical Backtest Performance (Out-Of-Sample Walk-Forward)")
            
            # Clean dataset for backtest
            df_backtest = features.build_dataset(df_raw, horizon=4, threshold=0.01)
            
            # Run walk-forward testing to generate out-of-sample signals
            with st.spinner("Simulating walk-forward backtest..."):
                try:
                    oos_df = strategy.walk_forward(
                        df_backtest,
                        init_train_frac=0.5,
                        step_frac=0.1,
                        prob_threshold=prob_threshold
                    )
                    
                    trades_df, summary = strategy.backtest(
                        oos_df, 
                        cost_per_trade=cost_per_trade,
                        atr_stop_mult=atr_stop_mult,
                        atr_target_mult=atr_target_mult
                    )
                    
                    # Compute Buy and Hold return
                    bh_entry = oos_df['open'].iloc[0]
                    bh_exit = oos_df['close'].iloc[-1]
                    bh_return = (bh_exit * (1 - cost_per_trade)) / (bh_entry * (1 + cost_per_trade)) - 1.0
                    
                    # Print results columns
                    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                    with col_met1:
                        st.metric("Strategy Total Return", f"{summary['total_return']*100:.2f}%")
                    with col_met2:
                        st.metric("Buy-and-Hold Return", f"{bh_return*100:.2f}%")
                    with col_met3:
                        st.metric("Total Trades Executed", f"{summary['n_trades']}")
                    with col_met4:
                        st.metric("Trade Win Rate", f"{summary['win_rate']*100:.1f}%")
                        
                    # Show Trades Log DataFrame
                    if not trades_df.empty:
                        with st.expander("🔍 View Complete Backtest Trade Log"):
                            # Format columns for nice display
                            display_trades = trades_df.copy()
                            display_trades['net_return'] = display_trades['net_return'].apply(lambda x: f"{x*100:.2f}%")
                            display_trades['entry_price'] = display_trades['entry_price'].round(2)
                            display_trades['exit_price'] = display_trades['exit_price'].round(2)
                            st.dataframe(display_trades, use_container_width=True)
                    else:
                        st.info("No trades were triggered during the out-of-sample validation period.")
                        
                except Exception as e:
                    st.warning(f"Could not complete walk-forward simulation: {e}")
            
            st.markdown("---")
            
            # --- MAIN AREA SECTION 4: DETAILED INDICATOR VALUES ---
            st.subheader("📋 Latest Computed Technical Indicator Features")
            # Show last 10 rows of clean feature values
            st.dataframe(df_valid_features[features.FEATURE_COLS].tail(10).round(4), use_container_width=True)
