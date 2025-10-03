import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import load_data_from_db
from engine import SimulationEngine

st.set_page_config(page_title="Live Signals", page_icon="📡", layout="wide")

st.title("📡 Live Signal Screener")
st.info("Scan for technical signals, filtered by historical win rate performance.")

# Sidebar parameters
st.sidebar.header("Filter Parameters")

min_win_rate = st.sidebar.slider(
    "Minimum Win Rate (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=5,
    help="Only show signals where the stock has achieved at least this win rate historically"
)

signal_filter = st.sidebar.multiselect(
    "Signal Types to Display",
    options=["Golden Cross (Buy)", "Death Cross (Sell)", "RSI Oversold (Buy)", "RSI Overbought (Sell)", "Seasonal"],
    default=["Golden Cross (Buy)", "RSI Oversold (Buy)", "Seasonal"],
    help="Select which signal types you want to see"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest Parameters")
st.sidebar.markdown("These affect win rate calculations:")

profit_target_pct = st.sidebar.slider(
    "Profit Target (%)",
    min_value=5,
    max_value=100,
    value=15,
    step=5
)

rsi_hold_days = st.sidebar.slider(
    "RSI Max Hold Days",
    min_value=10,
    max_value=1000,
    value=60,
    step=10
)


@st.cache_data(ttl=3600)
def calculate_all_signals(_df):
    """Calculate technical signals for all stocks."""
    if _df.empty:
        return pd.DataFrame()

    with st.spinner("Calculating technical indicators for all stocks..."):
        df = _df.copy()
        gb = df.groupby('ticker')

        # Calculate moving averages
        df['sma_50'] = gb['close'].transform(lambda x: x.rolling(window=50).mean())
        df['sma_200'] = gb['close'].transform(lambda x: x.rolling(window=200).mean())

        # Calculate RSI
        delta = gb['close'].transform('diff')
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Previous values for signal detection
        df['sma_50_prev'] = gb['sma_50'].shift(1)
        df['sma_200_prev'] = gb['sma_200'].shift(1)
        df['rsi_14_prev'] = gb['rsi_14'].shift(1)

        # Add month for seasonal signals
        df['month'] = df['date'].dt.month
        df['month_prev'] = gb['month'].shift(1)

        # Signal masks
        gc_mask = (df['sma_50'] > df['sma_200']) & (df['sma_50_prev'] < df['sma_200_prev'])
        dc_mask = (df['sma_50'] < df['sma_200']) & (df['sma_50_prev'] > df['sma_200_prev'])
        rsi_buy_mask = (df['rsi_14'] < 30) & (df['rsi_14_prev'] >= 30)
        rsi_sell_mask = (df['rsi_14'] > 70) & (df['rsi_14_prev'] <= 70)
        seasonal_mask = ((df['month'] != df['month_prev']) & (df['month'].isin([2, 10])))

        # Combine all signals
        all_signals_mask = gc_mask | dc_mask | rsi_buy_mask | rsi_sell_mask | seasonal_mask
        signals_df = df[all_signals_mask].copy()

        # Label signals
        conditions = [
            gc_mask[all_signals_mask],
            dc_mask[all_signals_mask],
            rsi_buy_mask[all_signals_mask],
            rsi_sell_mask[all_signals_mask],
            seasonal_mask[all_signals_mask]
        ]
        choices = [
            'Golden Cross (Buy)',
            'Death Cross (Sell)',
            'RSI Oversold (Buy)',
            'RSI Overbought (Sell)',
            'Seasonal'
        ]
        signals_df['Signal'] = np.select(conditions, choices, default='Unknown')

        # Format output
        signals_df = signals_df[['date', 'ticker', 'Signal', 'close']].rename(
            columns={'close': 'Price on Signal'}
        ).sort_values(by='date', ascending=False).reset_index(drop=True)

        return signals_df


@st.cache_data(ttl=86400)
def get_win_rates_by_ticker(_main_df, _dividends_df, ticker_list, profit_target, rsi_days):
    """Calculate win rates for each signal type for each ticker."""
    win_rate_data = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(ticker_list):
        status_text.info(f"Calculating win rates for {ticker} ({i + 1}/{len(ticker_list)})...")

        engine = SimulationEngine(_main_df, _dividends_df, ticker)
        results = engine.run_dca_vs_active(10000, 120000, profit_target, rsi_days)

        if 'signal_performance' in results and not results['signal_performance'].empty:
            ticker_win_rates = {}
            for _, row in results['signal_performance'].iterrows():
                ticker_win_rates[row['Signal Type']] = {
                    'win_rate': row['Win Rate %'],
                    'total_trades': row['Total Trades'],
                    'winning_trades': row['Winning Trades']
                }
            win_rate_data[ticker] = ticker_win_rates

        progress_bar.progress((i + 1) / len(ticker_list))

    status_text.empty()
    progress_bar.empty()

    return win_rate_data


# Load data
main_df, ticker_list, dividends_df = load_data_from_db()

if main_df.empty:
    st.warning("No data available in the database.")
    st.stop()

# Calculate win rates for filtering
with st.spinner("Calculating historical win rates for all stocks..."):
    win_rates = get_win_rates_by_ticker(
        main_df,
        dividends_df,
        ticker_list,
        profit_target_pct,
        rsi_hold_days
    )

# Calculate all signals
all_signals = calculate_all_signals(main_df)

if all_signals.empty:
    st.warning("No signals found in the entire historical dataset.")
    st.stop()

# Map signal names to internal names for filtering
signal_mapping = {
    'Golden Cross (Buy)': 'Golden Cross',
    'RSI Oversold (Buy)': 'RSI Oversold',
    'Seasonal': 'Seasonal',
    'Death Cross (Sell)': 'Golden Cross',  # Use inverse for sell signals
    'RSI Overbought (Sell)': 'RSI Oversold'  # Use inverse for sell signals
}


# Filter signals based on win rate
def passes_win_rate_filter(row):
    ticker = row['ticker']
    signal = row['Signal']

    if ticker not in win_rates:
        return False

    signal_type = signal_mapping.get(signal)
    if not signal_type:
        return False

    if signal_type not in win_rates[ticker]:
        return False

    # Now win_rates contains a dict, so access the 'win_rate' key
    win_rate_info = win_rates[ticker][signal_type]
    return win_rate_info['win_rate'] >= min_win_rate


# Apply filters
filtered_signals = all_signals[all_signals.apply(passes_win_rate_filter, axis=1)].copy()

# Filter by selected signal types
if signal_filter:
    filtered_signals = filtered_signals[filtered_signals['Signal'].isin(signal_filter)]


# Add win rate column with trade details
def get_win_rate_with_details(row):
    ticker = row['ticker']
    signal = row['Signal']
    signal_type = signal_mapping.get(signal)

    if ticker in win_rates and signal_type in win_rates[ticker]:
        stats = win_rates[ticker][signal_type]
        win_rate = stats['win_rate']
        wins = int(stats['winning_trades'])
        total = int(stats['total_trades'])
        return f"{win_rate:.1f}% ({wins}/{total})"
    return "N/A"


filtered_signals['Win Rate (Wins/Total)'] = filtered_signals.apply(get_win_rate_with_details, axis=1)


# Add target price and exit information
def calculate_exit_info(row):
    signal = row['Signal']
    price = row['Price on Signal']
    date = row['date']

    # For Golden Cross - use profit target
    if signal == 'Golden Cross (Buy)':
        target_price = price * (1 + profit_target_pct / 100)
        return target_price, f"${target_price:.2f}", "Price Target"

    # For RSI Oversold - use profit target with time stop
    elif signal == 'RSI Oversold (Buy)':
        target_price = price * (1 + profit_target_pct / 100)
        exit_date = date + pd.Timedelta(days=rsi_hold_days)
        return target_price, f"${target_price:.2f} or {exit_date.strftime('%Y-%m-%d')}", "Target or Time Stop"

    # For Seasonal - calculate sell month
    elif signal == 'Seasonal':
        month = date.month
        if month == 2:  # February buy -> August sell
            sell_month = 8
        elif month == 10:  # October buy -> April sell
            sell_month = 4
        else:
            return None, "N/A", "N/A"

        # Calculate approximate sell date (15th of sell month)
        year = date.year
        if sell_month < month:
            year += 1
        sell_date = pd.Timestamp(year=year, month=sell_month, day=15)
        return None, sell_date.strftime('%Y-%m-%d'), "Seasonal Exit"

    # Other signals
    else:
        return None, "N/A", "N/A"


filtered_signals[['Target Price', 'Exit Info', 'Exit Type']] = filtered_signals.apply(
    lambda row: pd.Series(calculate_exit_info(row)),
    axis=1
)

# Define current week
today = pd.to_datetime(datetime.now().date())
start_of_week = today - pd.to_timedelta(today.dayofweek, unit='d')

# Split signals
current_week_signals = filtered_signals[filtered_signals['date'] >= start_of_week]
historical_signals = filtered_signals[filtered_signals['date'] < start_of_week]

# Display current week signals
st.subheader(f"🔔 High-Quality Signals This Week (≥{min_win_rate}% Win Rate)")

if current_week_signals.empty:
    st.info(
        f"No signals with ≥{min_win_rate}% win rate have appeared this week. Try lowering the minimum win rate threshold.")
else:
    # Prepare display columns
    display_columns = {
        "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "ticker": st.column_config.TextColumn("Ticker"),
        "Signal": st.column_config.TextColumn("Signal Type"),
        "Price on Signal": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
        "Win Rate (Wins/Total)": st.column_config.TextColumn("Win Rate (Wins/Total)"),
        "Exit Info": st.column_config.TextColumn("Exit Date/Target"),
        "Exit Type": st.column_config.TextColumn("Exit Strategy")
    }

    st.dataframe(
        current_week_signals,
        use_container_width=True,
        hide_index=True,
        column_config=display_columns
    )

    # Summary metrics
    st.markdown("### Current Week Summary")
    signal_counts = current_week_signals['Signal'].value_counts()
    cols = st.columns(len(signal_counts))

    for i, (signal, count) in enumerate(signal_counts.items()):
        with cols[i]:
            st.metric(signal, count)

# Display historical signals
st.subheader("📜 Signal History (Filtered)")

with st.expander("View all historical high-quality signals"):
    if historical_signals.empty:
        st.text(f"No historical signals found with ≥{min_win_rate}% win rate.")
    else:
        # Additional filters
        col1, col2 = st.columns(2)

        with col1:
            hist_tickers = ['All'] + sorted(historical_signals['ticker'].unique().tolist())
            hist_selected_ticker = st.selectbox(
                "Filter by ticker:",
                hist_tickers,
                key="hist_ticker"
            )

        with col2:
            hist_signals = ['All'] + sorted(historical_signals['Signal'].unique().tolist())
            hist_selected_signal = st.selectbox(
                "Filter by signal:",
                hist_signals,
                key="hist_signal"
            )

        # Apply filters
        filtered_hist = historical_signals.copy()

        if hist_selected_ticker != 'All':
            filtered_hist = filtered_hist[filtered_hist['ticker'] == hist_selected_ticker]

        if hist_selected_signal != 'All':
            filtered_hist = filtered_hist[filtered_hist['Signal'] == hist_selected_signal]

        # Display
        display_columns = {
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "ticker": st.column_config.TextColumn("Ticker"),
            "Signal": st.column_config.TextColumn("Signal Type"),
            "Price on Signal": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
            "Win Rate (Wins/Total)": st.column_config.TextColumn("Win Rate (Wins/Total)"),
            "Exit Info": st.column_config.TextColumn("Exit Date/Target"),
            "Exit Type": st.column_config.TextColumn("Exit Strategy")
        }

        st.dataframe(
            filtered_hist,
            use_container_width=True,
            hide_index=True,
            column_config=display_columns
        )

        st.info(f"Showing {len(filtered_hist)} of {len(historical_signals)} filtered historical signals")

# Statistics
st.subheader("📊 Filtered Signal Statistics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Most Active High-Quality Stocks**")
    if not filtered_signals.empty:
        top_stocks = filtered_signals['ticker'].value_counts().head(10)
        for ticker, count in top_stocks.items():
            st.text(f"{ticker}: {count} signals")
    else:
        st.text("No signals pass the filter criteria")

with col2:
    st.markdown("**Signal Type Distribution**")
    if not filtered_signals.empty:
        signal_dist = filtered_signals['Signal'].value_counts()
        for signal, count in signal_dist.items():
            percentage = (count / len(filtered_signals)) * 100
            st.text(f"{signal}: {count} ({percentage:.1f}%)")
    else:
        st.text("No signals pass the filter criteria")