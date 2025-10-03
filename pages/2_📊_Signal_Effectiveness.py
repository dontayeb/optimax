import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import load_data_from_db
from engine import SimulationEngine

st.set_page_config(page_title="Signal Effectiveness", page_icon="🎯", layout="wide")

st.title("🎯 Signal Effectiveness Matrix")
st.info("Analyze the historical win rate of each signal for every stock in your database.")

# Load data
main_df, ticker_list, dividends_df = load_data_from_db()

if not ticker_list:
    st.warning("No stocks found in the database.")
    st.stop()

# Sidebar parameters
st.sidebar.header("Analysis Parameters")
st.sidebar.markdown("Adjust these settings to see how different exit rules affect signal performance.")

profit_target_pct = st.sidebar.slider(
    "Golden Cross Profit Target (%)",
    min_value=5,
    max_value=100,
    value=15,
    step=5,
    help="Target profit percentage for Golden Cross signals before taking profits"
)

rsi_hold_days = st.sidebar.slider(
    "RSI Max Hold Days",
    min_value=10,
    max_value=1000,
    value=60,
    step=10,
    help="Maximum days to hold RSI oversold trades. Use high values (e.g., 1000) to hold until present."
)

if rsi_hold_days >= 365:
    st.sidebar.info(f"📌 With {rsi_hold_days} days, RSI trades will be held long-term (similar to buy-and-hold)")


@st.cache_data(ttl=86400)
def get_per_ticker_signal_stats(_main_df, _dividends_df, ticker_list, profit_target, rsi_days):
    """Calculate signal performance for each ticker with custom parameters."""
    all_perf_data = []
    status_text = st.empty()

    for i, ticker in enumerate(ticker_list):
        status_text.info(f"Backtesting signals for {ticker} ({i + 1}/{len(ticker_list)})...")
        engine = SimulationEngine(_main_df, _dividends_df, ticker)

        # Use custom params
        results = engine.run_dca_vs_active(10000, 120000, profit_target, rsi_days)

        if 'signal_performance' in results and not results['signal_performance'].empty:
            perf_df = results['signal_performance'].copy()
            perf_df['Ticker'] = ticker
            all_perf_data.append(perf_df)

    status_text.empty()

    if not all_perf_data:
        return pd.DataFrame()

    return pd.concat(all_perf_data, ignore_index=True)


# Main analysis button
if st.button("🚀 Analyze Signal Effectiveness Across Market", type="primary"):
    with st.spinner("Running comprehensive backtest across all stocks..."):
        full_perf_df = get_per_ticker_signal_stats(
            main_df,
            dividends_df,
            ticker_list,
            profit_target_pct,
            rsi_hold_days
        )

    if not full_perf_df.empty:
        # Create pivot tables
        win_rate_matrix = full_perf_df.pivot_table(
            index='Ticker',
            columns='Signal Type',
            values='Win Rate %'
        )

        winning_trades_matrix = full_perf_df.pivot_table(
            index='Ticker',
            columns='Signal Type',
            values='Winning Trades'
        ).fillna(0).astype(int)

        total_trades_matrix = full_perf_df.pivot_table(
            index='Ticker',
            columns='Signal Type',
            values='Total Trades'
        ).fillna(0).astype(int)

        # Display heatmap
        st.subheader("📊 Signal Win Rate (%) by Stock")
        st.markdown("Each cell shows the win rate percentage. Sort by clicking any column header.")

        styler = win_rate_matrix.style.background_gradient(
            cmap='RdYlGn',
            vmin=30,
            vmax=80
        )

        height = (len(win_rate_matrix) + 1) * 35 + 3
        st.dataframe(
            styler.format("{:.1f}%", na_rep="N/A"),
            use_container_width=True,
            height=height
        )

        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3 = st.columns(3)

        signal_types = win_rate_matrix.columns
        for i, signal_type in enumerate(signal_types):
            with [col1, col2, col3][i % 3]:
                avg_win_rate = win_rate_matrix[signal_type].mean()
                total_trades = total_trades_matrix[signal_type].sum()
                st.metric(
                    label=f"{signal_type}",
                    value=f"{avg_win_rate:.1f}%",
                    help=f"Average win rate across all stocks. {total_trades} total trades."
                )

        # Detailed view with trade counts
        with st.expander("📋 View Detailed Version (with trade counts)"):
            formatted_text = win_rate_matrix.copy()

            for col in formatted_text.columns:
                for idx in formatted_text.index:
                    rate = win_rate_matrix.loc[idx, col]
                    wins = winning_trades_matrix.loc[idx, col]
                    total = total_trades_matrix.loc[idx, col]

                    if pd.notna(rate) and total > 0:
                        formatted_text.loc[idx, col] = f"{rate:.1f}% ({wins}/{total})"
                    else:
                        formatted_text.loc[idx, col] = "N/A"

            st.dataframe(formatted_text, use_container_width=True)

        # Best/Worst performers
        st.subheader("🏆 Top & Bottom Performers")

        for signal_type in signal_types:
            with st.expander(f"📊 {signal_type}"):
                signal_data = win_rate_matrix[signal_type].dropna().sort_values(ascending=False)

                if len(signal_data) > 0:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Top 5 Stocks**")
                        for ticker, win_rate in signal_data.head(5).items():
                            trades = total_trades_matrix.loc[ticker, signal_type]
                            st.markdown(f"- **{ticker}**: {win_rate:.1f}% ({trades} trades)")

                    with col2:
                        st.markdown("**Bottom 5 Stocks**")
                        for ticker, win_rate in signal_data.tail(5).items():
                            trades = total_trades_matrix.loc[ticker, signal_type]
                            st.markdown(f"- **{ticker}**: {win_rate:.1f}% ({trades} trades)")
                else:
                    st.info(f"No data available for {signal_type}")

    else:
        st.warning("No signal performance data could be generated.")

else:
    st.info(
        "👈 Adjust parameters in the sidebar, then click the button above to run the analysis. This may take several minutes depending on the number of stocks.")

    st.markdown("""
    ### About This Analysis

    This page performs a comprehensive backtest of all technical signals across every stock in your database:

    **Signals Analyzed:**
    - **Golden Cross**: 50-day SMA crosses above 200-day SMA (bullish)
    - **RSI Oversold**: RSI drops below 30 (potential reversal)
    - **Seasonal**: February and October buy signals (calendar-based)

    **Customizable Parameters:**
    - **Golden Cross Profit Target**: Adjust when to take profits on Golden Cross trades
    - **RSI Max Hold Days**: Control how long to hold RSI trades (use high values like 1000 to hold until present)

    **Win Rate Calculation:**
    - A trade is counted as a "win" if it exits with a profit
    - Win Rate % = (Winning Trades / Total Trades) × 100

    **Color Coding:**
    - Green: Higher win rates (>60%)
    - Yellow: Moderate win rates (45-60%)
    - Red: Lower win rates (<45%)

    **Note**: Changing parameters will require re-running the analysis across all stocks.
    """)