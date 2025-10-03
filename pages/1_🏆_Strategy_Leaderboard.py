import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

# Use parquet version if database doesn't exist (production)
if os.path.exists('market_data.db'):
    from database import load_data_from_db
else:
    from database_parq import load_data_from_db

from engine import SimulationEngine

st.set_page_config(page_title="Strategy Leaderboard", page_icon="🏆", layout="wide")

st.title("🏆 Strategy Leaderboard")
st.info("Analyze signal reliability and simulate strategies across all stocks in your portfolio.")

# Load data
main_df, ticker_list, dividends_df = load_data_from_db()

if not ticker_list:
    st.warning("No stocks found in the database.")
    st.stop()


@st.cache_data(ttl=86400)
def get_per_ticker_signal_stats(_main_df, _dividends_df, ticker_list):
    """Get signal performance stats for all tickers."""
    all_perf_data = []
    status_text = st.empty()

    for i, ticker in enumerate(ticker_list):
        status_text.info(f"Backtesting signals for {ticker} ({i + 1}/{len(ticker_list)})...")
        engine = SimulationEngine(_main_df, _dividends_df, ticker)
        results = engine.run_dca_vs_active(10000, 120000, 15, 60)

        if 'signal_performance' in results and not results['signal_performance'].empty:
            perf_df = results['signal_performance'].copy()
            perf_df['Ticker'] = ticker
            all_perf_data.append(perf_df)

    status_text.empty()

    if not all_perf_data:
        return pd.DataFrame()

    return pd.concat(all_perf_data, ignore_index=True)


@st.cache_data(ttl=86400)
def run_full_market_simulation(_main_df, _dividends_df, ticker_list, monthly_dca, annual_active):
    """Run full market simulation for all stocks."""
    all_results = []
    progress_bar = st.progress(0, text="Initializing simulations...")
    status_text = st.empty()

    for i, ticker in enumerate(ticker_list):
        status_text.info(f"Simulating {ticker} ({i + 1}/{len(ticker_list)})...")
        engine = SimulationEngine(_main_df, _dividends_df, ticker)
        results = engine.run_dca_vs_active(monthly_dca, annual_active, 15, 60)

        if 'summary' in results and not results['summary'].empty:
            summary = results['summary']
            stock_summary = {'Ticker': ticker}

            for _, row in summary.iterrows():
                strategy = row['Strategy']
                stock_summary[f'{strategy} Final Value'] = row['Final Value']
                stock_summary[f'{strategy} Total Invested'] = row['Total Invested']

            all_results.append(stock_summary)

        progress_bar.progress((i + 1) / len(ticker_list))

    status_text.success("All market simulations complete!")

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


# Market-Wide Signal Performance Section
st.subheader("📊 Market-Wide Signal Performance")

with st.spinner("Calculating global signal win rates..."):
    per_ticker_perf = get_per_ticker_signal_stats(main_df, dividends_df, ticker_list)

    if not per_ticker_perf.empty:
        # Aggregate global statistics
        global_stats = per_ticker_perf.groupby('Signal Type').agg(
            Total_Trades=('Total Trades', 'sum'),
            Total_Losses=('Losing Trades', 'sum')
        ).reset_index()

        global_stats['Win Rate'] = (
                                           (global_stats['Total_Trades'] - global_stats['Total_Losses']) /
                                           global_stats['Total_Trades']
                                   ) * 100

        # Display metrics
        cols = st.columns(len(global_stats))
        for i, row in global_stats.iterrows():
            with cols[i]:
                st.metric(
                    label=f"{row['Signal Type']} Win Rate",
                    value=f"{row['Win Rate']:.1f}%",
                    help=f"Based on {int(row['Total_Trades'])} total trades across all stocks."
                )
    else:
        st.warning("Could not calculate global signal performance.")

# Stock vs. Strategy Performance Section
st.subheader("💰 Stock vs. Strategy Performance")

# Sidebar parameters
st.sidebar.header("Leaderboard Parameters")
monthly_investment = st.sidebar.number_input(
    "Monthly DCA Investment ($)",
    min_value=100,
    max_value=50000,
    value=10000,
    step=500
)
initial_capital = st.sidebar.number_input(
    "Active Strategy Annual Capital ($)",
    min_value=1000,
    max_value=1000000,
    value=120000,
    step=1000
)

if st.sidebar.button("🚀 Run Full Market Simulation", type="primary"):
    results_df = run_full_market_simulation(
        main_df,
        dividends_df,
        ticker_list,
        monthly_investment,
        initial_capital
    )

    if not results_df.empty:
        # Calculate ROI percentages
        results_df['DCA ROI %'] = (
                                          results_df['DCA Final Value'] / results_df['DCA Total Invested'] - 1
                                  ) * 100

        results_df['Active Trader ROI %'] = (
                                                    results_df['Active Trader Final Value'] /
                                                    results_df['Active Trader Total Invested'] - 1
                                            ) * 100

        results_df['Strategic Accumulator ROI %'] = (
                                                            results_df['Strategic Accumulator Final Value'] /
                                                            results_df['Strategic Accumulator Total Invested'] - 1
                                                    ) * 100

        # Sort by Strategic Accumulator ROI
        sorted_results_df = results_df.sort_values(
            by='Strategic Accumulator ROI %',
            ascending=False
        )

        # Calculate grand totals
        total_row = {
            'Ticker': '**GRAND TOTAL**',
            'DCA Final Value': results_df['DCA Final Value'].sum(),
            'DCA Total Invested': results_df['DCA Total Invested'].sum(),
            'Active Trader Final Value': results_df['Active Trader Final Value'].sum(),
            'Active Trader Total Invested': results_df['Active Trader Total Invested'].sum(),
            'Strategic Accumulator Final Value': results_df['Strategic Accumulator Final Value'].sum(),
            'Strategic Accumulator Total Invested': results_df['Strategic Accumulator Total Invested'].sum()
        }

        total_row['DCA ROI %'] = (
            (total_row['DCA Final Value'] / total_row['DCA Total Invested'] - 1) * 100
            if total_row['DCA Total Invested'] > 0 else 0
        )

        total_row['Active Trader ROI %'] = (
            (total_row['Active Trader Final Value'] / total_row['Active Trader Total Invested'] - 1) * 100
            if total_row['Active Trader Total Invested'] > 0 else 0
        )

        total_row['Strategic Accumulator ROI %'] = (
            (total_row['Strategic Accumulator Final Value'] /
             total_row['Strategic Accumulator Total Invested'] - 1) * 100
            if total_row['Strategic Accumulator Total Invested'] > 0 else 0
        )

        # Combine results with totals
        total_df = pd.DataFrame([total_row])
        display_df = pd.concat([sorted_results_df, total_df], ignore_index=True)

        # Define display columns
        display_cols = [
            'Ticker',
            'DCA Final Value', 'DCA Total Invested', 'DCA ROI %',
            'Active Trader Final Value', 'Active Trader Total Invested', 'Active Trader ROI %',
            'Strategic Accumulator Final Value', 'Strategic Accumulator Total Invested',
            'Strategic Accumulator ROI %'
        ]

        # Calculate dynamic height
        height = (len(display_df) + 1) * 35 + 3

        # Style and display dataframe
        styler = display_df[display_cols].style.format({
            'DCA Final Value': '${:,.2f}',
            'DCA Total Invested': '${:,.2f}',
            'DCA ROI %': '{:.2f}%',
            'Active Trader Final Value': '${:,.2f}',
            'Active Trader Total Invested': '${:,.2f}',
            'Active Trader ROI %': '{:.2f}%',
            'Strategic Accumulator Final Value': '${:,.2f}',
            'Strategic Accumulator Total Invested': '${:,.2f}',
            'Strategic Accumulator ROI %': '{:.2f}%'
        }).background_gradient(
            cmap='RdYlGn',
            subset=['DCA ROI %', 'Active Trader ROI %', 'Strategic Accumulator ROI %']
        )

        st.dataframe(styler, use_container_width=True, height=height, hide_index=True)

        # Summary insights
        st.subheader("📈 Key Insights")

        col1, col2, col3 = st.columns(3)

        with col1:
            best_dca = sorted_results_df.nlargest(1, 'DCA ROI %').iloc[0]
            st.metric(
                "Best DCA Stock",
                best_dca['Ticker'],
                f"{best_dca['DCA ROI %']:.1f}%"
            )

        with col2:
            best_active = sorted_results_df.nlargest(1, 'Active Trader ROI %').iloc[0]
            st.metric(
                "Best Active Trader Stock",
                best_active['Ticker'],
                f"{best_active['Active Trader ROI %']:.1f}%"
            )

        with col3:
            best_accum = sorted_results_df.nlargest(1, 'Strategic Accumulator ROI %').iloc[0]
            st.metric(
                "Best Strategic Accumulator Stock",
                best_accum['Ticker'],
                f"{best_accum['Strategic Accumulator ROI %']:.1f}%"
            )

else:
    st.info("Click the 'Run Full Market Simulation' button in the sidebar to begin.")

    st.markdown("""
    ### How This Works

    This leaderboard runs a complete backtest simulation across all stocks in your database:

    1. **Market-Wide Signal Performance**: Shows the overall win rate for each signal type
    2. **Full Portfolio Simulation**: Compares how each strategy performs on every stock
    3. **Grand Total**: Aggregates results to show portfolio-level performance

    **Strategy Comparison:**
    - DCA: Simple monthly investment
    - Active Trader: Uses technical signals with profit targets
    - Strategic Accumulator: Combines signals with quarterly buys (buy-and-hold)

    The simulation helps identify which stocks work best with which strategies.
    """)