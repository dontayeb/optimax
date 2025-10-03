import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import load_data_from_db
from engine import SimulationEngine

st.set_page_config(page_title="Strategy Optimizer", page_icon="🔬", layout="wide")

st.title("🔬 Strategy Optimizer")
st.info("Discover the optimal trading rules for each stock through systematic parameter testing.")

# Load data
main_df, ticker_list, dividends_df = load_data_from_db()

if not ticker_list:
    st.warning("No stocks found in the database.")
    st.stop()

# Sidebar - Stock Selection
st.sidebar.header("Stock Selection")
selected_ticker = st.sidebar.selectbox("Choose a Stock:", ticker_list)

# Sidebar - Signal Selection
st.sidebar.header("Signal to Optimize")
signal_type = st.sidebar.radio(
    "Select Signal Type:",
    ["Golden Cross", "RSI Oversold", "Both"]
)

# Sidebar - Parameter Ranges
st.sidebar.header("Parameter Ranges to Test")

st.sidebar.subheader("Profit Target Range")
profit_min = st.sidebar.number_input("Minimum Profit Target (%)", 5, 100, 5, 5)
profit_max = st.sidebar.number_input("Maximum Profit Target (%)", 5, 100, 50, 5)
profit_step = st.sidebar.number_input("Step Size (%)", 1, 20, 5, 1)

st.sidebar.subheader("Hold Period Range (Days)")
hold_min = st.sidebar.number_input("Minimum Hold Days", 10, 500, 30, 10)
hold_max = st.sidebar.number_input("Maximum Hold Days", 10, 1000, 180, 10)
hold_step = st.sidebar.number_input("Step Size (Days)", 5, 50, 30, 5)

# Calculate combinations
profit_targets = list(range(profit_min, profit_max + 1, profit_step))
hold_periods = list(range(hold_min, hold_max + 1, hold_step))
total_combinations = len(profit_targets) * len(hold_periods)

st.sidebar.markdown("---")
st.sidebar.metric("Total Combinations to Test", f"{total_combinations:,}")
estimated_time = total_combinations * 0.5  # Rough estimate
st.sidebar.caption(f"Estimated time: ~{estimated_time:.0f} seconds")


@st.cache_data(ttl=86400)
def run_optimization(_main_df, _dividends_df, ticker, profit_targets, hold_periods, signal_filter):
    """Run parameter sweep to find optimal trading rules."""
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total = len(profit_targets) * len(hold_periods)
    current = 0

    for profit_target in profit_targets:
        for hold_days in hold_periods:
            current += 1
            status_text.info(f"Testing combination {current}/{total}: Profit={profit_target}%, Hold={hold_days} days")

            engine = SimulationEngine(_main_df, _dividends_df, ticker)

            if engine.df.empty:
                continue

            # Run simulation with these parameters
            sim_results = engine.run_dca_vs_active(10000, 120000, profit_target, hold_days)

            if 'summary' in sim_results and not sim_results['summary'].empty:
                # Extract Active Trader performance
                active_row = sim_results['summary'][
                    sim_results['summary']['Strategy'] == 'Active Trader'
                    ]

                if not active_row.empty:
                    final_value = active_row['Final Value'].values[0]
                    total_invested = active_row['Total Invested'].values[0]
                    roi = active_row['Return on Investment %'].values[0]

                    # Get risk metrics
                    risk_metrics = sim_results.get('risk_metrics', {})
                    active_risk = risk_metrics.get('Active Trader', {})

                    # Get signal-specific performance
                    signal_perf = sim_results.get('signal_performance', pd.DataFrame())

                    result_data = {
                        'Profit Target %': profit_target,
                        'Max Hold Days': hold_days,
                        'Final Value': final_value,
                        'Total Invested': total_invested,
                        'ROI %': roi,
                        'Max Drawdown %': active_risk.get('Max Drawdown %', 0),
                        'Sharpe Ratio': active_risk.get('Sharpe Ratio', 0),
                        'Volatility %': active_risk.get('Annualized Volatility %', 0)
                    }

                    # Add signal-specific win rates
                    for signal_name in ['Golden Cross', 'RSI Oversold', 'Seasonal']:
                        signal_row = signal_perf[signal_perf['Signal Type'] == signal_name]
                        if not signal_row.empty:
                            result_data[f'{signal_name} Win Rate %'] = signal_row['Win Rate %'].values[0]
                            result_data[f'{signal_name} Trades'] = signal_row['Total Trades'].values[0]
                        else:
                            result_data[f'{signal_name} Win Rate %'] = 0
                            result_data[f'{signal_name} Trades'] = 0

                    results.append(result_data)

            progress_bar.progress(current / total)

    status_text.empty()
    progress_bar.empty()

    return pd.DataFrame(results)


# Main optimization button
if st.button("🚀 Run Optimization", type="primary"):
    st.markdown("---")
    st.subheader(f"Optimization Results for {selected_ticker}")

    with st.spinner("Running comprehensive parameter sweep..."):
        results_df = run_optimization(
            main_df,
            dividends_df,
            selected_ticker,
            profit_targets,
            hold_periods,
            signal_type
        )

    # Store results in session state
    st.session_state['optimization_results'] = results_df
    st.session_state['optimization_ticker'] = selected_ticker

# Check if we have results to display
if 'optimization_results' in st.session_state and not st.session_state['optimization_results'].empty:
    results_df = st.session_state['optimization_results']
    result_ticker = st.session_state.get('optimization_ticker', selected_ticker)

    if results_df.empty:
        st.error("No results generated. Check if the stock has sufficient data.")
        st.stop()

    st.markdown("---")
    st.subheader(f"Optimization Results for {result_ticker}")

    # Display top performers
    st.markdown("### 🏆 Top 10 Parameter Combinations")

    # Sort by ROI
    top_roi = results_df.nlargest(10, 'ROI %')[
        ['Profit Target %', 'Max Hold Days', 'ROI %', 'Sharpe Ratio', 'Max Drawdown %']
    ]

    st.dataframe(
        top_roi.style.format({
            'ROI %': '{:.2f}%',
            'Sharpe Ratio': '{:.3f}',
            'Max Drawdown %': '{:.2f}%'
        }).background_gradient(cmap='RdYlGn', subset=['ROI %', 'Sharpe Ratio']),
        use_container_width=True,
        hide_index=True
    )

    # Key insights
    best_combo = results_df.loc[results_df['ROI %'].idxmax()]
    best_sharpe = results_df.loc[results_df['Sharpe Ratio'].idxmax()]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Best ROI Combination",
            f"{best_combo['ROI %']:.2f}%",
            f"Target: {best_combo['Profit Target %']:.0f}%, Hold: {best_combo['Max Hold Days']:.0f} days"
        )

    with col2:
        st.metric(
            "Best Risk-Adjusted (Sharpe)",
            f"{best_sharpe['Sharpe Ratio']:.3f}",
            f"Target: {best_sharpe['Profit Target %']:.0f}%, Hold: {best_sharpe['Max Hold Days']:.0f} days"
        )

    # Heatmap visualization
    st.markdown("### 🔥 Performance Heatmap")

    # Only show metrics that exist in the dataframe
    available_metrics = []
    potential_metrics = [
        "ROI %",
        "Sharpe Ratio",
        "Max Drawdown %",
        "Golden Cross Win Rate %",
        "RSI Oversold Win Rate %",
        "Volatility %"
    ]

    for metric in potential_metrics:
        if metric in results_df.columns:
            # Check if there's any non-zero data
            if results_df[metric].notna().any() and (results_df[metric] != 0).any():
                available_metrics.append(metric)

    if not available_metrics:
        st.warning("No metrics available for heatmap visualization.")
    else:
        metric_choice = st.selectbox(
            "Select metric to visualize:",
            available_metrics
        )

        # Create pivot table for heatmap
        heatmap_data = results_df.pivot_table(
            index='Max Hold Days',
            columns='Profit Target %',
            values=metric_choice
        )

        # Display heatmap
        st.dataframe(
            heatmap_data.style.background_gradient(
                cmap='RdYlGn' if 'Drawdown' not in metric_choice else 'RdYlGn_r',
                axis=None
            ).format("{:.2f}"),
            use_container_width=True
        )

        st.caption("Rows = Max Hold Days | Columns = Profit Target % | Brighter green = Better performance")

    # Signal-specific analysis
    if signal_type in ["Golden Cross", "Both"]:
        st.markdown("### 📈 Golden Cross Signal Analysis")

        gc_results = results_df[results_df['Golden Cross Trades'] > 0].copy()

        if not gc_results.empty:
            col1, col2, col3 = st.columns(3)

            best_gc_wr = gc_results.loc[gc_results['Golden Cross Win Rate %'].idxmax()]

            with col1:
                st.metric(
                    "Best Golden Cross Win Rate",
                    f"{best_gc_wr['Golden Cross Win Rate %']:.1f}%",
                    f"{int(best_gc_wr['Golden Cross Trades'])} trades"
                )

            with col2:
                st.metric(
                    "Optimal Profit Target",
                    f"{best_gc_wr['Profit Target %']:.0f}%"
                )

            with col3:
                st.metric(
                    "Optimal Hold Period",
                    f"{best_gc_wr['Max Hold Days']:.0f} days"
                )

    if signal_type in ["RSI Oversold", "Both"]:
        st.markdown("### 📉 RSI Oversold Signal Analysis")

        rsi_results = results_df[results_df['RSI Oversold Trades'] > 0].copy()

        if not rsi_results.empty:
            col1, col2, col3 = st.columns(3)

            best_rsi_wr = rsi_results.loc[rsi_results['RSI Oversold Win Rate %'].idxmax()]

            with col1:
                st.metric(
                    "Best RSI Win Rate",
                    f"{best_rsi_wr['RSI Oversold Win Rate %']:.1f}%",
                    f"{int(best_rsi_wr['RSI Oversold Trades'])} trades"
                )

            with col2:
                st.metric(
                    "Optimal Profit Target",
                    f"{best_rsi_wr['Profit Target %']:.0f}%"
                )

            with col3:
                st.metric(
                    "Optimal Hold Period",
                    f"{best_rsi_wr['Max Hold Days']:.0f} days"
                )

    # Export results
    st.markdown("### 💾 Export Full Results")

    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"{result_ticker}_optimization_results.csv",
        mime="text/csv"
    )

    # Add a button to clear results and start fresh
    if st.button("🔄 Clear Results & Run New Optimization"):
        del st.session_state['optimization_results']
        del st.session_state['optimization_ticker']
        st.rerun()

else:
    st.info("Configure your optimization parameters in the sidebar, then click 'Run Optimization' to begin.")

    st.markdown("""
    ### How the Optimizer Works

    This tool systematically tests every combination of:
    - **Profit Targets**: When to take profits on winning trades
    - **Hold Periods**: Maximum days to hold before forced exit

    For each combination, it runs a full backtest and records:
    - Return on Investment (ROI)
    - Risk metrics (Sharpe Ratio, Max Drawdown, Volatility)
    - Signal-specific win rates and trade counts

    ### What to Look For

    1. **Best ROI**: Highest absolute returns (but may be riskier)
    2. **Best Sharpe Ratio**: Best risk-adjusted returns (professional's choice)
    3. **Signal-Specific Patterns**: Some stocks work better with longer/shorter holds

    ### Pro Tips

    - Start with broader ranges, then narrow in on promising regions
    - Higher profit targets may have fewer trades but higher win rates
    - Longer hold periods reduce trading frequency but may miss exits
    - The "best" parameters may change over time - re-optimize periodically
    """)