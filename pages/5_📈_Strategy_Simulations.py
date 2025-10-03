import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import load_data_from_db
from engine import SimulationEngine

st.set_page_config(page_title="Strategy Simulations", page_icon="🎯", layout="wide")

st.title("🎯 Strategy Simulations")
st.info("Compare different investment strategies for individual stocks with customizable parameters.")

# Load data
main_df, ticker_list, dividends_df = load_data_from_db()

if not ticker_list:
    st.warning("No stocks found in the database. Please run the scraping scripts first.")
    st.stop()

# Sidebar controls
st.sidebar.header("Simulation Controls")
selected_ticker = st.sidebar.selectbox("Choose a Stock to Analyze:", options=ticker_list)

st.sidebar.header("Investment Parameters")
monthly_investment = st.sidebar.number_input(
    "Monthly DCA Investment ($)",
    min_value=100,
    max_value=50000,
    value=10000,
    step=500
)
annual_capital = st.sidebar.number_input(
    "Active Strategy Annual Capital ($)",
    min_value=1000,
    max_value=1000000,
    value=120000,
    step=1000
)

st.sidebar.subheader("Active Trader Rule Adjustments")
profit_target = st.sidebar.slider(
    "Technical Profit Target (%)",
    min_value=5,
    max_value=50,
    value=15,
    step=1,
    help="Target profit percentage for technical signals before taking profits"
)
rsi_hold_days = st.sidebar.slider(
    "RSI Max Hold Days",
    min_value=10,
    max_value=360,
    value=60,
    step=5,
    help="Maximum days to hold RSI oversold trades before forced exit"
)

# Run simulation button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    st.header(f"Results for {selected_ticker}")

    with st.spinner("Running complex backtest with your custom rules..."):
        engine = SimulationEngine(main_df, dividends_df, selected_ticker)

        if engine.df.empty:
            st.error(f"No data available for {selected_ticker}")
            st.stop()

        results = engine.run_dca_vs_active(
            monthly_investment,
            annual_capital,
            profit_target,
            rsi_hold_days
        )

        # Display summary
        st.subheader("📊 Final Portfolio Summary")
        summary_df = results['summary'].copy()

        # Format losing entries column
        summary_df['Losing Entries'] = summary_df.apply(
            lambda row: (
                f"{int(row['Losing Entries'])} (Underwater)"
                if row['Strategy'] == 'Strategic Accumulator'
                else int(row['Losing Entries'])
                if not pd.isna(row['Losing Entries'])
                else '-'
            ),
            axis=1
        )

        st.dataframe(
            summary_df.style.format({
                'Final Value': '${:,.2f}',
                'Total Invested': '${:,.2f}',
                'Return on Investment %': '{:.2f}%',
                'Total Entries': '{:.0f}'
            }).hide(axis="index"),
            use_container_width=True
        )

        # Display risk metrics
        if 'risk_metrics' in results and results['risk_metrics']:
            st.subheader("⚠️ Risk-Adjusted Performance Metrics")
            st.markdown("Professional metrics that measure risk-adjusted returns:")

            risk_df = pd.DataFrame(results['risk_metrics']).T
            risk_df.index.name = 'Strategy'
            risk_df = risk_df.reset_index()

            st.dataframe(
                risk_df.style.format({
                    'Max Drawdown %': '{:.2f}%',
                    'Annualized Volatility %': '{:.2f}%',
                    'Sharpe Ratio': '{:.3f}'
                }).background_gradient(cmap='RdYlGn_r', subset=['Max Drawdown %'])
                .background_gradient(cmap='RdYlGn_r', subset=['Annualized Volatility %'])
                .background_gradient(cmap='RdYlGn', subset=['Sharpe Ratio']),
                use_container_width=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Max Drawdown**")
                st.caption("Largest peak-to-trough decline. Lower is better. Shows the maximum 'pain' you'd endure.")

            with col2:
                st.markdown("**Annualized Volatility**")
                st.caption("How bumpy the ride is. Lower is better. Measures portfolio stability.")

            with col3:
                st.markdown("**Sharpe Ratio**")
                st.caption("Return per unit of risk. Higher is better. The professional's metric of choice.")

        # Display signal performance
        if not results['signal_performance'].empty:
            st.subheader("🎯 Active Trader Signal Performance")
            st.dataframe(
                results['signal_performance'].style.format({
                    'Win Rate %': '{:.1f}%'
                }).hide(axis="index"),
                use_container_width=True
            )

        # Display portfolio growth
        st.subheader("📈 Portfolio Growth Over Time")
        st.line_chart(results['portfolio_over_time'])

        # Display trade logs in expander
        if results.get('trade_logs'):
            with st.expander("📝 View Detailed Trade Logs"):
                for strategy, logs in results['trade_logs'].items():
                    if logs:
                        st.markdown(f"**{strategy} Trades:**")
                        for log in logs[-20:]:  # Show last 20 trades
                            st.text(log)
                        if len(logs) > 20:
                            st.info(f"Showing last 20 of {len(logs)} total trades")
                        st.markdown("---")

else:
    st.info("👈 Adjust parameters in the sidebar and click 'Run Simulation' to see results.")

    st.markdown("""
    ### How It Works

    **DCA (Dollar Cost Averaging)**
    - Invests a fixed amount monthly regardless of market conditions
    - Simple, passive strategy

    **Active Trader**
    - Uses technical signals: Golden Cross, RSI Oversold, and Seasonal patterns
    - Takes profits at target price or exits on opposite signals
    - Configurable profit targets and time stops

    **Hybrid**
    - Combines 75% disciplined monthly DCA with 25% tactical signal-based entries
    - Buy-and-hold approach with strategic timing
    - Balances consistency with opportunistic growth
    """)