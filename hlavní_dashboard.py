import streamlit as st
from database import load_data_from_db
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Page ---
st.title("📈 Stock Analysis & Strategy Dashboard")

st.markdown("""
Welcome to the Stock Analysis Dashboard! This application provides comprehensive tools 
for analyzing stock trading strategies and technical signals.

### Available Features:

**📊 Strategy Simulations**
- Compare DCA vs. Active Trading strategies
- Customize profit targets and holding periods
- View detailed backtest results for individual stocks

**🎯 Signal Effectiveness**
- Analyze win rates for technical signals across all stocks
- View performance matrix with color-coded heatmap
- Identify which signals work best for which stocks

**🏆 Strategy Leaderboard**
- Run simulations across your entire stock universe
- Compare performance metrics for all strategies
- See market-wide signal statistics

**📡 Live Signals**
- View current week's technical signals
- Browse historical signal occurrences
- Track Golden Cross, Death Cross, and RSI signals

**🔍 Database Viewer**
- Inspect your market data
- Filter by stock ticker
- Verify data integrity

---

### Getting Started

1. Make sure you've run the data scraper to populate `market_data.db`
2. Use the sidebar to navigate between different pages
3. Each page has its own controls and parameters

### Navigation

Use the sidebar menu on the left to explore different features of the dashboard.
""")

# Load and display basic stats
with st.spinner("Loading market data..."):
    main_df, ticker_list, dividends_df = load_data_from_db()

if not main_df.empty:
    st.success("✅ Database loaded successfully!")

    # Get last update date
    last_update = main_df['date'].max()
    days_since_update = (pd.Timestamp.now() - last_update).days

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Stocks", len(ticker_list))

    with col2:
        st.metric("Total Data Points", f"{len(main_df):,}")

    with col3:
        if not main_df.empty:
            date_range = (main_df['date'].max() - main_df['date'].min()).days
            st.metric("Data Range (Days)", f"{date_range:,}")

    with col4:
        st.metric(
            "Last Updated",
            last_update.strftime('%Y-%m-%d'),
            delta=f"{days_since_update} days ago" if days_since_update > 0 else "Today",
            delta_color="inverse"
        )

    with st.expander("📋 View Stock List"):
        st.write(", ".join(ticker_list))
else:
    st.error("⚠️ No data found. Please run the scraper to populate the database.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
Built with Streamlit | Data updated from market_data.db
</div>
""", unsafe_allow_html=True)