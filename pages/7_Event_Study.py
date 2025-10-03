import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import load_data_from_db
import sqlite3

st.set_page_config(page_title="Event Study", page_icon="📅", layout="wide")

st.title("📅 Event Study Analysis")
st.info("Analyze how stock prices behave around key events like dividend ex-dates and news releases.")

# Load data
main_df, ticker_list, dividends_df = load_data_from_db()

if not ticker_list:
    st.warning("No stocks found in the database.")
    st.stop()

# Sidebar configuration
st.sidebar.header("Event Selection")
selected_ticker = st.sidebar.selectbox("Choose a Stock:", ticker_list)

event_type = st.sidebar.radio(
    "Event Type:",
    ["Dividend Ex-Date", "News Release"]
)

st.sidebar.header("Analysis Window")
days_before = st.sidebar.slider("Days Before Event", 1, 30, 10, 1)
days_after = st.sidebar.slider("Days After Event", 1, 30, 10, 1)


def load_events(ticker, event_type):
    """Load event dates from database."""
    conn = sqlite3.connect('market_data.db')

    # Get stock_id
    stock_id_query = "SELECT id FROM stocks WHERE ticker = ?"
    stock_id = pd.read_sql_query(stock_id_query, conn, params=(ticker,))

    if stock_id.empty:
        conn.close()
        return pd.DataFrame()

    stock_id = stock_id['id'].values[0]

    if event_type == "Dividend Ex-Date":
        query = "SELECT ex_date as event_date, amount FROM dividends WHERE stock_id = ? AND ex_date IS NOT NULL"
        events = pd.read_sql_query(query, conn, params=(stock_id,))
        events['event_date'] = pd.to_datetime(events['event_date'], errors='coerce')
        events = events.dropna(subset=['event_date'])
    else:  # News Release
        query = "SELECT news_date as event_date, headline FROM news WHERE stock_id = ?"
        events = pd.read_sql_query(query, conn, params=(stock_id,))
        events['event_date'] = pd.to_datetime(events['event_date'], errors='coerce')
        events = events.dropna(subset=['event_date'])

    conn.close()
    return events


def calculate_event_returns(ticker_df, event_dates, days_before, days_after):
    """Calculate price returns around event dates."""
    all_event_data = []

    for event_date in event_dates:
        # Find the event date in the price data
        event_idx = ticker_df[ticker_df['date'] == event_date].index

        if len(event_idx) == 0:
            # Try to find the closest date
            closest_idx = (ticker_df['date'] - event_date).abs().argmin()
            event_idx = [closest_idx]

        event_idx = event_idx[0]

        # Get surrounding data
        start_idx = max(0, event_idx - days_before)
        end_idx = min(len(ticker_df) - 1, event_idx + days_after)

        if start_idx >= end_idx:
            continue

        event_window = ticker_df.iloc[start_idx:end_idx + 1].copy()

        # Calculate relative day from event
        event_window['days_from_event'] = range(-days_before, len(event_window) - days_before)

        # Normalize prices to event day (event day = 100)
        event_price = ticker_df.iloc[event_idx]['close']
        event_window['normalized_price'] = (event_window['close'] / event_price) * 100
        event_window['return_pct'] = ((event_window['close'] / event_price) - 1) * 100

        all_event_data.append(event_window[['days_from_event', 'normalized_price', 'return_pct']])

    return all_event_data


def aggregate_event_returns(all_event_data):
    """Calculate average returns across all events."""
    if not all_event_data:
        return pd.DataFrame()

    # Combine all events
    combined = pd.concat(all_event_data, ignore_index=True)

    # Calculate average for each day relative to event
    aggregated = combined.groupby('days_from_event').agg({
        'normalized_price': ['mean', 'std', 'count'],
        'return_pct': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    aggregated.columns = [
        'days_from_event',
        'avg_normalized_price',
        'std_normalized_price',
        'event_count',
        'avg_return_pct',
        'std_return_pct'
    ]

    return aggregated


# Main analysis
if st.button("🔍 Run Event Study", type="primary"):
    st.markdown("---")
    st.subheader(f"Event Study: {selected_ticker} - {event_type}")

    # Get stock data
    ticker_df = main_df[main_df['ticker'] == selected_ticker].copy().reset_index(drop=True)

    if ticker_df.empty:
        st.error(f"No price data found for {selected_ticker}")
        st.stop()

    # Load events
    with st.spinner("Loading events..."):
        events_df = load_events(selected_ticker, event_type)

    if events_df.empty:
        st.warning(f"No {event_type} events found for {selected_ticker}")
        st.stop()

    st.success(f"Found {len(events_df)} {event_type} events")

    # Calculate returns around events
    with st.spinner("Analyzing price behavior around events..."):
        event_data = calculate_event_returns(
            ticker_df,
            events_df['event_date'].tolist(),
            days_before,
            days_after
        )

    if not event_data:
        st.warning("Insufficient data to perform analysis")
        st.stop()

    # Aggregate results
    aggregated = aggregate_event_returns(event_data)

    # Display results
    col1, col2, col3 = st.columns(3)

    # Key metrics
    day_before = aggregated[aggregated['days_from_event'] == -1]['avg_return_pct'].values
    event_day = aggregated[aggregated['days_from_event'] == 0]['avg_return_pct'].values
    day_after = aggregated[aggregated['days_from_event'] == 1]['avg_return_pct'].values

    with col1:
        if len(day_before) > 0:
            st.metric("Avg Return (Day Before)", f"{day_before[0]:.2f}%")

    with col2:
        if len(event_day) > 0:
            st.metric("Avg Return (Event Day)", f"{event_day[0]:.2f}%")

    with col3:
        if len(day_after) > 0:
            st.metric("Avg Return (Day After)", f"{day_after[0]:.2f}%")

    # Visualizations
    st.markdown("### 📊 Average Price Behavior Around Events")

    # Line chart
    chart_data = aggregated[['days_from_event', 'avg_normalized_price']].copy()
    chart_data = chart_data.set_index('days_from_event')
    st.line_chart(chart_data, use_container_width=True)

    st.caption(f"Price normalized to 100 on event day (Day 0). Based on {len(event_data)} events.")

    # Cumulative return chart
    st.markdown("### 📈 Average Cumulative Return Pattern")

    return_chart = aggregated[['days_from_event', 'avg_return_pct']].copy()
    return_chart = return_chart.set_index('days_from_event')
    st.line_chart(return_chart, use_container_width=True)

    # Detailed table
    with st.expander("📋 View Detailed Statistics"):
        display_df = aggregated.copy()
        display_df['days_from_event'] = display_df['days_from_event'].astype(int)

        st.dataframe(
            display_df.style.format({
                'avg_normalized_price': '{:.2f}',
                'std_normalized_price': '{:.2f}',
                'avg_return_pct': '{:.2f}%',
                'std_return_pct': '{:.2f}%',
                'event_count': '{:.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )

    # Event list
    st.markdown("### 📅 Event History")

    if event_type == "Dividend Ex-Date":
        display_events = events_df.copy()
        display_events['event_date'] = display_events['event_date'].dt.strftime('%Y-%m-%d')
        display_events = display_events.sort_values('event_date', ascending=False)

        st.dataframe(
            display_events.style.format({'amount': '${:.2f}'}),
            use_container_width=True,
            hide_index=True
        )
    else:
        display_events = events_df.copy()
        display_events['event_date'] = display_events['event_date'].dt.strftime('%Y-%m-%d')
        display_events = display_events.sort_values('event_date', ascending=False).head(50)

        st.dataframe(
            display_events,
            use_container_width=True,
            hide_index=True
        )

        if len(events_df) > 50:
            st.caption(f"Showing most recent 50 of {len(events_df)} total news events")

    # Statistical significance note
    st.markdown("### 📝 Interpretation Guide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**What to look for:**")
        st.markdown("""
        - **Pre-event run-up**: Price increases before event
        - **Event day reaction**: Immediate price response
        - **Post-event drift**: Continued movement after event
        - **Pattern consistency**: Lower std dev = more reliable pattern
        """)

    with col2:
        st.markdown("**Sample size matters:**")
        event_count = aggregated['event_count'].iloc[0]

        if event_count >= 20:
            st.success(f"✓ Strong: {event_count} events (reliable patterns)")
        elif event_count >= 10:
            st.info(f"⚠ Moderate: {event_count} events (patterns emerging)")
        else:
            st.warning(f"⚠ Weak: {event_count} events (insufficient for conclusions)")

else:
    st.info("Configure your event study parameters in the sidebar, then click 'Run Event Study' to begin.")

    st.markdown("""
    ### What is Event Study Analysis?

    Event studies examine how stock prices behave around specific events. This helps answer questions like:

    **For Dividend Ex-Dates:**
    - Does the price typically run up before the ex-date?
    - How much does it drop on the ex-date?
    - Is there a recovery pattern afterward?

    **For News Releases:**
    - What is the average price impact of news?
    - How long does the impact last?
    - Are there predictable patterns?

    ### How It Works

    1. **Identify Events**: Find all occurrences of the selected event type
    2. **Extract Windows**: Get price data before and after each event
    3. **Normalize**: Set event day price to 100 for comparison
    4. **Aggregate**: Calculate average pattern across all events
    5. **Visualize**: Display the typical price behavior

    ### Reading the Charts

    - **Day 0** = Event day
    - **Negative days** = Days before event
    - **Positive days** = Days after event
    - **Normalized Price 100** = Event day price
    - **Higher/Lower** = Average price movement relative to event day

    ### Pro Tips

    - More events = more reliable patterns
    - Look for consistent trends across multiple time periods
    - Compare different stocks to find unique patterns
    - Use this to time entries/exits around known events
    """)