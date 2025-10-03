import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_table_names, get_table_data, get_stocks_list

st.set_page_config(page_title="Database Viewer", page_icon="🔍", layout="wide")

st.title("🔍 Database Inspector")
st.info("Use this tool to verify the data scraped into your `market_data.db` file.")

try:
    # Get table names
    table_names = get_table_names()

    if not table_names:
        st.error("No tables found in the database.")
        st.stop()

    # Table selection
    selected_table = st.selectbox("Select a table to view:", table_names)

    if selected_table:
        # Load table data
        df = get_table_data(selected_table)

        if df.empty:
            st.warning(f"No data found in table '{selected_table}'")
        else:
            # Get stocks list for filtering
            stocks_df = get_stocks_list()

            if not stocks_df.empty:
                ticker_list = sorted(stocks_df['ticker'].unique().tolist())
            else:
                ticker_list = []

            # Show filters if stock_id column exists
            if 'stock_id' in df.columns and ticker_list:
                st.subheader("Filters")

                col1, col2 = st.columns([3, 1])

                with col1:
                    selected_tickers = st.multiselect(
                        "Filter by Ticker(s):",
                        options=ticker_list,
                        help="Select one or more tickers to filter the data"
                    )

                with col2:
                    if st.button("Clear Filters"):
                        st.rerun()

                # Apply filter
                if selected_tickers:
                    ids_to_filter = stocks_df[
                        stocks_df['ticker'].isin(selected_tickers)
                    ]['id'].tolist()
                    df = df[df['stock_id'].isin(ids_to_filter)]
                    st.info(f"Filtered to {len(selected_tickers)} ticker(s)")

            # Display table info
            st.subheader(f"Table: {selected_table}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Rows", f"{len(df):,}")

            with col2:
                st.metric("Total Columns", len(df.columns))

            with col3:
                if 'date' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['date'])
                        date_range = (df['date'].max() - df['date'].min()).days
                        st.metric("Date Range (Days)", f"{date_range:,}")
                    except:
                        pass

            # Column information
            with st.expander("📋 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True, hide_index=True)

            # Sample data preview
            st.subheader("Data Preview")

            # Options for viewing
            col1, col2 = st.columns([1, 3])

            with col1:
                view_option = st.radio(
                    "View:",
                    ["First 100 rows", "Last 100 rows", "Random sample", "All data"],
                    help="Choose how to view the data"
                )

            with col2:
                if view_option == "Random sample":
                    sample_size = st.slider(
                        "Sample size:",
                        min_value=10,
                        max_value=min(1000, len(df)),
                        value=min(100, len(df)),
                        step=10
                    )

            # Display data based on selection
            if view_option == "First 100 rows":
                display_df = df.head(100)
            elif view_option == "Last 100 rows":
                display_df = df.tail(100)
            elif view_option == "Random sample":
                display_df = df.sample(n=min(sample_size, len(df)))
            else:
                display_df = df

            # Format date columns
            for col in display_df.columns:
                if 'date' in col.lower():
                    try:
                        display_df[col] = pd.to_datetime(display_df[col])
                    except:
                        pass

            # Display the dataframe
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )

            st.success(f"Displaying {len(display_df)} of {len(df)} rows from the '{selected_table}' table.")

            # Data statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

            if numeric_cols:
                with st.expander("📊 Numeric Column Statistics"):
                    stats_df = df[numeric_cols].describe()
                    st.dataframe(stats_df, use_container_width=True)

            # Export option
            st.subheader("💾 Export Data")

            col1, col2 = st.columns([1, 3])

            with col1:
                export_format = st.selectbox("Format:", ["CSV", "JSON"])

            with col2:
                if export_format == "CSV":
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{selected_table}.csv",
                        mime="text/csv"
                    )
                else:
                    json = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json,
                        file_name=f"{selected_table}.json",
                        mime="application/json"
                    )

except Exception as e:
    st.error(f"An error occurred while reading the database: {e}")
    st.exception(e)