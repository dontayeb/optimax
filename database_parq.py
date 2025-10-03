import streamlit as st
import pandas as pd
import os

# Check if we're running locally or in production
USE_PARQUET = not os.path.exists('market_data.db')

if USE_PARQUET:
    DATA_DIR = 'data'
else:
    import sqlite3


@st.cache_data(ttl=3600)
def load_data_from_db():
    """Load all necessary data from Parquet files or SQLite database."""
    try:
        if USE_PARQUET:
            # Load from Parquet files (for deployment)
            stocks_df = pd.read_parquet(f'{DATA_DIR}/stocks.parquet')
            daily_df = pd.read_parquet(f'{DATA_DIR}/daily_data.parquet')
            dividends_df = pd.read_parquet(f'{DATA_DIR}/dividends.parquet')

            # Merge to create price_df
            price_df = pd.merge(
                daily_df,
                stocks_df[['id', 'ticker']],
                left_on='stock_id',
                right_on='id',
                how='left'
            )
            price_df = price_df[['ticker', 'date', 'close', 'volume']]
            price_df['date'] = pd.to_datetime(price_df['date'])

            # Merge dividends with stocks
            dividends_df = pd.merge(
                dividends_df,
                stocks_df[['id', 'ticker']],
                left_on='stock_id',
                right_on='id',
                how='left'
            )

            tickers = sorted(stocks_df['ticker'].unique().tolist())

            return price_df, tickers, dividends_df
        else:
            # Load from SQLite (for local development)
            conn = sqlite3.connect('market_data.db', check_same_thread=False)

            price_query = """
                SELECT s.ticker, d.date, d.close, d.volume 
                FROM daily_data d 
                JOIN stocks s ON s.id = d.stock_id 
                ORDER BY s.ticker, d.date
            """
            price_df = pd.read_sql_query(price_query, conn)
            price_df['date'] = pd.to_datetime(price_df['date'])

            stocks_df = pd.read_sql_query("SELECT id, ticker FROM stocks", conn)

            dividends_df = pd.read_sql_query("SELECT * FROM dividends", conn)
            dividends_df = pd.merge(
                dividends_df,
                stocks_df,
                left_on='stock_id',
                right_on='id',
                how='left'
            )

            conn.close()

            tickers = sorted(price_df['ticker'].unique().tolist())
            return price_df, tickers, dividends_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), [], pd.DataFrame()


def get_table_names():
    """Get all table names from the database."""
    if USE_PARQUET:
        # Return available parquet files
        if os.path.exists(DATA_DIR):
            files = [f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')]
            return files
        return []
    else:
        try:
            conn = sqlite3.connect('market_data.db', check_same_thread=False)
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            table_names = pd.read_sql_query(query, conn)['name'].tolist()
            conn.close()
            return table_names
        except Exception as e:
            st.error(f"Error reading database: {e}")
            return []


def get_table_data(table_name):
    """Get all data from a specific table."""
    if USE_PARQUET:
        try:
            df = pd.read_parquet(f'{DATA_DIR}/{table_name}.parquet')
            return df
        except Exception as e:
            st.error(f"Error reading {table_name}: {e}")
            return pd.DataFrame()
    else:
        try:
            conn = sqlite3.connect('market_data.db', check_same_thread=False)
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error reading table {table_name}: {e}")
            return pd.DataFrame()


def get_stocks_list():
    """Get list of all stock tickers."""
    if USE_PARQUET:
        try:
            stocks_df = pd.read_parquet(f'{DATA_DIR}/stocks.parquet')
            return stocks_df
        except Exception as e:
            st.error(f"Error reading stocks: {e}")
            return pd.DataFrame()
    else:
        try:
            conn = sqlite3.connect('market_data.db', check_same_thread=False)
            stocks_df = pd.read_sql_query("SELECT id, ticker FROM stocks", conn)
            conn.close()
            return stocks_df
        except Exception as e:
            st.error(f"Error reading stocks: {e}")
            return pd.DataFrame()