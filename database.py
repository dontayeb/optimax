import streamlit as st
import pandas as pd
import sqlite3


@st.cache_data(ttl=3600)
def load_data_from_db():
    """Load all necessary data from the SQLite database."""
    try:
        conn = sqlite3.connect('market_data.db', check_same_thread=False)

        # Load price data
        price_query = """
            SELECT s.ticker, d.date, d.close, d.volume 
            FROM daily_data d 
            JOIN stocks s ON s.id = d.stock_id 
            ORDER BY s.ticker, d.date
        """
        price_df = pd.read_sql_query(price_query, conn)
        price_df['date'] = pd.to_datetime(price_df['date'])

        # Load stocks
        stocks_df = pd.read_sql_query("SELECT id, ticker FROM stocks", conn)

        # Load dividends
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
        st.error(f"Error loading database: {e}. Has the scraper been run?")
        return pd.DataFrame(), [], pd.DataFrame()


def get_table_names():
    """Get all table names from the database."""
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
    try:
        conn = sqlite3.connect('market_data.db', check_same_thread=False)
        stocks_df = pd.read_sql_query("SELECT id, ticker FROM stocks", conn)
        conn.close()
        return stocks_df
    except Exception as e:
        st.error(f"Error reading stocks: {e}")
        return pd.DataFrame()