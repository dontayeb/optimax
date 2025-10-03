"""
Convert SQLite database to Parquet files for deployment
Run this locally after scraping new data, then commit the parquet files
"""

import sqlite3
import pandas as pd
import os

DB_FILE = 'market_data.db'
OUTPUT_DIR = 'data'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_to_parquet():
    """Convert all tables from SQLite to Parquet format"""
    conn = sqlite3.connect(DB_FILE)

    # Export stocks table
    print("Converting stocks table...")
    stocks_df = pd.read_sql_query("SELECT * FROM stocks", conn)
    stocks_df.to_parquet(f'{OUTPUT_DIR}/stocks.parquet', index=False)
    print(f"  Saved {len(stocks_df)} stocks")

    # Export daily_data table
    print("Converting daily_data table...")
    daily_df = pd.read_sql_query("SELECT * FROM daily_data", conn)
    daily_df.to_parquet(f'{OUTPUT_DIR}/daily_data.parquet', index=False)
    print(f"  Saved {len(daily_df)} daily records")

    # Export dividends table
    print("Converting dividends table...")
    dividends_df = pd.read_sql_query("SELECT * FROM dividends", conn)
    dividends_df.to_parquet(f'{OUTPUT_DIR}/dividends.parquet', index=False)
    print(f"  Saved {len(dividends_df)} dividend records")

    # Export earnings table (if exists)
    try:
        print("Converting earnings table...")
        earnings_df = pd.read_sql_query("SELECT * FROM earnings", conn)
        earnings_df.to_parquet(f'{OUTPUT_DIR}/earnings.parquet', index=False)
        print(f"  Saved {len(earnings_df)} earnings records")
    except:
        print("  No earnings table found")

    # Export news table (if exists)
    try:
        print("Converting news table...")
        news_df = pd.read_sql_query("SELECT * FROM news", conn)
        news_df.to_parquet(f'{OUTPUT_DIR}/news.parquet', index=False)
        print(f"  Saved {len(news_df)} news records")
    except:
        print("  No news table found")

    conn.close()

    # Calculate total size
    total_size = sum(os.path.getsize(f'{OUTPUT_DIR}/{f}') for f in os.listdir(OUTPUT_DIR))
    print(f"\nTotal Parquet size: {total_size / (1024 * 1024):.2f} MB")
    print(f"Files saved to '{OUTPUT_DIR}/' directory")


if __name__ == "__main__":
    convert_to_parquet()