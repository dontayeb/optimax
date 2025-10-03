import pandas as pd
import numpy as np


# --- Phase 1 & 2: Data Loading and Cleaning (Final Version) ---
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads, cleans, and prepares stock data. This version correctly parses the
    "Low - High" format in the 'Today's Range ($)' column.
    """
    print("Step 1: Loading and cleaning data...")
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    required_cols_keys = [
        'Date', 'Instrument', 'Current Yr Div ($)', 'Volume',
        "Today's Range ($)", 'Last Traded Price ($)', 'Closing Bid ($)', 'Closing Ask ($)'
    ]
    missing_cols = [col for col in required_cols_keys if col not in df.columns]
    if missing_cols:
        print(f"\nError: The following required columns are missing: {missing_cols}")
        return pd.DataFrame()

    required_cols_map = {
        'Date': 'date', 'Instrument': 'instrument', 'Current Yr Div ($)': 'current_yr_div',
        'Volume': 'volume', 'Today\'s Range ($)': 'daily_range', 'Last Traded Price ($)': 'price',
        'Closing Bid ($)': 'bid', 'Closing Ask ($)': 'ask'
    }
    df = df[list(required_cols_map.keys())].rename(columns=required_cols_map)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    range_split = df['daily_range'].astype(str).str.split(' - ', expand=True)
    low_price = pd.to_numeric(range_split[0], errors='coerce')
    high_price = pd.to_numeric(range_split[1], errors='coerce')
    df['daily_range'] = high_price - low_price

    numeric_cols = ['current_yr_div', 'volume', 'price', 'bid', 'ask']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[['price', 'bid', 'ask', 'current_yr_div']] = df[['price', 'bid', 'ask', 'current_yr_div']].ffill()
    df[['volume', 'daily_range']] = df[['volume', 'daily_range']].fillna(0)

    df.dropna(inplace=True)

    df.sort_values(by=['instrument', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("Data loading and cleaning complete.")
    return df


# --- Phase 3: Feature Engineering (Corrected Version) ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Step 2: Engineering features...")
    gb = df.groupby('instrument')
    df['daily_return_%'] = gb['price'].transform(lambda x: x.pct_change() * 100)
    df['sma_50'] = gb['price'].transform(lambda x: x.rolling(window=50).mean())
    df['sma_200'] = gb['price'].transform(lambda x: x.rolling(window=200).mean())
    delta = gb['price'].transform('diff')
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['52_week_high'] = gb['price'].transform(lambda x: x.rolling(window=252, min_periods=1).max())
    df['52_week_low'] = gb['price'].transform(lambda x: x.rolling(window=252, min_periods=1).min())
    df['avg_volume_30d'] = gb['volume'].transform(lambda x: x.rolling(window=30).mean())
    df['bid_ask_spread'] = df['ask'] - df['bid']
    df['year'] = df['date'].dt.year

    annual_dividends = df.groupby(['instrument', 'year'])['current_yr_div'].max().reset_index()
    annual_dividends.rename(columns={'current_yr_div': 'total_annual_dividend'}, inplace=True)
    annual_dividends['prev_year'] = annual_dividends['year'] - 1
    annual_dividends = annual_dividends.merge(
        annual_dividends[['instrument', 'year', 'total_annual_dividend']],
        left_on=['instrument', 'prev_year'], right_on=['instrument', 'year'],
        suffixes=('', '_prev'), how='left'
    )
    annual_dividends.rename(columns={'total_annual_dividend_prev': 'previous_annual_dividend'}, inplace=True)

    df = df.merge(
        annual_dividends[['instrument', 'year', 'total_annual_dividend', 'previous_annual_dividend']],
        on=['instrument', 'year'], how='left'
    )

    df['dividend_growth_%'] = (df['total_annual_dividend'] - df['previous_annual_dividend']) / df[
        'previous_annual_dividend'] * 100
    df['annualized_yield_%'] = (df['previous_annual_dividend'] / df['price']) * 100
    df['dividend_payment_event'] = gb['current_yr_div'].transform('diff') > 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("Feature engineering complete.")
    return df


# --- Phase 4: Pattern Recognition ---
def scan_for_patterns(df: pd.DataFrame) -> pd.DataFrame:
    print("Step 3: Scanning for patterns...")
    signals = []
    for instrument, group in df.groupby('instrument'):
        prev_row = group.shift(1)
        for i, row in group.iterrows():
            if row['sma_50'] > row['sma_200'] and prev_row.loc[i, 'sma_50'] < prev_row.loc[i, 'sma_200']:
                signals.append(
                    {'date': row['date'], 'instrument': instrument, 'pattern': 'Golden Cross', 'signal': 'BUY',
                     'details': f"50-SMA ({row['sma_50']:.2f}) crossed above 200-SMA ({row['sma_200']:.2f})",
                     'price_on_signal': row['price']})
            if row['sma_50'] < row['sma_200'] and prev_row.loc[i, 'sma_50'] > prev_row.loc[i, 'sma_200']:
                signals.append(
                    {'date': row['date'], 'instrument': instrument, 'pattern': 'Death Cross', 'signal': 'SELL',
                     'details': f"50-SMA ({row['sma_50']:.2f}) crossed below 200-SMA ({row['sma_200']:.2f})",
                     'price_on_signal': row['price']})
            if row['rsi_14'] < 30 and prev_row.loc[i, 'rsi_14'] >= 30:
                signals.append(
                    {'date': row['date'], 'instrument': instrument, 'pattern': 'RSI Oversold', 'signal': 'BUY',
                     'details': f"RSI crossed below 30, currently {row['rsi_14']:.2f}",
                     'price_on_signal': row['price']})
            if row['rsi_14'] > 70 and prev_row.loc[i, 'rsi_14'] <= 70:
                signals.append(
                    {'date': row['date'], 'instrument': instrument, 'pattern': 'RSI Overbought', 'signal': 'SELL',
                     'details': f"RSI crossed above 70, currently {row['rsi_14']:.2f}",
                     'price_on_signal': row['price']})
            if row['price'] >= row['52_week_high'] and prev_row.loc[i, 'price'] < prev_row.loc[i, '52_week_high']:
                signals.append(
                    {'date': row['date'], 'instrument': instrument, 'pattern': 'New 52-Week High', 'signal': 'WATCH',
                     'details': f"Price hit a new 52-week high of ${row['price']:.2f}",
                     'price_on_signal': row['price']})
            if row['avg_volume_30d'] > 0 and row['volume'] > (row['avg_volume_30d'] * 2.0):
                signals.append(
                    {'date': row['date'], 'instrument': instrument, 'pattern': 'Unusual Volume', 'signal': 'WATCH',
                     'details': f"Volume {row['volume']:.0f} is >2x the 30-day avg of {row['avg_volume_30d']:.0f}",
                     'price_on_signal': row['price']})
            if row['dividend_payment_event']:
                signals.append(
                    {'date': row['date'], 'instrument': instrument, 'pattern': 'Dividend Paid', 'signal': 'INFO',
                     'details': f"Dividend payment detected. Cumulative div for year is now ${row['current_yr_div']:.2f}",
                     'price_on_signal': row['price']})

    print("Pattern scanning complete.")
    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        signals_df = signals_df[['date', 'instrument', 'pattern', 'signal', 'details', 'price_on_signal']]
    return signals_df


# --- NEW: Phase 6 - Historical Pattern Analysis ---
def analyze_historical_patterns(df: pd.DataFrame, signals_df: pd.DataFrame):
    """
    Analyzes historical data to find seasonal patterns and the performance of signals.
    """
    print("\n--- HISTORICAL PATTERN ANALYSIS ---")

    if df.empty:
        print("Cannot perform analysis because the initial DataFrame is empty.")
        return

    for instrument in df['instrument'].unique():
        print(f"\n--- Analyzing Instrument: {instrument} ---")
        instrument_df = df[instrument == df['instrument']].copy()
        instrument_signals = signals_df[signals_df['instrument'] == instrument].copy()

        # 1. Monthly Performance Analysis
        instrument_df['month'] = instrument_df['date'].dt.month
        monthly_performance = instrument_df.groupby('month')['daily_return_%'].mean()
        print("\nAverage Monthly Returns (%):")
        for month, avg_return in monthly_performance.items():
            print(f"  Month {month:02d}: {avg_return:.2f}%")

        # 2. Signal Performance Analysis
        print("\nSignal Performance (Average % Return After Signal):")
        # Define forward-looking periods (in trading days)
        periods = {'7_days': 7, '30_days': 30, '90_days': 90}

        # Add future prices to the main instrument dataframe for quick lookups
        for name, days in periods.items():
            instrument_df[f'future_price_{name}'] = instrument_df['price'].shift(-days)

        # Merge future prices into the signals dataframe
        merged_signals = pd.merge(instrument_signals, instrument_df[
            ['date', 'future_price_7_days', 'future_price_30_days', 'future_price_90_days']], on='date', how='left')

        for pattern in merged_signals['pattern'].unique():
            pattern_signals = merged_signals[merged_signals['pattern'] == pattern]
            print(f"  Pattern: {pattern} ({len(pattern_signals)} occurrences)")

            for name, days in periods.items():
                future_price_col = f'future_price_{name}'
                # Calculate the average return
                returns = (pattern_signals[future_price_col] - pattern_signals['price_on_signal']) / pattern_signals[
                    'price_on_signal'] * 100
                avg_return = returns.mean()
                if not pd.isna(avg_return):
                    print(f"    - Avg return after {days} days: {avg_return:.2f}%")


# --- Main Execution ---
if __name__ == "__main__":
    excel_file_path = 'tjh.xlsx'

    cleaned_df = load_and_clean_data(excel_file_path)

    if not cleaned_df.empty:
        features_df = engineer_features(cleaned_df)
        signals_df = scan_for_patterns(features_df)

        print("\n--- SCANNER RESULTS (EVENTS) ---")
        if signals_df.empty:
            print("No significant patterns were detected.")
        else:
            signals_df.sort_values(by='date', inplace=True)
            print(signals_df.to_string())

        # Run the new analysis function
        analyze_historical_patterns(features_df, signals_df)

    else:
        print("\nExecution halted because the data frame was empty after cleaning.")