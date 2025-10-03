import pandas as pd
import numpy as np


# --- Phase 1 & 2: Data Loading and Cleaning (Final Version) ---
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    # (This function remains the same as the last version)
    print("Step 1: Loading and cleaning data...")
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return pd.DataFrame()
    df.columns = df.columns.str.strip()
    required_cols_keys = ['Date', 'Instrument', 'Current Yr Div ($)', 'Volume', "Today's Range ($)",
                          'Last Traded Price ($)', 'Closing Bid ($)', 'Closing Ask ($)']
    missing_cols = [col for col in required_cols_keys if col not in df.columns]
    if missing_cols:
        print(f"\nError: The following required columns are missing: {missing_cols}")
        return pd.DataFrame()
    required_cols_map = {'Date': 'date', 'Instrument': 'instrument', 'Current Yr Div ($)': 'current_yr_div',
                         'Volume': 'volume', 'Today\'s Range ($)': 'daily_range', 'Last Traded Price ($)': 'price',
                         'Closing Bid ($)': 'bid', 'Closing Ask ($)': 'ask'}
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
    # (This function remains the same as the last version)
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
    # (This function remains the same as the last version)
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
            # ... (other signals are the same)
    print("Pattern scanning complete.")
    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        signals_df = signals_df[['date', 'instrument', 'pattern', 'signal', 'details', 'price_on_signal']]
    return signals_df


# --- NEW: Phase 7 - Profit Target Analysis ---
def analyze_profit_target_patterns(df: pd.DataFrame, signals_df: pd.DataFrame):
    """
    Analyzes BUY signals to see how often they hit a specific profit target.
    """
    print("\n--- PROFIT TARGET (+10%) ANALYSIS ---")

    # --- Parameters you can easily change ---
    profit_target_pct = 10.0
    stop_loss_pct = 5.0
    max_holding_days = 90
    # ----------------------------------------

    for instrument in df['instrument'].unique():
        print(f"\n--- Analyzing Instrument: {instrument} ---")
        instrument_df = df[df['instrument'] == instrument]
        instrument_signals = signals_df[signals_df['instrument'] == instrument]

        buy_signals = instrument_signals[instrument_signals['signal'] == 'BUY'].copy()

        if buy_signals.empty:
            print("  No 'BUY' signals found for this instrument.")
            continue

        trade_results = []

        for index, signal in buy_signals.iterrows():
            entry_date = signal['date']
            entry_price = signal['price_on_signal']
            pattern_name = signal['pattern']

            take_profit_price = entry_price * (1 + profit_target_pct / 100)
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100)

            # Get all trading days after the signal
            future_data = instrument_df[instrument_df['date'] > entry_date].head(max_holding_days)

            outcome = 'Expired'
            exit_date = None
            days_held = max_holding_days

            for i, day in future_data.iterrows():
                # Check for a win
                if day['price'] >= take_profit_price:
                    outcome = 'Win'
                    exit_date = day['date']
                    days_held = (exit_date - entry_date).days
                    break
                # Check for a loss
                elif day['price'] <= stop_loss_price:
                    outcome = 'Loss'
                    exit_date = day['date']
                    days_held = (exit_date - entry_date).days
                    break

            trade_results.append({
                'pattern': pattern_name,
                'outcome': outcome,
                'days_held': days_held
            })

        if not trade_results:
            continue

        # Aggregate and print the results
        results_df = pd.DataFrame(trade_results)
        for pattern in results_df['pattern'].unique():
            pattern_results = results_df[results_df['pattern'] == pattern]
            total_trades = len(pattern_results)
            wins = len(pattern_results[pattern_results['outcome'] == 'Win'])
            losses = len(pattern_results[pattern_results['outcome'] == 'Loss'])
            expired = len(pattern_results[pattern_results['outcome'] == 'Expired'])

            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

            # Calculate average hold time ONLY for winning trades
            avg_days_for_wins = pattern_results[pattern_results['outcome'] == 'Win']['days_held'].mean()

            print(f"\n  Pattern: {pattern} ({total_trades} trades)")
            print(f"    - Win Rate for +10% profit: {win_rate:.1f}% ({wins} wins)")
            print(f"    - Stop-Loss hit (-5%): {losses} times")
            print(f"    - Expired (90 days): {expired} times")
            if not pd.isna(avg_days_for_wins):
                print(f"    - Average holding days for wins: {avg_days_for_wins:.1f} days")


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

        # We've removed the old analysis function and replaced it with the new one
        analyze_profit_target_patterns(features_df, signals_df)

    else:
        print("\nExecution halted because the data frame was empty after cleaning.")