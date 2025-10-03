import pandas as pd
import numpy as np


# --- Phase 1 & 2: Data Loading and Cleaning (Final Version) ---
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    # (This function remains the same)
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


# --- Phase 3: Feature Engineering ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # (This function remains the same)
    print("Step 2: Engineering features...")
    gb = df.groupby('instrument')
    df['daily_return_%'] = gb['price'].transform(lambda x: x.pct_change() * 100)
    # ... (rest of function is the same, adding month and year is important)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    print("Feature engineering complete.")
    return df


# --- Phase 9 - Seasonal Strategy Backtester (with Detailed Breakdown and Error Fix) ---
def analyze_seasonal_strategy(df: pd.DataFrame, profit_target_pct: float):
    """
    Backtests a purely seasonal strategy with a detailed, trade-by-trade breakdown.
    """
    print(f"\n\n--- ANALYSIS: SEASONAL STRATEGY (Win Rate for +{profit_target_pct}%) ---")
    if df.empty: return

    for instrument in df['instrument'].unique():
        print(f"\n--- Analyzing Instrument: {instrument} ---")
        instrument_df = df[df['instrument'] == instrument]

        aggregated_results = []
        detailed_trade_log = []

        for buy_month in range(1, 13):
            for hold_months in range(1, 12):
                strategy_wins = 0
                strategy_attempts = 0
                days_to_win_list = []

                sell_month = (buy_month + hold_months - 1) % 12 + 1

                for year in instrument_df['year'].unique():
                    entry_year = year
                    entry_data = instrument_df[
                        (instrument_df['year'] == entry_year) & (instrument_df['month'] == buy_month)]
                    if entry_data.empty: continue

                    entry_day = entry_data.iloc[0]
                    entry_date = entry_day['date']
                    entry_price = entry_day['price']
                    profit_target_price = entry_price * (1 + profit_target_pct / 100)

                    sell_year = entry_year + (buy_month + hold_months - 1) // 12

                    sell_window_start = entry_date
                    if sell_month == 12:
                        sell_window_end = pd.Timestamp(year=sell_year, month=12, day=31)
                    else:
                        sell_window_end = pd.Timestamp(year=sell_year, month=sell_month + 1, day=1) - pd.Timedelta(
                            days=1)

                    sell_window_data = instrument_df[
                        (instrument_df['date'] > sell_window_start) & (instrument_df['date'] <= sell_window_end)]
                    if sell_window_data.empty: continue

                    strategy_attempts += 1

                    highest_price_in_window = sell_window_data['price'].max()
                    highest_potential_gain_pct = (highest_price_in_window - entry_price) / entry_price * 100
                    peak_date_index = sell_window_data['price'].idxmax()
                    peak_date = sell_window_data.loc[peak_date_index, 'date']

                    winning_trades = sell_window_data[sell_window_data['price'] >= profit_target_price]
                    outcome = 'Miss'
                    exit_date = None
                    days_to_win = None

                    if not winning_trades.empty:
                        strategy_wins += 1
                        outcome = 'Win'
                        first_win_date = winning_trades.iloc[0]['date']
                        exit_date = first_win_date
                        days_to_win = (first_win_date - entry_date).days
                        days_to_win_list.append(days_to_win)

                    detailed_trade_log.append({
                        'Buy Month': buy_month, 'Sell By Month': sell_month, 'Year': year,
                        'Entry Date': entry_date, 'Entry Price': entry_price,
                        'Outcome': outcome, 'Exit Date': exit_date, 'Days to Win': days_to_win,
                        'Highest Potential Gain %': highest_potential_gain_pct,
                        'Peak Date': peak_date
                    })

                if strategy_attempts > 0:
                    win_rate = (strategy_wins / strategy_attempts) * 100
                    avg_days_to_win = np.mean(days_to_win_list) if days_to_win_list else np.nan
                    aggregated_results.append({
                        'Buy Month': buy_month, 'Sell By Month': sell_month,
                        'Attempts': strategy_attempts, 'Win Rate %': win_rate,
                        'Avg Days to Win': avg_days_to_win
                    })

        # --- Display Part 1: Summary of Top Strategies ---
        if aggregated_results:
            results_df = pd.DataFrame(aggregated_results)
            top_strategies = results_df.sort_values(by='Win Rate %', ascending=False).head(15)

            print("\n--- Summary: Top 15 Most Successful Seasonal Strategies ---")
            top_strategies_display = top_strategies.copy()
            # *** THIS IS THE CORRECTED SECTION FOR THE SUMMARY TABLE ***
            top_strategies_display['Buy Month'] = top_strategies_display['Buy Month'].apply(
                lambda x: pd.to_datetime(str(int(x)), format='%m').strftime('%b'))
            top_strategies_display['Sell By Month'] = top_strategies_display['Sell By Month'].apply(
                lambda x: pd.to_datetime(str(int(x)), format='%m').strftime('%b'))
            top_strategies_display = top_strategies_display.round(1)
            print(top_strategies_display.to_string(index=False))

            # --- Display Part 2: Detailed Breakdown ---
            detailed_log_df = pd.DataFrame(detailed_trade_log)
            print("\n\n--- Detailed Breakdown of Top Strategies ---")

            for index, strategy in top_strategies.iterrows():
                buy_m = strategy['Buy Month']
                sell_m = strategy['Sell By Month']
                # *** THIS IS THE CORRECTED SECTION FOR THE DETAILED BREAKDOWN ***
                buy_month_name = pd.to_datetime(str(int(buy_m)), format='%m').strftime('%B')
                sell_month_name = pd.to_datetime(str(int(sell_m)), format='%m').strftime('%B')

                print(f"\n\n--- Strategy: Buy in {buy_month_name}, Sell by {sell_month_name} ---")

                breakdown_df = detailed_log_df[
                    (detailed_log_df['Buy Month'] == buy_m) & (detailed_log_df['Sell By Month'] == sell_m)]

                display_cols = breakdown_df[['Year', 'Entry Date', 'Entry Price', 'Outcome', 'Exit Date', 'Days to Win',
                                             'Highest Potential Gain %', 'Peak Date']].copy()
                display_cols['Entry Date'] = display_cols['Entry Date'].dt.strftime('%Y-%m-%d')
                display_cols['Exit Date'] = pd.to_datetime(display_cols['Exit Date']).dt.strftime('%Y-%m-%d').replace(
                    'NaT', 'N/A')
                display_cols['Peak Date'] = display_cols['Peak Date'].dt.strftime('%Y-%m-%d')
                display_cols['Entry Price'] = display_cols['Entry Price'].map('{:,.2f}'.format)
                display_cols['Highest Potential Gain %'] = display_cols['Highest Potential Gain %'].map(
                    '{:.2f}%'.format)

                print(display_cols.to_string(index=False))


# --- Main Execution ---
if __name__ == "__main__":
    excel_file_path = 'tjh.xlsx'

    cleaned_df = load_and_clean_data(excel_file_path)

    if not cleaned_df.empty:
        features_df = engineer_features(cleaned_df)

        analyze_seasonal_strategy(features_df, profit_target_pct=15.0)

    else:
        print("\nExecution halted because the data frame was empty after cleaning.")