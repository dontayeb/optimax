import pandas as pd
import numpy as np
from datetime import datetime


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
    print("Step 2: Engineering features...")
    gb = df.groupby('instrument')
    df['sma_50'] = gb['price'].transform(lambda x: x.rolling(window=50).mean())
    df['sma_200'] = gb['price'].transform(lambda x: x.rolling(window=200).mean())
    delta = gb['price'].transform('diff')
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Add signal flags directly to the dataframe for easier simulation
    df['golden_cross_signal'] = (df['sma_50'] > df['sma_200']) & (
                df.groupby('instrument')['sma_50'].shift(1) < df.groupby('instrument')['sma_200'].shift(1))
    df['rsi_oversold_signal'] = (df['rsi_14'] < 30) & (df.groupby('instrument')['rsi_14'].shift(1) >= 30)

    print("Feature engineering complete.")
    return df


# --- NEW: Phase 10 - Portfolio Strategy Simulation ---
def simulate_portfolio_strategies(df: pd.DataFrame):
    print("\n\n--- FINAL ANALYSIS: PORTFOLIO STRATEGY SIMULATION ---")

    # --- Parameters ---
    instrument_name = df['instrument'].unique()[0]
    print(f"\n--- Simulating for Instrument: {instrument_name} over the entire period ---")

    # --- Strategy Definitions ---
    # Active Strategy will invest in these months. Let's use the best two found previously.
    SEASONAL_BUY_MONTHS = [2, 10]  # Feb, Oct
    SEASONAL_SELL_MONTHS = {2: 8, 10: 4}  # Buy in Feb -> Sell in Aug; Buy in Oct -> Sell in Apr
    INVESTMENT_PER_TRADE = 10000
    TECHNICAL_PROFIT_TARGET = 1.15  # 15%

    # --- Initialize Portfolios ---
    dca = {'cash': 0, 'shares': 0, 'value': 0}
    active = {'cash': 120000, 'shares': 0, 'value': 120000, 'open_trades': []}

    # --- Initialize Logs ---
    dca_log = []
    active_log = []
    yearly_report = []

    # --- Main Simulation Loop (Day by Day) ---
    for i, day in df.iterrows():
        current_date = day['date']
        current_price = day['price']

        # --- Update Portfolio Values ---
        dca['value'] = dca['cash'] + (dca['shares'] * current_price)
        active['value'] = active['cash'] + (active['shares'] * current_price)

        # --- Yearly Capital Injection ---
        if i > 0 and day['year'] != df.iloc[i - 1]['year']:
            active['cash'] += 120000

        # --- DCA Strategy Logic ---
        if i == 0 or day['month'] != df.iloc[i - 1]['month']:
            dca['cash'] += 10000
            shares_bought = dca['cash'] / current_price
            dca['shares'] += shares_bought
            dca_log.append(
                f"{current_date.strftime('%Y-%m-%d')}: Invested ${dca['cash']:,.2f} for {shares_bought:.2f} shares at ${current_price:.2f}")
            dca['cash'] = 0

        # --- Active Strategy: Exit Logic (check before entering) ---
        for trade in active['open_trades'][:]:  # Iterate on a copy
            # Seasonal Exit
            if trade['type'] == 'Seasonal' and day['month'] == trade['sell_month']:
                proceeds = trade['shares'] * current_price
                active['cash'] += proceeds
                active['shares'] -= trade['shares']
                active['open_trades'].remove(trade)
                active_log.append(
                    f"{current_date.strftime('%Y-%m-%d')}: SOLD (Seasonal) {trade['shares']:.2f} shares at ${current_price:.2f}. Profit/Loss: ${proceeds - trade['cost']:.2f}")
            # Technical Exit
            elif trade['type'] == 'Technical' and current_price >= trade['target_price']:
                proceeds = trade['shares'] * current_price
                active['cash'] += proceeds
                active['shares'] -= trade['shares']
                active['open_trades'].remove(trade)
                active_log.append(
                    f"{current_date.strftime('%Y-%m-%d')}: SOLD (Target Met) {trade['shares']:.2f} shares at ${current_price:.2f}. Profit/Loss: ${proceeds - trade['cost']:.2f}")

        # --- Active Strategy: Entry Logic ---
        is_seasonal_buy_day = (i == 0 or day['month'] != df.iloc[i - 1]['month']) and day[
            'month'] in SEASONAL_BUY_MONTHS
        is_gc_signal = day['golden_cross_signal']
        is_rsi_signal = day['rsi_oversold_signal']

        if (is_seasonal_buy_day or is_gc_signal or is_rsi_signal) and active['cash'] > 0:
            investment_amount = min(INVESTMENT_PER_TRADE, active['cash'])
            shares_bought = investment_amount / current_price
            active['cash'] -= investment_amount
            active['shares'] += shares_bought

            trade_type = 'Seasonal' if is_seasonal_buy_day else 'Technical'
            signal = 'Seasonal Buy'
            if is_gc_signal: signal = 'Golden Cross'
            if is_rsi_signal: signal = 'RSI Oversold'

            new_trade = {
                'entry_date': current_date,
                'entry_price': current_price,
                'shares': shares_bought,
                'cost': investment_amount,
                'type': trade_type,
                'target_price': current_price * TECHNICAL_PROFIT_TARGET if trade_type == 'Technical' else None,
                'sell_month': SEASONAL_SELL_MONTHS.get(day['month']) if trade_type == 'Seasonal' else None
            }
            active['open_trades'].append(new_trade)
            active_log.append(
                f"{current_date.strftime('%Y-%m-%d')}: BOUGHT ({signal}) {shares_bought:.2f} shares for ${investment_amount:,.2f} at ${current_price:.2f}")

        # --- End of Year Reporting ---
        if i > 0 and day['year'] != df.iloc[i - 1]['year']:
            prev_year = df.iloc[i - 1]['year']
            yearly_report.append({
                'Year': prev_year,
                'DCA End Value': dca['value'],
                'Active Strategy End Value': active['value']
            })

    # Add final year to report
    final_year = df.iloc[-1]['year']
    yearly_report.append({
        'Year': final_year,
        'DCA End Value': dca['value'],
        'Active Strategy End Value': active['value']
    })

    # --- Final Report Generation ---
    print("\n\n--- FINAL RESULTS ---")
    print(f"\nEnd of Period Value (DCA Strategy): ${dca['value']:,.2f}")
    print(f"End of Period Value (Active Strategy): ${active['value']:,.2f}")

    print("\n--- Year-by-Year Portfolio Value Breakdown ---")
    report_df = pd.DataFrame(yearly_report)
    report_df['DCA End Value'] = report_df['DCA End Value'].map('${:,.2f}'.format)
    report_df['Active Strategy End Value'] = report_df['Active Strategy End Value'].map('${:,.2f}'.format)
    print(report_df.to_string(index=False))

    print("\n--- Active Strategy Trade Log ---")
    for entry in active_log:
        print(entry)

    print("\n--- DCA Strategy Monthly Entries (Example: First Year) ---")
    first_year = df['year'].min()
    for entry in dca_log:
        if str(first_year) in entry:
            print(entry)


# --- Main Execution ---
if __name__ == "__main__":
    excel_file_path = 'tjh.xlsx'

    cleaned_df = load_and_clean_data(excel_file_path)

    if not cleaned_df.empty:
        features_df = engineer_features(cleaned_df)
        simulate_portfolio_strategies(features_df)
    else:
        print("\nExecution halted because the data frame was empty after cleaning.")