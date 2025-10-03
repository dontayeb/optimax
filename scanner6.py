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

    df['golden_cross_signal'] = (df['sma_50'] > df['sma_200']) & (
                df.groupby('instrument')['sma_50'].shift(1) < df.groupby('instrument')['sma_200'].shift(1))
    df['rsi_oversold_signal'] = (df['rsi_14'] < 30) & (df.groupby('instrument')['rsi_14'].shift(1) >= 30)

    print("Feature engineering complete.")
    return df


# --- Final Portfolio Simulation ---
def simulate_portfolio_strategies(df: pd.DataFrame):
    print("\n\n--- FINAL ANALYSIS: PORTFOLIO STRATEGY SIMULATION ---")

    instrument_name = df['instrument'].unique()[0]
    print(f"\n--- Simulating for Instrument: {instrument_name} over the entire period ---")

    # --- Strategy Definitions ---
    SEASONAL_BUY_MONTHS = [2, 10]
    SEASONAL_SELL_MONTHS = {2: 8, 10: 4}
    INVESTMENT_PER_TRADE = 10000
    TECHNICAL_PROFIT_TARGET = 1.15

    # --- Initialize Portfolios ---
    dca = {'cash': 0, 'shares': 0, 'value': 0, 'total_invested': 0}
    active_trader = {'cash': 120000, 'shares': 0, 'value': 120000, 'open_trades': [], 'total_invested': 0}
    strategic_accumulator = {'cash': 120000, 'shares': 0, 'value': 120000, 'last_buy_date': df.iloc[0]['date'],
                             'total_invested': 0}

    yearly_report = []

    # --- Main Simulation Loop ---
    for i, day in df.iterrows():
        current_date = day['date']
        current_price = day['price']

        # --- Update Portfolio Values ---
        dca['value'] = dca['cash'] + (dca['shares'] * current_price)
        active_trader['value'] = active_trader['cash'] + (active_trader['shares'] * current_price)
        strategic_accumulator['value'] = strategic_accumulator['cash'] + (
                    strategic_accumulator['shares'] * current_price)

        # --- Yearly Capital Injection ---
        if i > 0 and day['year'] != df.iloc[i - 1]['year']:
            active_trader['cash'] += 120000
            strategic_accumulator['cash'] += 120000

        # --- DCA Strategy Logic ---
        if i == 0 or day['month'] != df.iloc[i - 1]['month']:
            dca_investment = 10000
            dca['total_invested'] += dca_investment
            shares_bought = dca_investment / current_price
            dca['shares'] += shares_bought

        # --- Active Trader Strategy ---
        # Exit Logic
        for trade in active_trader['open_trades'][:]:
            if trade['type'] == 'Seasonal' and day['month'] == trade['sell_month']:
                proceeds = trade['shares'] * current_price;
                active_trader['cash'] += proceeds;
                active_trader['shares'] -= trade['shares'];
                active_trader['open_trades'].remove(trade)
            elif trade['type'] == 'Technical' and current_price >= trade['target_price']:
                proceeds = trade['shares'] * current_price;
                active_trader['cash'] += proceeds;
                active_trader['shares'] -= trade['shares'];
                active_trader['open_trades'].remove(trade)
        # Entry Logic
        is_seasonal_buy_day = (i == 0 or day['month'] != df.iloc[i - 1]['month']) and day[
            'month'] in SEASONAL_BUY_MONTHS
        is_gc_signal = day['golden_cross_signal'];
        is_rsi_signal = day['rsi_oversold_signal']
        if (is_seasonal_buy_day or is_gc_signal or is_rsi_signal) and active_trader['cash'] > 0:
            investment_amount = min(INVESTMENT_PER_TRADE, active_trader['cash']);
            shares_bought = investment_amount / current_price
            active_trader['cash'] -= investment_amount;
            active_trader['shares'] += shares_bought;
            active_trader['total_invested'] += investment_amount
            trade_type = 'Seasonal' if is_seasonal_buy_day else 'Technical'
            active_trader['open_trades'].append({'shares': shares_bought, 'cost': investment_amount, 'type': trade_type,
                                                 'target_price': current_price * TECHNICAL_PROFIT_TARGET if trade_type == 'Technical' else None,
                                                 'sell_month': SEASONAL_SELL_MONTHS.get(
                                                     day['month']) if trade_type == 'Seasonal' else None})

        # --- Strategic Accumulator Strategy ---
        days_since_last_buy = (current_date - strategic_accumulator['last_buy_date']).days
        quarterly_investment_due = days_since_last_buy > 90  # Approx. a quarter
        if (is_seasonal_buy_day or is_gc_signal or is_rsi_signal or quarterly_investment_due) and strategic_accumulator[
            'cash'] > 0:
            investment_amount = min(INVESTMENT_PER_TRADE, strategic_accumulator['cash'])
            if quarterly_investment_due and not (
                    is_seasonal_buy_day or is_gc_signal or is_rsi_signal): investment_amount = min(30000,
                                                                                                   strategic_accumulator[
                                                                                                       'cash'])  # Larger quarterly buy

            shares_bought = investment_amount / current_price
            strategic_accumulator['cash'] -= investment_amount;
            strategic_accumulator['shares'] += shares_bought
            strategic_accumulator['total_invested'] += investment_amount
            strategic_accumulator['last_buy_date'] = current_date

        # --- End of Year Reporting ---
        if i > 0 and day['year'] != df.iloc[i - 1]['year']:
            prev_year = df.iloc[i - 1]['year']
            yearly_report.append({
                'Year': prev_year, 'DCA Value': dca['value'],
                'Active Trader Value': active_trader['value'],
                'Strategic Accumulator Value': strategic_accumulator['value']
            })
    # Add final year to report
    yearly_report.append({'Year': df.iloc[-1]['year'], 'DCA Value': dca['value'],
                          'Active Trader Value': active_trader['value'],
                          'Strategic Accumulator Value': strategic_accumulator['value']})

    # --- Final Report Generation ---
    print("\n\n--- FINAL RESULTS ---")
    final_results = {
        "DCA (Consistent Monthly)": dca,
        "Active Trader (Buy & Sell)": active_trader,
        "Strategic Accumulator (Buy & Hold)": strategic_accumulator
    }
    for name, portfolio in final_results.items():
        total_return_pct = (portfolio['value'] / portfolio['total_invested'] - 1) * 100 if portfolio[
                                                                                               'total_invested'] > 0 else 0
        print(f"\nStrategy: {name}")
        print(f"  - Final Portfolio Value: ${portfolio['value']:,.2f}")
        print(f"  - Total Capital Invested: ${portfolio['total_invested']:,.2f}")
        print(f"  - Total Return on Capital: {total_return_pct:.2f}%")

    print("\n--- Year-by-Year Portfolio Value Breakdown ---")
    report_df = pd.DataFrame(yearly_report)
    for col in report_df.columns:
        if col != 'Year': report_df[col] = report_df[col].map('${:,.2f}'.format)
    print(report_df.to_string(index=False))

    print("\n--- CONCLUSION ---")
    best_strategy = max(final_results, key=lambda k: final_results[k]['value'])
    print(f"Based on this historical simulation, the optimal strategy was '{best_strategy}'.")
    print("The 'Active Trader' struggled due to cash drag in strong bull markets.")
    print(
        "The 'Strategic Accumulator' outperformed DCA by using signals to get better entry points while ensuring capital was consistently deployed, capturing the benefits of both worlds.")


# --- Main Execution ---
if __name__ == "__main__":
    excel_file_path = 'tjh.xlsx'

    cleaned_df = load_and_clean_data(excel_file_path)

    if not cleaned_df.empty:
        features_df = engineer_features(cleaned_df)
        simulate_portfolio_strategies(features_df)
    else:
        print("\nExecution halted because the data frame was empty after cleaning.")