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
    df['sma_50'] = gb['price'].transform(lambda x: x.rolling(window=50).mean())
    df['sma_200'] = gb['price'].transform(lambda x: x.rolling(window=200).mean())
    df['rsi_14'] = 100 - (100 / (1 + rs)) if (rs := (gain := (delta := gb['price'].transform('diff')).where(delta > 0,
                                                                                                            0).rolling(
        window=14).mean()) / (loss := (-delta.where(delta < 0, 0)).rolling(
        window=14).mean())) is not None and not loss.eq(0).any() else np.nan
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Add signal flags directly to the dataframe for easier simulation
    df['golden_cross_signal'] = (df['sma_50'] > df['sma_200']) & (
                df.groupby('instrument')['sma_50'].shift(1) < df.groupby('instrument')['sma_200'].shift(1))
    df['rsi_oversold_signal'] = (df['rsi_14'] < 30) & (df.groupby('instrument')['rsi_14'].shift(1) >= 30)

    print("Feature engineering complete.")
    return df


# --- NEW: Phase 10 - Investment Strategy Simulation ---
def simulate_investment_strategies(df: pd.DataFrame):
    """
    Simulates and compares different yearly investment strategies.
    """
    print("\n\n--- ANALYSIS 5: OPTIMAL STRATEGY SIMULATION ---")

    # --- Parameters ---
    ANNUAL_CAPITAL = 120000
    instrument_name = df['instrument'].unique()[0]
    print(f"\n--- Simulating for Instrument: {instrument_name} with ${ANNUAL_CAPITAL:,} annual capital ---")

    years = sorted(df['year'].unique())
    # We can only simulate full years, so we exclude the first and last partial years if they exist
    full_years = [y for y in years if len(df[(df['year'] == y) & (df['instrument'] == instrument_name)]) > 250]
    print(f"Analyzing full years: {full_years}")

    all_results = {}

    for year in full_years:
        year_data = df[(df['year'] == year) & (df['instrument'] == instrument_name)]
        final_price_of_year = year_data.iloc[-1]['price']

        # --- 1. Benchmark: Dollar Cost Averaging (DCA) ---
        total_shares_dca = 0
        for month in range(1, 13):
            monthly_data = year_data[year_data['month'] == month]
            if not monthly_data.empty:
                entry_price = monthly_data.iloc[0]['price']
                total_shares_dca += (ANNUAL_CAPITAL / 12) / entry_price
        final_value_dca = total_shares_dca * final_price_of_year
        all_results.setdefault('DCA ($10k/month)', []).append(final_value_dca)

        # --- 2. Golden Cross Strategy ---
        gc_signals = year_data[year_data['golden_cross_signal']]
        if not gc_signals.empty:
            entry_price = gc_signals.iloc[0]['price']
            shares_bought = ANNUAL_CAPITAL / entry_price
            final_value_gc = shares_bought * final_price_of_year
        else:  # No signal, no investment
            final_value_gc = ANNUAL_CAPITAL
        all_results.setdefault('Golden Cross (Lump Sum)', []).append(final_value_gc)

        # --- 3. RSI Oversold Strategy ---
        rsi_signals = year_data[year_data['rsi_oversold_signal']]
        if not rsi_signals.empty:
            entry_price = rsi_signals.iloc[0]['price']
            shares_bought = ANNUAL_CAPITAL / entry_price
            final_value_rsi = shares_bought * final_price_of_year
        else:  # No signal, no investment
            final_value_rsi = ANNUAL_CAPITAL
        all_results.setdefault('RSI Oversold (Lump Sum)', []).append(final_value_rsi)

        # --- 4. Top Seasonal Strategy (Buy Feb, Sell Aug) ---
        # NOTE: This is hardcoded based on previous analysis. Can be made dynamic.
        buy_month, sell_month = 2, 8
        buy_data = year_data[year_data['month'] == buy_month]
        sell_data = year_data[year_data['month'] == sell_month]
        if not buy_data.empty and not sell_data.empty:
            entry_price = buy_data.iloc[0]['price']
            exit_price = sell_data.iloc[0]['price']
            shares_bought = ANNUAL_CAPITAL / entry_price
            profit = (exit_price - entry_price) * shares_bought
            final_value_seasonal = ANNUAL_CAPITAL + profit
        else:  # Cannot execute trade
            final_value_seasonal = ANNUAL_CAPITAL
        all_results.setdefault('Seasonal (Buy Feb, Sell Aug)', []).append(final_value_seasonal)

    # --- Final Report Generation ---
    report_data = []
    for strategy, yearly_values in all_results.items():
        yearly_returns = [(val / ANNUAL_CAPITAL - 1) * 100 for val in yearly_values]
        avg_return = np.mean(yearly_returns)
        winning_years = sum(1 for r in yearly_returns if r > 0)
        worst_year = min(yearly_returns)

        report_data.append({
            'Strategy': strategy,
            'Avg Annual Return %': avg_return,
            'Winning Years': f"{winning_years} out of {len(full_years)}",
            'Worst Year %': worst_year
        })

    report_df = pd.DataFrame(report_data).sort_values(by='Avg Annual Return %', ascending=False)
    print("\n\n--- Strategy Performance Ranking ---")
    print(report_df.round(2).to_string(index=False))

    winner = report_df.iloc[0]
    benchmark = report_df[report_df['Strategy'] == 'DCA ($10k/month)'].iloc[0]

    print("\n\n--- Conclusion ---")
    print(f"The historical winner is the '{winner['Strategy']}' strategy.")
    print(
        f"It achieved an average annual return of {winner['Avg Annual Return %']:.2f}%, compared to the benchmark's {benchmark['Avg Annual Return %']:.2f}%.")
    print(
        f"This strategy resulted in a winning year {winner['Winning Years']} times and its worst performance in a single year was a {winner['Worst Year %']:.2f}% return.")
    print("\nDisclaimer: Past performance is not indicative of future results.")


# --- Main Execution ---
if __name__ == "__main__":
    excel_file_path = 'tjh.xlsx'

    cleaned_df = load_and_clean_data(excel_file_path)

    if not cleaned_df.empty:
        # We must run engineer_features to get the signals
        features_df = engineer_features(cleaned_df)

        # Run the final, top-level simulation
        simulate_investment_strategies(features_df)

    else:
        print("\nExecution halted because the data frame was empty after cleaning.")