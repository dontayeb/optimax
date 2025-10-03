import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")


# --- 1. CORE DATA & ENGINE FUNCTIONS ---

@st.cache_data(ttl=3600)
def load_data_from_db():
    try:
        conn = sqlite3.connect('market_data.db', check_same_thread=False)
        price_query = "SELECT s.ticker, d.date, d.close, d.volume FROM daily_data d JOIN stocks s ON s.id = d.stock_id ORDER BY s.ticker, d.date"
        price_df = pd.read_sql_query(price_query, conn)
        price_df['date'] = pd.to_datetime(price_df['date'])
        stocks_df = pd.read_sql_query("SELECT id, ticker FROM stocks", conn)
        dividends_df = pd.read_sql_query("SELECT * FROM dividends", conn)
        dividends_df = pd.merge(dividends_df, stocks_df, left_on='stock_id', right_on='id', how='left')
        conn.close()
        tickers = sorted(price_df['ticker'].unique().tolist())
        return price_df, tickers, dividends_df
    except Exception as e:
        st.error(f"Error loading database: {e}. Has the scraper been run?")
        return pd.DataFrame(), [], pd.DataFrame()


class SimulationEngine:
    def __init__(self, full_price_df, full_dividends_df, ticker):
        self.ticker = ticker
        self.df = full_price_df[full_price_df['ticker'] == ticker].copy().reset_index(drop=True)
        if self.df.empty: return
        self.dividends = full_dividends_df[full_dividends_df['ticker'] == ticker].copy()
        if not self.dividends.empty: self.dividends['ex_date'] = pd.to_datetime(self.dividends['ex_date'],
                                                                                errors='coerce')
        self._engineer_features()

    def _engineer_features(self):
        if self.df.empty: return
        self.df['sma_50'] = self.df['close'].rolling(window=50).mean()
        self.df['sma_200'] = self.df['close'].rolling(window=200).mean()
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi_14'] = 100 - (100 / (1 + rs))
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['golden_cross_signal'] = (self.df['sma_50'] > self.df['sma_200']) & (
                    self.df['sma_50'].shift(1) < self.df['sma_200'].shift(1))
        self.df['death_cross_signal'] = (self.df['sma_50'] < self.df['sma_200']) & (
                    self.df['sma_50'].shift(1) > self.df['sma_200'].shift(1))
        self.df['rsi_oversold_signal'] = (self.df['rsi_14'] < 30) & (self.df['rsi_14'].shift(1) >= 30)

    def run_dca_vs_active(self, monthly_dca, annual_active, profit_target_pct, rsi_max_hold_days):
        if self.df.empty: return {}
        SEASONAL_BUY_MONTHS = [2, 10];
        SEASONAL_SELL_MONTHS = {2: 8, 10: 4}
        INVESTMENT_PER_TRADE = 10000;
        TECHNICAL_PROFIT_TARGET = 1 + (profit_target_pct / 100)
        portfolios = {'DCA': {'cash': 0, 'shares': 0, 'value': 0, 'total_invested': 0},
                      'Active Trader': {'cash': annual_active, 'shares': 0, 'value': annual_active, 'total_invested': 0,
                                        'open_trades': [], 'signal_stats': {s: {'entries': 0, 'losses': 0} for s in
                                                                            ["Golden Cross", "RSI Oversold",
                                                                             "Seasonal"]}},
                      'Strategic Accumulator': {'cash': annual_active, 'shares': 0, 'value': annual_active,
                                                'total_invested': 0, 'last_buy_date': self.df.iloc[0]['date'],
                                                'all_entries': []}}
        logs = {'Active Trader': [], 'Strategic Accumulator': []};
        yearly_report = [];
        portfolio_over_time_log = []
        for i, day in self.df.iterrows():
            current_date, current_price = day['date'], day['close']
            for name, p in portfolios.items(): p['value'] = p['cash'] + (p['shares'] * current_price)
            if i > 0 and day['year'] != self.df.iloc[i - 1]['year']: portfolios['Active Trader'][
                'cash'] += annual_active; portfolios['Strategic Accumulator']['cash'] += annual_active
            if i == 0 or day['month'] != self.df.iloc[i - 1]['month']: portfolios['DCA'][
                'total_invested'] += monthly_dca; portfolios['DCA']['shares'] += monthly_dca / current_price
            p_trader = portfolios['Active Trader']
            for trade in p_trader['open_trades'][:]:
                exit_reason = None
                days_held = (current_date - trade['entry_date']).days
                if trade['type'] == 'Seasonal' and day['month'] == trade['sell_month']:
                    exit_reason = "Seasonal Sell"
                elif trade['type'] == 'Technical' and current_price >= trade['target_price']:
                    exit_reason = "Target Met"
                elif trade['signal'] == 'Golden Cross' and day['death_cross_signal']:
                    exit_reason = "Death Cross"
                elif trade['signal'] == 'RSI Oversold' and days_held > rsi_max_hold_days:
                    exit_reason = "Time Stop"
                if exit_reason:
                    proceeds = trade['shares'] * current_price
                    if proceeds < trade['cost']: p_trader['signal_stats'][trade['signal']]['losses'] += 1
                    p_trader['cash'] += proceeds;
                    p_trader['shares'] -= trade['shares'];
                    p_trader['open_trades'].remove(trade)
                    logs['Active Trader'].append(
                        f"{current_date.strftime('%Y-%m-%d')}: SOLD ({exit_reason}) for ${proceeds:,.2f}. P/L: ${proceeds - trade['cost']:.2f}")
            is_seasonal = (i == 0 or day['month'] != self.df.iloc[i - 1]['month']) and day[
                'month'] in SEASONAL_BUY_MONTHS;
            is_gc, is_rsi = day['golden_cross_signal'], day['rsi_oversold_signal']
            if (is_seasonal or is_gc or is_rsi) and p_trader['cash'] > 0:
                signal = "Seasonal" if is_seasonal else "Golden Cross" if is_gc else "RSI Oversold";
                p_trader['signal_stats'][signal]['entries'] += 1
                amount = min(INVESTMENT_PER_TRADE, p_trader['cash']);
                shares = amount / current_price
                p_trader['cash'] -= amount;
                p_trader['shares'] += shares;
                p_trader['total_invested'] += amount
                p_trader['open_trades'].append(
                    {'shares': shares, 'cost': amount, 'type': "Seasonal" if is_seasonal else "Technical",
                     'signal': signal, 'entry_date': current_date,
                     'target_price': current_price * TECHNICAL_PROFIT_TARGET if not is_seasonal else None,
                     'sell_month': SEASONAL_SELL_MONTHS.get(day['month']) if is_seasonal else None})
                logs['Active Trader'].append(
                    f"{current_date.strftime('%Y-%m-%d')}: BOUGHT ({signal}) for ${amount:,.2f}")
            p_accum = portfolios['Strategic Accumulator'];
            days_since_buy = (current_date - p_accum['last_buy_date']).days;
            quarterly_buy = days_since_buy > 90
            if (is_seasonal or is_gc or is_rsi or quarterly_buy) and p_accum['cash'] > 0:
                amount = min(30000 if quarterly_buy and not (is_seasonal or is_gc or is_rsi) else INVESTMENT_PER_TRADE,
                             p_accum['cash']);
                shares = amount / current_price
                p_accum['cash'] -= amount;
                p_accum['shares'] += shares;
                p_accum['total_invested'] += amount;
                p_accum['last_buy_date'] = current_date
                p_accum['all_entries'].append({'entry_price': current_price})
            portfolio_over_time_log.append({'Date': current_date, 'DCA': portfolios['DCA']['value'],
                                            'Active Trader': portfolios['Active Trader']['value'],
                                            'Strategic Accumulator': portfolios['Strategic Accumulator']['value']})
        final_price = self.df.iloc[-1]['close'];
        underwater_entries = sum(
            1 for e in portfolios['Strategic Accumulator']['all_entries'] if e['entry_price'] > final_price)
        signal_perf_data = []
        for signal, stats in portfolios['Active Trader']['signal_stats'].items():
            entries = stats['entries']
            if entries > 0:
                losses = stats['losses'];
                wins = entries - losses;
                win_rate = (wins / entries) * 100
                signal_perf_data.append(
                    {'Signal Type': signal, 'Total Trades': entries, 'Winning Trades': wins, 'Losing Trades': losses,
                     'Win Rate %': win_rate})
        summary_data = []
        for name, p in portfolios.items():
            roi = (p['value'] / p['total_invested'] - 1) * 100 if p['total_invested'] > 0 else 0
            total_entries = sum(
                stats['entries'] for stats in p.get('signal_stats', {}).values()) if name == 'Active Trader' else len(
                p.get('all_entries', [])) if name == 'Strategic Accumulator' else np.nan
            losing_entries = sum(stats['losses'] for stats in p.get('signal_stats',
                                                                    {}).values()) if name == 'Active Trader' else underwater_entries if name == 'Strategic Accumulator' else np.nan
            summary_data.append({'Strategy': name, 'Final Value': p['value'], 'Total Invested': p['total_invested'],
                                 'Return on Investment %': roi, 'Total Entries': total_entries,
                                 'Losing Entries': losing_entries})
        return {'summary': pd.DataFrame(summary_data),
                'portfolio_over_time': pd.DataFrame(portfolio_over_time_log).set_index('Date'),
                'signal_performance': pd.DataFrame(signal_perf_data), 'trade_logs': logs}


# --- 2. GLOBAL HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def calculate_all_signals(_df):
    if _df.empty: return pd.DataFrame()
    st.info("Calculating technical indicators for all stocks...")
    df = _df.copy();
    gb = df.groupby('ticker');
    df['sma_50'] = gb['close'].transform(lambda x: x.rolling(window=50).mean());
    df['sma_200'] = gb['close'].transform(lambda x: x.rolling(window=200).mean())
    delta = gb['close'].transform('diff');
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean();
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean();
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs));
    df['sma_50_prev'] = gb['sma_50'].shift(1);
    df['sma_200_prev'] = gb['sma_200'].shift(1);
    df['rsi_14_prev'] = gb['rsi_14'].shift(1)
    gc_mask = (df['sma_50'] > df['sma_200']) & (df['sma_50_prev'] < df['sma_200_prev']);
    dc_mask = (df['sma_50'] < df['sma_200']) & (df['sma_50_prev'] > df['sma_200_prev'])
    rsi_buy_mask = (df['rsi_14'] < 30) & (df['rsi_14_prev'] >= 30);
    rsi_sell_mask = (df['rsi_14'] > 70) & (df['rsi_14_prev'] <= 70)
    all_signals_mask = gc_mask | dc_mask | rsi_buy_mask | rsi_sell_mask
    signals_df = df[all_signals_mask].copy()
    conditions = [gc_mask[all_signals_mask], dc_mask[all_signals_mask], rsi_buy_mask[all_signals_mask],
                  rsi_sell_mask[all_signals_mask]]
    choices = ['Golden Cross (Buy)', 'Death Cross (Sell)', 'RSI Oversold (Buy)', 'RSI Overbought (Sell)'];
    signals_df['Signal'] = np.select(conditions, choices, default='Unknown')
    signals_df = signals_df[['date', 'ticker', 'Signal', 'close']].rename(
        columns={'close': 'Price on Signal'}).sort_values(by='date', ascending=False).reset_index(drop=True)
    return signals_df


@st.cache_data(ttl=86400)
def get_per_ticker_signal_stats(_main_df, _dividends_df, ticker_list):
    all_perf_data = [];
    status_text = st.empty()
    for i, ticker in enumerate(ticker_list):
        status_text.info(f"Backtesting signals for {ticker} ({i + 1}/{len(ticker_list)})...")
        engine = SimulationEngine(_main_df, _dividends_df, ticker)
        results = engine.run_dca_vs_active(10000, 120000, 15, 60)  # Use default params for this global calc
        if 'signal_performance' in results and not results['signal_performance'].empty:
            perf_df = results['signal_performance'].copy();
            perf_df['Ticker'] = ticker
            all_perf_data.append(perf_df)
    status_text.empty()
    if not all_perf_data: return pd.DataFrame()
    return pd.concat(all_perf_data, ignore_index=True)


@st.cache_data(ttl=86400)
def run_full_market_simulation(_main_df, _dividends_df, ticker_list, monthly_dca, annual_active):
    all_results = [];
    progress_bar = st.progress(0, text="Initializing simulations...");
    status_text = st.empty()
    for i, ticker in enumerate(ticker_list):
        status_text.info(f"Simulating {ticker} ({i + 1}/{len(ticker_list)})...")
        engine = SimulationEngine(_main_df, _dividends_df, ticker)
        results = engine.run_dca_vs_active(monthly_dca, annual_active, 15, 60)  # Use default params for leaderboard
        if 'summary' in results and not results['summary'].empty:
            summary = results['summary'];
            stock_summary = {'Ticker': ticker}
            for _, row in summary.iterrows():
                strategy = row['Strategy']
                stock_summary[f'{strategy} Final Value'] = row['Final Value']
                stock_summary[f'{strategy} Total Invested'] = row['Total Invested']
            all_results.append(stock_summary)
        progress_bar.progress((i + 1) / len(ticker_list))
    status_text.success("All market simulations complete!");
    if not all_results: return pd.DataFrame()
    return pd.DataFrame(all_results)


# --- 3. PAGE DISPLAY FUNCTIONS ---
def show_leaderboard_page(main_df, ticker_list, dividends_df):
    st.header("🏆 Strategy Leaderboard");
    st.info("This page analyzes signal reliability and simulates strategies across all stocks.")
    st.subheader("Market-Wide Signal Performance")
    with st.spinner("Calculating global signal win rates..."):
        per_ticker_perf = get_per_ticker_signal_stats(main_df, dividends_df, ticker_list)
        if not per_ticker_perf.empty:
            global_stats = per_ticker_perf.groupby('Signal Type').agg(Total_Trades=('Total Trades', 'sum'),
                                                                      Total_Losses=('Losing Trades',
                                                                                    'sum')).reset_index()
            global_stats['Win Rate'] = ((global_stats['Total_Trades'] - global_stats['Total_Losses']) / global_stats[
                'Total_Trades']) * 100
            cols = st.columns(len(global_stats))
            for i, row in global_stats.iterrows():
                with cols[i]:
                    st.metric(label=f"{row['Signal Type']} Win Rate", value=f"{row['Win Rate']:.1f}%",
                              help=f"Based on {row['Total_Trades']} total trades across all stocks.")
        else:
            st.warning("Could not calculate global signal performance.")
    st.subheader("Stock vs. Strategy Performance")
    st.sidebar.header("Leaderboard Parameters");
    monthly_investment = st.sidebar.number_input("Monthly DCA Investment ($)", 100, 50000, 10000, 500)
    initial_capital = st.sidebar.number_input("Active Strategy Annual Capital ($)", 1000, 1000000, 120000, 1000)
    if st.sidebar.button("🚀 Run Full Market Simulation"):
        results_df = run_full_market_simulation(main_df, dividends_df, ticker_list, monthly_investment, initial_capital)
        if not results_df.empty:
            results_df['DCA ROI %'] = (results_df['DCA Final Value'] / results_df['DCA Total Invested'] - 1) * 100
            results_df['Active Trader ROI %'] = (results_df['Active Trader Final Value'] / results_df[
                'Active Trader Total Invested'] - 1) * 100
            results_df['Strategic Accumulator ROI %'] = (results_df['Strategic Accumulator Final Value'] / results_df[
                'Strategic Accumulator Total Invested'] - 1) * 100
            sorted_results_df = results_df.sort_values(by='Strategic Accumulator ROI %', ascending=False)
            total_row = {'Ticker': '**GRAND TOTAL**', 'DCA Final Value': results_df['DCA Final Value'].sum(),
                         'DCA Total Invested': results_df['DCA Total Invested'].sum(),
                         'Active Trader Final Value': results_df['Active Trader Final Value'].sum(),
                         'Active Trader Total Invested': results_df['Active Trader Total Invested'].sum(),
                         'Strategic Accumulator Final Value': results_df['Strategic Accumulator Final Value'].sum(),
                         'Strategic Accumulator Total Invested': results_df[
                             'Strategic Accumulator Total Invested'].sum()}
            total_row['DCA ROI %'] = (total_row['DCA Final Value'] / total_row['DCA Total Invested'] - 1) * 100 if \
            total_row['DCA Total Invested'] > 0 else 0
            total_row['Active Trader ROI %'] = (total_row['Active Trader Final Value'] / total_row[
                'Active Trader Total Invested'] - 1) * 100 if total_row['Active Trader Total Invested'] > 0 else 0
            total_row['Strategic Accumulator ROI %'] = (total_row['Strategic Accumulator Final Value'] / total_row[
                'Strategic Accumulator Total Invested'] - 1) * 100 if total_row[
                                                                          'Strategic Accumulator Total Invested'] > 0 else 0
            total_df = pd.DataFrame([total_row]);
            display_df = pd.concat([sorted_results_df, total_df], ignore_index=True)
            display_cols = ['Ticker', 'DCA Final Value', 'DCA Total Invested', 'DCA ROI %', 'Active Trader Final Value',
                            'Active Trader Total Invested', 'Active Trader ROI %', 'Strategic Accumulator Final Value',
                            'Strategic Accumulator Total Invested', 'Strategic Accumulator ROI %']
            height = (len(display_df) + 1) * 35 + 3
            styler = display_df[display_cols].style.format(
                {'DCA Final Value': '${:,.2f}', 'DCA Total Invested': '${:,.2f}', 'DCA ROI %': '{:.2f}%',
                 'Active Trader Final Value': '${:,.2f}', 'Active Trader Total Invested': '${:,.2f}',
                 'Active Trader ROI %': '{:.2f}%', 'Strategic Accumulator Final Value': '${:,.2f}',
                 'Strategic Accumulator Total Invested': '${:,.2f}',
                 'Strategic Accumulator ROI %': '{:.2f}%'}).background_gradient(cmap='RdYlGn', subset=['DCA ROI %',
                                                                                                       'Active Trader ROI %',
                                                                                                       'Strategic Accumulator ROI %'])
            st.dataframe(styler, use_container_width=True, height=height, hide_index=True)
    else:
        st.info("Click the 'Run Full Market Simulation' button in the sidebar to begin.")


def show_simulation_page(main_df, ticker_list, dividends_df):
    if not ticker_list: st.warning("No stocks found in the database. Please run the scraping scripts first."); return
    st.sidebar.header("Simulation Controls");
    selected_ticker = st.sidebar.selectbox("Choose a Stock to Analyze:", options=ticker_list)
    simulation_type = st.sidebar.radio("Select Simulation Type:", options=["DCA vs. Active Strategies"])
    st.sidebar.header("Parameters")
    if simulation_type == "DCA vs. Active Strategies":
        monthly_investment = st.sidebar.number_input("Monthly DCA Investment ($)", 100, 50000, 10000, 500)
        annual_capital = st.sidebar.number_input("Active Strategy Annual Capital ($)", 1000, 1000000, 120000, 1000)
        # --- NEW INTERACTIVE PARAMETERS ---
        st.sidebar.subheader("Active Trader Rule Adjustments")
        profit_target = st.sidebar.slider("Technical Profit Target (%)", 5, 50, 15, 1)
        rsi_hold_days = st.sidebar.slider("RSI Max Hold Days", 10, 360, 60, 5)

    if st.sidebar.button("🚀 Run Simulation"):
        st.header(f"Results for {selected_ticker}");
        engine = SimulationEngine(main_df, dividends_df, selected_ticker)
        if simulation_type == "DCA vs. Active Strategies":
            with st.spinner("Running complex backtest with your custom rules..."):
                results = engine.run_dca_vs_active(monthly_investment, annual_capital, profit_target, rsi_hold_days)
                st.subheader("Final Portfolio Summary");
                summary_df = results['summary'].copy()
                summary_df['Losing Entries'] = summary_df.apply(
                    lambda row: f"{int(row['Losing Entries'])} (Underwater)" if row[
                                                                                    'Strategy'] == 'Strategic Accumulator' else int(
                        row['Losing Entries']) if not pd.isna(row['Losing Entries']) else '-', axis=1)
                st.dataframe(summary_df.style.format(
                    {'Final Value': '${:,.2f}', 'Total Invested': '${:,.2f}', 'Return on Investment %': '{:.2f}%',
                     'Total Entries': '{:.0f}'}).hide(axis="index"))
                if not results['signal_performance'].empty:
                    st.subheader("Active Trader Signal Performance")
                    st.dataframe(
                        results['signal_performance'].style.format({'Win Rate %': '{:.1f}%'}).hide(axis="index"))
                st.subheader("Portfolio Growth Over Time");
                st.line_chart(results['portfolio_over_time'])
    else:
        st.info("Adjust parameters and click 'Run Simulation' to see results.")


def show_signal_effectiveness_page(main_df, ticker_list, dividends_df):
    st.header("📊 Signal Effectiveness Matrix");
    st.info("This page analyzes the historical win rate of each signal for every stock.")
    if st.button("🚀 Analyze Signal Effectiveness Across Market"):
        full_perf_df = get_per_ticker_signal_stats(main_df, dividends_df, ticker_list)
        if not full_perf_df.empty:
            win_rate_matrix = full_perf_df.pivot_table(index='Ticker', columns='Signal Type', values='Win Rate %')
            winning_trades_matrix = full_perf_df.pivot_table(index='Ticker', columns='Signal Type',
                                                             values='Winning Trades').fillna(0).astype(int)
            total_trades_matrix = full_perf_df.pivot_table(index='Ticker', columns='Signal Type',
                                                           values='Total Trades').fillna(0).astype(int)
            st.subheader("Signal Win Rate (%) by Stock");
            st.markdown("Each cell shows: `Win Rate %`. Sort by clicking any column header.")

            styler = win_rate_matrix.style.background_gradient(cmap='RdYlGn', vmin=30, vmax=80)

            formatted_text = win_rate_matrix.copy()
            for col in formatted_text.columns:
                for idx in formatted_text.index:
                    rate = win_rate_matrix.loc[idx, col]
                    wins = winning_trades_matrix.loc[idx, col]
                    total = total_trades_matrix.loc[idx, col]
                    if pd.notna(rate) and total > 0:
                        formatted_text.loc[idx, col] = f"{rate:.1f}% ({wins}/{total})"
                    else:
                        formatted_text.loc[idx, col] = "N/A"
            height = (len(win_rate_matrix) + 1) * 35 + 3
            st.dataframe(styler.format(na_rep="N/A"), use_container_width=True, height=height)
            with st.expander("View Full Text Version"):
                st.dataframe(formatted_text, use_container_width=True)
        else:
            st.warning("No signal performance data could be generated.")
    else:
        st.info("Click the button above to run the analysis.")


def show_signals_page(price_df):
    st.header("📈 Live Signal Screener");
    st.info("This page scans the entire database for the latest technical entry and exit signals.")
    all_signals = calculate_all_signals(price_df)
    if all_signals.empty: st.warning("No signals found in the entire historical dataset."); return
    today = pd.to_datetime(datetime.now().date());
    start_of_week = today - pd.to_timedelta(today.dayofweek, unit='d')
    current_week_signals = all_signals[all_signals['date'] >= start_of_week];
    historical_signals = all_signals[all_signals['date'] < start_of_week]
    st.subheader("Signals This Week")
    if current_week_signals.empty:
        st.success("No new signals have appeared in the current week.")
    else:
        st.dataframe(current_week_signals, use_container_width=True, hide_index=True,
                     column_config={"date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                                    "Price on Signal": st.column_config.NumberColumn(format="$%.2f")})
    st.subheader("Signal History")
    with st.expander("View all historical signals"):
        if historical_signals.empty:
            st.text("No older signals found.")
        else:
            st.dataframe(historical_signals, use_container_width=True, hide_index=True,
                         column_config={"date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                                        "Price on Signal": st.column_config.NumberColumn(format="$%.2f")})


def show_db_viewer():
    st.header("🔍 Database Inspector");
    st.info("Use this tool to verify the data scraped into your `market_data.db` file.")
    try:
        conn = sqlite3.connect('market_data.db', check_same_thread=False)
        table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
        selected_table = st.selectbox("Select a table to view:", table_names)
        if selected_table:
            df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
            stocks_df = pd.read_sql_query("SELECT id, ticker FROM stocks", conn);
            ticker_list = sorted(stocks_df['ticker'].unique().tolist())
            if 'stock_id' in df.columns:
                selected_tickers = st.multiselect("Filter by Ticker(s):", options=ticker_list)
                if selected_tickers:
                    ids_to_filter = stocks_df[stocks_df['ticker'].isin(selected_tickers)]['id'].tolist()
                    df = df[df['stock_id'].isin(ids_to_filter)]
            st.dataframe(df);
            st.success(f"Displaying **{len(df)}** rows from the **'{selected_table}'** table.")
    except Exception as e:
        st.error(f"An error occurred while reading the database: {e}")
    finally:
        if 'conn' in locals() and conn: conn.close()


# --- 3. Main Application Logic ---
st.title("📈 Stock Analysis & Strategy Dashboard")
st.sidebar.header("Main Menu")
app_mode = st.sidebar.radio("Choose a view:",
                            ["Strategy Simulations", "Signal Effectiveness", "Strategy Leaderboard", "Live Signals",
                             "Database Viewer"])

main_df, ticker_list, dividends_df = load_data_from_db()

if app_mode == "Strategy Simulations":
    show_simulation_page(main_df, ticker_list, dividends_df)
elif app_mode == "Signal Effectiveness":
    show_signal_effectiveness_page(main_df, ticker_list, dividends_df)
elif app_mode == "Strategy Leaderboard":
    show_leaderboard_page(main_df, ticker_list, dividends_df)
elif app_mode == "Live Signals":
    show_signals_page(main_df)
elif app_mode == "Database Viewer":
    show_db_viewer()