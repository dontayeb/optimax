import pandas as pd
import numpy as np


class SimulationEngine:
    """Handles all backtesting and simulation logic for trading strategies."""

    def __init__(self, full_price_df, full_dividends_df, ticker):
        self.ticker = ticker
        self.df = full_price_df[full_price_df['ticker'] == ticker].copy().reset_index(drop=True)

        if self.df.empty:
            return

        self.dividends = full_dividends_df[full_dividends_df['ticker'] == ticker].copy()

        if not self.dividends.empty:
            self.dividends['ex_date'] = pd.to_datetime(
                self.dividends['ex_date'],
                errors='coerce'
            )

        self._engineer_features()

    def _engineer_features(self):
        """Calculate technical indicators for the stock."""
        if self.df.empty:
            return

        # Moving averages
        self.df['sma_50'] = self.df['close'].rolling(window=50).mean()
        self.df['sma_200'] = self.df['close'].rolling(window=200).mean()

        # RSI calculation
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi_14'] = 100 - (100 / (1 + rs))

        # Date components
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month

        # Signal detection
        self.df['golden_cross_signal'] = (
                (self.df['sma_50'] > self.df['sma_200']) &
                (self.df['sma_50'].shift(1) < self.df['sma_200'].shift(1))
        )
        self.df['death_cross_signal'] = (
                (self.df['sma_50'] < self.df['sma_200']) &
                (self.df['sma_50'].shift(1) > self.df['sma_200'].shift(1))
        )
        self.df['rsi_oversold_signal'] = (
                (self.df['rsi_14'] < 30) &
                (self.df['rsi_14'].shift(1) >= 30)
        )

    def run_dca_vs_active(self, monthly_dca, annual_active, profit_target_pct, rsi_max_hold_days):
        """
        Run simulation comparing DCA, Active Trader, and Strategic Accumulator strategies.

        Args:
            monthly_dca: Monthly investment amount for DCA strategy
            annual_active: Annual capital injection for active strategies
            profit_target_pct: Profit target percentage for technical signals
            rsi_max_hold_days: Maximum holding period for RSI trades

        Returns:
            Dictionary with summary, portfolio_over_time, signal_performance, trade_logs, and risk_metrics
        """
        if self.df.empty:
            return {}

        # Constants
        SEASONAL_BUY_MONTHS = [2, 10]
        SEASONAL_SELL_MONTHS = {2: 8, 10: 4}
        INVESTMENT_PER_TRADE = 10000
        TECHNICAL_PROFIT_TARGET = 1 + (profit_target_pct / 100)
        RISK_FREE_RATE = 0.04  # 4% annual risk-free rate for Sharpe calculation

        # Initialize portfolios
        portfolios = {
            'DCA': {
                'cash': 0,
                'shares': 0,
                'value': 0,
                'total_invested': 0
            },
            'Active Trader': {
                'cash': annual_active,
                'shares': 0,
                'value': annual_active,
                'total_invested': 0,
                'open_trades': [],
                'signal_stats': {
                    s: {'entries': 0, 'losses': 0}
                    for s in ["Golden Cross", "RSI Oversold", "Seasonal"]
                }
            },
            'Strategic Accumulator': {
                'cash': annual_active,
                'shares': 0,
                'value': annual_active,
                'total_invested': 0,
                'last_buy_date': self.df.iloc[0]['date'],
                'all_entries': []
            }
        }

        logs = {'Active Trader': [], 'Strategic Accumulator': []}
        portfolio_over_time_log = []

        # Main simulation loop
        for i, day in self.df.iterrows():
            current_date = day['date']
            current_price = day['close']

            # Update portfolio values
            for name, p in portfolios.items():
                p['value'] = p['cash'] + (p['shares'] * current_price)

            # Add annual capital at year start
            if i > 0 and day['year'] != self.df.iloc[i - 1]['year']:
                portfolios['Active Trader']['cash'] += annual_active
                portfolios['Strategic Accumulator']['cash'] += annual_active

            # DCA monthly investment
            if i == 0 or day['month'] != self.df.iloc[i - 1]['month']:
                portfolios['DCA']['total_invested'] += monthly_dca
                portfolios['DCA']['shares'] += monthly_dca / current_price

            # Active Trader: Check exit conditions
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
                    if proceeds < trade['cost']:
                        p_trader['signal_stats'][trade['signal']]['losses'] += 1

                    p_trader['cash'] += proceeds
                    p_trader['shares'] -= trade['shares']
                    p_trader['open_trades'].remove(trade)

                    logs['Active Trader'].append(
                        f"{current_date.strftime('%Y-%m-%d')}: SOLD ({exit_reason}) "
                        f"for ${proceeds:,.2f}. P/L: ${proceeds - trade['cost']:.2f}"
                    )

            # Check for buy signals
            is_seasonal = (
                    (i == 0 or day['month'] != self.df.iloc[i - 1]['month']) and
                    day['month'] in SEASONAL_BUY_MONTHS
            )
            is_gc = day['golden_cross_signal']
            is_rsi = day['rsi_oversold_signal']

            # Active Trader: Execute buys
            if (is_seasonal or is_gc or is_rsi) and p_trader['cash'] > 0:
                signal = "Seasonal" if is_seasonal else "Golden Cross" if is_gc else "RSI Oversold"
                p_trader['signal_stats'][signal]['entries'] += 1

                amount = min(INVESTMENT_PER_TRADE, p_trader['cash'])
                shares = amount / current_price

                p_trader['cash'] -= amount
                p_trader['shares'] += shares
                p_trader['total_invested'] += amount

                p_trader['open_trades'].append({
                    'shares': shares,
                    'cost': amount,
                    'type': "Seasonal" if is_seasonal else "Technical",
                    'signal': signal,
                    'entry_date': current_date,
                    'target_price': current_price * TECHNICAL_PROFIT_TARGET if not is_seasonal else None,
                    'sell_month': SEASONAL_SELL_MONTHS.get(day['month']) if is_seasonal else None
                })

                logs['Active Trader'].append(
                    f"{current_date.strftime('%Y-%m-%d')}: BOUGHT ({signal}) for ${amount:,.2f}"
                )

            # Strategic Accumulator: Execute buys
            p_accum = portfolios['Strategic Accumulator']
            days_since_buy = (current_date - p_accum['last_buy_date']).days
            quarterly_buy = days_since_buy > 90

            if (is_seasonal or is_gc or is_rsi or quarterly_buy) and p_accum['cash'] > 0:
                amount = min(
                    30000 if quarterly_buy and not (is_seasonal or is_gc or is_rsi) else INVESTMENT_PER_TRADE,
                    p_accum['cash']
                )
                shares = amount / current_price

                p_accum['cash'] -= amount
                p_accum['shares'] += shares
                p_accum['total_invested'] += amount
                p_accum['last_buy_date'] = current_date
                p_accum['all_entries'].append({'entry_price': current_price})

            # Log portfolio values over time
            portfolio_over_time_log.append({
                'Date': current_date,
                'DCA': portfolios['DCA']['value'],
                'Active Trader': portfolios['Active Trader']['value'],
                'Strategic Accumulator': portfolios['Strategic Accumulator']['value']
            })

        # Calculate final metrics
        final_price = self.df.iloc[-1]['close']
        underwater_entries = sum(
            1 for e in portfolios['Strategic Accumulator']['all_entries']
            if e['entry_price'] > final_price
        )

        # Calculate risk metrics for each strategy
        risk_metrics = self._calculate_risk_metrics(
            portfolio_over_time_log,
            portfolios,
            RISK_FREE_RATE
        )

        # Build signal performance data
        signal_perf_data = []
        for signal, stats in portfolios['Active Trader']['signal_stats'].items():
            entries = stats['entries']
            if entries > 0:
                losses = stats['losses']
                wins = entries - losses
                win_rate = (wins / entries) * 100
                signal_perf_data.append({
                    'Signal Type': signal,
                    'Total Trades': entries,
                    'Winning Trades': wins,
                    'Losing Trades': losses,
                    'Win Rate %': win_rate
                })

        # Build summary data
        summary_data = []
        for name, p in portfolios.items():
            roi = (p['value'] / p['total_invested'] - 1) * 100 if p['total_invested'] > 0 else 0

            total_entries = (
                sum(stats['entries'] for stats in p.get('signal_stats', {}).values())
                if name == 'Active Trader'
                else len(p.get('all_entries', []))
                if name == 'Strategic Accumulator'
                else np.nan
            )

            losing_entries = (
                sum(stats['losses'] for stats in p.get('signal_stats', {}).values())
                if name == 'Active Trader'
                else underwater_entries
                if name == 'Strategic Accumulator'
                else np.nan
            )

            summary_data.append({
                'Strategy': name,
                'Final Value': p['value'],
                'Total Invested': p['total_invested'],
                'Return on Investment %': roi,
                'Total Entries': total_entries,
                'Losing Entries': losing_entries
            })

        return {
            'summary': pd.DataFrame(summary_data),
            'portfolio_over_time': pd.DataFrame(portfolio_over_time_log).set_index('Date'),
            'signal_performance': pd.DataFrame(signal_perf_data),
            'trade_logs': logs,
            'risk_metrics': risk_metrics
        }

    def _calculate_risk_metrics(self, portfolio_log, portfolios, risk_free_rate):
        """
        Calculate professional risk metrics for each strategy.

        Returns a dictionary with max drawdown, volatility, and Sharpe ratio for each strategy.
        """
        risk_metrics = {}

        # Convert to DataFrame for easier calculations
        df = pd.DataFrame(portfolio_log).set_index('Date')

        for strategy_name in ['DCA', 'Active Trader', 'Strategic Accumulator']:
            if strategy_name not in df.columns:
                continue

            values = df[strategy_name]

            # Calculate daily returns
            returns = values.pct_change().dropna()

            if len(returns) == 0:
                risk_metrics[strategy_name] = {
                    'Max Drawdown %': 0,
                    'Annualized Volatility %': 0,
                    'Sharpe Ratio': 0
                }
                continue

            # Max Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100  # Convert to percentage

            # Annualized Volatility
            # Assume daily data, annualize by multiplying by sqrt(252 trading days)
            daily_vol = returns.std()
            annualized_vol = daily_vol * np.sqrt(252) * 100  # Convert to percentage

            # Sharpe Ratio
            # Calculate total return
            total_invested = portfolios[strategy_name]['total_invested']
            final_value = portfolios[strategy_name]['value']

            if total_invested > 0:
                total_return = (final_value / total_invested) - 1
                # Annualize the return
                years = len(values) / 252  # Approximate trading days in a year
                if years > 0:
                    annualized_return = (1 + total_return) ** (1 / years) - 1
                else:
                    annualized_return = 0

                # Sharpe = (Return - Risk Free Rate) / Volatility
                if annualized_vol > 0:
                    sharpe_ratio = (annualized_return - risk_free_rate) / (annualized_vol / 100)
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0

            risk_metrics[strategy_name] = {
                'Max Drawdown %': max_drawdown,
                'Annualized Volatility %': annualized_vol,
                'Sharpe Ratio': sharpe_ratio
            }

        return risk_metrics