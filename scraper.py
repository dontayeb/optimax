# File: scraper.py
# Your dedicated, daily data-gathering tool (Complete Fixed Version)

import sqlite3
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright

# --- 1. CONFIGURATION ---
TICKER_MAP = {
'138SL': '311', 'ASBH': '1600148', 'BIL': '21', 'BRG': '23', 'CCC': '37', 'CPJ': '256', 'CAR': '39', 'CPFV': '1600091', 'EPLY': '279', 'FIRSTROCKJMD': '1600097', 'GENAC': '258', 'GK': '55', 'GHL': '57', 'ENERGY': '41', 'JBG': '65', 'JP': '75', 'JSE': '276', 'JMMBGL': '319', 'KEY': '336', 'KPREIT': '89', 'KW': '91', 'LASD': '99', 'LASM': '103', 'MTL': '1600081', 'MASSY': '1600126', 'MGL': '1600150', 'MJE': '1600072', 'MPCCEL': '1600078', 'NCBFG': '1600012', 'PAL': '113', 'PJAM': '121', 'PJX': '337', 'PROVEN': '321', 'PULS': '125', 'QWI': '1600092', 'RJR': '127', 'SJ': '297', 'XFUND': '290', 'SELECTF': '1600090', 'SELECTMD': '1600096', 'SALF': '131', 'SGJ': '133', 'SEP': '137', 'SML': '1600074', 'SIL': '309', 'SVL': '139', 'SCIJMD': '1600069', 'SRFJMD': '1600118', 'TJH': '1600100', 'TROPICAL': '1600104', 'VMIL': '1600034', 'WIG': '1600087', 'WISYNCO': '1600033', 'AFS': '13', 'AMG': '254', 'AHPC': '1600164', 'BPOW': '25', 'CAC': '323', 'CHL': '35', 'CABROKERS': '1600099', 'KREMI': '274', 'CFF': '285', 'PURITY': '270', 'DTL': '295', 'DOLLA': '1600130', 'DCOVE': '45', 'LEARN': '1600128', 'ELITE': '1600035', 'EFRESH': '1600071', 'ECL': '1600016', 'FTNA': '1600077', 'FOSRICH': '1600029', 'FESCO': '1600110', 'GWEST': '1600030', 'HONBUN': '61', 'IPCL': '1600138', 'INDIES': '1600073', 'ROC': '333', 'ISP': '335', 'JAMT': '87', 'JETCON': '334', 'JFP': '1600127', 'KLE': '268', 'KNTYR': '1600082', 'KEX': '301', 'LASF': '101', 'LUMBER': '1600095', 'MAILPAC': '1600094', 'MEEG': '1600010', 'MDS': '299', 'MFS': '1600075', 'OMNI': '1600156', '1GS': '1600147', 'ONE': '1600131', 'PTL': '272', 'RAWILL': '1600160', 'RPL': '1600136', 'SPURTREE': '1600125', 'SOS': '1600017', 'LAB': '1600089'
    #'ASBH': '1600148', 'BAR': '21', 'CHL': '35', 'CCC': '37', 'CPJ': '256',
    #'CAR': '39', 'FTNA': '1600077', 'FOSRICH': '1600029', 'FESCO': '1600110',
    #'GENAC': '258', 'GK': '55', 'GHL': '57', 'JBG': '65', 'JP': '75',
    #'JMMBGL': '319', 'KW': '91', 'MASSY': '1600126', 'NCBFG': '1600012',
    #'PJAM': '121', 'PULS': '125', 'SJ': '297', 'SGJ': '133', 'SEP': '137',
    #'SVL': '139', 'TJH': '1600100', 'WISYNCO': '1600033'
}
DB_FILE = 'market_data.db'


# --- 2. DATABASE SETUP ---
def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS stocks (id INTEGER PRIMARY KEY, ticker TEXT UNIQUE NOT NULL, instrument_code TEXT UNIQUE NOT NULL)''')
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS daily_data (stock_id INTEGER, date TEXT NOT NULL, close REAL, volume INTEGER, FOREIGN KEY (stock_id) REFERENCES stocks (id), UNIQUE(stock_id, date))''')
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS dividends (stock_id INTEGER, record_date TEXT, ex_date TEXT, payment_date TEXT, amount REAL, currency TEXT, FOREIGN KEY (stock_id) REFERENCES stocks (id), UNIQUE(stock_id, payment_date))''')
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS earnings (stock_id INTEGER, financial_year INTEGER, q1_earnings REAL, q2_earnings REAL, q3_earnings REAL, q4_earnings REAL, FOREIGN KEY (stock_id) REFERENCES stocks (id), UNIQUE(stock_id, financial_year))''')
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS news (stock_id INTEGER, news_date TEXT, headline TEXT, FOREIGN KEY (stock_id) REFERENCES stocks (id), UNIQUE(stock_id, headline))''')
    conn.commit()
    conn.close()


# --- 3. PARSING LOGIC ---
def parse_historical_table(soup):
    historical_data = []
    try:
        main_content = soup.find('main')
        table_container = main_content.find('div', class_='tw-border-b tw-border-gray-200')
        data_table = table_container.find('table')
        if not data_table:
            return []
        for row in data_table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < 10:
                continue
            try:
                date_str = cells[1].text.strip()
                volume_str = cells[6].text.strip()
                close_str = cells[9].text.strip()
                trade_date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                close_price = float(close_str.replace(',', ''))
                volume = int(volume_str.replace(',', ''))
                historical_data.append({'date': trade_date, 'close': close_price, 'volume': volume})
            except (ValueError, IndexError):
                continue
        return historical_data
    except AttributeError:
        return []


def parse_dividends_table(soup):
    dividends_data = []
    try:
        dividends_header = soup.find(lambda tag: tag.name in ['h3', 'h4'] and 'dividends' in tag.text.lower())
        if not dividends_header:
            return []
        dividends_table = dividends_header.find_next('table')
        if not dividends_table:
            return []
        for row in dividends_table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < 5:
                continue
            record_date = cells[0].text.strip()
            ex_date = cells[2].text.strip()
            payment_date = cells[3].text.strip()
            amount_str = cells[4].text.strip()
            if amount_str and len(amount_str.split()) > 1:
                parts = amount_str.split()
                currency = parts[0]
                amount = float(parts[1])
                dividends_data.append({
                    'record_date': record_date,
                    'ex_date': ex_date,
                    'payment_date': payment_date,
                    'amount': amount,
                    'currency': currency
                })
    except Exception as e:
        print(f"  -> Error parsing dividends table: {e}")
    return dividends_data


def parse_earnings_table(soup):
    earnings_data = []
    try:
        earnings_header = soup.find(lambda tag: tag.name in ['h3', 'h4'] and 'earnings' in tag.text.lower())
        if not earnings_header:
            return []
        earnings_table = earnings_header.find_next('table')
        if not earnings_table:
            return []
        for row in earnings_table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) == 5:  # This targets the quarterly table
                year = int(cells[0].text.strip())
                q1 = float(cells[1].text.strip().replace(',', ''))
                q2 = float(cells[2].text.strip().replace(',', ''))
                q3 = float(cells[3].text.strip().replace(',', ''))
                q4 = float(cells[4].text.strip().replace(',', ''))
                earnings_data.append({'year': year, 'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4})
    except Exception as e:
        print(f"  -> Error parsing earnings table: {e}")
    return earnings_data


def parse_news_page(soup):
    news_data = []
    try:
        news_container = soup.find('div', class_='elementor-posts-container')
        if not news_container:
            return []
        for article in news_container.find_all('article'):
            headline_tag = article.find('h3', class_='elementor-post__title')
            date_tag = article.find('span', class_='elementor-post-date')
            if headline_tag and date_tag:
                headline = headline_tag.text.strip()
                news_date = date_tag.text.strip()
                try:
                    clean_date = datetime.strptime(news_date, '%B %d, %Y').strftime('%Y-%m-%d')
                except ValueError:
                    clean_date = news_date
                news_data.append({'date': clean_date, 'headline': headline})
    except Exception as e:
        print(f"  -> Error parsing a news page: {e}")
    return news_data


# --- 4. SCRAPER FUNCTIONS ---

def fetch_historical_data_with_playwright(page, conn):
    print("\n--- Starting Intelligent Price Data Update ---")
    cursor = conn.cursor()

    for ticker, code in TICKER_MAP.items():
        print(f"\nProcessing Prices for {ticker} (Code: {code})...")

        # First check if this instrument_code already exists
        cursor.execute("SELECT id, ticker FROM stocks WHERE instrument_code = ?", (code,))
        existing = cursor.fetchone()

        if existing:
            stock_id = existing[0]
            old_ticker = existing[1]

            if old_ticker != ticker:
                # Update the ticker name (handles ticker changes like BAR -> BIL)
                print(f"  -> Updating ticker from {old_ticker} to {ticker}")
                cursor.execute("UPDATE stocks SET ticker = ? WHERE id = ?", (ticker, stock_id))
                conn.commit()
        else:
            # New stock, insert it
            cursor.execute("INSERT INTO stocks (ticker, instrument_code) VALUES (?, ?)", (ticker, code))
            conn.commit()
            stock_id = cursor.lastrowid

    #for ticker, code in TICKER_MAP.items():
        #print(f"\nProcessing Prices for {ticker} (Code: {code})...")
        #cursor.execute("INSERT OR IGNORE INTO stocks (ticker, instrument_code) VALUES (?, ?)", (ticker, code))
        #conn.commit()
        #cursor.execute("SELECT id FROM stocks WHERE ticker = ?", (ticker,))
        #stock_id = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(date) FROM daily_data WHERE stock_id = ?", (stock_id,))
        last_date_str = cursor.fetchone()[0]

        if last_date_str:
            start_date = datetime.strptime(last_date_str, '%Y-%m-%d').date() - timedelta(days=7)
        else:
            start_date = datetime.now().date() - timedelta(days=(5 * 365 - 1))

        end_date = datetime.now().date()

        if last_date_str and datetime.strptime(last_date_str, '%Y-%m-%d').date() >= end_date:
            print("  -> Price data is already up to date. Skipping.")
            continue

        url = f"https://www.jamstockex.com/trading/price-history/?instrumentCode={code}&fromDate={start_date.strftime('%Y-%m-%d')}&thruDate={end_date.strftime('%Y-%m-%d')}"
        print(f"  Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

        try:
            # Navigate and wait for content to render
            page.goto(url, timeout=60000, wait_until='networkidle')
            print("  Waiting for content to render...")
            time.sleep(5)

            html_content = page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            scraped_data = parse_historical_table(soup)

            if scraped_data:
                data_to_insert = [(stock_id, row['date'], row['close'], row['volume']) for row in scraped_data]
                cursor.executemany(
                    "INSERT OR IGNORE INTO daily_data (stock_id, date, close, volume) VALUES (?, ?, ?, ?)",
                    data_to_insert)
                conn.commit()
                print(f"  -> Processed {len(data_to_insert)} row(s). DB is now up to date.")
            else:
                print("  -> No new price data found on page for this date range.")
        except Exception as e:
            print(f"  -> An error occurred while processing prices for {ticker}: {e}")

        print("  Respecting crawl-delay of 30 seconds...")
        time.sleep(30)

    print("\n--- Price Data Update Complete ---")


def fetch_all_news_for_stock(page, ticker, deep_scrape=False):
    all_news = []
    try:
        news_url = f"https://www.jamstockex.com/?tag={ticker}"
        print(f"  -> Navigating to news page: {news_url}")
        page.goto(news_url, timeout=60000, wait_until='networkidle')
        time.sleep(3)  # Wait for content to render
        page_num = 1

        while True:
            html_content = page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            page_news = parse_news_page(soup)

            if not page_news and not all_news:
                break

            all_news.extend(page_news)

            if not deep_scrape:
                print("  -> Quick scrape complete (first page only).")
                break

            next_button = page.locator('a.next.page-numbers')
            if next_button.is_visible():
                print(f"  -> Page {page_num}: Found {len(page_news)} news items. Clicking to next page...")
                next_button.click()
                page.wait_for_load_state('networkidle', timeout=30000)
                time.sleep(2)
                page_num += 1
            else:
                print(f"  -> Page {page_num}: Found {len(page_news)} news items. End of news.")
                break
    except Exception as e:
        print(f"  -> An error occurred while fetching news for {ticker}: {e}")

    return all_news


def fetch_instrument_details(page, conn):
    print("\n--- Starting Instrument Details Scrape ---")
    cursor = conn.cursor()

    for ticker, code in TICKER_MAP.items():
        print(f"\nProcessing Details for {ticker} (Code: {code})...")
        cursor.execute("SELECT id FROM stocks WHERE ticker = ?", (ticker,))
        stock_id_result = cursor.fetchone()

        if not stock_id_result:
            continue

        stock_id = stock_id_result[0]
        details_url = f"https://www.jamstockex.com/trading/instruments/?instrument={code}"
        print(f"  Navigating to details page: {details_url}")

        try:
            # Navigate and wait for content to render
            page.goto(details_url, timeout=60000, wait_until='networkidle')
            print("  Waiting for content to render...")
            time.sleep(5)

            html_content = page.content()
            soup = BeautifulSoup(html_content, 'html.parser')

            # Parse dividends
            dividends = parse_dividends_table(soup)
            if dividends:
                cursor.executemany(
                    "INSERT OR IGNORE INTO dividends (stock_id, record_date, ex_date, payment_date, amount, currency) VALUES (?, ?, ?, ?, ?, ?)",
                    [(stock_id, d['record_date'], d['ex_date'], d['payment_date'], d['amount'], d['currency']) for d in
                     dividends]
                )
                print(f"  -> Stored {len(dividends)} dividend records.")
            else:
                print("  -> No dividend data was found or parsed.")

            # Parse earnings
            earnings = parse_earnings_table(soup)
            if earnings:
                cursor.executemany(
                    "INSERT OR IGNORE INTO earnings (stock_id, financial_year, q1_earnings, q2_earnings, q3_earnings, q4_earnings) VALUES (?, ?, ?, ?, ?, ?)",
                    [(stock_id, e['year'], e['q1'], e['q2'], e['q3'], e['q4']) for e in earnings]
                )
                print(f"  -> Stored {len(earnings)} earnings records.")
            else:
                print("  -> No earnings data was found or parsed.")

        except Exception as e:
            print(f"  -> An error occurred while processing details for {ticker}: {e}")

        # Fetch news
        cursor.execute("SELECT 1 FROM news WHERE stock_id = ?", (stock_id,))
        has_news_already = cursor.fetchone()

        news = fetch_all_news_for_stock(page, ticker, deep_scrape=not has_news_already)
        if news:
            news_to_insert = [(stock_id, n['date'], n['headline']) for n in news]
            cursor.executemany(
                "INSERT OR IGNORE INTO news (stock_id, news_date, headline) VALUES (?, ?, ?)",
                news_to_insert
            )
            print(f"  -> Stored {len(news)} new news headlines.")

        conn.commit()
        print("  Respecting crawl-delay of 30 seconds...")
        time.sleep(30)

    print("\n--- Instrument Details Scrape Complete ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    setup_database()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()
        conn = sqlite3.connect(DB_FILE)

        # --- CHOOSE WHICH SCRAPER TO RUN ---

        # To update daily prices (run this daily)
        fetch_historical_data_with_playwright(page, conn)

        # To update fundamentals (run this daily or weekly)
        fetch_instrument_details(page, conn)

        conn.close()
        browser.close()

    print("\n--- All Scraping Processes Finished ---")