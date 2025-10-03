import requests
from bs4 import BeautifulSoup
import re
import json
import csv


def scrape_jse_instruments():
    """
    Scrape all stock tickers and their instrument IDs from Jamaica Stock Exchange
    """
    url = "https://www.jamstockex.com/trading/instruments/"

    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the dropdown/select element containing instruments
        # The instrument IDs are typically in the URL parameter or select options
        instruments = []

        # Method 1: Look for select dropdown with instruments
        select_element = soup.find('select', {'name': 'instrument'}) or soup.find('select',
                                                                                  id=re.compile(r'instrument', re.I))

        if select_element:
            options = select_element.find_all('option')
            for option in options:
                value = option.get('value')
                text = option.get_text(strip=True)

                if value and text and value != '':
                    # Extract ticker from the text (usually in parentheses)
                    ticker_match = re.search(r'\(([A-Z0-9.]+)(?::|\))', text)
                    ticker = ticker_match.group(1) if ticker_match else text

                    instruments.append({
                        'ticker': ticker,
                        'instrument_id': value,
                        'full_name': text
                    })

        # Method 2: Look for links with instrument parameter
        if not instruments:
            links = soup.find_all('a', href=re.compile(r'instrument=\d+'))
            for link in links:
                href = link.get('href')
                text = link.get_text(strip=True)

                # Extract instrument ID from URL
                id_match = re.search(r'instrument=(\d+)', href)
                if id_match:
                    instrument_id = id_match.group(1)

                    # Extract ticker
                    ticker_match = re.search(r'\(([A-Z0-9.]+)(?::|\))', text)
                    ticker = ticker_match.group(1) if ticker_match else text

                    # Avoid duplicates
                    if not any(inst['instrument_id'] == instrument_id for inst in instruments):
                        instruments.append({
                            'ticker': ticker,
                            'instrument_id': instrument_id,
                            'full_name': text
                        })

        return instruments

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []


def save_to_csv(instruments, filename='jse_instruments.csv'):
    """Save instruments to CSV file"""
    if not instruments:
        print("No instruments to save")
        return

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ticker', 'instrument_id', 'full_name'])
        writer.writeheader()
        writer.writerows(instruments)

    print(f"Saved {len(instruments)} instruments to {filename}")


def save_to_json(instruments, filename='jse_instruments.json'):
    """Save instruments to JSON file"""
    if not instruments:
        print("No instruments to save")
        return

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(instruments, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(instruments)} instruments to {filename}")


# Main execution
if __name__ == "__main__":
    print("Scraping Jamaica Stock Exchange instruments...")

    instruments = scrape_jse_instruments()

    if instruments:
        print(f"\nFound {len(instruments)} instruments:\n")

        # Display first 10 as sample
        for inst in instruments[:10]:
            print(f"Ticker: {inst['ticker']:15} | ID: {inst['instrument_id']:10} | Name: {inst['full_name']}")

        if len(instruments) > 10:
            print(f"\n... and {len(instruments) - 10} more")

        # Save to files
        print("\nSaving to files...")
        save_to_csv(instruments)
        save_to_json(instruments)

    else:
        print("No instruments found. The website structure may have changed.")
        print("Please check the HTML structure manually.")