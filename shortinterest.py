
import requests
from bs4 import BeautifulSoup

def scrape_short_interest(stock_ticker):
    url = f"https://finance.yahoo.com/quote/{stock_ticker}/key-statistics?p={stock_ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, "html.parser")
    share_stats_section = soup.find("h3", string="Share Statistics")
    table = share_stats_section.find_next("table")
    rows = table.find_all("tr")

    short_interest = None
    short_ratio = None
    short_float = None

    for row in rows:
        data = row.find_all("td")
        if len(data) == 2:
            label = data[0].text.strip()
            value = data[1].text.strip()

            if "Shares Short" in label:
                short_interest = value
            elif "Short Ratio" in label:
                short_ratio = value
            elif "Short % of Float" in label:
                short_float = value

    print(f"Shares Short: {short_interest}")
    print(f"Short Ratio: {short_ratio}")
    print(f"Short % of Float: {short_float}")
    return short_interest, short_ratio, short_float




import os
from datetime import date, timedelta

SHORT_DATA_FILE = "short_interest_data.txt"
DATA_EXPIRY_DAYS = 1  # Number of days before considering data expired and fetching new data

def get_short_interest(item):
    symbol, short_interest, currency = item
    print(short_interest)
    if short_interest is not None:
        # Extract the three values from short_interest tuple
        shares_short, short_ratio, short_float = short_interest
        # Use any of the values for ranking, e.g., short_ratio
        return float(short_ratio) if short_ratio is not None else 0
    else:
        return 0
    
def compare_short_interest_symbols():
    symbols = [
        ("NVDA", "USD"),
        ("TSLA", "USD"),
        ("META", "USD"),
        ("AAPL", "USD"),
        ("MSFT", "USD"),
        ("AMZN", "USD"),
        ("GOOGL", "USD"),
        ("SNAP", "USD"),
        ("FOUR", "USD"),
        ("AMD", "USD"),
        ("ASML", "USD")
    ]

    # Check if the data file exists and if it's within the expiry period
    if os.path.exists(SHORT_DATA_FILE):
        modified_date = date.fromtimestamp(os.path.getmtime(SHORT_DATA_FILE))
        current_date = date.today()
        days_since_modified = (current_date - modified_date).days

        if days_since_modified <= DATA_EXPIRY_DAYS:
            print("Reading from the existing data file...")
            with open(SHORT_DATA_FILE, "r") as file:
                print(file.read())
            return

    # Fetch new short interest data
    short_interest_data = []
    for symbol, currency in symbols:
        ticker = symbol  # Replace with the appropriate ticker format for the data source
        short_interest = scrape_short_interest(ticker)
        short_interest_data.append((symbol, short_interest, currency))
    print(short_interest_data)
    # sorted_short_interest = sorted(short_interest_data, key=lambda x: (x[1] if x[1] is not None else 0), reverse=True)
    sorted_short_interest = sorted(short_interest_data, key=lambda x: get_short_interest(x), reverse=True)

    # Print and rank the symbols
    rank = 1
    for symbol, short_interest, currency in sorted_short_interest:
        print(f"Rank: {rank}")
        print(f"Symbol: {symbol}")
        print(f"Short Interest: {short_interest} {currency}")
        print("---")
        rank += 1

    # Store the data in the file
    with open(SHORT_DATA_FILE, "w") as file:
        for symbol, short_interest, currency in sorted_short_interest:
            file.write(f"Symbol: {symbol}\n")
            file.write(f"Short Interest: {short_interest} {currency}\n")
            file.write("---\n")

# Usage example
compare_short_interest_symbols()

# Usage example
compare_short_interest_symbols()


# Usage example
compare_short_interest_symbols()

# Usage example
stock_ticker = "TSLA"  # Replace with the desired stock ticker
scrape_short_interest(stock_ticker)

