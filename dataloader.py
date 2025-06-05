import yfinance as yf
import pandas as pd
import time
import requests
from datetime import datetime

def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)[0]
        tickers = table['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"Retrieved {len(tickers)} S&P500 tickers")
        return tickers
    except Exception as e:
        print(f"Error retrieving S&P500 tickers: {e}")
        return []

def get_additional_tickers():
    additional_tickers = [
        'TSLA', 'NVDA', 'PLTR', 'SNOW', 'COIN', 'RBLX', 'HOOD', 'SOFI',
        'BABA', 'TSM', 'ASML', 'NVO', 'UL', 'TM', 'SONY', 'SAP', 'SHOP',
        'MSTR', 'RIOT', 'MARA', 'CLSK',
        'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'GLD', 'SLV',
        'O', 'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'WELL', 'AVB', 'EXR',
        'BRK-B', 'BRK-A', 'UBER', 'LYFT', 'DASH', 'ABNB', 'SPOT', 'NFLX',
        'DIS', 'PYPL', 'SQ', 'ROKU', 'ZM', 'DOCU', 'CRWD', 'ZS', 'OKTA'
    ]
    return additional_tickers

def validate_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        return len(hist) > 0
    except:
        return False

def download_price_data_batch(tickers, start="2018-01-01", end="2023-01-01", interval='1mo', max_retries=3):
    all_data = pd.DataFrame()
    batch_size = 20
    
    print(f"Downloading {len(tickers)} tickers between {start} and {end}...")
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tickers)...")
        print(f"Tickers: {batch[:5]}{'...' if len(batch) > 5 else ''}")
        
        retries = 0
        while retries < max_retries:
            try:
                data = yf.download(
                    batch, 
                    start=start, 
                    end=end, 
                    interval=interval,
                    auto_adjust=True,
                    prepost=False,
                    threads=False,
                    progress=False
                )
                
                if data.empty:
                    print(f"No data for batch {batch_num}")
                    break
                
                if len(batch) == 1:
                    ticker = batch[0]
                    if not data.empty and 'Adj Close' in data.columns:
                        prices = data[['Adj Close']].rename(columns={'Adj Close': ticker})
                    elif not data.empty and 'Close' in data.columns:
                        prices = data[['Close']].rename(columns={'Close': ticker})
                    else:
                        prices = pd.DataFrame()
                else:
                    prices = pd.DataFrame()
                    if not data.empty and isinstance(data.columns, pd.MultiIndex):
                        for ticker in batch:
                            try:
                                if ('Adj Close', ticker) in data.columns:
                                    prices[ticker] = data[('Adj Close', ticker)]
                                elif ('Close', ticker) in data.columns:
                                    prices[ticker] = data[('Close', ticker)]
                            except KeyError:
                                continue
                    elif not data.empty:
                        if 'Adj Close' in data.columns:
                            prices = data[['Adj Close']]
                        elif 'Close' in data.columns:
                            prices = data[['Close']]
                
                if all_data.empty:
                    all_data = prices
                else:
                    all_data = pd.concat([all_data, prices], axis=1)
                
                print(f"✓ Batch {batch_num} downloaded ({prices.shape[1]} valid tickers)")
                break
                
            except Exception as e:
                retries += 1
                print(f"Error batch {batch_num}, attempt {retries}/{max_retries}: {e}")
                if retries < max_retries:
                    time.sleep(2)
                else:
                    print(f"✗ Failed batch {batch_num} after {max_retries} attempts")
        
        if i + batch_size < len(tickers):
            time.sleep(2)
    
    return all_data

def clean_data(data):
    print("Cleaning data...")
    
    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0, how='all')
    data = data.sort_index()
    
    print(f"Final data: {data.shape[0]} dates x {data.shape[1]} tickers")
    return data

def compute_returns(price_data):
    print("Computing returns...")
    
    returns = price_data.pct_change()
    
    print(f"Price data shape: {price_data.shape}")
    print(f"Returns shape before cleaning: {returns.shape}")
    print(f"Non-null values per column (first 5):")
    non_null_counts = returns.count()
    print(non_null_counts.head())
    
    returns_cleaned = returns.dropna(how='all')
    
    print(f"Returns shape after cleaning: {returns_cleaned.shape}")
    print(f"First date with returns: {returns_cleaned.index[0] if len(returns_cleaned) > 0 else 'None'}")
    print(f"Last date with returns: {returns_cleaned.index[-1] if len(returns_cleaned) > 0 else 'None'}")
    
    return returns_cleaned

def save_to_csv(data, filename):
    if data.empty:
        print(f"⚠️  Warning: Empty DataFrame for {filename}")
        return
    
    data.to_csv(filename, index=True)
    print(f"✓ Data saved to {filename}")
    print(f"Dimensions: {data.shape[0]} rows x {data.shape[1]} columns")
    
    print("Data preview:")
    print(data.head(3))

def main():
    print("=== Financial Data Retrieval ===")
    
    sp500_tickers = get_sp500_tickers()
    additional_tickers = get_additional_tickers()
    
    all_tickers = list(set(sp500_tickers + additional_tickers))
    print(f"Total of {len(all_tickers)} unique tickers")
    
    if len(all_tickers) > 500:
        all_tickers = all_tickers[:500]
        print(f"Limited to {len(all_tickers)} tickers")
    
    price_data = download_price_data_batch(
        all_tickers,
        start="2008-01-01",
        end="2018-01-01",
        interval='1mo'
    )
    
    if price_data.empty:
        print("No data retrieved!")
        return
    
    clean_prices = clean_data(price_data)
    returns = compute_returns(clean_prices)
    
    save_to_csv(clean_prices, "monthly_prices.csv")
    
    if not returns.empty:
        save_to_csv(returns, "monthly_returns.csv")
    else:
        print("⚠️  No returns computed - monthly_returns.csv not created")
    
    print("\n=== Summary ===")
    print(f"Period: 2018-01-01 to 2023-01-01")
    print(f"Frequency: Monthly")
    print(f"Tickers retrieved: {clean_prices.shape[1]}")
    print(f"Data points: {clean_prices.shape[0]}")
    print(f"Files created: monthly_prices.csv, monthly_returns.csv")

if __name__ == "__main__":
    main()
