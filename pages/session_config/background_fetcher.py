import yfinance as yf
import time
import json
import os

TICKERS = ["TLKM", "ISAT", "EXCL"]
PERIOD = "5d"
CHECK_INTERVAL = 3600

def fetch_all():
    last_dates = {}

    while True:
        status = {}
        for ticker in TICKERS:
            data = yf.download(ticker, period=PERIOD)
            latest_date = data.index[-1].strftime('%Y-%m-%d')
            is_updated = (
                ticker not in last_dates or latest_date != last_dates[ticker]
            )
            data_filename = f"stock_data_{ticker}.json"
            data.to_json(data_filename)
            status[ticker] = is_updated

            last_dates[ticker] = latest_date

        with open("data_status.json", "w") as f:
            json.dump(status, f)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    fetch_all()