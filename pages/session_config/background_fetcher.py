import os
import json
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf 

DATA_DIR = "data"
CACHED_FILE = os.path.join(DATA_DIR, "cached_data.json")
features = ["Open","High","Low","Close"]
DATA_DIR = "data"
HISTORY_FILE = "training_history"
companies = {
    "TLKM.JK": "Telkom Indonesia Tbk [TLKM]",
    "ISAT.JK": "Indosat Tbk [ISAT]",
    "EXCL.JK": "XL Axiata Tbk [EXCL]"
}

def fetch_all():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load cached data
    cached_data = {}
    if os.path.exists(CACHED_FILE):
        with open(CACHED_FILE, "r") as f:
            raw = json.load(f)
            for ticker, records in raw["cached_data"].items():
                df = pd.DataFrame(records)
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                df.set_index("Date", inplace=True)
                cached_data[ticker] = df
    else:
        return True, {}  # Kalau belum ada cache, anggap perlu update

    updated = False
    all_data = {}
    current_time = datetime.now().replace(tzinfo=None)

    for ticker in companies.keys():
        # Ambil cache terakhir
        df_cached = cached_data.get(ticker, pd.DataFrame())
        last_cached_date = df_cached.index.max() if not df_cached.empty else None

        # Fetch 5 hari terakhir
        ticker = yf.Ticker(ticker)
        start_date = current_time - timedelta(days=5)
        df_new = ticker.history(start=start_date, end=current_time, auto_adjust=False)
        if df_new is None or df_new.empty:
            all_data[ticker] = df_cached
            continue

        df_new = df_new[features]
        df_new.index = pd.to_datetime(df_new.index).tz_localize(None)
        df_new = df_new[~df_new.index.duplicated(keep='last')]
        last_fetched_date = df_new.index.max()

        last_cached_date = df_cached.index.max() if not df_cached.empty else None
        last_fetched_date = df_new.index.max()
        if last_cached_date is None or last_fetched_date > last_cached_date:
            updated = True

    return updated
