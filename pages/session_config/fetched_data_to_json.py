import os
import json
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import tempfile

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CACHE_FILE = "cached_data.json"
STATUS_FILE = "data_status.json"
UPDATE_INTERVAL = timedelta(minutes=30)
PERIOD = "5d"  
TICKERS = ["TLKM.JK", "ISAT.JK", "EXCL.JK"]
features = ["Open","High","Low","Close"]

def load_cached_data():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            saved_data = json.load(f)
            cached_data = {
                ticker: pd.DataFrame(data)
                for ticker, data in saved_data.get('cached_data', {}).items()
            }
            last_update_time = {
                key: datetime.fromisoformat(value)
                for key, value in saved_data.get('last_update_time', {}).items()
            }
            return cached_data, last_update_time
    else:
        return {}, {}
def save_cached_data(cached_data):
    current_time = datetime.now()

    serializable_data = {
        'cached_data': {
            ticker: [
                {
                    k: (v.isoformat() if isinstance(v, (datetime, pd.Timestamp)) else v)
                    for k, v in row.items()
                }
                for row in df.to_dict(orient='records')
            ]
            for ticker, df in cached_data.items()
        },
        'last_update_time': {
            'time_yfinance_fetched': current_time.isoformat()
        }
    }

    # Save to temporary file first
    with tempfile.NamedTemporaryFile('w', delete=False, dir=DATA_DIR, suffix='.json') as tmp_file:
        json.dump(serializable_data, tmp_file, indent=2)
        temp_path = tmp_file.name

    # Rename to final file
    final_path = os.path.join(DATA_DIR, "cached_data.json")
    os.replace(temp_path, final_path)


def is_need_update(loaded_data):
    for ticker in TICKERS:
        local_last_date = pd.to_datetime(loaded_data.index[-1]).date()

        # Fetch data terbaru
        remote_data = yf.download(ticker, period=PERIOD)
        if remote_data.empty:
            continue  # Skip, atau bisa raise warning
        remote_last_date = remote_data.index[-1].date()

        # Bandingkan tanggal terakhir
        if remote_last_date != local_last_date:
            return True

    return False

def is_over_one_month(current_time, last_update_time):
    if not last_update_time:
        return True

    if isinstance(last_update_time, str):
        last_update_time = datetime.fromisoformat(last_update_time)

    if current_time.date() != last_update_time.date():
        return True

    return (current_time - last_update_time) > timedelta(days=30)


