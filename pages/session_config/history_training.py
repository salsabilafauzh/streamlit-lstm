import json
import os
import datetime

DATA_DIR = "data"
HISTORY_FILE = "training_history"

def save_training_history(loss, val_loss, training_time, ticker):
    os.makedirs(DATA_DIR, exist_ok=True)
    current_time = datetime.datetime.now()
    date_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    history_data = {
        "loss": loss,
        "val_loss": val_loss,
        "training_time": training_time,
        "date": date_str,
    }
    file_path = os.path.join(DATA_DIR, f"{HISTORY_FILE}_{ticker}.json")
    with open(file_path, "w") as f:
        json.dump(history_data, f)

def load_training_history(ticker):
    file_path = os.path.join(DATA_DIR, f"{HISTORY_FILE}_{ticker}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None
