import json
import os

HISTORY_FILE = "training_history"

def save_training_history(loss, val_loss,training_time, ticker):
    history_data = {"loss": loss, "val_loss": val_loss, "training_time": training_time}
    
    with open(f"{HISTORY_FILE}_{ticker}.json", "w") as f:
        json.dump(history_data, f)

def load_training_history(ticker):
    if os.path.exists(f"{HISTORY_FILE}_{ticker}.json"):
        with open(f"{HISTORY_FILE}_{ticker}.json", "r") as f:
            return json.load(f)
    return None