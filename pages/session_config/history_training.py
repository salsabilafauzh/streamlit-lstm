import json
import os

HISTORY_FILE = "training_history.json"

def save_training_history(loss, val_loss,ticker):
    history_data = {"loss": loss, "val_loss": val_loss}
    
    with open(f"{HISTORY_FILE}_{ticker}", "w") as f:
        json.dump(history_data, f)

def load_training_history(ticker):
    if os.path.exists(f"{HISTORY_FILE}_{ticker}"):
        with open(f"{HISTORY_FILE}_{ticker}", "r") as f:
            return json.load(f)
    return None