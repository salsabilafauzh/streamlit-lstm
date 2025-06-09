import json
import os
import pandas as pd
DATA_DIR = "data"

def save_predicted_line_json(predicted_result, test_dates, ticker):
    data = []
    print(test_dates)
    for i in range(len(test_dates)):
        item = {
            "date": test_dates[i] if isinstance(test_dates[i], str) else test_dates[i].strftime("%Y-%m-%d"),
            "prediction": {
                "Open": float(predicted_result[i][0]),
                "High": float(predicted_result[i][1]), 
                "Low": float(predicted_result[i][2]),
                "Close":float(predicted_result[i][3]),
            }
        }
        print(item)
        data.append(item)

    filename = f"{ticker}_predicted_line.json"
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)



def load_predicted_line_json(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}_predicted_line.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            saved_data = json.load(f)
        df = pd.DataFrame(saved_data)
        return df