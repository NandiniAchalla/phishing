import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import json
import joblib
import pandas as pd

from models.nn_model import PhishingNN


def predict(csv_path):
    with open("data/processed/feature_columns.json") as f:
        feature_cols = json.load(f)

    df = pd.read_csv(csv_path)
    X = df[feature_cols]

    scaler = joblib.load("data/processed/feature_scaler.pkl")
    X = scaler.transform(X)

    model = PhishingNN(input_dim=X.shape[1])
    model.load_state_dict(torch.load("data/processed/global_model.pt"))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32)))
        preds = (probs.numpy() > 0.5).astype(int)

    return preds


if __name__ == "__main__":
    print(predict("data/processed/features.csv")[:10])
