import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils.metrics_logger import save_metrics

from models.nn_model import PhishingNN


def evaluate_federated_model():
    df = pd.read_csv("data/processed/features.csv")

    feature_cols = [c for c in df.columns if c not in ["label", "sample_id", "source_folder"]]
    X = df[feature_cols]
    y = df["label"].values

    scaler = joblib.load("data/processed/feature_scaler.pkl")
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = PhishingNN(input_dim=X.shape[1])
    model.load_state_dict(torch.load("data/processed/global_model.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs = torch.sigmoid(logits)
        preds = (probs.numpy() > 0.5).astype(int)

    print("Federated Model Accuracy:", accuracy_score(y_test, preds))
    print("\nFederated Classification Report:\n")
    print(classification_report(y_test, preds))

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "classification_report": classification_report(y_test, preds, output_dict=True)
    }

    save_metrics(metrics, "federated_results.json")


if __name__ == "__main__":
    evaluate_federated_model()
