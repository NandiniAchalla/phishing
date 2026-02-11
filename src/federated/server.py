import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler

from models.nn_model import PhishingNN
from federated.client import train_client
from federated.fedavg import fedavg


def run_federated_training(rounds=7):
    df = pd.read_csv("data/processed/features.csv")

    feature_cols = [c for c in df.columns if c not in ["label", "sample_id", "source_folder"]]
    input_dim = len(feature_cols)

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, "data/processed/feature_scaler.pkl")

    global_model = PhishingNN(input_dim)

    client_groups = df.groupby("source_folder")

    for rnd in range(rounds):
        print(f"\nFederated Round {rnd + 1}")

        client_states = []

        for client_id, client_df in client_groups:
            print(f"Client: {client_id}")

            X = client_df[feature_cols].values
            y = client_df["label"]

            local_model = PhishingNN(input_dim)
            local_model.load_state_dict(global_model.state_dict())

            client_state = train_client(local_model, X, y)
            client_states.append(client_state)

        global_weights = fedavg(client_states)
        global_model.load_state_dict(global_weights)

    torch.save(global_model.state_dict(), "data/processed/global_model.pt")
    print("\nFederated training complete!")


if __name__ == "__main__":
    run_federated_training()
