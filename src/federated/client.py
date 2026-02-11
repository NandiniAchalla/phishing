import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_client(model, X, y, epochs=5, lr=0.001):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # CLASS-WEIGHTED LOSS
    pos_weight = torch.tensor([2.5])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model.state_dict()
