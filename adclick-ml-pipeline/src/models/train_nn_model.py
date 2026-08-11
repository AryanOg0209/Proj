"""
PyTorch MLP as an alternative CTR model — demonstrates the neural-network
leg of the JD's "clustering, decision trees, neural networks" requirement,
and reports the same metrics as the sklearn models for a fair comparison.

Usage:
    python -m src.models.train_nn_model --features data/features/impressions \
        --out models/nn_model.pt
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.evaluate import evaluate_binary
from src.models.train_ctr_model import CATEGORICAL_COLS, NUMERIC_COLS, TARGET, load_features


class CTRNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )


def train(
    features_path: str,
    out_path: str,
    epochs: int = 8,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    df = load_features(features_path)
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    preprocessor = _build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_t = preprocessor.transform(X_test).astype(np.float32)
    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
        X_test_t = X_test_t.toarray()

    train_ds = TensorDataset(torch.from_numpy(X_train_t), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = CTRNet(input_dim=X_train_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        print(f"epoch {epoch + 1}/{epochs} — loss: {epoch_loss / len(train_ds):.4f}")

    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test_t))
        test_proba = torch.sigmoid(test_logits).numpy()

    metrics = evaluate_binary(y_test, test_proba)
    print(f"[neural_net] {metrics.as_dict()}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": X_train_t.shape[1]}, out_path
    )
    joblib.dump(preprocessor, out_path.replace(".pt", "_preprocessor.joblib"))
    with open(out_path.replace(".pt", "_metrics.json"), "w") as f:
        json.dump(metrics.as_dict(), f, indent=2)

    return metrics.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/features/impressions")
    parser.add_argument("--out", default="models/nn_model.pt")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    train(args.features, args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()
