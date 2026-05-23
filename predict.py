"""
predict.py — Zero-Day IDS inference module (PyTorch)

Usage:
    from predict import IDSPredictor

    predictor = IDSPredictor()
    result = predictor.predict({"FLOW_DURATION_MILLISECONDS": 500, ...})
    print(result)
    # → {"label": 1, "confidence": 0.91, "verdict": "ATTACK", "proba": [0.09, 0.91]}
"""

import numpy as np
import joblib
import torch
import torch.nn as nn


# ── Model definition (must match train.py) ────────────────────────────────────
class DQNNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNet, self).__init__()
        self.fc1   = nn.Linear(state_size, 128)
        self.bn1   = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.2)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64,  batch_first=True)
        self.lstm3 = nn.LSTM(input_size=64,  hidden_size=64,  batch_first=True)
        self.fc2   = nn.Linear(64, 64)
        self.drop2 = nn.Dropout(0.2)
        self.fc3   = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = x.unsqueeze(1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        return torch.softmax(self.fc3(x), dim=-1)


# ── Predictor class ───────────────────────────────────────────────────────────
class IDSPredictor:
    def __init__(self,
                 model_path: str      = "deployment/best_dqn_model.pt",
                 scaler_path: str     = "deployment/scaler.pkl",
                 encoder_path: str    = "deployment/label_encoder.pkl",
                 features_path: str   = "deployment/feature_cols.pkl"):

        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler        = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.feature_cols  = joblib.load(features_path)

        # Rebuild model and load weights
        state_size   = len(self.feature_cols)
        self.model   = DQNNet(state_size, 2).to(self.device)
        ckpt         = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def _to_array(self, flow: dict) -> np.ndarray:
        return np.array([[flow.get(f, 0.0) for f in self.feature_cols]], dtype=np.float32)

    def predict(self, flow: dict) -> dict:
        x        = self._to_array(flow)
        x_scaled = self.scaler.transform(x).astype(np.float32)
        x_tensor = torch.FloatTensor(x_scaled).to(self.device)
        with torch.no_grad():
            proba = self.model(x_tensor).cpu().numpy()[0]
        label = int(np.argmax(proba))
        return {
            "label":      label,
            "confidence": float(np.max(proba)),
            "verdict":    "ATTACK" if label == 1 else "BENIGN",
            "proba":      proba.tolist(),
        }

    def predict_batch(self, flows: list) -> list:
        X        = np.array([[f.get(fn, 0.0) for fn in self.feature_cols] for f in flows], dtype=np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        results  = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), 512):
                batch  = torch.FloatTensor(X_scaled[i:i+512]).to(self.device)
                probas = self.model(batch).cpu().numpy()
                for proba in probas:
                    label = int(np.argmax(proba))
                    results.append({
                        "label":      label,
                        "confidence": float(np.max(proba)),
                        "verdict":    "ATTACK" if label == 1 else "BENIGN",
                        "proba":      proba.tolist(),
                    })
        return results
