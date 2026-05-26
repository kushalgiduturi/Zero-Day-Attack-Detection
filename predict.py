"""
predict.py — Zero-Day IDS inference module (PyTorch)
Supports any dataset via UniversalFeatureMapper.

Usage:
    from predict import IDSPredictor

    predictor = IDSPredictor()

    # With your original NF-UQ-NIDS features
    result = predictor.predict({"FLOW_DURATION_MILLISECONDS": 500, ...})

    # With any other dataset features (auto-mapped)
    result = predictor.predict({"duration": 500, "src_bytes": 12, ...})

    print(result)
    # → {"label": 1, "confidence": 0.91, "verdict": "ATTACK", "proba": [0.09, 0.91]}
"""

import numpy as np
import joblib
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
#  UNIVERSAL FEATURES — 20 core features present in almost every network dataset
# ══════════════════════════════════════════════════════════════════════════════
UNIVERSAL_FEATURES = [
    "FLOW_DURATION_MILLISECONDS",
    "TOTAL_FWDPACKETS",
    "TOTAL_BWDPACKETS",
    "TOTAL_LENGTH_OF_FWD_PACKETS",
    "TOTAL_LENGTH_OF_BWD_PACKETS",
    "FWDPACKET_LENGTH_MEAN",
    "FWDPACKET_LENGTH_STD",
    "BWDPACKET_LENGTH_MEAN",
    "BWDPACKET_LENGTH_STD",
    "FLOW_BYTES_PER_SECOND",
    "FLOW_PACKETS_PER_SECOND",
    "FLOW_IAT_MEAN",
    "FLOW_IAT_STD",
    "FWD_IAT_MEAN",
    "BWD_IAT_MEAN",
    "FWD_PSH_FLAGS",
    "BWD_PSH_FLAGS",
    "FWD_URG_FLAGS",
    "ACTIVE_MEAN",
    "IDLE_MEAN",
]

# Aliases — same feature, different column names across datasets
FEATURE_ALIASES = {
    "FLOW_DURATION_MILLISECONDS": [
        "duration", "flow_duration", "Duration",
        "FLOW_DURATION", "fl_dur", "Flow Duration",
    ],
    "TOTAL_FWDPACKETS": [
        "total_fwd_packets", "fwd_packets", "Tot Fwd Pkts",
        "src_bytes", "fwd_pkts_tot", "spkts",
        " Total Fwd Packets",
    ],
    "TOTAL_BWDPACKETS": [
        "total_bwd_packets", "bwd_packets", "Tot Bwd Pkts",
        "dst_bytes", "bwd_pkts_tot", "dpkts",
        " Total Bwd packets",
    ],
    "TOTAL_LENGTH_OF_FWD_PACKETS": [
        "total_len_fwd_pkts", "fwd_bytes", "TotLen Fwd Pkts",
        "sbytes", " Total Length of Fwd Packets",
    ],
    "TOTAL_LENGTH_OF_BWD_PACKETS": [
        "total_len_bwd_pkts", "bwd_bytes", "TotLen Bwd Pkts",
        "dbytes", " Total Length of Bwd Packets",
    ],
    "FWDPACKET_LENGTH_MEAN": [
        "fwd_pkt_len_mean", "Fwd Pkt Len Mean",
        "fwd_seg_size_avg", " Fwd Packet Length Mean",
    ],
    "FWDPACKET_LENGTH_STD": [
        "fwd_pkt_len_std", "Fwd Pkt Len Std",
        " Fwd Packet Length Std",
    ],
    "BWDPACKET_LENGTH_MEAN": [
        "bwd_pkt_len_mean", "Bwd Pkt Len Mean",
        " Bwd Packet Length Mean",
    ],
    "BWDPACKET_LENGTH_STD": [
        "bwd_pkt_len_std", "Bwd Pkt Len Std",
        " Bwd Packet Length Std",
    ],
    "FLOW_BYTES_PER_SECOND": [
        "flow_byts_s", "bytes_per_sec", "Flow Bytes/s",
        "byterate", " Flow Bytes/s",
    ],
    "FLOW_PACKETS_PER_SECOND": [
        "flow_pkts_s", "packets_per_sec", "Flow Pkts/s",
        "pktrate", "rate", " Flow Packets/s",
    ],
    "FLOW_IAT_MEAN": [
        "flow_iat_mean", "Flow IAT Mean",
        "iat_mean", "mean_iat", " Flow IAT Mean",
    ],
    "FLOW_IAT_STD": [
        "flow_iat_std", "Flow IAT Std",
        " Flow IAT Std",
    ],
    "FWD_IAT_MEAN": [
        "fwd_iat_mean", "Fwd IAT Mean",
        " Fwd IAT Mean",
    ],
    "BWD_IAT_MEAN": [
        "bwd_iat_mean", "Bwd IAT Mean",
        " Bwd IAT Mean",
    ],
    "FWD_PSH_FLAGS": [
        "fwd_psh_flags", "FWD_PSH_FLAGS",
        " Fwd PSH Flags",
    ],
    "BWD_PSH_FLAGS": [
        "bwd_psh_flags", "BWD_PSH_FLAGS",
        " Bwd PSH Flags",
    ],
    "FWD_URG_FLAGS": [
        "fwd_urg_flags", "FWD_URG_FLAGS",
        " Fwd URG Flags",
    ],
    "ACTIVE_MEAN": [
        "active_mean", "Active Mean",
        " Active Mean",
    ],
    "IDLE_MEAN": [
        "idle_mean", "Idle Mean",
        " Idle Mean",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  UNIVERSAL FEATURE MAPPER
# ══════════════════════════════════════════════════════════════════════════════
class UniversalFeatureMapper:
    """
    Automatically maps any dataset's column names to the
    20 universal core features. Missing features are filled with 0.

    Works with: NF-UQ-NIDS, NSL-KDD, UNSW-NB15, CIC-IDS2017,
                raw dicts from live capture, or any unknown dataset.
    """

    def __init__(self):
        # Maps incoming key → universal feature name
        self._alias_lookup = {}
        for universal_col, aliases in FEATURE_ALIASES.items():
            # direct name
            self._alias_lookup[universal_col.lower()] = universal_col
            # all aliases
            for alias in aliases:
                self._alias_lookup[alias.lower().strip()] = universal_col

    def transform_dict(self, flow: dict) -> dict:
        """
        Map a single flow dict (any format) → universal feature dict.
        Missing features default to 0.0.
        """
        mapped = {f: 0.0 for f in UNIVERSAL_FEATURES}
        for key, value in flow.items():
            universal = self._alias_lookup.get(key.lower().strip())
            if universal:
                mapped[universal] = float(value)
        return mapped

    def to_array(self, flow: dict) -> np.ndarray:
        """Map flow dict → numpy array in universal feature order."""
        mapped = self.transform_dict(flow)
        return np.array(
            [[mapped[f] for f in UNIVERSAL_FEATURES]],
            dtype=np.float32
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION  (must match train.py)
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTOR CLASS
# ══════════════════════════════════════════════════════════════════════════════
class IDSPredictor:
    """
    Loads the trained model and runs inference on any network flow.

    Two modes:
      1. Native mode  — flow dict uses exact NF-UQ-NIDS feature names
                        (uses saved feature_cols.pkl + scaler.pkl)
      2. Universal mode — flow dict uses any column names from any dataset
                          (auto-mapped via UniversalFeatureMapper)

    The predictor auto-detects which mode to use based on the flow keys.
    """

    def __init__(self,
                 model_path:    str = "deployment/best_dqn_model.pt",
                 scaler_path:   str = "deployment/scaler.pkl",
                 encoder_path:  str = "deployment/label_encoder.pkl",
                 features_path: str = "deployment/feature_cols.pkl",
                 mapper_path:   str = "deployment/feature_mapper.pkl"):

        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler        = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.feature_cols  = joblib.load(features_path)

        # Load saved mapper if available, otherwise create a fresh one
        try:
            self.mapper = joblib.load(mapper_path)
            print("Universal feature mapper loaded from disk.")
        except FileNotFoundError:
            self.mapper = UniversalFeatureMapper()
            print("Universal feature mapper created (default aliases).")

        # Rebuild model and load weights
        state_size = len(self.feature_cols)
        self.model = DQNNet(state_size, 2).to(self.device)
        ckpt       = torch.load(model_path, map_location=self.device,
                                weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"Model loaded on : {self.device}")
        print(f"Features used   : {state_size}")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _is_native(self, flow: dict) -> bool:
        """
        Returns True if the flow uses native NF-UQ-NIDS feature names.
        Returns False if it needs universal mapping.
        """
        native_keys = set(self.feature_cols)
        flow_keys   = set(flow.keys())
        overlap     = len(flow_keys & native_keys)
        return overlap >= len(flow_keys) * 0.5   # >50% native keys = native mode

    def _native_to_array(self, flow: dict) -> np.ndarray:
        """Convert native NF-UQ-NIDS flow dict → numpy array."""
        return np.array(
            [[flow.get(f, 0.0) for f in self.feature_cols]],
            dtype=np.float32
        )

    def _universal_to_array(self, flow: dict) -> np.ndarray:
        """
        Map any dataset's flow dict → universal features → 
        align to model's expected feature order → numpy array.
        """
        # Step 1 — map to universal 20 features
        universal_mapped = self.mapper.transform_dict(flow)

        # Step 2 — align to model's native feature order
        # Use universal value if native feature is missing
        aligned = []
        for f in self.feature_cols:
            if f in flow:
                aligned.append(float(flow[f]))
            else:
                aligned.append(universal_mapped.get(f, 0.0))

        return np.array([aligned], dtype=np.float32)

    def _run_model(self, x_scaled: np.ndarray) -> np.ndarray:
        """Scale → tensor → model → probabilities."""
        x_tensor = torch.FloatTensor(x_scaled).to(self.device)
        with torch.no_grad():
            proba = self.model(x_tensor).cpu().numpy()
        return proba

    def _build_result(self, proba: np.ndarray) -> dict:
        label = int(np.argmax(proba))
        return {
            "label":      label,
            "confidence": float(np.max(proba)),
            "verdict":    "ATTACK" if label == 1 else "BENIGN",
            "proba":      proba.tolist(),
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, flow: dict) -> dict:
        """
        Predict on a single network flow.
        Accepts any dataset format — auto-detects native vs universal mode.

        Args:
            flow: dict of {feature_name: value}
                  Works with NF-UQ-NIDS, NSL-KDD, UNSW-NB15,
                  CIC-IDS2017, or any custom feature names.

        Returns:
            {
                "label"     : 0 or 1,
                "confidence": 0.0 to 1.0,
                "verdict"   : "BENIGN" or "ATTACK",
                "proba"     : [p_benign, p_attack]
            }
        """
        # Auto-detect mode
        if self._is_native(flow):
            x = self._native_to_array(flow)
        else:
            x = self._universal_to_array(flow)

        x_scaled = self.scaler.transform(x).astype(np.float32)
        proba    = self._run_model(x_scaled)[0]
        return self._build_result(proba)

    def predict_batch(self, flows: list) -> list:
        """
        Predict on a list of flows (any format, mixed datasets supported).
        Processes in chunks of 512 for GPU memory efficiency.

        Args:
            flows: list of flow dicts

        Returns:
            list of prediction dicts
        """
        # Build array — handle each flow independently for format detection
        arrays = []
        for flow in flows:
            if self._is_native(flow):
                arrays.append(self._native_to_array(flow)[0])
            else:
                arrays.append(self._universal_to_array(flow)[0])

        X        = np.array(arrays, dtype=np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)

        results = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), 512):
                batch  = torch.FloatTensor(X_scaled[i:i+512]).to(self.device)
                probas = self.model(batch).cpu().numpy()
                for proba in probas:
                    results.append(self._build_result(proba))
        return results

    def predict_dataset(self, dataset_name: str, csv_path: str) -> list:
        """
        Predict on an entire CSV file from a known dataset.
        Supported: 'nsl-kdd', 'unsw-nb15', 'cic-ids2017', 'nf-uq-nids', 'auto'

        Args:
            dataset_name : name of the dataset format
            csv_path     : path to the CSV file

        Returns:
            list of prediction dicts
        """
        import pandas as pd

        print(f"Loading {dataset_name} dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Dataset-specific label column cleanup
        label_col_map = {
            "nsl-kdd"    : ("label",  lambda x: 0 if x == "normal" else 1),
            "unsw-nb15"  : ("label",  lambda x: int(x)),
            "cic-ids2017": (" Label", lambda x: 0 if x.strip() == "BENIGN" else 1),
            "nf-uq-nids" : ("Label",  lambda x: int(x)),
        }

        flows   = df.to_dict(orient="records")
        results = self.predict_batch(flows)

        print(f"Predicted {len(results)} flows.")
        return results