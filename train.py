"""
train.py — Zero-Day Attack Detection (ZDAD) — PyTorch Version
DQN + LSTM model trained on NF-UQ-NIDS dataset.
Supports any dataset via UniversalFeatureMapper + DatasetAdapters.

Usage:
    python train.py --data "C:\\Users\\Kushal\\Downloads\\Zero Day Attack\\NF-UQ-NIDS-v2.csv"
    python train.py --data "path/to/NF-UQ-NIDS-v2.csv" --episodes 100 --batch 64
    python train.py --data "path/to/KDDTrain+.csv" --dataset nsl-kdd
    python train.py --data "path/to/UNSW_NB15.csv" --dataset unsw-nb15
    python train.py --data "path/to/Friday.csv"    --dataset cic-ids2017
    python train.py --data "path/to/unknown.csv"   --dataset auto
"""

import argparse
import os
import random
import time
from collections import deque

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
ZERO_DAY_ATTACKS = {"Shellcode", "Brute Force", "Theft", "Ransomware", "Backdoor"}
DROP_COLS        = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
CHUNK_SIZE       = 10_000


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
        "fwd_psh_flags", " Fwd PSH Flags",
    ],
    "BWD_PSH_FLAGS": [
        "bwd_psh_flags", " Bwd PSH Flags",
    ],
    "FWD_URG_FLAGS": [
        "fwd_urg_flags", " Fwd URG Flags",
    ],
    "ACTIVE_MEAN": [
        "active_mean", "Active Mean", " Active Mean",
    ],
    "IDLE_MEAN": [
        "idle_mean", "Idle Mean", " Idle Mean",
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
                or any unknown dataset.
    """

    def __init__(self):
        self.col_mapping  = {}   # dataset_col → universal_col
        self.missing_cols = []

    def fit(self, df: pd.DataFrame) -> "UniversalFeatureMapper":
        """Auto-detect which columns in df match universal features."""
        df_cols_lower    = {c.lower().strip(): c for c in df.columns}
        self.col_mapping = {}

        for universal_col, aliases in FEATURE_ALIASES.items():
            # Check exact match first
            if universal_col in df.columns:
                self.col_mapping[universal_col] = universal_col
                continue
            # Check aliases
            found = False
            for alias in aliases:
                if alias.lower().strip() in df_cols_lower:
                    self.col_mapping[df_cols_lower[alias.lower().strip()]] = universal_col
                    found = True
                    break
            if not found:
                self.missing_cols.append(universal_col)

        print(f"  Mapped features  : {len(self.col_mapping)}")
        print(f"  Missing features : {len(self.missing_cols)} (filled with 0)")
        if self.missing_cols:
            print(f"  Missing cols     : {self.missing_cols}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame with exactly the universal features."""
        result = df.rename(columns=self.col_mapping)
        for col in self.missing_cols:
            result[col] = 0.0
        return result[UNIVERSAL_FEATURES].astype(np.float32)

    def transform_dict(self, flow: dict) -> dict:
        """Transform a single flow dict."""
        mapped = {}
        for original, universal in self.col_mapping.items():
            mapped[universal] = flow.get(original, 0.0)
        for col in self.missing_cols:
            mapped[col] = 0.0
        return {f: mapped.get(f, 0.0) for f in UNIVERSAL_FEATURES}


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET ADAPTERS — one per known dataset format
# ══════════════════════════════════════════════════════════════════════════════
class NFUQNIDSAdapter:
    """Native format — no column renaming needed."""
    label_col = "Label"
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

class NSLKDDAdapter:
    """Converts NSL-KDD format → NF-UQ-NIDS compatible."""
    label_col = "labels"   # NSL-KDD uses 'labels' (plural) with no header issues
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 'labels' column has values like "normal", "neptune", "smurf" etc.
        # Anything that is not "normal" is an attack
        lbl = df["labels"] if "labels" in df.columns else df["label"]
        # Always convert via string — handles object, StringDtype, and numeric
        try:
            lbl_str = lbl.astype(str).str.strip().str.lower()
            # If all values are digits it's already 0/1 numeric
            if lbl_str.str.isdigit().all():
                df["Label"] = lbl_str.astype(int)
            else:
                df["Label"] = (lbl_str != "normal").astype(int)
        except Exception:
            df["Label"] = (lbl.astype(str).str.strip().str.lower() != "normal").astype(int)
        df.rename(columns={
            "duration"        : "FLOW_DURATION_MILLISECONDS",
            "src_bytes"       : "TOTAL_LENGTH_OF_FWD_PACKETS",
            "dst_bytes"       : "TOTAL_LENGTH_OF_BWD_PACKETS",
            "count"           : "TOTAL_FWDPACKETS",
            "srv_count"       : "TOTAL_BWDPACKETS",
            "serror_rate"     : "FLOW_IAT_MEAN",
            "rerror_rate"     : "FLOW_IAT_STD",
            "same_srv_rate"   : "FLOW_PACKETS_PER_SECOND",
            "diff_srv_rate"   : "FLOW_BYTES_PER_SECOND",
        }, inplace=True)
        return df

class UNSWNB15Adapter:
    """Converts UNSW-NB15 format → NF-UQ-NIDS compatible."""
    label_col = "label"   # UNSW-NB15 uses lowercase
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Label"] = df["label"].astype(int)
        df.rename(columns={
            "dur"    : "FLOW_DURATION_MILLISECONDS",
            "spkts"  : "TOTAL_FWDPACKETS",
            "dpkts"  : "TOTAL_BWDPACKETS",
            "sbytes" : "TOTAL_LENGTH_OF_FWD_PACKETS",
            "dbytes" : "TOTAL_LENGTH_OF_BWD_PACKETS",
            "rate"   : "FLOW_PACKETS_PER_SECOND",
        }, inplace=True)
        return df

class CICIDS2017Adapter:
    """Converts CIC-IDS2017 format → NF-UQ-NIDS compatible."""
    label_col = " Label"   # CIC-IDS2017 has a leading space in column name
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Strip all column names first to remove leading/trailing spaces
        df.columns = df.columns.str.strip()
        df["Label"] = (df["Label"].str.strip() != "BENIGN").astype(int)
        df.rename(columns={
            "Flow Duration"     : "FLOW_DURATION_MILLISECONDS",
            "Total Fwd Packets" : "TOTAL_FWDPACKETS",
            "Total Bwd packets" : "TOTAL_BWDPACKETS",
            "Flow Bytes/s"      : "FLOW_BYTES_PER_SECOND",
            "Flow Packets/s"    : "FLOW_PACKETS_PER_SECOND",
            "Flow IAT Mean"     : "FLOW_IAT_MEAN",
            "Fwd Packet Length Mean" : "FWDPACKET_LENGTH_MEAN",
            "Fwd Packet Length Std"  : "FWDPACKET_LENGTH_STD",
            "Bwd Packet Length Mean" : "BWDPACKET_LENGTH_MEAN",
            "Bwd Packet Length Std"  : "BWDPACKET_LENGTH_STD",
            "Flow IAT Std"           : "FLOW_IAT_STD",
            "Fwd IAT Mean"           : "FWD_IAT_MEAN",
            "Bwd IAT Mean"           : "BWD_IAT_MEAN",
            "Fwd PSH Flags"          : "FWD_PSH_FLAGS",
            "Bwd PSH Flags"          : "BWD_PSH_FLAGS",
            "Fwd URG Flags"          : "FWD_URG_FLAGS",
            "Active Mean"            : "ACTIVE_MEAN",
            "Idle Mean"              : "IDLE_MEAN",
            "Total Length of Fwd Packets" : "TOTAL_LENGTH_OF_FWD_PACKETS",
            "Total Length of Bwd Packets" : "TOTAL_LENGTH_OF_BWD_PACKETS",
        }, inplace=True)
        return df

# Registry — maps --dataset flag → adapter class
DATASET_ADAPTERS = {
    "nf-uq-nids" : NFUQNIDSAdapter,
    "nsl-kdd"    : NSLKDDAdapter,
    "unsw-nb15"  : UNSWNB15Adapter,
    "cic-ids2017": CICIDS2017Adapter,
    "auto"       : None,   # auto-detect via UniversalFeatureMapper
}


# ══════════════════════════════════════════════════════════════════════════════
#  GPU CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
def configure_device():
    """
    Auto-detects CUDA GPU. PyTorch supports CUDA 13.0 on Windows natively.
    Falls back to CPU if no GPU is found.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu    = torch.cuda.get_device_name(0)
        mem    = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU detected: {gpu} ({mem:.1f} GB VRAM)")
        print(f"  CUDA version : {torch.version.cuda}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True
    else:
        device = torch.device("cpu")
        print("⚠ No CUDA GPU found — training on CPU.")
    return device


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_and_preprocess(csv_path: str, dataset_name: str):
    """
    Two-pass chunk reader to handle large CSV files without memory overflow.
    Applies the correct dataset adapter before encoding.
    """
    print("\nPass 1: scanning label classes...")

    # Get adapter
    adapter_cls = DATASET_ADAPTERS.get(dataset_name)
    adapter     = adapter_cls() if adapter_cls else NFUQNIDSAdapter()

    # Auto-detect actual label column name (handles leading/trailing spaces)
    sample_cols  = pd.read_csv(csv_path, nrows=0).columns.tolist()
    raw_label    = adapter.label_col
    # Find the real column name by stripping spaces for comparison
    matched      = next(
        (c for c in sample_cols if c.strip() == raw_label.strip()),
        raw_label   # fallback to original if not found
    )
    label_col    = matched
    print(f"  Label column detected: '{label_col}'")

    all_labels = set()
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, usecols=[label_col]):
        all_labels.update(chunk[label_col].unique())

    # Pass 1b — re-scan after applying adapter to get the real final labels
    # Needed for datasets like CIC-IDS2017 that transform labels in convert()
    real_labels = set()
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        chunk = adapter.convert(chunk)
        real_labels.update(chunk["Label"].astype(str).unique())

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(real_labels))
    print(f"  Classes found (after adapter): {list(label_encoder.classes_)}")

    print("Pass 2: loading and cleaning data...")
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        # Apply dataset adapter
        chunk = adapter.convert(chunk)

        chunk.drop([c for c in DROP_COLS if c in chunk.columns], axis=1, inplace=True)
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        num_cols = chunk.select_dtypes(include="number").columns
        str_cols = chunk.select_dtypes(include="object").columns
        chunk[num_cols] = chunk[num_cols].fillna(0)
        chunk[str_cols] = chunk[str_cols].fillna("")

        # Drop rows with labels not seen during fitting (corrupted/junk rows)
        valid_mask = chunk["Label"].astype(str).isin(set(label_encoder.classes_))
        dropped    = (~valid_mask).sum()
        if dropped > 0:
            print(f"  Dropped {dropped} rows with unrecognised labels")
        chunk = chunk[valid_mask]

        chunk["Label"] = label_encoder.transform(chunk["Label"].astype(str))
        chunks.append(chunk)

    data = pd.concat(chunks, ignore_index=True)
    print(f"  Total rows  : {len(data):,}")
    print(f"  Total cols  : {data.shape[1]}")
    return data, label_encoder


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / ZERO-DAY TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def split_zero_day(data):
    """
    Holds out zero-day attack types entirely from training.
    They only appear in the test set to simulate real-world unseen threats.
    """
    attack_col = "Attack" if "Attack" in data.columns else "Label"

    if attack_col == "Attack":
        mask       = data["Attack"].isin(ZERO_DAY_ATTACKS)
        train_data = data[~mask].copy()
        test_data  = data[mask].copy()
        print(f"\nTraining rows (known attacks)  : {len(train_data):,}")
        print(f"Test rows     (zero-day attacks): {len(test_data):,}")
        print(f"\nTraining attack types:\n{train_data['Attack'].value_counts()}")
        print(f"\nZero-day attack types:\n{test_data['Attack'].value_counts()}\n")
    else:
        print("  No 'Attack' column found — using random 80/20 split.")
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=SEED)

    return train_data, test_data


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE SCALING + OVERSAMPLING
# ══════════════════════════════════════════════════════════════════════════════
def prepare_features(train_data, test_data, data, out_dir, dataset_name):
    """
    For NF-UQ-NIDS: uses all 41 native features.
    For other datasets: auto-maps to 20 universal features via UniversalFeatureMapper.
    Applies fast random oversampling to balance class distribution.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Feature selection ──────────────────────────────────────────────────────
    if dataset_name in ("nf-uq-nids", "auto") and "auto" != dataset_name:
        # Native mode — use all numeric columns
        feature_cols = [
            c for c in data.columns
            if c not in ["Label", "Attack"]
            and data[c].dtype in ["float64", "float32", "int64", "int32"]
        ]
        print(f"Feature columns (native): {len(feature_cols)}")

        X_train_raw = train_data[feature_cols].values.astype(np.float64)
        X_test_raw  = test_data[feature_cols].values.astype(np.float64)
        mapper      = None

    else:
        # Universal mode — auto-map to 20 core features
        print("Auto-mapping features to universal format...")
        mapper = UniversalFeatureMapper()
        mapper.fit(data)

        X_train_raw  = mapper.transform(train_data).values.astype(np.float64)
        X_test_raw   = mapper.transform(test_data).values.astype(np.float64)
        feature_cols = UNIVERSAL_FEATURES
        print(f"Feature columns (universal): {len(feature_cols)}")

        # Save mapper for inference
        joblib.dump(mapper, f"{out_dir}/feature_mapper.pkl")
        print(f"Mapper saved → {out_dir}/feature_mapper.pkl")

    y_train_raw = train_data["Label"].values.astype(np.int64)
    y_test_raw  = test_data["Label"].values.astype(np.int64)

    # ── Clean infinite / extreme values ───────────────────────────────────────
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw  = np.nan_to_num(X_test_raw,  nan=0.0, posinf=0.0, neginf=0.0)
    X_train_raw = np.clip(X_train_raw, -1e15, 1e15)
    X_test_raw  = np.clip(X_test_raw,  -1e15, 1e15)

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test_raw).astype(np.float32)

    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    print(f"Scaler saved → {out_dir}/scaler.pkl")

    # ── Fast random oversampling (replaces slow SMOTE) ────────────────────────
    print("Balancing classes with random oversampling (fast)...")
    counts       = np.bincount(y_train_raw)
    if len(counts) < 2:
        print(f"  ⚠ Only one class found ({counts}) — skipping oversampling.")
        X_res = X_train_scaled.astype(np.float32)
        y_res = y_train_raw.astype(np.int64)
    else:
        majority_cls = int(np.argmax(counts))
        minority_cls = int(np.argmin(counts))
        majority_idx = np.where(y_train_raw == majority_cls)[0]
        minority_idx = np.where(y_train_raw == minority_cls)[0]
        oversample_idx = np.random.choice(minority_idx, size=len(majority_idx), replace=True)
        all_idx        = np.concatenate([majority_idx, oversample_idx])
        np.random.shuffle(all_idx)
        X_res = X_train_scaled[all_idx].astype(np.float32)
        y_res = y_train_raw[all_idx].astype(np.int64)
    print(f"  Class distribution after oversampling: {np.bincount(y_res)}")

    # ── 80/20 validation split ────────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=SEED)

    # Final test set = validation + zero-day samples
    X_test = np.vstack([X_val, X_test_scaled]).astype(np.float32)
    y_test = np.concatenate([y_val, y_test_raw]).astype(np.int64)

    print(f"Train shape : {X_tr.shape}")
    print(f"Test  shape : {X_test.shape}\n")
    return X_tr, y_tr, X_test, y_test, feature_cols, scaler, mapper


# ══════════════════════════════════════════════════════════════════════════════
#  PYTORCH MODEL — DQN with LSTM
# ══════════════════════════════════════════════════════════════════════════════
class DQNNet(nn.Module):
    """
    Architecture:
      Linear(128) → BatchNorm → ReLU → Dropout
      → Reshape (seq_len=1, hidden=128)
      → LSTM(128) → LSTM(64) → LSTM(64)
      → Linear(64) → ReLU → Dropout
      → Linear(action_size) → Softmax
    """
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
        x = torch.softmax(self.fc3(x), dim=-1)
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  DQN AGENT
# ══════════════════════════════════════════════════════════════════════════════
class DQNAgent:
    """
    Double DQN with Prioritized Experience Replay.
    Main model selects actions; target model evaluates Q-values (stable targets).
    """

    def __init__(self, state_size, action_size, device):
        self.state_size  = state_size
        self.action_size = action_size
        self.device      = device

        self.memory    = deque(maxlen=10_000)
        self.priorities = deque(maxlen=10_000)
        self.alpha      = 0.6
        self.beta       = 0.4

        self.gamma              = 0.97
        self.epsilon            = 1.0
        self.epsilon_min        = 0.05
        self.epsilon_decay      = 0.995
        self.learning_rate      = 0.001
        self.target_update_freq = 5

        self.model        = DQNNet(state_size, action_size).to(device)
        self.target_model = DQNNet(state_size, action_size).to(device)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn      = nn.MSELoss()
        self.update_target_network()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        max_p = max(self.priorities, default=1.0)
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_p)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_t)
        self.model.train()
        return int(torch.argmax(q_values).item())

    def replay(self, batch_size, class_weights):
        if len(self.memory) < batch_size:
            return

        priorities = np.array(self.priorities, dtype=np.float32)
        probs      = priorities ** self.alpha
        probs     /= probs.sum()
        indices    = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        minibatch  = [self.memory[i] for i in indices]

        states      = torch.FloatTensor(np.array([s for s,a,r,ns,d in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([ns for s,a,r,ns,d in minibatch])).to(self.device)
        actions     = [a for s,a,r,ns,d in minibatch]
        rewards     = [r for s,a,r,ns,d in minibatch]
        dones       = [d for s,a,r,ns,d in minibatch]

        self.model.train()
        current_qs = self.model(states)

        self.target_model.eval()
        with torch.no_grad():
            target_qs = self.target_model(next_states)

        targets        = current_qs.clone()
        sample_weights = []

        for i in range(batch_size):
            target = rewards[i] if dones[i] else rewards[i] + self.gamma * torch.max(target_qs[i]).item()
            targets[i][actions[i]] = target
            is_w = (len(self.memory) * probs[indices[i]]) ** (-self.beta)
            sample_weights.append(class_weights.get(actions[i], 1.0) * is_w)

        sample_weights  = torch.FloatTensor(sample_weights).to(self.device)
        sample_weights /= sample_weights.max()

        loss = (sample_weights * ((current_qs - targets.detach()) ** 2).mean(dim=1)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict_batch(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch  = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                output = self.model(batch)
                all_preds.append(output.cpu().numpy())
        return np.vstack(all_preds)

    def save(self, path):
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon":         self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epsilon = ckpt["epsilon"]
        self.update_target_network()


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train(agent, X_tr, y_tr, X_test, y_test,
          class_weights_dict, episodes, batch_size,
          steps_per_ep, early_stop_patience, out_dir):

    cumulative_rewards, average_rewards, f1_history, time_history = [], [], [], []
    best_f1, no_improve = 0.0, 0
    n_train = len(X_tr)

    print(f"\nTraining on : {agent.device}")
    print(f"Episodes    : {episodes}")
    print(f"Steps/ep    : {steps_per_ep}")
    print(f"Batch size  : {batch_size}\n")

    for e in range(episodes):
        t0           = time.time()
        total_reward = 0

        start_idx = np.random.randint(0, n_train - steps_per_ep - 1)
        state     = X_tr[start_idx]

        for t in range(steps_per_ep):
            idx    = start_idx + t
            action = agent.act(state)
            reward = 1 if action == y_tr[idx] else -1

            total_reward += reward
            next_state    = X_tr[idx + 1]
            done          = (t == steps_per_ep - 1)

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size, class_weights_dict)

        if e % agent.target_update_freq == 0:
            agent.update_target_network()

        ep_time = time.time() - t0
        avg_r   = total_reward / steps_per_ep

        cumulative_rewards.append(total_reward)
        average_rewards.append(avg_r)
        time_history.append(ep_time)

        y_pred_prob = agent.predict_batch(X_test)
        y_pred      = np.argmax(y_pred_prob, axis=1)
        ep_f1       = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_history.append(ep_f1)

        conv = abs(average_rewards[-1] - average_rewards[-2]) if e > 0 else 0.0
        print(f"Ep {e+1:03d}/{episodes} | Reward: {total_reward:+.0f} | "
              f"Avg: {avg_r:+.3f} | F1: {ep_f1:.4f} | "
              f"ε: {agent.epsilon:.3f} | Δ: {conv:.4f} | t: {ep_time:.1f}s")

        if ep_f1 > best_f1:
            best_f1    = ep_f1
            no_improve = 0
            agent.save(f"{out_dir}/best_dqn_model.pt")
            print(f"  ✓ New best F1 = {best_f1:.4f} — model saved")
        else:
            no_improve += 1

        if (e + 1) % 20 == 0:
            print(f"\n--- Episode {e+1} Report ---")
            print(confusion_matrix(y_test, y_pred))
            n_cls = len(np.unique(np.concatenate([y_test, y_pred])))
            tnames = ["Benign", "Attack"] if n_cls == 2 else None
            print(classification_report(y_test, y_pred,
                  target_names=tnames, zero_division=0))

        if no_improve >= early_stop_patience:
            print(f"\nEarly stopping at episode {e+1} "
                  f"(no F1 improvement for {early_stop_patience} episodes)")
            break

    print(f"\nTraining complete. Best F1 : {best_f1:.4f}")
    print(f"Avg time/episode           : {np.mean(time_history):.1f}s")
    return cumulative_rewards, average_rewards, f1_history, time_history, best_f1, y_pred


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(cumulative_rewards, f1_history, time_history, best_f1, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(cumulative_rewards, color="steelblue")
    axes[0].set_title("Cumulative Reward per Episode")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)

    axes[1].plot(f1_history, color="darkorange")
    axes[1].axhline(best_f1, color="red", linestyle="--", alpha=0.6,
                    label=f"Best = {best_f1:.3f}")
    axes[1].set_title("Test F1-Score per Episode")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("F1 (weighted)")
    axes[1].legend()

    axes[2].plot(time_history, color="mediumseagreen")
    axes[2].set_title("Time per Episode (s)")
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Seconds")

    plt.tight_layout()
    path = f"{out_dir}/training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_confusion_matrix(y_test, y_pred, out_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Attack"],
                yticklabels=["Benign", "Attack"])
    plt.title("Confusion Matrix — Zero-Day Test Set")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    path = f"{out_dir}/confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(agent, out_dir, X_tr, y_tr, X_test, y_test):
    agent.load(f"{out_dir}/best_dqn_model.pt")

    print("\n=== TRAINING SET ===")
    train_prob = agent.predict_batch(X_tr)
    train_pred = np.argmax(train_prob, axis=1)
    n_classes_tr = len(np.unique(np.concatenate([y_tr, train_pred])))
    tnames = ["Benign", "Attack"] if n_classes_tr == 2 else None
    print(classification_report(y_tr, train_pred,
          target_names=tnames, zero_division=0))
    if n_classes_tr == 2:
        print(f"AUC-ROC (train): {roc_auc_score(y_tr, train_pred):.4f}")
    else:
        print("⚠ Only one class in training set — AUC-ROC not applicable.")

    print("\n=== TEST SET (Zero-Day Attacks) ===")
    test_prob  = agent.predict_batch(X_test)
    test_pred  = np.argmax(test_prob, axis=1)
    n_classes_te = len(np.unique(np.concatenate([y_test, test_pred])))
    tnames = ["Benign", "Attack"] if n_classes_te == 2 else None
    print(classification_report(y_test, test_pred,
          target_names=tnames, zero_division=0))
    if n_classes_te == 2:
        print(f"AUC-ROC  (test): {roc_auc_score(y_test, test_pred):.4f}")
    else:
        print("⚠ Only one class in test set — AUC-ROC not applicable.")
    print(f"Accuracy (test): {accuracy_score(y_test, test_pred):.4f}")
    print(f"F1       (test): {f1_score(y_test, test_pred, average='weighted', zero_division=0):.4f}")

    return test_pred


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="ZDAD — Zero-Day Attack Detector (PyTorch)")
    parser.add_argument("--data",     required=True,  help="Path to dataset CSV file")
    parser.add_argument("--dataset",  default="nf-uq-nids",
                        choices=list(DATASET_ADAPTERS.keys()),
                        help="Dataset format (default: nf-uq-nids)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--steps",    type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--out",      default="deployment")
    args = parser.parse_args()

    print(f"Dataset format  : {args.dataset}")

    # GPU setup
    device = configure_device()

    # Pipeline
    data, label_encoder = load_and_preprocess(args.data, args.dataset)

    os.makedirs(args.out, exist_ok=True)
    joblib.dump(label_encoder, f"{args.out}/label_encoder.pkl")

    train_data, test_data = split_zero_day(data)

    X_tr, y_tr, X_test, y_test, feature_cols, scaler, mapper = prepare_features(
        train_data, test_data, data, args.out, args.dataset
    )

    # Class weights
    unique_cls         = np.unique(y_tr)
    cw                 = compute_class_weight("balanced", classes=unique_cls, y=y_tr)
    class_weights_dict = {int(c): float(w) for c, w in zip(unique_cls, cw)}
    print(f"Class weights: {class_weights_dict}")

    # Agent
    agent = DQNAgent(X_tr.shape[1], 2, device)

    # Train
    cum_r, avg_r, f1_hist, t_hist, best_f1, y_pred = train(
        agent, X_tr, y_tr, X_test, y_test,
        class_weights_dict,
        episodes=args.episodes,
        batch_size=args.batch,
        steps_per_ep=args.steps,
        early_stop_patience=args.patience,
        out_dir=args.out,
    )

    # Save feature list for inference
    joblib.dump(feature_cols, f"{args.out}/feature_cols.pkl")

    # Plots
    plot_training_curves(cum_r, f1_hist, t_hist, best_f1, args.out)
    plot_confusion_matrix(y_test, y_pred, args.out)

    # Final evaluation
    evaluate(agent, args.out, X_tr, y_tr, X_test, y_test)

    print(f"\nAll artifacts saved to ./{args.out}/")


if __name__ == "__main__":
    main()