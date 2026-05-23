"""
train.py — Zero-Day Attack Detection (ZDAD) — PyTorch Version
DQN + LSTM model trained on NF-UQ-NIDS dataset.

Usage:
    python train.py --data "C:\\Users\\Kushal\\Downloads\\Zero Day Attack\\NF-UQ-NIDS-v2.csv"
    python train.py --data "path/to/NF-UQ-NIDS-v2.csv" --episodes 100 --batch 64
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
from imblearn.over_sampling import SMOTE
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
        # Allow PyTorch to use TF32 for faster matmul on Ampere/Ada GPUs (RTX 40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True   # auto-tune convolutions
    else:
        device = torch.device("cpu")
        print("⚠ No CUDA GPU found — training on CPU.")
    return device


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_and_preprocess(csv_path: str):
    """
    Two-pass chunk reader to handle large CSV files without memory overflow.
    Pass 1 — collect all unique Label values and fit LabelEncoder once.
    Pass 2 — load, clean, and encode every chunk.
    """
    print("\nPass 1: scanning label classes...")
    all_labels = set()
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, usecols=["Label"]):
        all_labels.update(chunk["Label"].unique())

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(all_labels))
    print(f"  Classes found: {list(label_encoder.classes_)}")

    print("Pass 2: loading and cleaning data...")
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        chunk.drop([c for c in DROP_COLS if c in chunk.columns], axis=1, inplace=True)
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        num_cols = chunk.select_dtypes(include="number").columns
        str_cols = chunk.select_dtypes(include="object").columns
        chunk[num_cols] = chunk[num_cols].fillna(0)
        chunk[str_cols] = chunk[str_cols].fillna("")
        chunk["Label"]  = label_encoder.transform(chunk["Label"])
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
    # Check which zero-day attack column exists
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
        # Fallback: random 80/20 split if no Attack column
        print("  No 'Attack' column found — using random 80/20 split.")
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=SEED)

    return train_data, test_data


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE SCALING + SMOTE
# ══════════════════════════════════════════════════════════════════════════════
def prepare_features(train_data, test_data, data, out_dir):
    """
    Scales features with StandardScaler (fit on train only) and
    applies SMOTE to balance the training class distribution.
    """
    feature_cols = [
        c for c in data.columns
        if c not in ["Label", "Attack"]
        and data[c].dtype in ["float64", "float32", "int64", "int32"]
    ]
    print(f"Feature columns: {len(feature_cols)}")

    X_train_raw    = train_data[feature_cols].values.astype(np.float64)
    y_train_raw    = train_data["Label"].values.astype(np.int64)
    X_test_raw     = test_data[feature_cols].values.astype(np.float64)
    y_test_raw     = test_data["Label"].values.astype(np.int64)
    # Replace inf/-inf and clip extreme values before scaling
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw  = np.nan_to_num(X_test_raw,  nan=0.0, posinf=0.0, neginf=0.0)
    X_train_raw = np.clip(X_train_raw, -1e15, 1e15)
    X_test_raw  = np.clip(X_test_raw,  -1e15, 1e15)

    # Single scaler — fit on training data only, transform both sets
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test_raw).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    print(f"Scaler saved → {out_dir}/scaler.pkl")

    # SMOTE oversampling
    print("Applying SMOTE...")
    try:
        smote        = SMOTE(random_state=SEED, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X_train_scaled, y_train_raw)
        X_res        = X_res.astype(np.float32)
        y_res        = y_res.astype(np.int64)
        print(f"  Class distribution after SMOTE: {np.bincount(y_res)}")
    except ValueError as e:
        print(f"  SMOTE failed ({e}), using original data.")
        X_res, y_res = X_train_scaled, y_train_raw

    # 80/20 validation split on resampled data
    X_tr, X_val, y_tr, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=SEED)

    # Final test set = validation + zero-day samples
    X_test = np.vstack([X_val, X_test_scaled]).astype(np.float32)
    y_test = np.concatenate([y_val, y_test_raw]).astype(np.int64)

    print(f"Train shape : {X_tr.shape}")
    print(f"Test  shape : {X_test.shape}\n")
    return X_tr, y_tr, X_test, y_test, feature_cols, scaler


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

        # Three stacked LSTM layers
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64,  batch_first=True)
        self.lstm3 = nn.LSTM(input_size=64,  hidden_size=64,  batch_first=True)

        self.fc2   = nn.Linear(64, 64)
        self.drop2 = nn.Dropout(0.2)
        self.fc3   = nn.Linear(64, action_size)

    def forward(self, x):
        # x shape: (batch, state_size)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)

        # Add sequence dimension: (batch, 1, 128)
        x = x.unsqueeze(1)

        x, _ = self.lstm1(x)   # (batch, 1, 128)
        x, _ = self.lstm2(x)   # (batch, 1, 64)
        x, _ = self.lstm3(x)   # (batch, 1, 64)

        # Take the last time step
        x = x[:, -1, :]        # (batch, 64)

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

        # Replay buffer
        self.memory     = deque(maxlen=10_000)
        self.priorities  = deque(maxlen=10_000)
        self.alpha       = 0.6
        self.beta        = 0.4

        # Hyperparameters
        self.gamma             = 0.97
        self.epsilon           = 1.0
        self.epsilon_min       = 0.05
        self.epsilon_decay     = 0.995
        self.learning_rate     = 0.001
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
        """ε-greedy action selection."""
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

        # Prioritized sampling
        priorities  = np.array(self.priorities, dtype=np.float32)
        probs       = priorities ** self.alpha
        probs      /= probs.sum()
        indices     = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        minibatch   = [self.memory[i] for i in indices]

        states      = torch.FloatTensor(np.array([s for s,a,r,ns,d in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([ns for s,a,r,ns,d in minibatch])).to(self.device)
        actions     = [a for s,a,r,ns,d in minibatch]
        rewards     = [r for s,a,r,ns,d in minibatch]
        dones       = [d for s,a,r,ns,d in minibatch]

        # Current Q-values from main model
        self.model.train()
        current_qs = self.model(states)

        # Target Q-values from target model (stable)
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

        sample_weights = torch.FloatTensor(sample_weights).to(self.device)
        sample_weights /= sample_weights.max()

        # Weighted loss
        loss = (sample_weights * ((current_qs - targets.detach()) ** 2).mean(dim=1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict_batch(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Run inference in chunks to avoid OOM on large arrays."""
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
        ckpt = torch.load(path, map_location=self.device)
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

        # Random start index each episode (avoids always starting from index 0)
        start_idx = np.random.randint(0, n_train - steps_per_ep - 1)
        state     = X_tr[start_idx]

        for t in range(steps_per_ep):
            idx    = start_idx + t
            action = agent.act(state)
            reward = 1 if action == y_tr[idx] else -1

            total_reward += reward

            next_state = X_tr[idx + 1]
            done       = (t == steps_per_ep - 1)

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size, class_weights_dict)

        # Sync target network
        if e % agent.target_update_freq == 0:
            agent.update_target_network()

        ep_time = time.time() - t0
        avg_r   = total_reward / steps_per_ep

        cumulative_rewards.append(total_reward)
        average_rewards.append(avg_r)
        time_history.append(ep_time)

        # Evaluate on test set
        y_pred_prob = agent.predict_batch(X_test)
        y_pred      = np.argmax(y_pred_prob, axis=1)
        ep_f1       = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_history.append(ep_f1)

        conv = abs(average_rewards[-1] - average_rewards[-2]) if e > 0 else 0.0
        print(f"Ep {e+1:03d}/{episodes} | Reward: {total_reward:+.0f} | "
              f"Avg: {avg_r:+.3f} | F1: {ep_f1:.4f} | "
              f"ε: {agent.epsilon:.3f} | Δ: {conv:.4f} | t: {ep_time:.1f}s")

        # Save best checkpoint
        if ep_f1 > best_f1:
            best_f1    = ep_f1
            no_improve = 0
            agent.save(f"{out_dir}/best_dqn_model.pt")
            print(f"  ✓ New best F1 = {best_f1:.4f} — model saved")
        else:
            no_improve += 1

        # Detailed report every 20 episodes
        if (e + 1) % 20 == 0:
            print(f"\n--- Episode {e+1} Report ---")
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred,
                  target_names=["Benign", "Attack"], zero_division=0))

        # Early stopping
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
    # Load best checkpoint
    agent.load(f"{out_dir}/best_dqn_model.pt")

    print("\n=== TRAINING SET ===")
    train_prob = agent.predict_batch(X_tr)
    train_pred = np.argmax(train_prob, axis=1)
    print(classification_report(y_tr, train_pred,
          target_names=["Benign", "Attack"], zero_division=0))
    print(f"AUC-ROC (train): {roc_auc_score(y_tr, train_pred):.4f}")

    print("\n=== TEST SET (Zero-Day Attacks) ===")
    test_prob  = agent.predict_batch(X_test)
    test_pred  = np.argmax(test_prob, axis=1)
    print(classification_report(y_test, test_pred,
          target_names=["Benign", "Attack"], zero_division=0))
    print(f"AUC-ROC  (test): {roc_auc_score(y_test, test_pred):.4f}")
    print(f"Accuracy (test): {accuracy_score(y_test, test_pred):.4f}")
    print(f"F1       (test): {f1_score(y_test, test_pred, average='weighted', zero_division=0):.4f}")

    return test_pred


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="ZDAD — Zero-Day Attack Detector (PyTorch)")
    parser.add_argument("--data",     required=True,  help="Path to NF-UQ-NIDS CSV file")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--steps",    type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--out",      default="deployment")
    args = parser.parse_args()

    # GPU setup
    device = configure_device()

    # Pipeline
    data, label_encoder = load_and_preprocess(args.data)
    joblib.dump(label_encoder, f"{args.out}/label_encoder.pkl") if os.makedirs(args.out, exist_ok=True) or True else None

    train_data, test_data = split_zero_day(data)

    X_tr, y_tr, X_test, y_test, feature_cols, scaler = prepare_features(
        train_data, test_data, data, args.out
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
