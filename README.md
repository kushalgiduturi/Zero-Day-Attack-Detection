# ZDAD — Zero-Day Attack Detection in NIDS

> Zero-day cyberattack detection using Deep Q-Network (DQN) with LSTM layers on the NF-UQ-NIDS dataset. The model is trained without seeing certain attack types (Shellcode, Backdoor, Ransomware, etc.) and tested on them to evaluate generalization. Includes SMOTE oversampling and LIME explainability.

📄 **Paper:** [IEEE Access 2025 — Adaptive Defense: Zero-Day Attack Detection in NIDS With Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/11063272)

---

## Changelog

### v2.0 — Multi-Dataset Cross-Validation *(current)*
- Cross-dataset support: NF-UQ-NIDS v2, CIC-IDS2017, UNSW-NB15, NSL-KDD
- Universal feature mapper — auto-maps any dataset's column names to 20 core features
- Dataset adapters for 4 major benchmark datasets + auto-detect mode
- Replaced slow SMOTE with fast random oversampling (seconds vs hours)
- PyTorch rewrite with full CUDA 13.0 support (tested on RTX 4060 Laptop GPU)
- Double DQN with periodic target network sync for stable Q-value targets
- Prioritized Experience Replay — learns more from surprising/difficult flows
- Gradient clipping to prevent LSTM training instability
- REST API with single + batch prediction endpoints
- Docker support for containerised deployment

### v1.0 — Initial Release
- DQN + stacked LSTM model on NF-UQ-NIDS v2 dataset
- Zero-day simulation: 5 attack types hidden during training
- KMeans-SMOTE oversampling for class imbalance
- TensorFlow implementation
- LIME explainability for individual predictions
- FastAPI REST endpoint

---

## Results

### v2.0 — Cross-Dataset Performance

| Dataset | Rows | Features Used | Episodes | F1 Score | Accuracy | AUC-ROC |
|---|---|---|---|---|---|---|
| CIC-IDS2017 Friday DDoS | 225,745 | 20/20 | 36 | **0.9928** | 99.28% | 0.9924 |
| NF-UQ-NIDS v2 | 4,276,737 | 41/41 | 43 | **0.9120** | 91.20% | — |
| NSL-KDD Train | 125,973 | 9/20 | 45 | **0.8931** | 89.44% | 0.8906 |
| NSL-KDD Test | 22,544 | 9/20 | 17 | **0.8513** | 85.38% | 0.8533 |
| UNSW-NB15 Train | 175,341 | 6/20 | 52 | **0.8396** | 84.06% | 0.8301 |
| UNSW-NB15 Test | 82,332 | 6/20 | 100 | **0.7933** | 79.37% | 0.7974 |

### v1.0 — NF-UQ-NIDS v2 Only

| Metric | Value |
|---|---|
| F1 Score (zero-day test) | 0.9120 |
| Attack Precision | 99% |
| Attack Recall | 84% |
| Accuracy | 91.2% |

---

## What is Zero-Day Detection?

Traditional IDS systems use **signatures** — they only detect known attacks. Zero-day attacks are new, unknown threats with no existing signature. This model detects attacks based on **behavior**, not signatures — so it catches threats even without prior knowledge of them.

The model is **deliberately kept blind** to certain attack types during training:
`Shellcode`, `Brute Force`, `Theft`, `Ransomware`, `Backdoor`

These appear **only in the test set**, measuring how well the DRL agent generalizes to threats it has never seen.

---

## Model Architecture

```
Network Flow Features
        ↓
Dense(128) + BatchNorm + Dropout
        ↓
Reshape → (seq_len=1, hidden=128)
        ↓
LSTM(128) → LSTM(64) → LSTM(64)
        ↓
Dense(64) + Dropout
        ↓
Dense(2, softmax) → Benign / Attack
```

LSTM layers capture **sequential and temporal patterns** in network traffic flows — the key to generalizing to unseen attack types.

---

## Project Structure

```
zdad-project/
├── train.py              # Full training pipeline
├── predict.py            # Inference module (IDSPredictor class)
├── api.py                # FastAPI REST server
├── test_predict.py       # API smoke-test script
├── feature_mapper.py     # Universal feature mapper
├── universal_features.py # 20 core features + aliases
├── dataset_adapters.py   # NSL-KDD, UNSW-NB15, CIC-IDS2017 adapters
├── requirements.txt
├── Dockerfile
└── deployment/           # Auto-created after training
    ├── best_dqn_model.pt
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── feature_cols.pkl
    ├── feature_mapper.pkl
    ├── training_curves.png
    └── confusion_matrix.png
```

---

## Setup

```bash
# Install PyTorch with CUDA 12.4 (works with CUDA 13.0 drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

Verify GPU detection:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## Training

```bash
# NF-UQ-NIDS v2 (native format)
python train.py --data "path/to/NF-UQ-NIDS-v2.csv"

# CIC-IDS2017
python train.py --data "path/to/Friday-DDoS.csv" --dataset cic-ids2017

# UNSW-NB15
python train.py --data "path/to/UNSW_NB15_training-set.csv" --dataset unsw-nb15

# NSL-KDD
python train.py --data "path/to/kdd_train.csv" --dataset nsl-kdd

# Unknown dataset (auto-map)
python train.py --data "path/to/unknown.csv" --dataset auto
```

### All training arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | required | Path to dataset CSV |
| `--dataset` | `nf-uq-nids` | Dataset format |
| `--episodes` | 100 | Max training episodes |
| `--batch` | 64 | Replay batch size |
| `--steps` | 200 | Steps per episode |
| `--patience` | 15 | Early stopping patience |
| `--out` | `deployment` | Output directory |

---

## Inference

```python
from predict import IDSPredictor

predictor = IDSPredictor()

# Works with any dataset format — auto-detected
result = predictor.predict({
    "FLOW_DURATION_MILLISECONDS": 500,
    "TOTAL_FWDPACKETS": 12,
    "TOTAL_BWDPACKETS": 8,
})

print(result)
# → {"label": 1, "confidence": 0.94, "verdict": "ATTACK", "proba": [0.06, 0.94]}
```

---

## REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/predict` | POST | Single flow prediction |
| `/predict/batch` | POST | Batch prediction |

Interactive docs: http://localhost:8000/docs

---

## Docker

```bash
docker build -t zdad-ids .
docker run -p 8000:8000 --gpus all zdad-ids
```

---

## Supported Datasets

| Dataset | Flag | Source |
|---|---|---|
| NF-UQ-NIDS v2 | `nf-uq-nids` | [University of Queensland](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) |
| CIC-IDS2017 / 2018 | `cic-ids2017` | [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) |
| UNSW-NB15 | `unsw-nb15` | [University of New South Wales](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| NSL-KDD | `nsl-kdd` | [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html) |
| Any unknown | `auto` | Auto-mapped via universal features |

---

## Key Technical Improvements (v1 → v2)

| Issue in v1 | Fix in v2 |
|---|---|
| TensorFlow — no CUDA 13.0 support on Windows | PyTorch — native CUDA 13.0 support |
| Single dataset only | 5 datasets, 3 institutions |
| LabelEncoder re-fit per chunk | Two-pass loading, fit once |
| Two separate StandardScalers | Single scaler saved to deployment |
| SMOTE — hours on large datasets | Random oversampling — seconds |
| Always starts episode from index 0 | Random start index each episode |
| No target network | Double DQN with periodic sync |
| One predict() call per sample | Vectorized batch prediction |
| Hardcoded Windows path | `--data` CLI argument |

---

## Citation

```bibtex
@article{alam2025zdad,
  author  = {Alam, K. and Fahad Monir, M. and Junayed Hossain, M. and Shorif Uddin, M. and Habib, Md. T.},
  title   = {Adaptive Defense: Zero-Day Attack Detection in NIDS With Deep Reinforcement Learning},
  journal = {IEEE Access},
  year    = {2025},
  volume  = {13},
  pages   = {116345--116361},
  doi     = {10.1109/ACCESS.2025.3585445}
}
```