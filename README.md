# ZDAD — Zero-Day Attack Detection in NIDS

> Zero-day cyberattack detection using Deep Q-Network (DQN) with LSTM layers on the NF-UQ-NIDS dataset. The model is trained without seeing certain attack types (Shellcode, Backdoor, Ransomware, etc.) and tested on them to evaluate generalization. Includes SMOTE oversampling and LIME explainability.

📄 **Paper:** [IEEE Access — Adaptive Defense: Zero-Day Attack Detection in NIDS With Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/11063272)

---

## Project Structure

```
zdad-project/
├── train.py            # Full training pipeline (data → model → evaluation)
├── predict.py          # Inference module (IDSPredictor class)
├── api.py              # FastAPI REST server
├── test_predict.py     # API smoke-test script
├── requirements.txt
├── Dockerfile
└── deployment/         # Auto-created after training
    ├── best_dqn_model.keras
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── training_curves.png
    └── confusion_matrix.png
```

---

## How It Works

### Zero-Day Simulation
The model is **deliberately kept blind** to certain attack types during training:
`Shellcode`, `Brute Force`, `Theft`, `Ransomware`, `Backdoor`

These appear **only in the test set**, measuring how well the DRL agent generalizes to threats it has never seen — the core zero-day detection challenge.

### Model Architecture
```
Dense(128) + BatchNorm + Dropout
    → Reshape(1, 128)
    → LSTM(128) → LSTM(64) → LSTM(64)
    → Dense(64) + Dropout
    → Dense(2, softmax)   # Benign / Attack
```

LSTM layers capture **temporal and sequential patterns** in network traffic flows.

### Key Improvements over the Original Notebook
| Issue | Fix |
|---|---|
| `LabelEncoder` re-fit inside chunk loop → inconsistent label mapping | Two-pass loading: scan all labels first, fit encoder once |
| Two separate `StandardScaler` objects → LIME used the wrong one | Single scaler, saved to `deployment/scaler.pkl` |
| Always starts from `features_train[0]` each episode | Random start index each episode |
| Uniform experience replay | Prioritized Experience Replay (PER) |
| No target network → unstable Q-values | Double DQN with periodic target-network sync |
| One `model.predict()` call per sample → very slow | Vectorized batch prediction |

---

## Setup

```bash
pip install -r requirements.txt
```

For GPU training, install the CUDA-compatible TensorFlow build:
```bash
pip install tensorflow[and-cuda]   # Linux/WSL2
# or
pip install tensorflow-gpu         # Windows (TF ≤ 2.10)
```

Verify GPU is detected:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Training

```bash
# Basic (auto-detects GPU)
python train.py --data path/to/NF-UQ-NIDS.csv

# Full options
python train.py \
    --data     path/to/NF-UQ-NIDS.csv \
    --episodes 100 \
    --batch    64 \
    --steps    200 \
    --patience 15 \
    --out      deployment \
    --gpu
```

| Argument | Default | Description |
|---|---|---|
| `--data` | required | Path to NF-UQ-NIDS CSV |
| `--episodes` | 100 | Max training episodes |
| `--batch` | 64 | Replay batch size |
| `--steps` | 200 | Steps per episode |
| `--patience` | 15 | Early stopping (no F1 improvement) |
| `--out` | `deployment` | Output directory for model artifacts |
| `--gpu` | false | Error if no GPU found (instead of silently using CPU) |

Training auto-detects and uses your GPU. Memory growth is enabled so TensorFlow doesn't claim all VRAM at startup.

---

## Inference

```python
from predict import IDSPredictor

predictor = IDSPredictor()

result = predictor.predict({
    "FLOW_DURATION_MILLISECONDS": 500,
    "TOTAL_FWDPACKETS": 12,
    "TOTAL_BWDPACKETS": 8,
    # ... all feature columns
})

print(result)
# → {"label": 1, "confidence": 0.91, "verdict": "ATTACK", "proba": [0.09, 0.91]}
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
| `/predict/batch` | POST | Batch flow prediction |

Interactive docs: http://localhost:8000/docs

Test the API:
```bash
python test_predict.py
```

---

## Docker

```bash
docker build -t zero-day-ids .
docker run -p 8000:8000 zero-day-ids
```

---

## Dataset

[NF-UQ-NIDS](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) — University of Queensland network intrusion detection dataset.

> The CSV file is not included in this repo (too large). Download it separately and pass the path via `--data`.

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
