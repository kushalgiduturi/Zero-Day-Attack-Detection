"""
api.py — FastAPI REST server for Zero-Day IDS (PyTorch)

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from predict import IDSPredictor

app = FastAPI(
    title="Zero-Day IDS API (PyTorch)",
    description="DQN + LSTM intrusion detection for NF-UQ-NIDS network flows",
    version="2.0.0",
)

predictor = IDSPredictor()


class FlowRequest(BaseModel):
    features: Dict[str, float]

class BatchFlowRequest(BaseModel):
    flows: List[Dict[str, float]]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: FlowRequest):
    try:
        return predictor.predict(req.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(req: BatchFlowRequest):
    try:
        return {"predictions": predictor.predict_batch(req.flows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
