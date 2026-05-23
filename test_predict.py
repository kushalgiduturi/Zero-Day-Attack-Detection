"""
test_predict.py — API smoke test

Run server first:
    uvicorn api:app --host 0.0.0.0 --port 8000

Then:
    python test_predict.py
"""

import requests

BASE_URL = "http://localhost:8000"

print("Health:", requests.get(f"{BASE_URL}/health").json())

sample_flow = {
    "features": {
        "FLOW_DURATION_MILLISECONDS": 500,
        "TOTAL_FWDPACKETS": 12,
        "TOTAL_BWDPACKETS": 8,
        "TOTAL_LENGTH_OF_FWD_PACKETS": 2048,
        "TOTAL_LENGTH_OF_BWD_PACKETS": 512,
        # add all feature columns from your dataset
    }
}

response = requests.post(f"{BASE_URL}/predict", json=sample_flow)
print("\nSingle prediction:", response.json())

batch = {"flows": [sample_flow["features"], {**sample_flow["features"], "FLOW_DURATION_MILLISECONDS": 9999}]}
batch_response = requests.post(f"{BASE_URL}/predict/batch", json=batch)
print("\nBatch predictions:")
for i, pred in enumerate(batch_response.json()["predictions"]):
    print(f"  Flow {i+1}: {pred['verdict']} ({pred['confidence']:.2%})")
