from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np
from utils import FEATURE_COLS

app = FastAPI(title="SentinelFlow – Trustworthy Risk Scoring")

model = joblib.load("models/model_calibrated.joblib")
try:
    novelty = joblib.load("models/novelty_iforest.joblib")
except Exception:
    novelty = None

class ScoreRequest(BaseModel):
    features: dict  # keys must include V1..V28 and Amount

class ScoreResponse(BaseModel):
    probability: float
    novelty: float | None = None

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    x = np.array([[req.features.get(c, 0.0) for c in FEATURE_COLS]])
    p = float(model.predict_proba(x)[:,1][0])
    nv = None
    if novelty is not None:
        nv = float(-novelty.decision_function(x)[0])  # higher = more novel
    return ScoreResponse(probability=p, novelty=nv)
