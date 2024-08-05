# SentinelFlow – Trustworthy Risk Scoring

End-to-end, trustworthy-ML fraud triage system with:
- Calibrated probabilities (LightGBM + isotonic)
- Selective prediction (review band)
- local feature contributions (SHAP).
- Novelty detector (IsolationForest)
- Drift monitor (PSI/KL) with retrain suggestion
- Review queue with simulated label delay
- Streamlit UI + FastAPI scorer

## Quickstart
```bash
pip install -r requirements.txt
python train_full.py --data_path data/creditcard.csv
streamlit run app_pro.py
```
API:
```bash
uvicorn api:app --reload
# POST {"features": {...}} to /score
```
## Design decisions
```bash
- **Calibrated probabilities** (isotonic) so 0.20 ≈ 20% risk over time.
- **Cost-aware policy**: choose threshold to minimize FP/FN cost; use an **uncertainty zone** to defer borderline cases.
- **Local explanations**: show top feature contributions per decision to aid analyst review.
- **Stability**: monitor drift (PSI/KL) on scores and key inputs; recommend recalibration when drift rises.
- **Safety**: no PII; demo dataset only; artifacts versioned for audit.
```


See `MODEL_CARD.md` for details.
