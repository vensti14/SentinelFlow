# SentinelFlow – Trustworthy Risk Scoring

## Overview
- **Goal:** Score card-not-present transactions for fraud risk and route to *approve / review / block*.
- **Model:** LightGBM classifier with **isotonic calibration** for probabilities; **IsolationForest** for novelty.
- **Data:** ULB credit card dataset (anonymized), time-based split 60/20/20.
- **Intended use:** Educational prototype and internal experimentation; **not** production.

## Data
- **Features:** V1–V28 (PCA components from anonymized raw fields) + `Amount`.
- **Target:** `Class` (1 = fraud, 0 = non-fraud).
- **Preprocessing:** Sorted by `Time`; no leakage across splits.

## Evaluation
- **Metrics (validation/test):** ROC-AUC, PR-AUC, Brier (calibration).  
- **Selective prediction:** abstain (review) band around threshold.  
- **Cost-aware threshold:** chosen from validation for given FP/FN costs.

## Calibration
- Method: **Isotonic** on validation (prefit).  
- Report reliability diagram and ECE (if added).  
- Note calibration can **drift** over time.

## Robustness & Drift
- **Drift monitoring:** PSI/KL on score and `Amount` vs validation baseline; banner when PSI > 0.2/0.3.  
- **Novelty detection:** IsolationForest flags unusual patterns; shown alongside main probability.

## Explainability
- **Local:** SHAP reason codes for each decision.  
- **Global:** Mean |SHAP| feature importance.  
- Limitations: explanations reflect model correlations, **not causation**; can vary with small perturbations.

## Fairness & Ethics
- Dataset is anonymized; avoid sensitive features.  
- Segment checks recommended (merchant type/region/time-of-day) for FP/FN disparities.  
- Add an appeal process and human-in-the-loop review in real deployments.

## Risks & Limitations
- Non-stationarity (fraud tactics change).  
- Label delays and noise.  
- Small/imbalanced data → unstable metrics on tiny windows.  
- Prototype UI (Streamlit) is not production SLOs or security-hardened.

## Usage
- **Training:** `python train_full.py --data_path data/creditcard.csv`  
- **App:** `streamlit run app_pro.py`  
- **API:** `uvicorn api:app --reload` then POST to `/score` with JSON:  
  ```json
  {"features": {"V1":0.1, "...":"...", "V28":-0.2, "Amount": 25.0}}
  ```

## Versioning
- Artifacts saved under `models/` with metrics in `artifacts.json` and baseline distributions in `baseline.json`.

## Authors & Contact
- Maintainer: 221B, prasadpp015@gmail.com.
