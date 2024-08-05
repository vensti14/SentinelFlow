import argparse, json, os, joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import class_weight
from sklearn.ensemble import IsolationForest
from utils import load_credit_data, time_based_split, FEATURE_COLS, TARGET_COL, metrics

def main(data_path: str, out_dir: str = "models"):
    os.makedirs(out_dir, exist_ok=True)
    df = load_credit_data(data_path)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    splits = time_based_split(len(df))
    X_train, y_train = X[splits.train_idx], y[splits.train_idx]
    X_valid, y_valid = X[splits.valid_idx], y[splits.valid_idx]
    X_test, y_test   = X[splits.test_idx],  y[splits.test_idx]

    # Class weights for imbalance
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}

    base = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8, class_weight=cw, random_state=42
    )
    base.fit(X_train, y_train)

    calib = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calib.fit(X_valid, y_valid)

    # Novelty model trained on non-fraud in train only
    iforest = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    iforest.fit(X_train[y_train==0])

    # Metrics
    y_prob_valid = calib.predict_proba(X_valid)[:,1]
    y_prob_test  = calib.predict_proba(X_test)[:,1]
    m_valid = metrics(y_valid, y_prob_valid)
    m_test  = metrics(y_test,  y_prob_test)

    # Save baseline distributions from validation slice for drift monitoring
    baseline = {
        "score_valid": y_prob_valid.tolist(),
        "amount_valid": df.iloc[splits.valid_idx]["Amount"].tolist()
    }

    artifacts = {
        "feature_cols": FEATURE_COLS,
        "metrics_valid": m_valid,
        "metrics_test": m_test
    }

    joblib.dump(calib, os.path.join(out_dir, "model_calibrated.joblib"))
    joblib.dump(iforest, os.path.join(out_dir, "novelty_iforest.joblib"))
    with open(os.path.join(out_dir, "baseline.json"), "w") as f:
        json.dump(baseline, f)
    with open(os.path.join(out_dir, "artifacts.json"), "w") as f:
        json.dump(artifacts, f, indent=2)

    print("Saved models and artifacts to", out_dir)
    print(json.dumps(artifacts, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="models")
    args = p.parse_args()
    main(args.data_path, args.out_dir)
