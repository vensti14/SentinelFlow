import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"

@dataclass
class SplitIndices:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray

def load_credit_data(path_or_buf) -> pd.DataFrame:
    df = pd.read_csv(path_or_buf)
    missing = [c for c in FEATURE_COLS + ["Time", TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.sort_values("Time").reset_index(drop=True)
    return df

def time_based_split(n: int, train_frac=0.6, valid_frac=0.2) -> SplitIndices:
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    idx = np.arange(n)
    return SplitIndices(train_idx=idx[:train_end], valid_idx=idx[train_end:valid_end], test_idx=idx[valid_end:])

def metrics(y_true, y_prob) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    def safe(fn):
        try:
            return float(fn)
        except Exception:
            return float("nan")
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = float("nan")
    try:
        prauc = average_precision_score(y_true, y_prob)
    except Exception:
        prauc = float("nan")
    try:
        brier = brier_score_loss(y_true, y_prob)
    except Exception:
        brier = float("nan")
    return {"roc_auc": safe(roc), "pr_auc": safe(prauc), "brier": safe(brier)}

def psi(base: np.ndarray, curr: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D arrays (continuous)."""
    base = np.asarray(base); curr = np.asarray(curr)
    edges = np.quantile(base, np.linspace(0,1,bins+1))
    # ensure unique edges
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    base_hist, _ = np.histogram(base, bins=edges)
    curr_hist, _ = np.histogram(curr, bins=edges)
    # Add small value to avoid 0 div/log
    base_pct = np.clip(base_hist / max(1, base_hist.sum()), 1e-6, 1)
    curr_pct = np.clip(curr_hist / max(1, curr_hist.sum()), 1e-6, 1)
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))

def kl_divergence(base: np.ndarray, curr: np.ndarray, bins: int = 20) -> float:
    base = np.asarray(base); curr = np.asarray(curr)
    edges = np.quantile(np.concatenate([base, curr]), np.linspace(0,1,bins+1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    b_hist, _ = np.histogram(base, bins=edges)
    c_hist, _ = np.histogram(curr, bins=edges)
    p = np.clip(b_hist / max(1,b_hist.sum()), 1e-12, 1)
    q = np.clip(c_hist / max(1,c_hist.sum()), 1e-12, 1)
    return float(np.sum(p * np.log(p / q)))
