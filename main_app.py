import time, os, json, io, joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

from utils import load_credit_data, time_based_split, FEATURE_COLS, TARGET_COL, metrics, psi, kl_divergence

st.set_page_config(page_title="SentinelFlow – Trustworthy Risk Scoring", layout="wide")
st.title("💳 SentinelFlow – Trustworthy Risk Scoring")
st.caption("Calibrated probabilities • uncertainty-aware review • transparent drivers • drift & novelty monitoring.")

# -------- Sidebar --------
with st.sidebar:
    st.header("1) Data")
    data_file = st.file_uploader("Upload creditcard.csv", type=["csv"])
    speed = st.slider("Stream speed (transactions/sec)", 1, 50, 20)

    st.header("2) Policy — Threshold & Uncertainty Zone")
    fp_cost = st.number_input("False positive cost", min_value=0.0, value=1.0, step=0.5)
    fn_cost = st.number_input("False negative cost", min_value=0.0, value=10.0, step=0.5)
    review_band = st.slider("Abstain band width (± around threshold)", 0.0, 0.2, 0.01, 0.005)

    st.header("3) Training")
    retrain = st.checkbox("Retrain models on upload", value=True)
    start_btn = st.button("Start / Retrain")

    st.header("4) Analyst Review Queue")
    label_delay = st.slider("Simulated label delay (seconds)", 0, 120, 10)
    process_labels_btn = st.button("Process arrived labels")

placeholder = st.empty()
explain_placeholder = st.empty()
summary_placeholder = st.empty()
queue_placeholder = st.empty()
drift_placeholder = st.empty()
download_placeholder = st.empty()

def fmt_seconds(s: int) -> str:
    s = int(max(0, s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"

@st.cache_resource
def load_or_train(df: pd.DataFrame, retrain: bool):
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.utils import class_weight
    from sklearn.ensemble import IsolationForest

    if (not retrain) and os.path.exists("models/model_calibrated.joblib"):
        model = joblib.load("models/model_calibrated.joblib")
        novelty = joblib.load("models/novelty_iforest.joblib") if os.path.exists("models/novelty_iforest.joblib") else None
        with open("models/baseline.json") as f:
            baseline = json.load(f)
        with open("models/artifacts.json") as f:
            artifacts = json.load(f)
        return model, novelty, baseline, artifacts

    # Train from scratch
    from train_full import main as train_main
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tmp.csv")
        df.to_csv(path, index=False)
        train_main(path, out_dir="models")

    model = joblib.load("models/model_calibrated.joblib")
    novelty = joblib.load("models/novelty_iforest.joblib")
    with open("models/baseline.json") as f:
        baseline = json.load(f)
    with open("models/artifacts.json") as f:
        artifacts = json.load(f)
    return model, novelty, baseline, artifacts

def build_shap_explainer(model):
    try:
        base_model = model
        if hasattr(model, "calibrated_classifiers_"):
            cc = model.calibrated_classifiers_[0]
            if hasattr(cc, "base_estimator"):
                base_model = cc.base_estimator
            elif hasattr(cc, "estimator"):
                base_model = cc.estimator
        elif hasattr(model, "base_estimator"):
            base_model = model.base_estimator
        return shap.TreeExplainer(base_model)
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")
        return None

def compute_cost_threshold(y_valid, p_valid, fp_cost, fn_cost):
    ths = np.linspace(0,1,401)
    best_t, best_cost = 0.5, float("inf")
    for t in ths:
        y_hat = (p_valid >= t).astype(int)
        fp = ((y_hat==1)&(y_valid==0)).sum()
        fn = ((y_hat==0)&(y_valid==1)).sum()
        cost = fp*fp_cost + fn*fn_cost
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t, best_cost

def drift_panel(baseline, recent_scores, recent_amounts):
    score_base = np.array(baseline["score_valid"])
    amt_base = np.array(baseline["amount_valid"])
    psi_score = psi(score_base, np.array(recent_scores), bins=10) if len(recent_scores)>0 else 0.0
    psi_amt = psi(amt_base, np.array(recent_amounts), bins=10) if len(recent_amounts)>0 else 0.0
    kl_score = kl_divergence(score_base, np.array(recent_scores), bins=20) if len(recent_scores)>0 else 0.0
    with drift_placeholder.container():
        st.subheader("📈 Drift & Stability monitor")
        st.write({"PSI(score)": round(psi_score,4), "PSI(amount)": round(psi_amt,4), "KL(score)": round(kl_score,4)})
        msg = None
        if psi_score > 0.3 or psi_amt > 0.3:
            msg = "High drift detected – retraining recommended."
        elif psi_score > 0.2 or psi_amt > 0.2:
            msg = "Moderate drift – monitor closely."
        if msg:
            st.warning(msg)

def init_session():
    if "review_queue" not in st.session_state:
        st.session_state["review_queue"] = []  # list of dicts
    if "decisions" not in st.session_state:
        st.session_state["decisions"] = []

def stream(df: pd.DataFrame, model, novelty_model, baseline, threshold: float, review_band: float, speed: int):
    init_session()
    splits = time_based_split(len(df))
    test_df = df.iloc[splits.test_idx].copy()
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values
    proba = model.predict_proba(X_test)[:,1]

    # novelty score: IsolationForest gives -1 (anomaly) or 1 (normal) via predict; decision_function for score
    novelty_score = None
    if novelty_model is not None:
        novelty_score = novelty_model.decision_function(X_test)  # higher = more normal; we invert for "novelty"
        novelty_score = -novelty_score

    explainer = build_shap_explainer(model)
    stats = {"tp":0,"fp":0,"tn":0,"fn":0,"review":0}
    total = len(test_df)
    pb = st.progress(0)
    start_time = time.time()

    recent_scores = []
    recent_amounts = []

    for i in range(total):
        p = float(proba[i])
        true = int(y_test[i])
        x_row = X_test[i:i+1]
        nv = float(novelty_score[i]) if novelty_score is not None else None

        decision = "approve"
        if abs(p - threshold) <= review_band:
            decision = "review"
            stats["review"] += 1
        elif p >= threshold:
            decision = "block"

        if decision == "block":
            if true == 1: stats["tp"] += 1
            else: stats["fp"] += 1
        elif decision == "approve":
            if true == 1: stats["fn"] += 1
            else: stats["tn"] += 1

        # Add to review queue with label delay simulation info
        if decision == "review":
            st.session_state["review_queue"].append({
                "time": int(test_df.iloc[i]["Time"]),
                "amount": float(test_df.iloc[i]["Amount"]),
                "prob": p,
                "novelty": nv,
                "true_label": int(true),
                "insert_ts": time.time(),
                "label_available": False,
                "final_label": None
            })

        # Save decision for download
        st.session_state["decisions"].append({
            "Time": int(test_df.iloc[i]["Time"]),
            "Amount": float(test_df.iloc[i]["Amount"]),
            "TrueLabel": int(true),
            "RiskProb": p,
            "Novelty": nv,
            "Decision": decision
        })

        processed = i + 1
        remaining = total - processed
        eta_seconds = int(remaining / max(1e-9, float(speed)))
        elapsed = int(time.time() - start_time)

        with placeholder.container():
            c1, c2, c3 = st.columns([2,2,1])
            with c1:
                st.subheader("Live transaction")
                st.write(test_df.iloc[i][["Time","Amount"]].to_frame().T)
                st.metric("Risk probability", f"{p:.3f}")
                if nv is not None:
                    st.metric("Novelty score (↑=more novel)", f"{nv:.3f}")
                st.write(f"Decision: **{decision.upper()}**  |  Threshold: {threshold:.2f} ± {review_band:.2f}")
                st.metric("Progress", f"{processed:,} / {total:,}")
                st.metric("ETA (hh:mm:ss)", fmt_seconds(eta_seconds))
                st.metric("Elapsed", fmt_seconds(elapsed))
            with c2:
                st.subheader("Cumulative stats (test slice)")
                st.write(stats)
                st.caption("tp: true positives, fp: false positives, tn: true negatives, fn: false negatives")
            with c3:
                st.subheader("Test set metrics (static)")
                st.json({"roc_auc": "see train metrics", "note": "demo stream"})
        pb.progress(int(processed * 100 / total))

        # Explanations
        with explain_placeholder.container():
            st.subheader("Reason codes (SHAP)")
            if explainer is not None:
                try:
                    sv = explainer.shap_values(x_row)
                    if isinstance(sv, list) and len(sv)==2:
                        sv = sv[1]
                    sv = np.array(sv).flatten()
                    top_idx = np.argsort(np.abs(sv))[-5:][::-1]
                    top = [(FEATURE_COLS[j], float(sv[j])) for j in top_idx]
                    st.write(pd.DataFrame(top, columns=["Feature","SHAP value (impact on risk)"]))
                except Exception as e:
                    st.info(f"SHAP explanation unavailable: {e}")
            else:
                st.info("SHAP explainer not available for this model.")

        # Drift monitoring (rolling)
        recent_scores.append(p)
        recent_amounts.append(float(test_df.iloc[i]["Amount"]))
        if len(recent_scores) >= 200 or i == total-1:
            drift_panel(baseline, recent_scores, recent_amounts)
            recent_scores.clear(); recent_amounts.clear()

        time.sleep(1.0/float(speed))

    st.success("Streaming complete.")
    # Download
    dec_df = pd.DataFrame(st.session_state["decisions"])
    csv_bytes = dec_df.to_csv(index=False).encode("utf-8")
    download_placeholder.download_button("Download decisions CSV", data=csv_bytes, file_name="decisions.csv", mime="text/csv")

def show_review_queue():
    init_session()
    st.subheader("🧾 Review queue")
    dfq = pd.DataFrame(st.session_state["review_queue"])
    if dfq.empty:
        st.info("No items in review.")
        return
    st.dataframe(dfq[["time","amount","prob","novelty","label_available","final_label"]])

def process_arrived_labels(delay_sec: int):
    """Simulate label arrival by revealing the true label after delay for oldest items without a label."""
    now = time.time()
    changed = 0
    for item in st.session_state.get("review_queue", []):
        if (not item["label_available"]) and (now - item["insert_ts"] >= delay_sec):
            # reveal label (simulation: use true_label)
            item["label_available"] = True
            item["final_label"] = item["true_label"]
            changed += 1
    return changed

# -------- Main flow --------
if data_file is None:
    st.info("Upload `creditcard.csv` to start. Columns required: Time, V1..V28, Amount, Class.")
else:
    df = load_credit_data(data_file)
    model, novelty, baseline, artifacts = load_or_train(df, retrain)
    st.success("Models ready.")
    st.write("Validation metrics:", artifacts["metrics_valid"])

    # auto threshold from validation
    splits = time_based_split(len(df))
    X_valid = df.iloc[splits.valid_idx][FEATURE_COLS].values
    y_valid = df.iloc[splits.valid_idx][TARGET_COL].values
    p_valid = model.predict_proba(X_valid)[:,1]
    auto_thr, _ = compute_cost_threshold(y_valid, p_valid, fp_cost, fn_cost)

    thr = st.slider("Decision threshold", 0.0, 1.0, float(auto_thr), 0.01)

    # Tabs for Stream / Review / Summary
    tab1, tab2 = st.tabs(["▶️ Stream", "🧾 Review queue"])

    with tab1:
        if start_btn:
            stream(df, model, novelty, baseline, thr, review_band, speed)

    with tab2:
        show_review_queue()
        if process_labels_btn:
            n = process_arrived_labels(label_delay)
            st.success(f"Processed {n} items with newly arrived labels (simulated).")
