"""
End-to-End Fair ML Pipeline for Public Welfare Decisions
Simulates SNAP/Medicaid/Unemployment datasets and runs full fairness analysis.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import json, random
random.seed(42)
np.random.seed(42)


import numpy as np

def compute_reweigh_weights(race_labels, y_labels):
    """Reweighing algorithm (Kamiran & Calders 2012)."""
    n = len(race_labels)
    weights = np.ones(n)

    for r in np.unique(race_labels):
        for y in [0, 1]:
            mask = (race_labels == r) & (y_labels == y)

            p_r  = (race_labels == r).mean()
            p_y  = (y_labels == y).mean()
            p_ry = mask.mean()

            if p_ry > 0:
                weights[mask] = (p_r * p_y) / p_ry

    return weights

# ─────────────────────────────────────────────
# 1. DATASET 
# ─────────────────────────────────────────────
from data_loader import load_real_data, load_synthetic_data

USE_REAL_DATA = False

if USE_REAL_DATA:
    df = load_real_data()
else:
    df = load_synthetic_data()

print(df.head())
# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
from data_loader import load_real_data, load_synthetic_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 🔥 Toggle dataset
USE_REAL_DATA = True

# =========================
# 🔹 LOAD DATA
# =========================
if USE_REAL_DATA:
    df = load_real_data()
else:
    df = load_synthetic_data()

# =========================
# 🔹 CLEAN DATA
# =========================
df = df.fillna("Unknown")

# =========================
# 🔹 HANDLE TARGET (VERY IMPORTANT)
# =========================
if USE_REAL_DATA:
    # Clean income column
    df["income"] = df["income"].astype(str).str.strip()
    df["income"] = df["income"].replace({
        "<=50K.": "<=50K",
        ">50K.": ">50K"
    })

    # Keep only valid rows
    df = df[df["income"].isin(["<=50K", ">50K"])]

    # Convert to binary
    df["income"] = df["income"].map({
        "<=50K": 0,
        ">50K": 1
    })

    FEATURES = [
        "age", "fnlwgt", "education_num",
        "capital_gain", "capital_loss", "hours_per_week"
    ]
    TARGET = "income"

else:
    FEATURES = [
        "age","income","hh_size",
        "employed","disability","prior_benefit"
    ]
    TARGET = "truly_eligible"

# =========================
# 🔹 ENCODE CATEGORICAL
# =========================
df_encoded = df.copy()

for col in df_encoded.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# =========================
# 🔹 SPLIT FEATURES / TARGET
# =========================
X = df_encoded[FEATURES]
y = df_encoded[TARGET]

# =========================
# 🔹 SAVE SENSITIVE ATTRIBUTES (for fairness)
# =========================
race = df_encoded["race"]

if USE_REAL_DATA:
    gender = df_encoded["sex"]
else:
    gender = df_encoded["gender"]

# =========================
# 🔹 SCALE FEATURES
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 🔹 TRAIN-TEST SPLIT (FIXED)
# =========================
X_tr, X_te, y_tr, y_te, race_tr, race_te, gen_tr, gen_te = train_test_split(
    X_scaled,
    y,
    race,
    gender,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================
# 🔹 DEBUG PRINTS
# =========================
print("Preprocessing done ✅")
print("Classes distribution:\n", y.value_counts())
print("Train size:", len(X_tr))

RACE_LABELS = {0:"White",1:"Black",2:"Hispanic",3:"Asian"}
def fairness_metrics(y_true, y_pred, sensitive, labels):
    out = {}
    for g, lbl in labels.items():
        mask = sensitive == g
        if mask.sum() == 0: continue
        yt, yp = y_true[mask], y_pred[mask]
        tn,fp,fn,tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        tpr  = tp/(tp+fn) if (tp+fn)>0 else 0
        fpr  = fp/(fp+tn) if (fp+tn)>0 else 0
        sr   = yp.mean()
        acc  = (yt==yp).mean()
        ppv  = tp/(tp+fp) if (tp+fp)>0 else 0
        out[lbl] = {"selection_rate": round(sr,4), "tpr": round(tpr,4),
                    "fpr": round(fpr,4), "accuracy": round(acc,4),
                    "precision": round(ppv,4), "n": int(mask.sum())}
    # Disparate Impact (ratio to privileged group = White)
    priv_sr = out.get("White",{}).get("selection_rate",1)
    for lbl in out:
        sr = out[lbl]["selection_rate"]
        out[lbl]["disparate_impact"] = round(sr/priv_sr, 4) if priv_sr>0 else None
        out[lbl]["statistical_parity_diff"] = round(sr - priv_sr, 4)
    return out

fairness_by_model_race   = {}
fairness_by_model_gender = {}

# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, mdl in models.items():
    mdl.fit(X_tr, y_tr)

    y_pred = mdl.predict(X_te)

    # 🔥 Safe probability handling
    if hasattr(mdl, "predict_proba"):
        y_prob = mdl.predict_proba(X_te)[:, 1]
    else:
        y_prob = y_pred  # fallback

    results[name] = {
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_te, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_te, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_te, y_prob), 4),
    }
    
    fairness_by_model_race[name]   = fairness_metrics(y_te, y_pred, race_te, RACE_LABELS)
    
    GENDER_LABELS = {0: "Male", 1: "Female"}
    fairness_by_model_gender[name] = fairness_metrics(y_te, y_pred, gen_te, GENDER_LABELS)

    print(f"\n{name}:")
    print(f"Accuracy: {results[name]['accuracy']}")
    print(f"F1 Score: {results[name]['f1']}")
    print(f"AUC: {results[name]['auc']}")

# ─────────────────────────────────────────────
# 4. FAIRNESS METRICS (AIF360-style formulas)
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
# 5. BIAS MITIGATION (Reweighing + Threshold Calibration)
# ─────────────────────────────────────────────
race_tr_np = np.asarray(race_tr)
y_tr_np    = np.asarray(y_tr)

weights_tr = compute_reweigh_weights(race_tr_np, y_tr_np)

rf_mit = RandomForestClassifier(n_estimators=100, random_state=42)
rf_mit.fit(X_tr, y_tr, sample_weight=weights_tr)

yp_mit  = rf_mit.predict(X_te)
ypr_mit = rf_mit.predict_proba(X_te)[:,1]

mitigated_overall = {
    "accuracy":  round(accuracy_score(y_te, yp_mit), 4),
    "precision": round(precision_score(y_te, yp_mit, zero_division=0), 4),
    "recall":    round(recall_score(y_te, yp_mit, zero_division=0), 4),
    "f1":        round(f1_score(y_te, yp_mit, zero_division=0), 4),
    "auc":       round(roc_auc_score(y_te, ypr_mit), 4),
}

mitigated_fairness = fairness_metrics(y_te, yp_mit, race_te, RACE_LABELS)

print("\nMitigated RF:", mitigated_overall)
print("\nMitigated Fairness:", mitigated_fairness)

# ─────────────────────────────────────────────
from sklearn.inspection import permutation_importance
import numpy as np

# 🔥 Ensure model exists
rf_base = models["Random Forest"]

# =========================
# 🔹 GLOBAL FEATURE IMPORTANCE
# =========================
perm = permutation_importance(
    rf_base,
    X_te,
    y_te,
    n_repeats=10,
    random_state=42
)

shap_approx = {
    FEATURES[i]: round(float(perm.importances_mean[i]), 5)
    for i in range(len(FEATURES))
}

print("\nGlobal Feature Importance:")
print(shap_approx)


# =========================
# 🔹 GROUP-WISE IMPORTANCE (RACE)
# =========================
shap_by_race = {}

race_te_np = np.asarray(race_te)

for g, lbl in RACE_LABELS.items():
    mask = race_te_np == g

    if mask.sum() < 30:
        continue

    pi = permutation_importance(
        rf_base,
        X_te[mask],
        y_te[mask],
        n_repeats=5,
        random_state=42
    )

    shap_by_race[lbl] = {
        FEATURES[i]: round(float(pi.importances_mean[i]), 5)
        for i in range(len(FEATURES))
    }

print("\nFeature Importance by Race:")
print(shap_by_race)

# ─────────────────────────────────────────────
# 7. DIFFERENTIAL PRIVACY NOISE (Laplace mechanism)
# ─────────────────────────────────────────────
import numpy as np

def dp_aggregate(values, sensitivity=1.0, epsilon=1.0, seed=42):
    np.random.seed(seed)  # 🔥 reproducibility
    noise = np.random.laplace(0, sensitivity/epsilon, len(values))

    noisy = []
    for v, n in zip(values, noise):
        val = v + n

        # 🔥 clamp between 0 and 1 (important for probabilities)
        val = max(0, min(1, val))

        noisy.append(round(float(val), 4))

    return noisy
# ─────────────────────────────────────────────
# 8. CASE STUDY: WELFARE FRAUD DETECTION SIMULATION
# ─────────────────────────────────────────────
# Simulate a fraud-flag dataset (high-stakes: false positives harm legitimate beneficiaries)
def run_fraud_case_study():
    import numpy as np

    np.random.seed(99)
    n_fraud = 5000

    race_f = np.random.choice([0,1,2,3], n_fraud, p=[0.60,0.15,0.18,0.07])
    income_f = np.random.normal(24000, 10000, n_fraud).clip(0)
    income_f -= (race_f==1)*3000 + (race_f==2)*2000

    actual_fraud = (income_f > 32000).astype(int)

    biased_flag = np.zeros(n_fraud, dtype=int)
    for i in range(n_fraud):
        base = 0.4 if actual_fraud[i] else 0.05
        bias = 0.10 if race_f[i]==1 else (0.07 if race_f[i]==2 else 0)
        biased_flag[i] = int(np.random.random() < base + bias)

    fraud_metrics_biased = fairness_metrics(actual_fraud, biased_flag, race_f, RACE_LABELS)

    fraud_score = np.random.beta(2, 5, n_fraud)
    fraud_score[actual_fraud==1] += 0.35

    fair_flag = np.zeros(n_fraud, dtype=int)
    for g in range(4):
        mask = race_f == g
        if mask.sum() < 10:
            continue
        thr = np.percentile(fraud_score[mask], 85)
        fair_flag[mask] = (fraud_score[mask] > thr).astype(int)

    fraud_metrics_fair = fairness_metrics(actual_fraud, fair_flag, race_f, RACE_LABELS)

    return {
        "biased": fraud_metrics_biased,
        "fair": fraud_metrics_fair
    }

    # 🔴 Biased system
    biased_flag = np.zeros(n_fraud, dtype=int)
    for i in range(n_fraud):
        base = 0.4 if actual_fraud[i] else 0.05
        bias = 0.10 if race_f[i]==1 else (0.07 if race_f[i]==2 else 0)
        biased_flag[i] = int(np.random.random() < base + bias)

    fraud_metrics_biased = fairness_metrics(
        actual_fraud, biased_flag, race_f, RACE_LABELS
    )

    # 🟢 Fair system (group thresholding)
    fraud_score = np.random.beta(2, 5, n_fraud)
    fraud_score[actual_fraud==1] += 0.35

    fair_flag = np.zeros(n_fraud, dtype=int)
    for g in range(4):
        mask = race_f == g
        if mask.sum() < 10:
            continue
        thr = np.percentile(fraud_score[mask], 85)
        fair_flag[mask] = (fraud_score[mask] > thr).astype(int)

    fraud_metrics_fair = fairness_metrics(
        actual_fraud, fair_flag, race_f, RACE_LABELS
    )

    return {
        "biased": fraud_metrics_biased,
        "fair": fraud_metrics_fair
    }

# ─────────────────────────────────────────────
# 9. COMPILE ALL RESULTS TO JSON
# ─────────────────────────────────────────────
import json
import numpy as np

# =========================
# 🔹 DP FUNCTION (IMPROVED)
# =========================
def dp_aggregate(values, sensitivity=1.0, epsilon=1.0, seed=42):
    np.random.seed(seed)
    noise = np.random.laplace(0, sensitivity/epsilon, len(values))

    return [
        round(max(0, min(1, float(v + n))), 4)
        for v, n in zip(values, noise)
    ]


# =========================
# 🔹 SAFE DP GENERATION
# =========================
dp_selection_rates = {}

if "fairness_by_model_race" in globals():
    for name in fairness_by_model_race:
        dp_selection_rates[name] = {}

        for lbl in fairness_by_model_race[name]:
            sr = fairness_by_model_race[name][lbl]["selection_rate"]
            dp_selection_rates[name][lbl] = dp_aggregate([sr], epsilon=2.0)[0]


# =========================
# 🔹 SAFE JSON CONVERTER
# =========================
def convert_to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =========================
# 🔹 DATASET STATS (SAFE)
# =========================
if "all_df" in globals():
    dataset_stats = {
        "total_records": int(len(all_df)),
        "race_distribution": all_df["race"].map(RACE_LABELS).value_counts().to_dict(),
        "overall_eligibility_rate": round(float(all_df.get("truly_eligible", 0).mean()), 4),
        "historical_approval_rate": round(float(all_df.get("historical_approved", 0).mean()), 4),
        "approval_by_race": all_df.groupby(all_df["race"].map(RACE_LABELS))["historical_approved"].mean().round(3).to_dict() if "historical_approved" in all_df else {},
        "income_by_race": all_df.groupby(all_df["race"].map(RACE_LABELS))["income"].mean().round(0).to_dict() if "income" in all_df else {}
    }
else:
    dataset_stats = {"info": "Dataset stats available only for synthetic dataset"}

# ✅ ADD THIS before building payload
fraud_results       = run_fraud_case_study()
fraud_metrics_biased = fraud_results["biased"]
fraud_metrics_fair   = fraud_results["fair"]
# =========================
# 🔹 BUILD PAYLOAD (SAFE)
# =========================
payload = {
    "dataset_stats": dataset_stats,

    "model_performance": {
        k: {kk: convert_to_native(vv) for kk, vv in v.items() if kk not in ["y_pred", "y_prob"]}
        for k, v in results.items()
    } if "results" in globals() else {},

    "fairness_race": fairness_by_model_race if "fairness_by_model_race" in globals() else {},
    "fairness_gender": fairness_by_model_gender if "fairness_by_model_gender" in globals() else {},

    "mitigated_performance": mitigated_overall if "mitigated_overall" in globals() else {},
    "mitigated_fairness_race": mitigated_fairness if "mitigated_fairness" in globals() else {},

    "feature_importance": shap_approx if "shap_approx" in globals() else {},
    "feature_importance_by_race": shap_by_race if "shap_by_race" in globals() else {},

    "dp_selection_rates": dp_selection_rates,

    "fraud_case_study": {
    "biased": fraud_metrics_biased,
    "fair":   fraud_metrics_fair
}
}

# =========================
# 🔹 SAVE FILE
# =========================
with open("results.json", "w") as f:
    json.dump(payload, f, indent=2, default=convert_to_native)

print("\n✅ Results saved to results.json")


# =========================
# 🔹 SUMMARY OUTPUT
# =========================
print("\n=== KEY FAIRNESS FINDINGS ===")

model_name = "Random Forest"

if "fairness_by_model_race" in globals() and model_name in fairness_by_model_race:
    print(f"\n{model_name} - Selection Rates by Race:")

    for race_lbl, metrics in fairness_by_model_race[model_name].items():
        print(
            f"  {race_lbl:12s}: "
            f"SR={metrics.get('selection_rate',0):.3f}  "
            f"DI={metrics.get('disparate_impact',0):.3f}  "
            f"TPR={metrics.get('tpr',0):.3f}"
        )


if "mitigated_fairness" in globals():
    print("\nMitigated RF - Selection Rates by Race:")

    for race_lbl, metrics in mitigated_fairness.items():
        print(
            f"  {race_lbl:12s}: "
            f"SR={metrics.get('selection_rate',0):.3f}  "
            f"DI={metrics.get('disparate_impact',0):.3f}  "
            f"TPR={metrics.get('tpr',0):.3f}"
        )


if "fraud_metrics_biased" in globals():
    print("\nFraud Case Study - False Positive Rate by Race:")

    for race_lbl in RACE_LABELS.values():
        if race_lbl in fraud_metrics_biased:
            print(
                f"  {race_lbl:12s}: "
                f"Biased FPR={fraud_metrics_biased[race_lbl].get('fpr',0):.3f}  "
                f"Fair FPR={fraud_metrics_fair[race_lbl].get('fpr',0):.3f}"
            )