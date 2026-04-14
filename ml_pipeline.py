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

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────
def generate_welfare_dataset(n=3000, program="SNAP"):
    """
    Simulates a realistic welfare eligibility / fraud-flag dataset with
    known demographic disparities reflecting documented historical bias.
    """
    race      = np.random.choice([0,1,2,3], n, p=[0.60,0.15,0.18,0.07])  # 0=White,1=Black,2=Hispanic,3=Asian
    gender    = np.random.choice([0,1], n, p=[0.48,0.52])
    age       = np.random.randint(18, 65, n)
    income    = np.random.normal(28000, 12000, n).clip(0)
    income   -= (race==1)*3000 + (race==2)*2000   # historical income gap
    hh_size   = np.random.poisson(2.8, n).clip(1,8).astype(int)
    employed  = (income > 20000).astype(int)
    disability= np.random.binomial(1, 0.12, n)
    prior_ben = np.random.binomial(1, 0.35, n)

    # True eligibility (income-based rule)
    fpl_threshold = 16000 + hh_size * 4800
    truly_eligible = ((income < fpl_threshold) | (disability==1)).astype(int)

    # Biased historical approval: penalises minority groups
    bias_noise = (race==1)*0.12 + (race==2)*0.08 + (gender==1)*0.04
    noise      = np.random.normal(0, 0.1, n)
    approval_prob = (0.60 * truly_eligible - bias_noise + noise).clip(0,1)
    historical_approval = (approval_prob > 0.5).astype(int)

    df = pd.DataFrame({
        "race": race, "gender": gender, "age": age,
        "income": income.astype(int), "hh_size": hh_size,
        "employed": employed, "disability": disability,
        "prior_benefit": prior_ben,
        "truly_eligible": truly_eligible,
        "historical_approved": historical_approval
    })
    return df

# Generate three program datasets
snap   = generate_welfare_dataset(3000, "SNAP")
medi   = generate_welfare_dataset(2500, "Medicaid")
unemp  = generate_welfare_dataset(2000, "Unemployment")
all_df = pd.concat([snap, medi, unemp], ignore_index=True)

print(f"Dataset sizes: SNAP={len(snap)}, Medicaid={len(medi)}, Unemployment={len(unemp)}")
print(all_df.describe().to_string())

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
FEATURES = ["age","income","hh_size","employed","disability","prior_benefit"]
TARGET    = "truly_eligible"

X = all_df[FEATURES].values
y = all_df[TARGET].values
race   = all_df["race"].values
gender = all_df["gender"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te, race_tr, race_te, gen_tr, gen_te = train_test_split(
    X_scaled, y, race, gender, test_size=0.25, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42)
}
results = {}
for name, mdl in models.items():
    mdl.fit(X_tr, y_tr)
    y_pred = mdl.predict(X_te)
    y_prob = mdl.predict_proba(X_te)[:,1]
    results[name] = {
        "accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred), 4),
        "recall":    round(recall_score(y_te, y_pred), 4),
        "f1":        round(f1_score(y_te, y_pred), 4),
        "auc":       round(roc_auc_score(y_te, y_prob), 4),
        "y_pred":    y_pred.tolist(),
        "y_prob":    y_prob.tolist()
    }
    print(f"\n{name}: acc={results[name]['accuracy']} f1={results[name]['f1']} auc={results[name]['auc']}")

# ─────────────────────────────────────────────
# 4. FAIRNESS METRICS (AIF360-style formulas)
# ─────────────────────────────────────────────
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
for name in results:
    yp = np.array(results[name]["y_pred"])
    fairness_by_model_race[name]   = fairness_metrics(y_te, yp, race_te,   RACE_LABELS)
    fairness_by_model_gender[name] = fairness_metrics(y_te, yp, gen_te,    {0:"Male",1:"Female"})

# ─────────────────────────────────────────────
# 5. BIAS MITIGATION (Reweighing + Threshold Calibration)
# ─────────────────────────────────────────────
# Reweighing: compute sample weights that equalise selection rates across groups
def compute_reweigh_weights(race_labels, y_labels):
    """Reweighing algorithm (Kamiran & Calders 2012)."""
    n = len(race_labels)
    weights = np.ones(n)
    for r in np.unique(race_labels):
        for y in [0,1]:
            mask = (race_labels==r) & (y_labels==y)
            p_r  = (race_labels==r).mean()
            p_y  = (y_labels==y).mean()
            p_ry = mask.mean()
            if p_ry > 0:
                weights[mask] = (p_r * p_y) / p_ry
    return weights

weights_tr = compute_reweigh_weights(race_tr, y_tr)
rf_mit = RandomForestClassifier(n_estimators=100, random_state=42)
rf_mit.fit(X_tr, y_tr, sample_weight=weights_tr)
yp_mit = rf_mit.predict(X_te)
ypr_mit= rf_mit.predict_proba(X_te)[:,1]

mitigated_overall = {
    "accuracy":  round(accuracy_score(y_te, yp_mit), 4),
    "precision": round(precision_score(y_te, yp_mit), 4),
    "recall":    round(recall_score(y_te, yp_mit), 4),
    "f1":        round(f1_score(y_te, yp_mit), 4),
    "auc":       round(roc_auc_score(y_te, ypr_mit), 4),
}
mitigated_fairness = fairness_metrics(y_te, yp_mit, race_te, RACE_LABELS)
print("\nMitigated RF:", mitigated_overall)

# ─────────────────────────────────────────────
# 6. SHAP-STYLE FEATURE IMPORTANCE (permutation-based approximation)
# ─────────────────────────────────────────────
from sklearn.inspection import permutation_importance
rf_base = models["Random Forest"]
perm = permutation_importance(rf_base, X_te, y_te, n_repeats=10, random_state=42)
shap_approx = {
    FEATURES[i]: round(float(perm.importances_mean[i]), 5)
    for i in range(len(FEATURES))
}
print("\nFeature importance (permutation):", shap_approx)

# Group-conditional feature importances
shap_by_race = {}
for g, lbl in RACE_LABELS.items():
    mask = race_te == g
    if mask.sum() < 30: continue
    pi = permutation_importance(rf_base, X_te[mask], y_te[mask], n_repeats=5, random_state=42)
    shap_by_race[lbl] = {FEATURES[i]: round(float(pi.importances_mean[i]),5) for i in range(len(FEATURES))}

# ─────────────────────────────────────────────
# 7. DIFFERENTIAL PRIVACY NOISE (Laplace mechanism)
# ─────────────────────────────────────────────
def dp_aggregate(values, sensitivity=1.0, epsilon=1.0):
    noise = np.random.laplace(0, sensitivity/epsilon, len(values))
    return [round(float(v+n), 4) for v,n in zip(values, noise)]

# Add DP noise to selection rates for public reporting
dp_selection_rates = {}
for name in fairness_by_model_race:
    dp_selection_rates[name] = {
        lbl: dp_aggregate([fairness_by_model_race[name][lbl]["selection_rate"]], epsilon=2.0)[0]
        for lbl in fairness_by_model_race[name]
    }

# ─────────────────────────────────────────────
# 8. CASE STUDY: WELFARE FRAUD DETECTION SIMULATION
# ─────────────────────────────────────────────
# Simulate a fraud-flag dataset (high-stakes: false positives harm legitimate beneficiaries)
np.random.seed(99)
n_fraud = 5000
race_f   = np.random.choice([0,1,2,3], n_fraud, p=[0.60,0.15,0.18,0.07])
income_f = np.random.normal(24000, 10000, n_fraud).clip(0)
income_f-= (race_f==1)*3000 + (race_f==2)*2000
actual_fraud = (income_f > 32000).astype(int)  # true fraud: over-income
# Biased flag: over-flags minorities
biased_flag = np.zeros(n_fraud, dtype=int)
for i in range(n_fraud):
    base = 0.4 if actual_fraud[i] else 0.05
    bias = 0.10 if race_f[i]==1 else (0.07 if race_f[i]==2 else 0)
    biased_flag[i] = int(np.random.random() < base + bias)

fraud_metrics_biased   = fairness_metrics(actual_fraud, biased_flag, race_f, RACE_LABELS)

# Fair fraud detector (threshold-equalized)
fraud_score = np.random.beta(2, 5, n_fraud)
fraud_score[actual_fraud==1] += 0.35
# Equalize thresholds per group
fair_flag = np.zeros(n_fraud, dtype=int)
for g in range(4):
    mask = race_f == g
    if mask.sum() < 10: continue
    thr = np.percentile(fraud_score[mask], 85)  # top 15% per group
    fair_flag[mask] = (fraud_score[mask] > thr).astype(int)

fraud_metrics_fair = fairness_metrics(actual_fraud, fair_flag, race_f, RACE_LABELS)

# ─────────────────────────────────────────────
# 9. COMPILE ALL RESULTS TO JSON
# ─────────────────────────────────────────────
payload = {
    "dataset_stats": {
        "total_records": len(all_df),
        "race_distribution": all_df["race"].map(RACE_LABELS).value_counts().to_dict(),
        "overall_eligibility_rate": round(float(all_df["truly_eligible"].mean()), 4),
        "historical_approval_rate": round(float(all_df["historical_approved"].mean()), 4),
        "approval_by_race": all_df.groupby(all_df["race"].map(RACE_LABELS))["historical_approved"].mean().round(3).to_dict(),
        "income_by_race":   all_df.groupby(all_df["race"].map(RACE_LABELS))["income"].mean().round(0).to_dict()
    },
    "model_performance": {k: {kk:vv for kk,vv in v.items() if kk not in ["y_pred","y_prob"]} for k,v in results.items()},
    "fairness_race":     fairness_by_model_race,
    "fairness_gender":   fairness_by_model_gender,
    "mitigated_performance": mitigated_overall,
    "mitigated_fairness_race": mitigated_fairness,
    "feature_importance": shap_approx,
    "feature_importance_by_race": shap_by_race,
    "dp_selection_rates": dp_selection_rates,
    "fraud_case_study": {
        "biased_detector": fraud_metrics_biased,
        "fair_detector":   fraud_metrics_fair
    }
}

with open("results.json", "w") as f:
    json.dump(payload, f, indent=2)

print("\n✅ All results saved to /home/claude/results.json")

# Print a summary of key fairness findings
print("\n=== KEY FAIRNESS FINDINGS ===")
for model in ["Random Forest"]:
    print(f"\n{model} - Selection Rates by Race:")
    for race_lbl, metrics in fairness_by_model_race[model].items():
        print(f"  {race_lbl:12s}: SR={metrics['selection_rate']:.3f}  DI={metrics['disparate_impact']:.3f}  TPR={metrics['tpr']:.3f}")

print("\nMitigated RF - Selection Rates by Race:")
for race_lbl, metrics in mitigated_fairness.items():
    print(f"  {race_lbl:12s}: SR={metrics['selection_rate']:.3f}  DI={metrics['disparate_impact']:.3f}  TPR={metrics['tpr']:.3f}")

print("\nFraud Case Study - False Positive Rate by Race:")
for race_lbl in RACE_LABELS.values():
    if race_lbl in fraud_metrics_biased:
        print(f"  {race_lbl:12s}: Biased FPR={fraud_metrics_biased[race_lbl]['fpr']:.3f}  Fair FPR={fraud_metrics_fair[race_lbl]['fpr']:.3f}")
