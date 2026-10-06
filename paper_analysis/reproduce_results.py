#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""Reproduce every quantitative claim in the paper, from cached features.

Outputs a single results.json + human-readable table so the manuscript
never contains a number that isn't regenerated here.
"""
import sys, json, csv
import numpy as np
sys.path.insert(0, _ROOT)
from evaluate_model import compute_auroc_age
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

ROOT = _ROOT
R = {}

# ---------------------------------------------------------------- data
d = np.load(f"{ROOT}/features_cache.npz", allow_pickle=True)
X, y, sites, ages = d["features"], d["labels"].astype(int), d["sites"], d["ages"]
rows = list(csv.DictReader(open(f"{ROOT}/training_data/demographics.csv")))
sex = np.array([1.0 if r["Sex"].strip().lower().startswith("m") else 0.0 for r in rows])
US = sorted(set(sites))
SPW = (y == 0).sum() / (y == 1).sum()

R["cohort"] = {
    "n_records": int(len(y)), "n_positive": int(y.sum()),
    "prevalence": round(float(y.mean()), 4),
    "per_site": {s: {"n": int((sites == s).sum()),
                     "pos": int(y[sites == s].sum()),
                     "prev": round(float(y[sites == s].mean()), 4)} for s in US},
    "age_range": [float(np.nanmin(ages)), float(np.nanmax(ages))],
}

def xgb():
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("c", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                         scale_pos_weight=SPW, subsample=0.8, colsample_bytree=0.7,
                                         min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0,
                                         random_state=42, eval_metric="logloss",
                                         verbosity=0, n_jobs=-1))])

def loso_auroc(Xu, model_fn=xgb, seeds=(42,)):
    """Leave-one-site-out pooled age-conditioned AUROC (mean over seeds)."""
    out = []
    for sd in seeds:
        pp = np.full(len(y), np.nan)
        for s in US:
            te, tr = sites == s, sites != s
            m = model_fn()
            try: m.named_steps["c"].set_params(random_state=sd)
            except Exception: pass
            m.fit(Xu[tr], y[tr]); pp[te] = m.predict_proba(Xu[te])[:, 1]
        ok = np.isfinite(pp)
        out.append(compute_auroc_age(y[ok], pp[ok], ages[ok], gap=2))
    return float(np.mean(out)), float(np.std(out))

# ------------------------------------------- CLAIM 1: demographics floor
demo = np.c_[ages, sex]
def logreg(): return Pipeline([("i", SimpleImputer(strategy="median")),
                               ("s", StandardScaler()),
                               ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
m_demo, _ = loso_auroc(demo, logreg)
m_full, s_full = loso_auroc(X, xgb, seeds=(42, 7, 2024))
R["claim1_demographics_floor"] = {
    "demographics_only_loso_auroc": round(m_demo, 4),
    "full_model_loso_auroc": round(m_full, 4),
    "full_model_std_over_3_seeds": round(s_full, 4),
    "gain_over_demographics": round(m_full - m_demo, 4),
}

# --------------------------------------- CLAIM 2: foundation model fails
emb = np.load(f"{ROOT}/osf_embeddings_cache.npz", allow_pickle=True)
E, Ey, Es, Ea, Ex = emb["features"], emb["labels"], emb["sites"], emb["ages"], emb["sex"]
ok = np.isfinite(E).all(1) & np.isfinite(Ea)
E, Ey, Es, Ea, Ex = E[ok], Ey[ok].astype(int), Es[ok], Ea[ok], Ex[ok]
EUS = sorted(set(Es))

def loso_emb(M, k=None, C=1.0):
    pp = np.full(len(Ey), np.nan)
    for s in EUS:
        te, tr = Es == s, Es != s
        if len(set(Ey[te])) < 2: continue
        A, B = M[tr], M[te]
        sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(B)
        if k and k < A.shape[1]:
            p = PCA(k, random_state=0).fit(A); A, B = p.transform(A), p.transform(B)
        pp[te] = LogisticRegression(max_iter=3000, class_weight="balanced", C=C).fit(A, Ey[tr]).predict_proba(B)[:, 1]
    m = np.isfinite(pp)
    return float(compute_auroc_age(Ey[m], pp[m], Ea[m], gap=2))

emb_demo = loso_emb(np.c_[Ea, Ex])
grid = {}
for name, M in [("mean", E[:, :768]), ("std", E[:, 768:]), ("mean+std", E)]:
    for k in (16, 32, 64):
        for C in (0.05, 0.2, 1.0):
            grid[f"{name}|PCA{k}|C{C}"] = round(loso_emb(M, k, C), 4)
best = max(grid, key=grid.get)
R["claim2_foundation_model"] = {
    "n_records_with_embeddings": int(len(Ey)), "n_positive": int(Ey.sum()),
    "embedding_dim": int(E.shape[1]),
    "demographics_only_same_subset": round(emb_demo, 4),
    "best_embedding_config": best, "best_embedding_auroc": grid[best],
    "n_configs_tested": len(grid),
    "n_configs_beating_demographics": int(sum(v > emb_demo for v in grid.values())),
    "full_grid": grid,
}

# ---------------------------------------- CLAIM 3: site-confounded features
site_prev = np.array([y[sites == s].mean() for s in sites])
ratios, n_bad, n_ok = {}, 0, 0
for j in range(X.shape[1]):
    col = X[:, j].astype(float); m = np.isfinite(col)
    if m.sum() < 50 or np.std(col[m]) == 0: continue
    cl = abs(np.corrcoef(col[m], y[m])[0, 1])
    cs = abs(np.corrcoef(col[m], site_prev[m])[0, 1])
    r = cs / cl if cl > 1e-6 else np.inf
    ratios[j] = (round(cl, 4), round(cs, 4), round(float(r), 2) if np.isfinite(r) else None)
    if r > 3: n_bad += 1
    else: n_ok += 1
R["claim3_site_confounding"] = {
    "n_features_total": int(X.shape[1]), "n_features_usable": len(ratios),
    "n_site_confounded_ratio_gt3": n_bad, "n_clean": n_ok,
    "pct_confounded": round(100 * n_bad / len(ratios), 1),
    "max_ratio": max(v[2] for v in ratios.values() if v[2]),
    "per_site_prevalence": {s: round(float(y[sites == s].mean()), 4) for s in US},
}

# ------------------------------- CLAIM 4: internal CV vs real leaderboard
# Real held-out-site scores from the official challenge (recorded from result emails).
LB = [
    {"id": 2083, "commit": "443af5c", "desc": "baseline XGBoost (76 feat)",        "auroc": 0.630, "reward": 0.007},
    {"id": 2102, "commit": "713b80f", "desc": "ensemble + 12-feat selection",      "auroc": 0.579, "reward": -0.080},
    {"id": 2177, "commit": "fb37529", "desc": "de-aged model + threshold",         "auroc": 0.602, "reward": 0.067},
    {"id": 2184, "commit": "bf584d0", "desc": "baseline + age-conditioned Bayes",  "auroc": 0.609, "reward": 0.036},
    {"id": 2204, "commit": "5514a6a", "desc": "baseline + 14 CAISR features",      "auroc": 0.576, "reward": -0.005},
    {"id": 2225, "commit": "7025584", "desc": "baseline + very aggressive thresh", "auroc": 0.630, "reward": -0.186},
]
R["claim4_cv_vs_leaderboard"] = {"leaderboard": LB}

json.dump(R, open(f"{_HERE}/results.json", "w"), indent=2, ensure_ascii=False)

# ------------------------------------------------------------- report
print("=" * 68)
print("COHORT:", R["cohort"]["n_records"], "records,", R["cohort"]["n_positive"],
      f"positive ({R['cohort']['prevalence']*100:.1f}%)")
for s, v in R["cohort"]["per_site"].items():
    print(f"   {s}: n={v['n']:4d}  pos={v['pos']:3d}  prevalence={v['prev']*100:.1f}%")
print("=" * 68)
c1 = R["claim1_demographics_floor"]
print(f"CLAIM 1 — demographics floor (LOSO age-conditioned AUROC)")
print(f"   age+sex only        : {c1['demographics_only_loso_auroc']}")
print(f"   full 76-feature model: {c1['full_model_loso_auroc']} (SD {c1['full_model_std_over_3_seeds']} over 3 seeds)")
print(f"   gain from all PSG features: {c1['gain_over_demographics']:+.4f}")
c2 = R["claim2_foundation_model"]
print(f"\nCLAIM 2 — foundation-model embeddings (n={c2['n_records_with_embeddings']}, {c2['n_positive']} pos)")
print(f"   age+sex on same subset : {c2['demographics_only_same_subset']}")
print(f"   best of {c2['n_configs_tested']} embedding configs: {c2['best_embedding_auroc']}  ({c2['best_embedding_config']})")
print(f"   configs beating demographics: {c2['n_configs_beating_demographics']}/{c2['n_configs_tested']}")
c3 = R["claim3_site_confounding"]
print(f"\nCLAIM 3 — site confounding")
print(f"   {c3['n_site_confounded_ratio_gt3']}/{c3['n_features_usable']} usable features ({c3['pct_confounded']}%) have ratio>3")
print(f"   worst ratio: {c3['max_ratio']}")
print(f"\nCLAIM 4 — {len(LB)} real held-out-site submissions recorded")
print("=" * 68)
print("saved -> paper/results.json")
