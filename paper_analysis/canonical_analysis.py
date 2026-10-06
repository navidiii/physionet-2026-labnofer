#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""SINGLE canonical run. Every number in the paper comes from here.

Convention (fixed once, used everywhere):
  * out-of-fold LOSO predictions, averaged over 5 seeds
  * point estimate  = age-conditioned AUROC of those averaged predictions
  * interval        = 2000-resample bootstrap percentile CI over records
  * seed SD         = SD of the per-seed AUROCs (algorithmic stability only)
"""
import sys, json, csv
import numpy as np
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
from fast_metric import auroc_age_fast
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr, f_oneway

ROOT = _ROOT
d  = np.load(f"{ROOT}/features_cache.npz", allow_pickle=True)
X, y, sites, ages = d["features"], d["labels"].astype(int), d["sites"], d["ages"]
Xs = np.load(f"{ROOT}/spectral_features_cache.npz")["features"]
c  = np.load(f"{ROOT}/caisr_features_cache.npz", allow_pickle=True)
KEYS = list(c["keys"])
SEL14 = ['arou_idx_rem','frac_r','frac_w','sleep_eff','tst_hr','waso_min','resp_idx_rem',
         'bout_w','limb_idx','rem_frac_last','n_rem_bouts','frac_n3','bout_n3','trans_entropy']
X_caisr = np.hstack([X, c["features"][:, [KEYS.index(k) for k in SEL14]]])
Xd = X.copy(); Xd[:, 0] = 0.0
SEL12 = [9,12,18,24,28,31,34,49,60,70,71,75]
rows = list(csv.DictReader(open(f"{ROOT}/training_data/demographics.csv")))
sex = np.array([1.0 if r["Sex"].strip().lower().startswith("m") else 0.0 for r in rows])
DEMO = np.c_[ages, sex]
SPW = (y==0).sum()/(y==1).sum(); US = sorted(set(sites)); SEEDS = (42,7,2024,13,99)

def xgb(sd):
    return Pipeline([("i",SimpleImputer(strategy="median")),("c",XGBClassifier(n_estimators=300,
        max_depth=4,learning_rate=0.05,scale_pos_weight=SPW,subsample=0.8,colsample_bytree=0.7,
        min_child_weight=5,reg_alpha=0.5,reg_lambda=2.0,random_state=sd,eval_metric="logloss",
        verbosity=0,n_jobs=-1))])
def lreg(sd):
    return Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),
                     ("c",LogisticRegression(max_iter=2000,class_weight="balanced"))])
def ens(Xtr,ytr,Xte,sd):
    a=Pipeline([("i",SimpleImputer(strategy="median")),("c",XGBClassifier(n_estimators=357,
        max_depth=6,learning_rate=0.05,scale_pos_weight=SPW,subsample=0.8,colsample_bytree=0.7,
        min_child_weight=9,reg_alpha=0.5,reg_lambda=2.0,random_state=sd,eval_metric="logloss",
        verbosity=0,n_jobs=-1))]).fit(Xtr,ytr)
    b=Pipeline([("i",SimpleImputer(strategy="median")),("c",LGBMClassifier(n_estimators=500,
        max_depth=6,learning_rate=0.05,scale_pos_weight=SPW,subsample=0.8,colsample_bytree=0.7,
        random_state=sd,verbose=-1,n_jobs=-1))]).fit(Xtr,ytr)
    return 0.10*a.predict_proba(Xte)[:,1]+0.90*b.predict_proba(Xte)[:,1]

VAR = {"ensemble_12":(Xs[:,SEL12],"ens",2102,0.579), "plus_14_caisr":(X_caisr,"xgb",2204,0.576),
       "plus_spectral":(Xs,"xgb",2184,0.609), "baseline_76":(X,"xgb",2083,0.630),
       "de_aged":(Xd,"xgb",2177,0.602), "demographics":(DEMO,"lreg",None,None)}

print("computing canonical out-of-fold predictions (5 seeds)...", flush=True)
OOF, PERSEED = {}, {}
for k,(Xu,kind,_,_) in VAR.items():
    per = []
    for sd in SEEDS:
        pp = np.full(len(y), np.nan)
        for s in US:
            te,tr = sites==s, sites!=s
            pp[te] = ens(Xu[tr],y[tr],Xu[te],sd) if kind=="ens" else \
                     (lreg if kind=="lreg" else xgb)(sd).fit(Xu[tr],y[tr]).predict_proba(Xu[te])[:,1]
        per.append(pp)
    OOF[k] = np.mean(per,axis=0)
    PERSEED[k] = [auroc_age_fast(y,p,ages) for p in per]
np.savez(f"{_HERE}/canonical_oof.npz", **OOF, y=y, sites=sites, ages=ages)

B = 2000
rng = np.random.default_rng(20260819)
BI = [rng.choice(len(y), len(y), replace=True) for _ in range(B)]
BOOT = {k: np.array([auroc_age_fast(y[b], OOF[k][b], ages[b]) for b in BI]) for k in VAR}

R = {"convention": "5-seed-averaged OOF predictions; 2000-resample bootstrap over records",
     "n_bootstrap": B, "seeds": list(SEEDS)}

# ---------- canonical point estimates + CI ----------
R["models"] = {}
print(f"\n{'variant':16} {'LOSO [95% CI]':>26} {'seedSD':>7} {'hidden':>7}")
for k,(_,_,sub,real) in VAR.items():
    pt = auroc_age_fast(y, OOF[k], ages)
    lo,hi = np.percentile(BOOT[k][np.isfinite(BOOT[k])], [2.5,97.5])
    R["models"][k] = {"submission": sub, "hidden_site": real, "loso": round(pt,6),
                      "ci95": [round(float(lo),4), round(float(hi),4)],
                      "ci_width": round(float(hi-lo),4), "seed_sd": round(float(np.std(PERSEED[k])),4)}
    print(f"{k:16} {pt:.3f} [{lo:.3f}, {hi:.3f}]  {np.std(PERSEED[k]):>7.3f} "
          f"{('%.3f'%real) if real else '     — ':>7}")

# ---------- paired difference: full vs demographics ----------
dif = np.array([auroc_age_fast(y[b],OOF['baseline_76'][b],ages[b]) -
                auroc_age_fast(y[b],OOF['demographics'][b],ages[b]) for b in BI])
dif = dif[np.isfinite(dif)]
pt_dif = R["models"]["baseline_76"]["loso"] - R["models"]["demographics"]["loso"]
lo,hi = np.percentile(dif,[2.5,97.5])
R["paired_full_minus_demo"] = {"point": round(pt_dif,6), "ci95":[round(float(lo),4),round(float(hi),4)],
                               "p_two_sided": round(float(2*min((dif>0).mean(),(dif<0).mean())),3)}
print(f"\nfull - demographics: {pt_dif:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
      f"p={R['paired_full_minus_demo']['p_two_sided']:.2f}")

# ---------- NEW: bootstrap rank stability ----------
MODELS = [k for k in VAR if k != "demographics"]
M = np.vstack([BOOT[k] for k in MODELS])
ok = np.isfinite(M).all(0); M = M[:, ok]
ranks = (-M).argsort(0).argsort(0) + 1          # 1 = best
top1 = {k: round(float((ranks[i]==1).mean()),3) for i,k in enumerate(MODELS)}
pt_rank = {k: int(sorted(MODELS, key=lambda m:-R["models"][m]["loso"]).index(k)+1) for k in MODELS}
pair = {}
for i,a in enumerate(MODELS):
    for j,b in enumerate(MODELS):
        if i<j: pair[f"{a}>{b}"] = round(float((M[i]>M[j]).mean()),3)
rho_to_point = [spearmanr(ranks[:,t], [pt_rank[k] for k in MODELS]).statistic for t in range(ranks.shape[1])]
R["rank_stability"] = {"p_rank1": top1, "point_rank": pt_rank, "pairwise_p_a_beats_b": pair,
    "spearman_vs_point_ranking": {"median": round(float(np.median(rho_to_point)),3),
        "pct2.5": round(float(np.percentile(rho_to_point,2.5)),3),
        "pct97.5": round(float(np.percentile(rho_to_point,97.5)),3),
        "frac_negative": round(float(np.mean(np.array(rho_to_point)<0)),3)}}
print("\nbootstrap rank stability — P(model ranked #1 by LOSO):")
for k in sorted(top1, key=lambda m:-top1[m]):
    print(f"  {k:16} {top1[k]*100:5.1f}%   (point rank {pt_rank[k]})")
rs = R["rank_stability"]["spearman_vs_point_ranking"]
print(f"  Spearman(bootstrap ranking, point ranking): median {rs['median']:+.2f} "
      f"[{rs['pct2.5']:+.2f}, {rs['pct97.5']:+.2f}]; negative in {rs['frac_negative']*100:.0f}% of resamples")

# ---------- NEW: leakage-free site association (eta-squared, outcome not used) ----------
def eta2(col, groups):
    m = np.isfinite(col)
    if m.sum() < 50 or np.std(col[m]) == 0: return np.nan
    cv, gv = col[m], groups[m]
    gm = cv.mean(); ss_t = ((cv-gm)**2).sum()
    if ss_t == 0: return np.nan
    ss_b = sum(len(cv[gv==g])*(cv[gv==g].mean()-gm)**2 for g in np.unique(gv))
    return float(ss_b/ss_t)

e_site  = np.array([eta2(X[:,j].astype(float), sites) for j in range(X.shape[1])])
e_label = np.array([eta2(X[:,j].astype(float), y.astype(str)) for j in range(X.shape[1])])
val = np.isfinite(e_site) & np.isfinite(e_label)
R["site_association"] = {"n_features": int(val.sum()),
    "eta2_site": [round(float(v),5) for v in e_site[val]],
    "eta2_label": [round(float(v),5) for v in e_label[val]],
    "n_site_gt_label": int((e_site[val] > e_label[val]).sum()),
    "n_site_10x_label": int((e_site[val] > 10*e_label[val]).sum()),
    "median_eta2_site": round(float(np.median(e_site[val])),4),
    "median_eta2_label": round(float(np.median(e_label[val])),4)}
sa = R["site_association"]
print(f"\nleakage-free site association (eta-squared; site identity, outcome never used):")
print(f"  {sa['n_features']} usable features")
print(f"  variance explained by site  > by outcome : {sa['n_site_gt_label']} features "
      f"({100*sa['n_site_gt_label']/sa['n_features']:.0f}%)")
print(f"  site explains >10x the outcome           : {sa['n_site_10x_label']} features")
print(f"  median eta2: site {sa['median_eta2_site']:.4f} vs outcome {sa['median_eta2_label']:.4f}")

# ---------- per-site folds + pooling sensitivity (incl. vs hidden ranking) ----------
ps = {}
for k in VAR:
    per = {s: auroc_age_fast(y[sites==s], OOF[k][sites==s], ages[sites==s]) for s in US}
    ps[k] = {"per_site": {s: round(v,4) for s,v in per.items()},
             "mean_of_site": round(float(np.mean(list(per.values()))),4),
             "spread": round(float(max(per.values())-min(per.values())),4)}
R["per_site"] = ps
pooled  = [R["models"][k]["loso"] for k in MODELS]
meansit = [ps[k]["mean_of_site"] for k in MODELS]
hidden  = [R["models"][k]["hidden_site"] for k in MODELS]
R["pooling_sensitivity"] = {
  "rho_pooled_vs_meanofsite": round(float(spearmanr(pooled,meansit).statistic),3),
  "rho_pooled_vs_hidden":     round(float(spearmanr(pooled,hidden).statistic),3),
  "rho_meanofsite_vs_hidden": round(float(spearmanr(meansit,hidden).statistic),3)}
print("\npooling sensitivity (Spearman rho):")
for kk,vv in R["pooling_sensitivity"].items(): print(f"  {kk:28} {vv:+.2f}")

R["precision_note"] = "loso stored to 6 dp; round for display, never truncate"
json.dump(R, open(f"{_HERE}/canonical_results.json","w"), indent=2)
print("\nsaved -> paper/canonical_results.json  +  canonical_oof.npz")
