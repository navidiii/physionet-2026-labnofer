#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""Multivariate site probe (complements per-feature eta^2 in section 4.3).
Question: from the signal/annotation features alone, can a classifier tell which site a record came from?
Protocol: stratified 5-fold CV over all 1,103 records, 3-class XGBoost, macro one-vs-rest AUC.
NB: all three sites are in training here, so this measures separability of KNOWN sites,
not detectability of an unseen site."""
import json, numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

ROOT = _ROOT
d=np.load(f"{ROOT}/features_cache.npz",allow_pickle=True)
X,y,sites=d["features"].astype(float),d["labels"].astype(int),d["sites"]
US=sorted(set(sites)); S=np.array([US.index(s) for s in sites])
Xf=X[:,10:]                                    # drop the 10 demographic columns
keep=[j for j in range(Xf.shape[1]) if np.isfinite(Xf[:,j]).sum()>50 and np.nanstd(Xf[:,j])>0]
Xf=Xf[:,keep]

nan_by_site=np.array([[np.isnan(Xf[S==g,j]).mean() for g in range(3)] for j in range(Xf.shape[1])])
nuniq=np.array([[len(np.unique(Xf[S==g,j][np.isfinite(Xf[S==g,j])])) for g in range(3)] for j in range(Xf.shape[1])])
clean=[j for j in range(Xf.shape[1]) if (nan_by_site[j]<0.05).all() and (nuniq[j]>20).all()]

def eta2(col,g):
    m=np.isfinite(col); cv,gv=col[m],g[m]; gm=cv.mean(); st=((cv-gm)**2).sum()
    return sum(len(cv[gv==k])*(cv[gv==k].mean()-gm)**2 for k in np.unique(gv))/st if st>0 else 0.0
es=np.array([eta2(Xf[:,j],S) for j in range(Xf.shape[1])])

def srn(M,S):
    out=np.full_like(M,np.nan)
    for g in np.unique(S):
        ix=np.where(S==g)[0]
        for j in range(M.shape[1]):
            c=M[ix,j]; ok=np.isfinite(c)
            if ok.sum()>1:
                r=np.full(len(c),np.nan); r[ok]=(rankdata(c[ok])-1)/(ok.sum()-1); out[ix,j]=r
    return out

def mk(seed): return Pipeline([("i",SimpleImputer(strategy="median")),("c",XGBClassifier(
        n_estimators=200,max_depth=4,learning_rate=0.08,subsample=0.8,colsample_bytree=0.7,
        verbosity=0,n_jobs=-1,random_state=seed,eval_metric="logloss"))])
def probe_site(M,seed=0):
    P=np.zeros((len(S),3))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(M,S):
        P[te]=mk(seed).fit(M[tr],S[tr]).predict_proba(M[te])
    return round(float(roc_auc_score(S,P,multi_class="ovr",average="macro")),3), round(float(balanced_accuracy_score(S,P.argmax(1))),3)
def probe_outcome(M,seed=0):
    P=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(M,y):
        P[te]=mk(seed).fit(M[tr],y[tr]).predict_proba(M[te])[:,1]
    return round(float(roc_auc_score(y,P)),3)

order=np.argsort(-es); Xn=srn(Xf,S)
R={"n_features_probed":int(Xf.shape[1]),"n_clean_features":len(clean),
   "chance_balanced_accuracy":round(1/3,3),
   "n_features_mostly_missing_in_some_site":int((nan_by_site>0.5).any(1).sum()),
   "n_features_constant_in_some_site":int((nuniq<=1).any(1).sum()),
   "n_features_le5_unique_in_some_site":int((nuniq<=5).any(1).sum())}
a,b=probe_site(Xf);               R["site_all_raw"]={"auc":a,"balanced_acc":b}
for k in (5,10,20):
    a,b=probe_site(np.delete(Xf,order[:k],axis=1)); R[f"site_raw_without_top{k}_site_features"]={"auc":a,"balanced_acc":b}
a,b=probe_site(Xf[:,np.argsort(es)[:20]]); R["site_raw_only_20_least_site_associated"]={"auc":a,"balanced_acc":b,"max_eta2_site":round(float(np.sort(es)[19]),4)}
a,b=probe_site(Xn);               R["site_all_per_site_rank_normalised"]={"auc":a,"balanced_acc":b}
a,b=probe_site(Xf[:,clean]);      R["site_clean_raw"]={"auc":a,"balanced_acc":b}
a,b=probe_site(Xn[:,clean]);      R["site_clean_per_site_rank_normalised"]={"auc":a,"balanced_acc":b}
R["outcome_from_same_features_pooled_5fold_auroc"]=probe_outcome(Xf)
json.dump(R,open(f"{_HERE}/site_probe.json","w"),indent=2)
print(json.dumps(R,indent=2))
