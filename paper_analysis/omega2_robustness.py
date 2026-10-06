#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""Robustness of the site-vs-outcome variance comparison: eta^2 (raw) and omega^2 (df-adjusted).
omega^2 = (SS_b - (k-1) MS_w) / (SS_t + MS_w); can be negative (reported, not clipped)."""
import json, numpy as np
ROOT = _ROOT
d=np.load(f"{ROOT}/features_cache.npz",allow_pickle=True)
X,y,sites=d["features"],d["labels"].astype(int),d["sites"]

def var_explained(col, groups):
    m=np.isfinite(col)
    if m.sum()<50 or np.std(col[m])==0: return np.nan, np.nan
    cv,gv=col[m],groups[m]; k=len(np.unique(gv)); n=len(cv)
    if k<2 or n-k<=0: return np.nan,np.nan
    gm=cv.mean(); ss_t=((cv-gm)**2).sum()
    if ss_t==0: return np.nan,np.nan
    ss_b=sum(len(cv[gv==g])*(cv[gv==g].mean()-gm)**2 for g in np.unique(gv))
    ms_w=(ss_t-ss_b)/(n-k)
    return float(ss_b/ss_t), float((ss_b-(k-1)*ms_w)/(ss_t+ms_w))

res=[var_explained(X[:,j].astype(float), sites) for j in range(X.shape[1])]
resl=[var_explained(X[:,j].astype(float), y.astype(str)) for j in range(X.shape[1])]
es=np.array([r[0] for r in res]); os_=np.array([r[1] for r in res])
el=np.array([r[0] for r in resl]); ol=np.array([r[1] for r in resl])
v=np.isfinite(es)&np.isfinite(el)&np.isfinite(os_)&np.isfinite(ol); n=int(v.sum())
out={"n":n,
  "eta2_site_gt_label":int((es[v]>el[v]).sum()),
  "omega2_site_gt_label":int((os_[v]>ol[v]).sum()),
  "omega2_pct":round(100*float((os_[v]>ol[v]).sum())/n,1),
  "omega2_median_site":round(float(np.median(os_[v])),5),
  "omega2_median_label":round(float(np.median(ol[v])),5),
  "omega2_site_10x":int((os_[v]>10*np.maximum(ol[v],0)).sum()),
  "omega2_label_negative":int((ol[v]<0).sum())}
json.dump(out,open(f"{_HERE}/omega2_robustness.json","w"),indent=2)
print(json.dumps(out,indent=1))
