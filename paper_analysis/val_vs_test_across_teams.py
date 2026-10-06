#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""How well did the hidden-validation score predict the hidden-test score across all teams?"""
import pandas as pd, numpy as np, json
from scipy.stats import spearmanr, pearsonr
df=pd.read_csv(_os.path.join(_HERE,"final_results","challenge_2026_results.tsv"),sep="\t")
v="Age-conditioned AUROC on the validation set"; t="Age-conditioned AUROC on the test set"
for c in (v,t): df[c]=pd.to_numeric(df[c],errors="coerce")
d=df.dropna(subset=[v,t]).copy(); d["delta"]=d[t]-d[v]; n=len(d)
d["rank_val"]=d[v].rank(ascending=False,method="min"); d["rank_test"]=d[t].rank(ascending=False,method="min")
rho,p=spearmanr(d[v],d[t]); r,_=pearsonr(d[v],d[t])
us=d[d["Team Name"]=="labnofer"].iloc[0]
R={"n_teams":int(n),"spearman_val_test":round(float(rho),3),"spearman_p":float(p),"pearson_val_test":round(float(r),3),
   "delta_mean":round(float(d.delta.mean()),4),"delta_sd":round(float(d.delta.std()),4),"delta_median":round(float(d.delta.median()),4),
   "n_abs_delta_ge_0_047":int((d.delta.abs()>=0.047).sum()),"our_delta":round(float(us.delta),4),
   "our_rank_val":int(us.rank_val),"our_rank_test":int(us.rank_test),
   "frac_teams_with_larger_drop_than_ours":round(float((d.delta<us.delta).mean()),3)}
print(json.dumps(R,indent=1))
print("\nپنج تیمِ برتر بر اساسِ validation → رتبه‌ی واقعی‌شان روی test:")
for _,x in d.sort_values(v,ascending=False).head(5).iterrows():
    print(f"  {str(x['Team Name']):22} val {x[v]:.3f} (رتبه {x.rank_val:.0f}) → test {x[t]:.3f} (رتبه {x.rank_test:.0f})")
print("\nبزرگ‌ترین جابه‌جایی‌ها:")
for _,x in d.reindex(d.delta.abs().sort_values(ascending=False).index).head(4).iterrows():
    print(f"  {str(x['Team Name']):22} val {x[v]:.3f} → test {x[t]:.3f} (Δ {x.delta:+.3f})")
json.dump(R,open(_os.path.join(_HERE,"val_vs_test_across_teams.json"),"w"),indent=2)
