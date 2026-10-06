#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""Figure 2 — leakage-free: variance explained by site vs by outcome, per feature."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _ROOT
R=json.load(open(f"{_HERE}/canonical_results.json"))["site_association"]
es=np.array(R["eta2_site"]); el=np.array(R["eta2_label"])
BLUE,ORANGE="#2a78d6","#eb6834"; INK,MUTED,GRID="#0b0b0b","#52514e","#d8d8d4"
plt.rcParams.update({"font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,
 "xtick.labelsize":7.5,"ytick.labelsize":7.5,"axes.edgecolor":MUTED,"axes.linewidth":.7,
 "xtick.color":MUTED,"ytick.color":MUTED,"text.color":INK,"axes.labelcolor":INK,
 "figure.dpi":200,"savefig.dpi":300,"savefig.bbox":"tight"})

FL=1e-5
esc=np.clip(es,FL,None); elc=np.clip(el,FL,None)
above = esc>elc
fig,ax=plt.subplots(figsize=(4.4,3.9))
lim=(FL*0.6, max(esc.max(),elc.max())*2.2)
ax.plot(lim,lim,color=INK,lw=.8,ls=(0,(3,3)),zorder=2)
ax.annotate("equal",(2.5e-1,2.5e-1),rotation=45,fontsize=6.8,color=MUTED,
            ha="center",va="bottom",rotation_mode="anchor")
ax.scatter(elc[~above],esc[~above],s=22,color=BLUE,edgecolor="white",lw=.7,zorder=3,
           label=f"outcome ≥ site  ({(~above).sum()})")
ax.scatter(elc[above],esc[above],s=22,color=ORANGE,edgecolor="white",lw=.7,zorder=4,
           label=f"site > outcome  ({above.sum()})")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel(r"variance explained by outcome  ($\eta^2$)")
ax.set_ylabel(r"variance explained by site  ($\eta^2$)")
ax.set_title("Most features carry more site than outcome information",loc="left",pad=7)
leg=ax.legend(loc="lower right",frameon=False,fontsize=7.2,handletextpad=.4,borderpad=.2)
ax.annotate("features with zero outcome variance\nare clipped to the axis floor",(0.035,0.115),
            xycoords="axes fraction",fontsize=6.5,color=MUTED,ha="left",va="top",linespacing=1.4)
ax.grid(color=GRID,lw=.5,which="major"); ax.set_axisbelow(True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.annotate(f"median $\\eta^2$\n  site      {R['median_eta2_site']:.3f}\n  outcome  {R['median_eta2_label']:.4f}",
            (0.035,0.30),xycoords="axes fraction",fontsize=7.1,color=INK,ha="left",va="top",
            linespacing=1.45)
fig.savefig(f"{_HERE}/fig2_site_association.png")
print("✅ fig2_site_association.png")
