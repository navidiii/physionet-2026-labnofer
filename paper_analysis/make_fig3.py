#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""Figure 3 — internal LOSO estimate vs true unseen-site score, all five variants."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = _ROOT
CAN = json.load(open(f"{_HERE}/canonical_results.json"))["models"]
V = {k: {"loso_mean": v["loso"], "ci": v["ci95"], "real_auroc": v["hidden_site"],
         "loso_sd": v["seed_sd"]} for k, v in CAN.items() if v["hidden_site"] is not None}
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": .7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})

NAMES = {"ensemble_12": "12-feature ensemble",
         "plus_14_caisr": "+ 14 CAISR features",
         "plus_spectral": "+ EEG spectral",
         "baseline_76": "baseline (76 features)",
         "de_aged": "age feature removed"}
# label offsets (dx, dy, ha) tuned to avoid collisions
OFF = {"ensemble_12":   ( .000,  .0042, "center"),
       "plus_14_caisr": ( .000, -.0055, "center"),
       "plus_spectral": ( .000,  .0042, "center"),
       "baseline_76":   ( .000,  .0042, "center"),
       "de_aged":       ( .000, -.0055, "center")}

L = np.array([V[k]["loso_mean"] for k in NAMES])
R = np.array([V[k]["real_auroc"] for k in NAMES])
CI = np.array([V[k]["ci"] for k in NAMES])
E = np.abs(CI - L[:, None]).T   # asymmetric bootstrap CI as error bars
rho, p = spearmanr(L, R)

fig, ax = plt.subplots(figsize=(4.8, 4.2))
lim = (0.520, 0.675)
ax.plot(lim, lim, color=MUTED, lw=.8, ls=(0, (3, 3)), zorder=1)
ax.annotate("perfect agreement", (0.617, 0.6195), rotation=26, fontsize=6.8,
            color=MUTED, ha="center", va="bottom", rotation_mode="anchor")

# trend through the points (the whole point: it slopes the wrong way)
m, b = np.polyfit(L, R, 1)
xs = np.array([L.min() - .008, L.max() + .008])
ax.plot(xs, m * xs + b, color=ORANGE, lw=1.3, zorder=2)

ax.errorbar(L, R, xerr=E, fmt="none", ecolor=BLUE, lw=1, capsize=2.5, zorder=3)
ax.scatter(L, R, s=46, color=BLUE, edgecolor="white", linewidth=1.2, zorder=4)
for k in NAMES:
    dx, dy, ha = OFF[k]
    ax.annotate(NAMES[k], (V[k]["loso_mean"] + dx, V[k]["real_auroc"] + dy),
                fontsize=7.2, color=INK, ha=ha, va="center")

ax.set_xlim(0.435, 0.760); ax.set_ylim(0.560, 0.648)
ax.set_xlabel("leave-one-site-out estimate   (bars: 95% bootstrap CI)")
ax.set_ylabel("hidden validation-site score")
ax.set_title("Internal estimates ordered the models differently from the hidden site", loc="left", pad=8)
top1 = json.load(open(f"{_HERE}/canonical_results.json"))["rank_stability"]["p_rank1"]
best = max(top1, key=top1.get)
ax.annotate(f"Spearman $\\rho$ = {rho:+.2f}  (n = 5, p = {p:.2f})\n"
            f"paired bootstrap: LOSO's top model stays #1 in {top1[best]*100:.0f}% of resamples",
            (0.032, 0.965), xycoords="axes fraction", fontsize=7.4, color=ORANGE,
            ha="left", va="top", linespacing=1.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
fig.savefig(f"{_HERE}/fig3_cv_vs_reality.png")
print(f"✅ fig3 rebuilt with 5 points; rho={rho:+.2f} p={p:.2f}")
