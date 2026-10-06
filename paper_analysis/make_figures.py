#!/usr/bin/env python3
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.environ.get("PHYSIO_ROOT", _os.path.dirname(_HERE))  # repo root: holds the data caches, which are NOT distributed
"""Generate paper figures from results.json + cached features. Print-safe palette."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = _ROOT
R = json.load(open(f"{_HERE}/results.json"))
CAN = json.load(open(f"{_HERE}/canonical_results.json"))
# canonical values override the older 3-seed run
R["claim1_demographics_floor"]["demographics_only_loso_auroc"] = CAN["models"]["demographics"]["loso"]
R["claim1_demographics_floor"]["full_model_loso_auroc"] = CAN["models"]["baseline_76"]["loso"]
R["claim1_demographics_floor"]["full_model_std_over_3_seeds"] = CAN["models"]["baseline_76"]["seed_sd"]

# validated categorical slots 1 & 2 (all-pairs safe, light mode)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})

def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)

# ---------------------------------------------------------------- FIG 1
c1, c2 = R["claim1_demographics_floor"], R["claim2_foundation_model"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.3))
fig.subplots_adjust(wspace=0.55)

def floor_panel(ax, vals, labs, colors, errs, title):
    ax.barh([0, 1], vals, xerr=errs, height=.38,
            color=colors, error_kw=dict(ecolor=INK, lw=.8, capsize=2), zorder=3)
    ax.axvline(0.5, color=INK, lw=.8, ls=(0, (3, 3)), zorder=4)
    ax.text(0.502, 1.42, "chance", ha="left", va="center", fontsize=7, color=MUTED)
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(v + e + .005, i, f"{v:.3f}", va="center", ha="left",
                fontsize=7.5, color=INK)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labs)
    ax.set_xlim(0.48, 0.605); ax.set_ylim(-0.55, 1.75)
    ax.set_xticks([0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60])
    ax.set_xlabel("LOSO age-conditioned AUROC")
    ax.set_title(title, loc="left", pad=6)
    style(ax)

floor_panel(a1,
    [c1["demographics_only_loso_auroc"], c1["full_model_loso_auroc"]],
    ["age + sex\n(2 features)", "all features\n(76 features)"],
    [MUTED, BLUE], [0, c1["full_model_std_over_3_seeds"]],
    f"a  Full cohort  (n={R['cohort']['n_records']}, {R['cohort']['n_positive']} positive)")

floor_panel(a2,
    [c2["demographics_only_same_subset"], c2["best_embedding_auroc"]],
    ["age + sex", f"best of {c2['n_configs_tested']}\nembedding configs"],
    [MUTED, ORANGE], [0, 0],
    f"b  Embedding subset  (n={c2['n_records_with_embeddings']}, {c2['n_positive']} positive)")

fig.savefig(f"{_HERE}/fig1_demographics_floor.png")
plt.close(fig)

print("✅ fig1_demographics_floor.png")

