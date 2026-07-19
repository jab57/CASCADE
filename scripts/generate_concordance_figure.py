"""
Generates figure_concordance_summary.pdf for the CASCADE research paper:
observed directional concordance vs. permutation baseline, for every
gene x cancer-type combination tested in the patient-data concordance
experiments (Sections 3.2-3.5).

Values for the twelve combinations already reported in earlier drafts are
the exact figures published in the paper text/tables (recovered from the
vector paths of the previous figure_concordance_summary.pdf, since the
live cBioPortal-derived permutation baseline is not bit-for-bit
reproducible run-to-run). Values for AURKA, CCNE1, and MDM2 are read
directly from this session's experiment4_tcga_myc_concordance.py output.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

GREEN = "#2B7A3F"   # proliferation-machinery-associated, observed
RED = "#A83232"     # lineage-identity TF / RTK, observed
GREY = "#B0B0B0"     # permutation baseline

# (label, observed_pct, permutation_pct, category)
DATA = [
    ("MYC\nBRCA", 90.00, 53.76, "green"),
    ("MYC\nCOAD", 72.00, 48.39, "green"),
    ("MYC\nSTAD", 85.71, 50.43, "green"),
    ("MYC\nMETABRIC", 87.23, 49.17, "green"),
    ("E2F3\nBRCA", 96.00, 47.70, "green"),
    ("CCND1\nBRCA", 96.00, 48.40, "green"),
    ("AURKA\nBRCA", 100.00, 43.52, "green"),
    ("CCNE1\nBRCA", 100.00, 50.12, "green"),
    ("MDM2\nBRCA", 68.00, 49.94, "green"),
    ("ERBB2\nBRCA", 18.00, 45.90, "red"),
    ("ERBB2\nCOAD", 14.29, 49.55, "red"),
    ("ERBB2\nSTAD", 93.62, 53.75, "red"),
    ("SOX9\nBRCA", 38.78, 50.54, "red"),
    ("FOXA1\nBRCA", 34.00, 48.73, "red"),
    ("GATA3\nBRCA", 0.00, 51.60, "red"),
]

labels = [d[0] for d in DATA]
observed = [d[1] for d in DATA]
permutation = [d[2] for d in DATA]
colors = [GREEN if d[3] == "green" else RED for d in DATA]

x = np.arange(len(DATA))
width = 0.4

fig, ax = plt.subplots(figsize=(15.0, 6.0))
ax.bar(x - width / 2, observed, width, color=colors, zorder=3)
ax.bar(x + width / 2, permutation, width, color=GREY, zorder=3)
ax.axhline(50, color="black", linestyle=":", linewidth=1, zorder=2)

ax.set_ylim(0, 100)
ax.set_ylabel("Concordance rate (%)", fontsize=13)
ax.set_title(
    "Directional concordance vs. permutation baseline, all tested gene×cancer-type combinations",
    fontsize=15,
)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.tick_params(axis="y", labelsize=11)

legend_handles = [
    Rectangle((0, 0), 1, 1, color=GREEN, label="Observed (proliferation-machinery gene)"),
    Rectangle((0, 0), 1, 1, color=RED, label="Observed (lineage-identity TF / RTK)"),
    Rectangle((0, 0), 1, 1, color=GREY, label="Permutation baseline (chance)"),
]
ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=3, frameon=False, fontsize=12)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("manuscript/figure_concordance_summary.pdf", bbox_inches="tight")
print("Wrote manuscript/figure_concordance_summary.pdf")
