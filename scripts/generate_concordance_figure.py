"""
Generates figure_concordance_summary.pdf for the CASCADE research paper:
observed directional concordance vs. permutation baseline, for every
gene-cancer-type combination tested in the patient-data concordance
experiments (Sections 3.2-3.5).

Values for the twelve combinations already reported in earlier drafts are
the exact figures published in the paper text/tables (recovered from the
vector paths of the previous figure_concordance_summary.pdf, since the
live cBioPortal-derived permutation baseline was not bit-for-bit
reproducible run-to-run before experiment4_tcga_myc_concordance.py's
background-pool ordering fix). Values for AURKA, CCNE1, MDM2, ESR1,
FOXM1, TOP2A, and RPS6KB1 are read directly from that script's
(now-reproducible) output. AURKA/COAD, CCND1/STAD, CCNE1/STAD, MDM2/STAD,
and TOP2A/STAD were added in the cross-cancer-type extension of the
generalization panel (five proliferation-machinery genes re-tested in a
second cancer type wherever amplified-sample eligibility allowed it);
their bars are colored green (category) regardless of individual outcome,
matching the ERBB2 convention already used for the original panel.
CCND2/BRCA and CCND3/BRCA were added once a local gene-identifier
resolution gap for those two genes (unrelated to CASCADE itself) was
fixed; both are colored green (proliferation-machinery category)
regardless of CCND2's own non-concordant outcome, same convention.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

GREEN = "#2B7A3F"   # proliferation-machinery-associated, observed
RED = "#A83232"     # lineage-identity TF / RTK, observed
GREY = "#B0B0B0"     # permutation baseline

# (label, observed_pct, permutation_pct, category)
DATA = [
    ("MYC/BRCA", 90.00, 53.76, "green"),
    ("MYC/COAD", 72.00, 48.39, "green"),
    ("MYC/STAD", 85.71, 50.43, "green"),
    ("MYC/METABRIC", 87.23, 49.17, "green"),
    ("E2F3/BRCA", 96.00, 47.70, "green"),
    ("CCND1/BRCA", 96.00, 48.40, "green"),
    ("CCND1/STAD", 57.14, 49.33, "green"),
    ("CCND2/BRCA", 38.00, 43.41, "green"),
    ("CCND3/BRCA", 76.00, 51.49, "green"),
    ("AURKA/BRCA", 100.00, 43.52, "green"),
    ("AURKA/COAD", 92.00, 46.83, "green"),
    ("CCNE1/BRCA", 100.00, 50.12, "green"),
    ("CCNE1/STAD", 97.96, 53.74, "green"),
    ("MDM2/BRCA", 68.00, 49.94, "green"),
    ("MDM2/STAD", 42.00, 50.93, "green"),
    ("FOXM1/BRCA", 100.00, 53.04, "green"),
    ("TOP2A/BRCA", 100.00, 43.56, "green"),
    ("TOP2A/STAD", 100.00, 51.79, "green"),
    ("RPS6KB1/BRCA", 89.58, 47.47, "green"),
    ("ERBB2/BRCA", 18.00, 45.90, "red"),
    ("ERBB2/COAD", 14.29, 49.55, "red"),
    ("ERBB2/STAD", 93.62, 53.75, "red"),
    ("SOX9/BRCA", 38.78, 50.54, "red"),
    ("FOXA1/BRCA", 34.00, 48.73, "red"),
    ("GATA3/BRCA", 0.00, 51.60, "red"),
    ("ESR1/BRCA", 4.00, 51.24, "red"),
]

labels = [d[0] for d in DATA]
observed = [d[1] for d in DATA]
permutation = [d[2] for d in DATA]
colors = [GREEN if d[3] == "green" else RED for d in DATA]

x = np.arange(len(DATA))
width = 0.4

fig, ax = plt.subplots(figsize=(17.0, 7.5))
ax.bar(x - width / 2, observed, width, color=colors, zorder=3)
ax.bar(x + width / 2, permutation, width, color=GREY, zorder=3)
ax.axhline(50, color="black", linestyle=":", linewidth=1, zorder=2)

ax.set_ylim(0, 100)
ax.set_ylabel("Concordance rate (%)", fontsize=13)
ax.set_title(
    "Directional concordance vs. permutation baseline, all tested gene-cancer-type combinations",
    fontsize=15,
)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10.5, rotation=45, ha="right", rotation_mode="anchor")
ax.tick_params(axis="y", labelsize=11)
ax.set_xlim(-0.7, len(DATA) - 0.3)

legend_handles = [
    Rectangle((0, 0), 1, 1, color=GREEN, label="Observed (proliferation-machinery gene)"),
    Rectangle((0, 0), 1, 1, color=RED, label="Observed (lineage-identity TF / RTK)"),
    Rectangle((0, 0), 1, 1, color=GREY, label="Permutation baseline (chance)"),
]
ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.28),
          ncol=3, frameon=False, fontsize=12)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("manuscript/figure_concordance_summary.pdf", bbox_inches="tight")
print("Wrote manuscript/figure_concordance_summary.pdf")
