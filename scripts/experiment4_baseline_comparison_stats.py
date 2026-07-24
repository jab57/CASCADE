"""
Baseline comparison statistics (Section 3.3 / Table tab:baseline_comparison,
Appendix app:baseline).

Combines CASCADE's own concordance counts (from experiment4_tcga_myc_concordance.py
and experiment4_e2f3_brca_concordance.py runs) with each public baseline gene set's
concordance counts (from experiment4_hallmark_baseline.py runs) into the nine
Fisher's-exact-test comparisons reported in the paper, then applies
Benjamini-Hochberg FDR correction across all nine.

This step reads only the cached JSON files already written to outputs/ by the
scripts above -- it does not re-fetch any data or re-run CASCADE. It exists to
make the paper's final combination step (previously done as an unrecorded
one-off calculation) reproducible from committed inputs.
"""

import json
from pathlib import Path

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

# Each row: (comparison label, cancer type, CASCADE result file, baseline result file)
COMPARISONS = [
    ("MYC-identity (MYC_TARGETS_V1)", "BRCA",
     "experiment4_myc_brca_concordance_n50_embedding.json", "experiment4_hallmark_baseline_brca.json"),
    ("MYC-identity (MYC_TARGETS_V1)", "COAD",
     "experiment4_myc_coad_concordance_n50_embedding.json", "experiment4_hallmark_baseline_coad.json"),
    ("MYC-identity (MYC_TARGETS_V1)", "STAD",
     "experiment4_myc_stad_concordance_n50_embedding.json", "experiment4_hallmark_baseline_stad.json"),
    ("E2F-identity (E2F_TARGETS), vs. E2F3", "BRCA",
     "experiment4_e2f3_brca_concordance_n50_embedding.json", "experiment4_hallmark_baseline_e2f_brca.json"),
    ("Generic (G2M_CHECKPOINT)", "BRCA",
     "experiment4_myc_brca_concordance_n50_embedding.json", "experiment4_hallmark_baseline_g2m_brca.json"),
    ("Generic (MITOTIC_SPINDLE)", "BRCA",
     "experiment4_myc_brca_concordance_n50_embedding.json", "experiment4_hallmark_baseline_spindle_brca.json"),
    ("Generic (E2F_TARGETS vs. MYC)", "BRCA",
     "experiment4_myc_brca_concordance_n50_embedding.json", "experiment4_hallmark_baseline_e2f_vs_myc_brca.json"),
    ("Generic (G2M_CHECKPOINT)", "COAD",
     "experiment4_myc_coad_concordance_n50_embedding.json", "experiment4_hallmark_baseline_g2m_coad.json"),
    ("Generic (G2M_CHECKPOINT)", "STAD",
     "experiment4_myc_stad_concordance_n50_embedding.json", "experiment4_hallmark_baseline_g2m_stad.json"),
]


def load_counts(filename: str) -> tuple[int, int]:
    data = json.loads((OUTPUTS / filename).read_text(encoding="utf-8"))
    return data["n_concordant"], data["n_tested"]


def main() -> None:
    rows = []
    for label, cancer_type, cascade_file, baseline_file in COMPARISONS:
        cascade_concordant, cascade_tested = load_counts(cascade_file)
        baseline_concordant, baseline_tested = load_counts(baseline_file)

        table = [
            [cascade_concordant, cascade_tested - cascade_concordant],
            [baseline_concordant, baseline_tested - baseline_concordant],
        ]
        _, p_value = fisher_exact(table, alternative="two-sided")

        rows.append({
            "comparison": label,
            "cancer_type": cancer_type,
            "cascade_rate": round(cascade_concordant / cascade_tested, 4),
            "cascade_n": cascade_tested,
            "baseline_rate": round(baseline_concordant / baseline_tested, 4),
            "baseline_n": baseline_tested,
            "fisher_p": p_value,
        })

    _, q_values, _, _ = multipletests([r["fisher_p"] for r in rows], method="fdr_bh")
    for row, q in zip(rows, q_values):
        row["bh_fdr_q"] = float(q)

    print(f"{'Comparison':<38} {'Cancer':<6} {'CASCADE':>8} {'Baseline':>9} {'Fisher p':>10} {'BH-FDR q':>10}")
    for row in rows:
        print(f"{row['comparison']:<38} {row['cancer_type']:<6} "
              f"{row['cascade_rate']*100:>7.1f}% {row['baseline_rate']*100:>8.1f}% "
              f"{row['fisher_p']:>10.4f} {row['bh_fdr_q']:>10.4f}")

    out_path = OUTPUTS / "experiment4_baseline_comparison_stats.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
