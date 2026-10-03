"""
Generalization-panel BH-FDR correction (Section sec:gene_specificity /
Table tab:other_genes).

Applies Benjamini-Hochberg FDR correction across all twenty-seven
gene-cancer-type combinations in the generalization sweep: the twenty-four
rows of Table tab:other_genes plus MYC's three cancer types from Table
tab:myc_main. The p-value corrected is each combination's one-sided binomial
test against 50% concordance (binomial_p_value, written by
experiment4_tcga_myc_concordance.py with the embedding method, top 50 targets).

This step reads only the cached JSON files already written to outputs/ -- it
does not re-fetch any data or re-run CASCADE. It exists to make the table's
BH-FDR column (previously an unrecorded one-off calculation) reproducible from
committed inputs.
"""

import json
from pathlib import Path

from stats_utils import bh_fdr

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

# (focal gene, cancer type), in Table tab:other_genes row order, then MYC.
COMBINATIONS = [
    ("E2F3", "brca"),
    ("CCND1", "brca"), ("CCND1", "stad"),
    ("CCND2", "brca"), ("CCND2", "coad"),
    ("CCND3", "brca"), ("CCND3", "stad"),
    ("AURKA", "brca"), ("AURKA", "coad"),
    ("CCNE1", "brca"), ("CCNE1", "stad"),
    ("MDM2", "brca"), ("MDM2", "stad"),
    ("FOXM1", "brca"),
    ("TOP2A", "brca"), ("TOP2A", "stad"),
    ("RPS6KB1", "brca"),
    ("ERBB2", "brca"), ("ERBB2", "coad"), ("ERBB2", "stad"),
    ("SOX9", "brca"),
    ("FOXA1", "brca"),
    ("GATA3", "brca"),
    ("ESR1", "brca"),
    ("MYC", "brca"), ("MYC", "coad"), ("MYC", "stad"),
]


def main() -> None:
    rows = []
    for gene, cancer_type in COMBINATIONS:
        path = OUTPUTS / f"experiment4_{gene.lower()}_{cancer_type}_concordance_n50_embedding.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "gene": gene,
            "cancer_type": cancer_type.upper(),
            "n_tested": data["n_tested"],
            "concordance_rate": data["observed_concordance_rate"],
            "binomial_p": data["binomial_p_value"],
        })

    q_values = bh_fdr([r["binomial_p"] for r in rows])
    for row, q in zip(rows, q_values):
        row["bh_fdr_q"] = q

    print(f"{'Gene':<8} {'Cancer':<6} {'n':>3} {'Concordance':>12} {'Binomial p':>12} {'BH-FDR q':>10}")
    for row in rows:
        q_str = "<0.0001" if row["bh_fdr_q"] < 0.0001 else f"{row['bh_fdr_q']:.4f}"
        print(f"{row['gene']:<8} {row['cancer_type']:<6} {row['n_tested']:>3} "
              f"{row['concordance_rate']*100:>11.1f}% {row['binomial_p']:>12.3g} {q_str:>10}")

    n_sig = sum(r["bh_fdr_q"] < 0.05 for r in rows)
    print(f"\n{n_sig} of {len(rows)} combinations significant at q<0.05")

    out_path = OUTPUTS / "experiment4_generalization_bh_fdr.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
