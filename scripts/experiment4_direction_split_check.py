"""
Direction-split diagnostic for the baseline comparison (Section 3.3,
experiment4_hallmark_baseline.py / experiment4_hallmark_matched_n.py).

Question: is CASCADE's own predicted-target panel itself close to a uniform
"down" call (matching the naive assumption used for the Hallmark baselines),
or does it make a genuinely mixed set of up/down calls? If CASCADE's panel is
heavily skewed toward "down" already, the baseline comparison is closer to
"two near-uniform guesses" than "gene-specific signal vs. a crude uniform
one," which changes how Section 3.3's finding should be read.

Reads already-saved per-gene results from outputs/experiment4_*_n50_embedding.json
(no new API/CASCADE calls). For each of MYC/BRCA, MYC/COAD, MYC/STAD, and
E2F3/BRCA:
  1. up/down split over the full panel and over just the top 15 by |effect|
     (list order in gene_results already reflects CASCADE's |effect| ranking).
  2. CASCADE's actual concordance rate (as reported in the paper) vs. the
     concordance rate if every "up" call were instead forced to "down"
     (recomputed directly from each gene's amp/non-amp mean z-scores).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"

FILES = {
    "MYC/BRCA": "experiment4_myc_brca_concordance_n50_embedding.json",
    "MYC/COAD": "experiment4_myc_coad_concordance_n50_embedding.json",
    "MYC/STAD": "experiment4_myc_stad_concordance_n50_embedding.json",
    "E2F3/BRCA": "experiment4_e2f3_brca_concordance_n50_embedding.json",
}


def forced_down_concordant(amp_mean: float, nonamp_mean: float) -> bool:
    """What 'concordant' would be if this gene had been uniformly called 'down'."""
    return amp_mean > nonamp_mean


def main() -> None:
    print(f"{'Combo':<10} {'N panel':>7} {'down/up (full)':>16} {'down/up (top15)':>18} "
          f"{'CASCADE rate':>13} {'Forced-down rate':>17} {'N tested':>9}")

    for label, fname in FILES.items():
        data = json.loads((OUTPUTS_DIR / fname).read_text(encoding="utf-8"))
        gene_results = data["gene_results"]
        n_panel = len(gene_results)

        n_down_full = sum(1 for r in gene_results if r["direction"] == "down")
        n_up_full = n_panel - n_down_full

        top15 = gene_results[:15]
        n_down_15 = sum(1 for r in top15 if r["direction"] == "down")
        n_up_15 = len(top15) - n_down_15

        tested = [r for r in gene_results if r["tested"]]
        n_tested = len(tested)
        actual_concordant = sum(1 for r in tested if r["concordant"])
        actual_rate = actual_concordant / n_tested

        forced_concordant = sum(
            1 for r in tested if forced_down_concordant(r["amp_mean_zscore"], r["nonamp_mean_zscore"])
        )
        forced_rate = forced_concordant / n_tested

        print(f"{label:<10} {n_panel:>7} {f'{n_down_full}/{n_up_full}':>16} "
              f"{f'{n_down_15}/{n_up_15}':>18} {actual_rate:>12.1%} {forced_rate:>16.1%} {n_tested:>9}")

        # How much of CASCADE's actual concordance comes specifically from genes
        # where its call *differs* from a uniform "down" guess (i.e. "up" calls),
        # versus genes where it agrees with the uniform guess ("down" calls)?
        up_genes = [r for r in tested if r["direction"] == "up"]
        down_genes = [r for r in tested if r["direction"] == "down"]
        if up_genes:
            up_concordant = sum(1 for r in up_genes if r["concordant"])
            print(f"{'':<10} of {len(up_genes)} 'up'-called genes (where CASCADE disagrees with uniform-down): "
                  f"{up_concordant}/{len(up_genes)} concordant ({up_concordant/len(up_genes):.1%})")
        if down_genes:
            down_concordant = sum(1 for r in down_genes if r["concordant"])
            print(f"{'':<10} of {len(down_genes)} 'down'-called genes (where CASCADE agrees with uniform-down): "
                  f"{down_concordant}/{len(down_genes)} concordant ({down_concordant/len(down_genes):.1%})")
        print()


if __name__ == "__main__":
    main()
