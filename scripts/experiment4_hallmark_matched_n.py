"""
Matched-N sensitivity check for the Hallmark-baseline comparison
(experiment4_hallmark_baseline.py).

The full Hallmark gene sets (~191-194 testable genes) are much larger than
CASCADE's own N=50 predicted-target panel, since Hallmark sets have no
natural "top-N by effect" ranking. This script asks: if you subsample the
Hallmark set down to N=50 (matching CASCADE's panel size exactly) many
times, does the concordance rate stay flat (confirming the full-set
comparison already reported is not an artifact of N), or does it vary
enough that "ties at matched size" would be an overstatement?

Reads already-fetched per-gene results from outputs/experiment4_hallmark_baseline_*.json
(no new API calls) and reports the distribution of concordance rates
across 1000 random N=50 subsamples (without replacement) of the tested
genes in each file.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"

N_SUBSAMPLE = 50
N_DRAWS = 1000
RNG_SEED = 42

FILES = {
    "MYC/BRCA": ("experiment4_hallmark_baseline_brca.json", 45, 50),   # CASCADE: 45/50 = 90.0%
    "MYC/COAD": ("experiment4_hallmark_baseline_coad.json", 36, 50),  # CASCADE: 36/50 = 72.0%
    "MYC/STAD": ("experiment4_hallmark_baseline_stad.json", 42, 49),  # CASCADE: 42/49 = 85.7%
    "E2F3/BRCA": ("experiment4_hallmark_baseline_e2f_brca.json", 48, 50),  # CASCADE: 96.0% of 50 = 48/50
    "G2M/BRCA-vs-MYC": ("experiment4_hallmark_baseline_g2m_brca.json", 45, 50),
    "Spindle/BRCA-vs-MYC": ("experiment4_hallmark_baseline_spindle_brca.json", 45, 50),
    "E2Fgeneric/BRCA-vs-MYC": ("experiment4_hallmark_baseline_e2f_vs_myc_brca.json", 45, 50),
    "G2M/COAD-vs-MYC": ("experiment4_hallmark_baseline_g2m_coad.json", 36, 50),
    "G2M/STAD-vs-MYC": ("experiment4_hallmark_baseline_g2m_stad.json", 42, 49),
}


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print(f"{'Combo':<12} {'Full-set rate':>14} {'CASCADE rate':>13} "
          f"{'Subsample mean':>15} {'Subsample SD':>13} {'Subsample [5th,95th]':>22} {'N tested':>9}")

    for label, (fname, cascade_n_concordant, cascade_n) in FILES.items():
        data = json.loads((OUTPUTS_DIR / fname).read_text(encoding="utf-8"))
        tested = [r for r in data["gene_results"] if r["tested"]]
        concordant_arr = np.array([r["concordant"] for r in tested])
        n_tested = len(concordant_arr)
        full_rate = concordant_arr.mean()
        cascade_rate = cascade_n_concordant / cascade_n

        if n_tested < N_SUBSAMPLE:
            print(f"{label:<12} only {n_tested} tested genes, cannot subsample to N={N_SUBSAMPLE}")
            continue

        subsample_rates = np.empty(N_DRAWS)
        for i in range(N_DRAWS):
            idx = rng.choice(n_tested, size=N_SUBSAMPLE, replace=False)
            subsample_rates[i] = concordant_arr[idx].mean()

        mean_rate = subsample_rates.mean()
        sd_rate = subsample_rates.std()
        p5, p95 = np.percentile(subsample_rates, [5, 95])
        # Unambiguous, non-overlapping partition (avoids conflating a tie-inclusive
        # complement with a strict "CASCADE wins" rate, as an earlier draft of this
        # analysis did): strict baseline win / exact tie / strict CASCADE win.
        # These three always sum to 1.000, by construction.
        frac_baseline_strictly_beats = float((subsample_rates > cascade_rate).mean())
        frac_tie = float((subsample_rates == cascade_rate).mean())
        frac_cascade_strictly_beats = float((subsample_rates < cascade_rate).mean())
        assert abs(frac_baseline_strictly_beats + frac_tie + frac_cascade_strictly_beats - 1.0) < 1e-9

        print(f"{label:<12} {full_rate:>13.1%} {cascade_rate:>12.1%} "
              f"{mean_rate:>14.1%} {sd_rate:>12.1%} [{p5:.1%}, {p95:.1%}]{'':>6} {n_tested:>9}   "
              f"P(baseline>CASCADE)={frac_baseline_strictly_beats:.3f}  P(tie)={frac_tie:.3f}  "
              f"P(CASCADE>baseline)={frac_cascade_strictly_beats:.3f}")


if __name__ == "__main__":
    main()
