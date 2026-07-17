"""
Independent-cohort replication of experiment4_tcga_myc_concordance.py (BRCA/MYC).

CASCADE's TCGA ARACNe networks were built from TCGA data downloaded April 2015
(raw, hg19-era pipeline). The original concordance test validated predictions
against cBioPortal's TCGA PanCancer Atlas 2018 data (GDC-harmonized, GRCh38) --
a different processing pipeline, but very likely overlapping/identical
underlying TCGA-BRCA patients. That shared-cohort overlap means real
patient-specific biology could inflate apparent "independent" validation even
though the two datasets are not literally the same numbers.

This script re-runs the same concordance test against METABRIC (Curtis et al.
2012 / Pereira et al. 2016), a completely independent breast cancer cohort:
different patients (UK/Canada, not TCGA/US), different institutions, and a
different measurement technology entirely (microarray, not RNA-seq). No
connection whatsoever to the data used to build CASCADE's TCGA network. This
is the clean test of whether the MYC result reflects real, generalizable
biology rather than a shared-cohort artifact.

No core CASCADE server code is modified. Results cached to outputs/.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loader import load_tcga_network
from tools.perturb import simulate_knockdown_with_embeddings
from tools.model_inference import get_model
from tools.gene_id_mapper import get_mapper

OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
RESULTS_PATH = OUTPUTS_DIR / "experiment4_myc_metabric_replication.json"

_BASE_URL = "https://www.cbioportal.org/api"
_SSL_VERIFY = os.environ.get("CASCADE_SSL_NO_VERIFY", "0") != "1"
_STUDY = "brca_metabric"
_CNA_PROFILE = f"{_STUDY}_cna"
_CNA_LIST = f"{_STUDY}_cna"
_EXPR_PROFILE = f"{_STUDY}_mrna_median_all_sample_Zscores"
_EXPR_LIST = f"{_STUDY}_mrna"

TOP_N = 50
PROPAGATION_DEPTH = 2
ALPHA = 0.7  # CASCADE's own default (cascade_langgraph_workflow.py)
EMBEDDING_THRESHOLD = 0.1  # CASCADE's own default for the TCGA path
N_PERMUTATIONS = 1000
BACKGROUND_POOL_SIZE = 200
RNG_SEED = 42


def batch_resolve_entrez(symbols: list[str]) -> dict[str, int]:
    resp = requests.post(f"{_BASE_URL}/genes/fetch", params={"geneIdType": "HUGO_GENE_SYMBOL"},
                          json=symbols, timeout=30, verify=_SSL_VERIFY)
    resp.raise_for_status()
    return {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in resp.json()}


def fetch_cna_per_sample(entrez_id: int) -> dict[str, int]:
    resp = requests.get(f"{_BASE_URL}/molecular-profiles/{_CNA_PROFILE}/molecular-data",
                         params={"sampleListId": _CNA_LIST, "entrezGeneId": entrez_id, "projection": "SUMMARY"},
                         timeout=30, verify=_SSL_VERIFY)
    resp.raise_for_status()
    return {d["sampleId"]: d["value"] for d in resp.json()}


def batch_fetch_expression(entrez_ids: list[int]) -> dict[int, dict[str, float]]:
    resp = requests.post(f"{_BASE_URL}/molecular-profiles/{_EXPR_PROFILE}/molecular-data/fetch",
                          params={"projection": "SUMMARY"},
                          json={"entrezGeneIds": entrez_ids, "sampleListId": _EXPR_LIST},
                          timeout=60, verify=_SSL_VERIFY)
    resp.raise_for_status()
    out: dict[int, dict[str, float]] = {eid: {} for eid in entrez_ids}
    for d in resp.json():
        if d.get("value") is not None:
            out[d["entrezGeneId"]][d["sampleId"]] = d["value"]
    return out


def get_myc_predicted_targets(network_df, top_n: int) -> list[tuple[str, str]]:
    """CASCADE's actual default embedding-enhanced propagation (Section 2.1 of
    the paper): network propagation blended with GREmLN embedding similarity."""
    model = get_model()
    ensembl_id = get_mapper().symbol_to_ensembl("MYC")
    result = simulate_knockdown_with_embeddings(
        network_df, "MYC", model, depth=PROPAGATION_DEPTH, top_k=top_n, alpha=ALPHA,
        embedding_gene=ensembl_id, embedding_threshold=EMBEDDING_THRESHOLD,
    )
    return [(g["symbol"], g["direction"]) for g in result.get("top_affected_genes", [])]


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("Getting CASCADE's MYC knockdown predictions (from TCGA BRCA network, as before)...")
    network_df = load_tcga_network("brca")
    predicted = get_myc_predicted_targets(network_df, TOP_N)
    print(f"  {len(predicted)} predicted targets")

    print("Fetching MYC copy-number status per patient (METABRIC, independent cohort)...")
    myc_entrez = batch_resolve_entrez(["MYC"])["MYC"]
    myc_cna = fetch_cna_per_sample(myc_entrez)
    amplified = {s for s, v in myc_cna.items() if v == 2}
    nonamplified = {s for s, v in myc_cna.items() if v == 0}
    print(f"  {len(amplified)} MYC-amplified, {len(nonamplified)} non-amplified samples")

    pred_symbols = [g for g, _ in predicted]
    pred_entrez_map = batch_resolve_entrez(pred_symbols)
    resolved_predicted = [(g, d, pred_entrez_map[g]) for g, d in predicted if g in pred_entrez_map]
    print(f"  {len(resolved_predicted)}/{len(predicted)} resolved in METABRIC's gene panel")

    print("Fetching per-sample expression for predicted targets (METABRIC)...")
    pred_expr = batch_fetch_expression([eid for _, _, eid in resolved_predicted])

    gene_results = []
    for gene, direction, eid in resolved_predicted:
        expr = pred_expr.get(eid, {})
        amp_vals = [v for s, v in expr.items() if s in amplified]
        nonamp_vals = [v for s, v in expr.items() if s in nonamplified]
        if len(amp_vals) < 10 or len(nonamp_vals) < 10:
            gene_results.append({"gene": gene, "direction": direction, "tested": False,
                                  "reason": "insufficient_samples", "n_amp": len(amp_vals), "n_nonamp": len(nonamp_vals)})
            continue
        amp_mean = float(np.mean(amp_vals))
        nonamp_mean = float(np.mean(nonamp_vals))
        concordant = (amp_mean > nonamp_mean) if direction == "down" else (amp_mean < nonamp_mean)
        gene_results.append({"gene": gene, "direction": direction, "tested": True,
                              "amp_mean_zscore": round(amp_mean, 4), "nonamp_mean_zscore": round(nonamp_mean, 4),
                              "n_amp": len(amp_vals), "n_nonamp": len(nonamp_vals), "concordant": concordant})

    tested = [r for r in gene_results if r["tested"]]
    n_concordant = sum(1 for r in tested if r["concordant"])
    n_tested = len(tested)
    print(f"  {n_concordant}/{n_tested} concordant in METABRIC")

    binom_result = binomtest(n_concordant, n_tested, p=0.5, alternative="greater")
    print(f"  Binomial test: p={binom_result.pvalue:.6f}")

    print(f"\nBuilding background pool ({BACKGROUND_POOL_SIZE} random network genes) for permutation control...")
    all_genes = list(set(network_df["regulator"].unique()) | set(network_df["target"].unique()))
    predicted_gene_set = {g for g, _, _ in resolved_predicted}
    candidate_bg = [g for g in all_genes if g not in predicted_gene_set and g != "MYC"]
    bg_sample = list(rng.choice(candidate_bg, size=min(BACKGROUND_POOL_SIZE, len(candidate_bg)), replace=False))
    bg_entrez_map = batch_resolve_entrez(bg_sample)
    bg_resolved = [(g, bg_entrez_map[g]) for g in bg_sample if g in bg_entrez_map]
    bg_expr = batch_fetch_expression([eid for _, eid in bg_resolved])

    bg_directions = []
    for gene, eid in bg_resolved:
        expr = bg_expr.get(eid, {})
        amp_vals = [v for s, v in expr.items() if s in amplified]
        nonamp_vals = [v for s, v in expr.items() if s in nonamplified]
        if len(amp_vals) < 10 or len(nonamp_vals) < 10:
            continue
        bg_directions.append(np.mean(amp_vals) > np.mean(nonamp_vals))
    print(f"  {len(bg_directions)} background genes usable for permutation")

    frac_down = sum(1 for r in tested if r["direction"] == "down") / n_tested
    bg_directions_arr = np.array(bg_directions)
    perm_concordant_rates = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        sample_idx = rng.integers(0, len(bg_directions_arr), size=n_tested)
        sampled_amp_higher = bg_directions_arr[sample_idx]
        sampled_pred_down = rng.random(n_tested) < frac_down
        perm_concordant_rates[i] = (sampled_amp_higher == sampled_pred_down).mean()

    observed_rate = n_concordant / n_tested
    empirical_p = float((perm_concordant_rates >= observed_rate).sum() / N_PERMUTATIONS)
    print(f"  Permutation empirical p = {empirical_p:.4f} (observed rate={observed_rate:.3f}, "
          f"permutation mean={perm_concordant_rates.mean():.3f})")

    output = {
        "cohort": "METABRIC (independent, non-TCGA)",
        "n_predicted_targets": len(predicted),
        "n_tested": n_tested,
        "n_concordant": n_concordant,
        "observed_concordance_rate": round(observed_rate, 4),
        "binomial_p_value": float(binom_result.pvalue),
        "permutation_empirical_p": empirical_p,
        "permutation_mean_rate": round(float(perm_concordant_rates.mean()), 4),
        "n_myc_amplified_samples": len(amplified),
        "n_myc_nonamplified_samples": len(nonamplified),
        "gene_results": gene_results,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
