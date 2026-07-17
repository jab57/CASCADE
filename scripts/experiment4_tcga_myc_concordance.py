"""
Experiment 4 (from RESEARCH_PAPER_PLAN.md): TCGA MYC/BRCA directional
concordance test.

Tests CASCADE's predicted MYC-knockdown effects against real patient tumor
data, rather than a curated gene list (OncoKB). Hypothesis: genes CASCADE
predicts as positively regulated by MYC (down upon knockdown) should show
higher expression in MYC-amplified BRCA tumors than non-amplified tumors;
genes predicted as negatively regulated (up upon knockdown) should show
lower expression in amplified tumors. This is a directional concordance
test against real patient copy-number/expression data (TCGA PanCancer
Atlas via cBioPortal), independent of the OncoKB-enrichment axis already
tested in experiment_cascade_validation.py.

No core CASCADE server code is modified -- read-only queries against local
TCGA network files and the public cBioPortal API. Results cached to outputs/.
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
from tools.perturb import _build_adjacency, _propagate_effect, simulate_knockdown_with_embeddings
from tools.model_inference import get_model
from tools.gene_id_mapper import get_mapper

OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# CASCADE's TCGA network folder name -> cBioPortal PanCancer Atlas study prefix.
# Differ for colorectal: CASCADE/aracne.networks uses "coad", cBioPortal's TCGA
# PanCancer Atlas only has the combined "coadread" (colon + rectal) study.
CASCADE_TO_CBIOPORTAL = {"brca": "brca", "coad": "coadread", "stad": "stad"}
CASCADE_CANCER_TYPE = sys.argv[1] if len(sys.argv) > 1 else "brca"
FOCAL_GENE = sys.argv[2] if len(sys.argv) > 2 else "MYC"
_TOP_N_ARG = int(sys.argv[3]) if len(sys.argv) > 3 else None
# METHOD: "network" (default, matches all prior runs -- bare BFS propagation,
# no embedding blending) or "embedding" (CASCADE's actual default behavior
# when the GREmLN model is loaded: network + embedding-similarity blend,
# alpha=0.7, matching cascade_langgraph_workflow.py's real defaults).
METHOD = sys.argv[4] if len(sys.argv) > 4 else "network"
_STUDY = CASCADE_TO_CBIOPORTAL[CASCADE_CANCER_TYPE]
_n_suffix = f"_n{_TOP_N_ARG}" if _TOP_N_ARG else ""
_method_suffix = "_embedding" if METHOD == "embedding" else ""
RESULTS_PATH = OUTPUTS_DIR / f"experiment4_{FOCAL_GENE.lower()}_{CASCADE_CANCER_TYPE}_concordance{_n_suffix}{_method_suffix}.json"

_BASE_URL = "https://www.cbioportal.org/api"
_SSL_VERIFY = os.environ.get("CASCADE_SSL_NO_VERIFY", "0") != "1"
_SUFFIX = "_tcga_pan_can_atlas_2018"
_CNA_PROFILE = f"{_STUDY}{_SUFFIX}_gistic"
_CNA_LIST = f"{_STUDY}{_SUFFIX}_cna"
_EXPR_PROFILE = f"{_STUDY}{_SUFFIX}_rna_seq_v2_mrna_median_Zscores"
_EXPR_LIST = f"{_STUDY}{_SUFFIX}_rna_seq_v2_mrna"

TOP_N = _TOP_N_ARG or 50  # predicted-gene panel size (RESEARCH_PAPER_PLAN.md suggested 50/100/200; 50 chosen for scope; overridable via sys.argv[3] for sensitivity checks)
PROPAGATION_DEPTH = 2  # CASCADE's own default
ALPHA = 0.7  # CASCADE's own default embedding/network blend weight (cascade_langgraph_workflow.py)
EMBEDDING_THRESHOLD = 0.1  # CASCADE's own default for the TCGA embedding path
BACKGROUND_POOL_SIZE = 200
N_PERMUTATIONS = 1000
RNG_SEED = 42


def batch_resolve_entrez(symbols: list[str]) -> dict[str, int]:
    resp = requests.post(
        f"{_BASE_URL}/genes/fetch",
        params={"geneIdType": "HUGO_GENE_SYMBOL"},
        json=symbols,
        timeout=30,
        verify=_SSL_VERIFY,
    )
    resp.raise_for_status()
    return {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in resp.json()}


def fetch_cna_per_sample(entrez_id: int) -> dict[str, int]:
    resp = requests.get(
        f"{_BASE_URL}/molecular-profiles/{_CNA_PROFILE}/molecular-data",
        params={"sampleListId": _CNA_LIST, "entrezGeneId": entrez_id, "projection": "SUMMARY"},
        timeout=30,
        verify=_SSL_VERIFY,
    )
    resp.raise_for_status()
    return {d["sampleId"]: d["value"] for d in resp.json()}


def batch_fetch_expression(entrez_ids: list[int]) -> dict[int, dict[str, float]]:
    """Returns {entrez_id: {sample_id: z_score}}."""
    resp = requests.post(
        f"{_BASE_URL}/molecular-profiles/{_EXPR_PROFILE}/molecular-data/fetch",
        params={"projection": "SUMMARY"},
        json={"entrezGeneIds": entrez_ids, "sampleListId": _EXPR_LIST},
        timeout=60,
        verify=_SSL_VERIFY,
    )
    resp.raise_for_status()
    out: dict[int, dict[str, float]] = {eid: {} for eid in entrez_ids}
    for d in resp.json():
        if d.get("value") is not None:
            out[d["entrezGeneId"]][d["sampleId"]] = d["value"]
    return out


def get_predicted_targets(network_df, focal_gene: str, top_n: int) -> tuple[list[tuple[str, str]], dict]:
    """Returns (list of (gene_symbol, direction), diagnostics) for CASCADE's
    top-N predicted knockdown targets of focal_gene in the given TCGA network,
    using either the network-only or embedding-enhanced path per METHOD."""
    if METHOD == "embedding":
        model = get_model()
        mapper = get_mapper()
        ensembl_id = mapper.symbol_to_ensembl(focal_gene)
        result = simulate_knockdown_with_embeddings(
            network_df, focal_gene, model,
            depth=PROPAGATION_DEPTH, top_k=top_n, alpha=ALPHA,
            embedding_gene=ensembl_id, embedding_threshold=EMBEDDING_THRESHOLD,
        )
        affected = result.get("top_affected_genes", [])
        diagnostics = {
            "method": "embedding",
            "n_total_affected": result.get("total_affected_genes"),
            "n_embedding_only_additions": sum(1 for g in affected if g.get("source") == "embedding_only"),
        }
        return [(g["symbol"], g["direction"]) for g in affected], diagnostics

    adj = _build_adjacency(network_df)
    effects = _propagate_effect(adj, focal_gene, initial_effect=-1.0, depth=PROPAGATION_DEPTH)
    effects.pop(focal_gene, None)
    ranked = sorted(effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    diagnostics = {"method": "network", "n_total_affected": len(effects)}
    return [(g, "down" if e < 0 else "up") for g, e in ranked], diagnostics


def concordance_direction(amp_mean: float, nonamp_mean: float, predicted_direction: str) -> bool:
    """predicted 'down' upon KD = positively regulated by focal gene -> amp should be higher.
    predicted 'up' upon KD = negatively regulated by focal gene -> amp should be lower."""
    if predicted_direction == "down":
        return amp_mean > nonamp_mean
    return amp_mean < nonamp_mean


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print(f"Loading TCGA {CASCADE_CANCER_TYPE.upper()} network and getting CASCADE's {FOCAL_GENE} knockdown "
          f"predictions (method={METHOD})...")
    network_df = load_tcga_network(CASCADE_CANCER_TYPE)
    predicted, diagnostics = get_predicted_targets(network_df, FOCAL_GENE, TOP_N)
    print(f"  {len(predicted)} predicted targets (top {TOP_N} by |effect|)")
    if METHOD == "embedding":
        print(f"  {diagnostics['n_embedding_only_additions']} embedding-only additions "
              f"(genes not reachable via network propagation alone)")

    print(f"Fetching {FOCAL_GENE} copy-number status per patient ({CASCADE_CANCER_TYPE.upper()}, cBioPortal)...")
    focal_entrez = batch_resolve_entrez([FOCAL_GENE])[FOCAL_GENE]
    focal_cna = fetch_cna_per_sample(focal_entrez)
    amplified = {s for s, v in focal_cna.items() if v == 2}
    nonamplified = {s for s, v in focal_cna.items() if v == 0}
    print(f"  {len(amplified)} {FOCAL_GENE}-amplified samples, {len(nonamplified)} non-amplified samples")

    print("Resolving predicted-gene symbols to Entrez IDs...")
    pred_symbols = [g for g, _ in predicted]
    pred_entrez_map = batch_resolve_entrez(pred_symbols)
    resolved_predicted = [(g, d, pred_entrez_map[g]) for g, d in predicted if g in pred_entrez_map]
    print(f"  {len(resolved_predicted)}/{len(predicted)} resolved")

    print("Fetching per-sample expression for predicted targets...")
    pred_expr = batch_fetch_expression([eid for _, _, eid in resolved_predicted])

    gene_results = []
    for gene, direction, eid in resolved_predicted:
        expr = pred_expr.get(eid, {})
        amp_vals = [v for s, v in expr.items() if s in amplified]
        nonamp_vals = [v for s, v in expr.items() if s in nonamplified]
        if len(amp_vals) < 10 or len(nonamp_vals) < 10:
            gene_results.append({"gene": gene, "direction": direction, "tested": False,
                                  "reason": "insufficient_samples"})
            continue
        amp_mean = float(np.mean(amp_vals))
        nonamp_mean = float(np.mean(nonamp_vals))
        concordant = concordance_direction(amp_mean, nonamp_mean, direction)
        gene_results.append({
            "gene": gene, "direction": direction, "tested": True,
            "amp_mean_zscore": round(amp_mean, 4), "nonamp_mean_zscore": round(nonamp_mean, 4),
            "n_amp": len(amp_vals), "n_nonamp": len(nonamp_vals), "concordant": concordant,
        })

    tested = [r for r in gene_results if r["tested"]]
    n_concordant = sum(1 for r in tested if r["concordant"])
    n_tested = len(tested)
    print(f"  {n_concordant}/{n_tested} concordant")

    binom_result = binomtest(n_concordant, n_tested, p=0.5, alternative="greater")
    print(f"  Binomial test: p={binom_result.pvalue:.4f}")

    print(f"\nBuilding background pool ({BACKGROUND_POOL_SIZE} random network genes) for permutation control...")
    all_genes = list(set(network_df["regulator"].unique()) | set(network_df["target"].unique()))
    predicted_gene_set = {g for g, _, _ in resolved_predicted}
    candidate_bg = [g for g in all_genes if g not in predicted_gene_set and g != FOCAL_GENE]
    bg_sample = list(rng.choice(candidate_bg, size=min(BACKGROUND_POOL_SIZE, len(candidate_bg)), replace=False))

    bg_entrez_map = batch_resolve_entrez(bg_sample)
    bg_resolved = [(g, bg_entrez_map[g]) for g in bg_sample if g in bg_entrez_map]
    print(f"  {len(bg_resolved)}/{len(bg_sample)} background genes resolved")

    bg_expr = batch_fetch_expression([eid for _, eid in bg_resolved])

    bg_directions = []  # sign of (amp_mean - nonamp_mean) for each background gene, no prediction attached
    for gene, eid in bg_resolved:
        expr = bg_expr.get(eid, {})
        amp_vals = [v for s, v in expr.items() if s in amplified]
        nonamp_vals = [v for s, v in expr.items() if s in nonamplified]
        if len(amp_vals) < 10 or len(nonamp_vals) < 10:
            continue
        bg_directions.append(np.mean(amp_vals) > np.mean(nonamp_vals))  # True = amp higher

    print(f"  {len(bg_directions)} background genes usable for permutation")

    # Permutation: for each iteration, sample n_tested genes (with replacement) from the
    # background pool, randomly assign each a "predicted direction" matching the true
    # proportion of down/up calls in the real predicted set, and compute concordance rate.
    frac_down = sum(1 for r in tested if r["direction"] == "down") / n_tested
    bg_directions_arr = np.array(bg_directions)
    perm_concordant_rates = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        sample_idx = rng.integers(0, len(bg_directions_arr), size=n_tested)
        sampled_amp_higher = bg_directions_arr[sample_idx]
        sampled_pred_down = rng.random(n_tested) < frac_down  # True = predicted "down" (expect amp higher)
        concordant = sampled_amp_higher == sampled_pred_down
        perm_concordant_rates[i] = concordant.mean()

    observed_rate = n_concordant / n_tested
    empirical_p = float((perm_concordant_rates >= observed_rate).sum() / N_PERMUTATIONS)
    print(f"  Permutation empirical p = {empirical_p:.4f} (observed rate={observed_rate:.3f}, "
          f"permutation mean={perm_concordant_rates.mean():.3f})")

    output = {
        "method": METHOD,
        "diagnostics": diagnostics,
        "n_predicted_targets": len(predicted),
        "n_resolved": len(resolved_predicted),
        "n_tested": n_tested,
        "n_concordant": n_concordant,
        "observed_concordance_rate": round(observed_rate, 4),
        "binomial_p_value": float(binom_result.pvalue),
        "permutation_empirical_p": empirical_p,
        "permutation_mean_rate": round(float(perm_concordant_rates.mean()), 4),
        "n_myc_amplified_samples": len(amplified),
        "n_myc_nonamplified_samples": len(nonamplified),
        "n_background_pool": len(bg_directions),
        "gene_results": gene_results,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
