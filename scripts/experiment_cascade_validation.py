"""
CASCADE cs.AI paper validation experiment.

Tests whether CASCADE's predicted *downstream* perturbation-propagation output
carries real cancer-driver signal. This is deliberately a different question
than RegNetAgents (arXiv:2607.14097) asks: RegNetAgents classifies *upstream*
regulator candidates by which network (TCGA vs. GREmLN) they come from and
tests enrichment of that candidate list against OncoKB. CASCADE's core
capability runs the other direction -- given a knockdown of a focal gene,
what downstream genes does network propagation predict are affected -- so
this experiment evaluates a distinct analytical output, while reusing the
same general-purpose statistical scaffold (Fisher's exact enrichment,
permutation control, BH-FDR, Stouffer Z, negative controls).

Two independent validation axes:
  1. OncoKB enrichment of predicted downstream targets (mirrors RegNetAgents
     Section 2.5, applied to a different CASCADE output).
  2. DepMap CRISPR essentiality: are predicted downstream targets more
     essential (more negative Chronos score) in lineage-matched cancer cell
     lines than random gene sets of the same size? This axis has no
     RegNetAgents analog -- it draws on a data source (DepMap) that CASCADE
     integrates and RegNetAgents does not.

Negative controls (same panels as RegNetAgents Section 2.6, for direct
comparability): housekeeping genes and tumor-expressed non-driver genes.

No core CASCADE server code is modified. Results are cached to outputs/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loader import load_tcga_network
from tools.perturb import _build_adjacency, _propagate_effect, simulate_knockdown_with_embeddings
from tools.depmap import load_depmap_data
from tools.model_inference import get_model
from tools.gene_id_mapper import get_mapper

OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
ONCOKB_CACHE_PATH = OUTPUTS_DIR / "oncokb_cancer_gene_list_cache.json"

N_PERMUTATIONS = 1000
RNG_SEED = 42
MIN_CANDIDATES_FOR_TEST = 3
# depth=2 (CASCADE's own default) compounds two ARACNe edges of noise; depth=1
# tests direct regulatory edges only -- the network's highest-confidence signal.
# Pass "1" or "2" as sys.argv[1] to select; defaults to 2 (CASCADE's default).
PROPAGATION_DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 2
# METHOD: "network" (default) or "embedding" (CASCADE's actual default for TCGA
# networks when the GREmLN model is loaded -- see experiment4_tcga_myc_concordance.py).
METHOD = sys.argv[2] if len(sys.argv) > 2 else "network"
ALPHA = 0.7
EMBEDDING_THRESHOLD = 0.1
# CASCADE's own server-wide default top_k (cascade_langgraph_mcp_server.py:1544) --
# the ranked shortlist an actual single query returns, not the full transitive
# propagation footprint (which can reach 900+ genes at depth=2 and would dilute
# any enrichment signal into an unrepresentative test of tool usage).
TOP_K = 25
_method_suffix = "_embedding" if METHOD == "embedding" else ""
RESULTS_PATH = OUTPUTS_DIR / f"cascade_validation_results_depth{PROPAGATION_DEPTH}{_method_suffix}.json"

# Focal gene panels. BRCA/COAD reuse RegNetAgents' exact OncoKB-annotated
# panels (minus genes absent from CASCADE's TCGA networks) for direct
# comparability. STAD is a cancer type RegNetAgents does not cover.
FOCAL_PANELS = {
    "brca": ["TP53", "MYC", "CTNNB1", "CCND1", "BRCA2", "PIK3CA", "PTEN",
             "RB1", "ERBB2", "ESR1", "GATA3"],
    "coad": ["TP53", "MYC", "CTNNB1", "CCND1", "KRAS", "APC", "SMAD4",
             "BRAF", "PIK3CA", "PTEN", "FBXW7", "TCF7L2"],
    "stad": ["TP53", "ARID1A", "PIK3CA", "CDH1", "KRAS", "ERBB2", "SMAD4",
             "RHOA", "CTNNB1", "APC", "FBXW7", "CCNE1"],
}

# Negative control panels. RegNetAgents' original housekeeping/neutral gene
# lists (ACTB, GAPDH, HPRT1, LDHA, TUBB, FASN, PCNA, PKM, PABPC1, VIM) were
# built for its upstream-regulator-search direction, where any gene can be a
# candidate (targets are common). CASCADE's downstream propagation instead
# requires the *perturbed* gene itself to be a network regulator (have
# outgoing edges) -- a much stricter condition that most classic
# housekeeping/structural genes fail (confirmed: 8/10 of RegNetAgents' genes
# have out-degree 0 in all three TCGA networks here, producing empty
# candidate sets). These replacement panels were selected by confirming
# out-degree >= 25 (TOP_K) in brca, coad, and stad simultaneously, and
# confirming absence from the OncoKB cancer gene list.
HOUSEKEEPING_GENES = ["SP1", "NFYA", "NFYB", "ELF1", "RBM39"]  # general/basal transcriptional machinery
NEUTRAL_GENES = ["JUND", "ATF4", "ELK1", "NFKB1", "E2F4"]  # tumor-network-active regulators, not OncoKB-annotated

# DepMap OncotreeLineage values matched to each TCGA cancer type.
CANCER_TO_LINEAGE = {
    "brca": "Breast",
    "coad": "Bowel",
    "stad": "Esophagus/Stomach",
}


def fetch_oncokb_genes() -> tuple[set[str], dict]:
    """Fetch the OncoKB cancer gene list from the public API, disk-cached."""
    if ONCOKB_CACHE_PATH.exists():
        cached = json.loads(ONCOKB_CACHE_PATH.read_text(encoding="utf-8"))
        return set(cached["genes"]), cached["meta"]

    import os
    ssl_verify = os.environ.get("CASCADE_SSL_NO_VERIFY", "0") != "1"
    url = "https://www.oncokb.org/api/v1/utils/cancerGeneList"
    resp = requests.get(url, timeout=30, verify=ssl_verify)
    resp.raise_for_status()
    data = resp.json()
    genes = sorted({entry["hugoSymbol"] for entry in data if entry.get("hugoSymbol")})

    meta = {"accessed": pd.Timestamp.now().isoformat(), "n_genes": len(genes), "source": url}
    ONCOKB_CACHE_PATH.write_text(
        json.dumps({"genes": genes, "meta": meta}, indent=2), encoding="utf-8"
    )
    return set(genes), meta


def get_candidate_set(adj: dict, gene: str, network_df: pd.DataFrame = None) -> tuple[set[str], int]:
    """Candidate set = top_k genes by |predicted effect|, matching CASCADE's own
    default single-query output (top_k=25), not the full propagation footprint.

    METHOD="network": bypasses simulate_knockdown's symbol-mapper wrapper (a
    no-op for TCGA networks -- they are already symbol-native -- but triggers
    live Ensembl API calls per gene that are unreachable in this environment),
    calling the bare BFS propagation directly instead.

    METHOD="embedding": CASCADE's actual default for TCGA networks when the
    GREmLN model is loaded -- network propagation blended with embedding
    similarity (alpha=0.7); does not have the symbol-mapper issue (it uses
    only the local gene-symbol cache by design)."""
    if METHOD == "embedding":
        model = get_model()
        ensembl_id = get_mapper().symbol_to_ensembl(gene)
        result = simulate_knockdown_with_embeddings(
            network_df, gene, model, depth=PROPAGATION_DEPTH, top_k=TOP_K, alpha=ALPHA,
            embedding_gene=ensembl_id, embedding_threshold=EMBEDDING_THRESHOLD,
        )
        top = result.get("top_affected_genes", [])
        n_total = result.get("total_affected_genes", len(top))
        return {g["symbol"] for g in top}, n_total

    effects = _propagate_effect(adj, gene, initial_effect=-1.0, depth=PROPAGATION_DEPTH)
    effects.pop(gene, None)
    n_total = len(effects)
    top = sorted(effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_K]
    return {g for g, _ in top}, n_total


def enrichment_test(candidate_set: set[str], oncokb_genes: set[str],
                     background: set[str], rng: np.random.Generator) -> dict:
    """Fisher's exact enrichment vs. OncoKB + permutation control."""
    n_candidates = len(candidate_set)
    if n_candidates < MIN_CANDIDATES_FOR_TEST:
        return {"tested": False, "n_candidates": n_candidates, "reason": "too_few_candidates"}

    background = background | candidate_set  # candidates must be in universe
    oncokb_bg = oncokb_genes & background
    overlap = len(candidate_set & oncokb_bg)

    a = overlap
    b = n_candidates - overlap
    c = len(oncokb_bg) - overlap
    d = len(background) - n_candidates - c
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")

    background_list = list(background)
    perm_ors = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        sample = set(rng.choice(background_list, size=n_candidates, replace=False))
        perm_overlap = len(sample & oncokb_bg)
        pa, pb = perm_overlap, n_candidates - perm_overlap
        pc = len(oncokb_bg) - perm_overlap
        pd_ = len(background) - n_candidates - pc
        with np.errstate(divide="ignore", invalid="ignore"):
            perm_or = ((pa + 0.5) * (pd_ + 0.5)) / ((pb + 0.5) * (pc + 0.5))
        perm_ors[i] = perm_or

    empirical_p = float((perm_ors >= odds_ratio).sum() / N_PERMUTATIONS)

    return {
        "tested": True,
        "n_candidates": n_candidates,
        "oncokb_overlap": overlap,
        "odds_ratio": round(float(odds_ratio), 4),
        "p_value": float(p_value),
        "empirical_permutation_p": empirical_p,
        "background_size": len(background),
        "oncokb_background_size": len(oncokb_bg),
    }


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in input order."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.array(p_values)[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank down
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out.tolist()


def stouffer_z(p_values: list[float], weights: list[float]) -> tuple[float, float]:
    """Weighted Stouffer's Z combining independent one-tailed p-values."""
    from scipy.stats import norm
    p_arr = np.clip(np.array(p_values), 1e-300, 1 - 1e-16)
    w_arr = np.array(weights, dtype=float)
    z_scores = norm.isf(p_arr)
    combined_z = float((w_arr * z_scores).sum() / np.sqrt((w_arr ** 2).sum()))
    combined_p = float(norm.sf(combined_z))
    return combined_z, combined_p


def depmap_essentiality_test(candidate_set: set[str], background: set[str],
                              lineage: str, depmap_scores: pd.DataFrame,
                              lineage_map: pd.Series, rng: np.random.Generator) -> dict:
    """Permutation test: are candidate genes more essential (lower Chronos) in
    lineage-matched cell lines than random gene sets of the same size?"""
    lineage_lines = lineage_map[lineage_map == lineage].index
    lineage_lines = [l for l in lineage_lines if l in depmap_scores.index]
    if not lineage_lines:
        return {"tested": False, "reason": "no_matching_cell_lines"}

    lineage_scores = depmap_scores.loc[lineage_lines]
    available_genes = set(lineage_scores.columns) & background
    candidates_available = candidate_set & available_genes
    if len(candidates_available) < MIN_CANDIDATES_FOR_TEST:
        return {"tested": False, "reason": "too_few_candidates_in_depmap"}

    observed_mean = float(lineage_scores[list(candidates_available)].mean().mean())

    background_list = list(available_genes)
    n = len(candidates_available)
    perm_means = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        sample = rng.choice(background_list, size=n, replace=False)
        perm_means[i] = lineage_scores[list(sample)].mean().mean()

    empirical_p = float((perm_means <= observed_mean).sum() / N_PERMUTATIONS)

    return {
        "tested": True,
        "n_candidates_in_depmap": n,
        "n_cell_lines": len(lineage_lines),
        "observed_mean_chronos": round(observed_mean, 4),
        "permutation_mean_chronos": round(float(perm_means.mean()), 4),
        "empirical_p": empirical_p,
    }


def run_panel(cancer_type: str, genes: list[str], adj: dict,
              background: set[str], oncokb_genes: set[str],
              depmap_scores: pd.DataFrame, lineage_map: pd.Series,
              rng: np.random.Generator, network_df: pd.DataFrame = None) -> dict:
    lineage = CANCER_TO_LINEAGE[cancer_type]
    panel_results = {}
    for gene in genes:
        if gene not in background:
            panel_results[gene] = {"skipped": True, "reason": "absent_from_network"}
            continue
        candidate_set, n_total = get_candidate_set(adj, gene, network_df)
        enrichment = enrichment_test(candidate_set, oncokb_genes, background, rng)
        essentiality = depmap_essentiality_test(
            candidate_set, background, lineage, depmap_scores, lineage_map, rng
        )
        panel_results[gene] = {
            "n_total_affected": n_total,
            "oncokb_enrichment": enrichment,
            "depmap_essentiality": essentiality,
        }
    return panel_results


def summarize_panel(panel_results: dict) -> dict:
    tested = {g: r["oncokb_enrichment"] for g, r in panel_results.items()
              if not r.get("skipped") and r["oncokb_enrichment"]["tested"]}
    if not tested:
        return {"n_tested": 0}
    p_values = [r["p_value"] for r in tested.values()]
    weights = [r["n_candidates"] for r in tested.values()]
    fdr = bh_fdr(p_values)
    for (gene, r), q in zip(tested.items(), fdr):
        r["bh_fdr"] = round(float(q), 4)
    z, combined_p = stouffer_z(p_values, weights)
    return {
        "n_tested": len(tested),
        "stouffer_z": round(z, 4),
        "stouffer_p": combined_p,
        "n_significant_fdr05": sum(1 for q in fdr if q < 0.05),
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("Fetching OncoKB cancer gene list...")
    oncokb_genes, oncokb_meta = fetch_oncokb_genes()
    print(f"  {oncokb_meta['n_genes']} genes (accessed {oncokb_meta.get('accessed', 'cached')})")

    print("Loading DepMap CRISPR essentiality data...")
    depmap_scores, lineage_map = load_depmap_data()

    all_results = {"oncokb_meta": oncokb_meta, "cancer_types": {}}

    for cancer_type in FOCAL_PANELS:
        print(f"\n=== {cancer_type.upper()} ===")
        network_df = load_tcga_network(cancer_type)
        if isinstance(network_df, dict) and "error" in network_df:
            print(f"  ERROR loading network: {network_df['error']}")
            continue
        background = set(network_df["regulator"].unique()) | set(network_df["target"].unique())
        print(f"  Network background: {len(background)} genes")
        adj = _build_adjacency(network_df)

        focal_results = run_panel(
            cancer_type, FOCAL_PANELS[cancer_type], adj, background,
            oncokb_genes, depmap_scores, lineage_map, rng, network_df
        )
        hk_results = run_panel(
            cancer_type, HOUSEKEEPING_GENES, adj, background,
            oncokb_genes, depmap_scores, lineage_map, rng, network_df
        )
        neutral_results = run_panel(
            cancer_type, NEUTRAL_GENES, adj, background,
            oncokb_genes, depmap_scores, lineage_map, rng, network_df
        )

        focal_summary = summarize_panel(focal_results)
        hk_summary = summarize_panel(hk_results)
        neutral_summary = summarize_panel(neutral_results)

        print(f"  Focal panel:   Stouffer Z={focal_summary.get('stouffer_z')}, "
              f"p={focal_summary.get('stouffer_p')}, "
              f"{focal_summary.get('n_significant_fdr05')}/{focal_summary.get('n_tested')} FDR<0.05")
        print(f"  Housekeeping:  Stouffer Z={hk_summary.get('stouffer_z')}, "
              f"p={hk_summary.get('stouffer_p')}")
        print(f"  Neutral:       Stouffer Z={neutral_summary.get('stouffer_z')}, "
              f"p={neutral_summary.get('stouffer_p')}")

        all_results["cancer_types"][cancer_type] = {
            "focal_panel": {"genes": focal_results, "summary": focal_summary},
            "housekeeping_panel": {"genes": hk_results, "summary": hk_summary},
            "neutral_panel": {"genes": neutral_results, "summary": neutral_summary},
        }

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
