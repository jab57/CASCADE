"""
PAM50 subtype control for experiment4_tcga_myc_concordance.py (BRCA).

RESEARCH_PAPER_PLAN.md flagged a real confound: MYC amplification is not
uniformly distributed across PAM50 molecular subtypes (observed here:
BRCA_Basal 61 amp/13 non-amp, BRCA_LumA 36/251, BRCA_LumB 32/33, BRCA_Her2
18/20, BRCA_Normal 6/21). If predicted-target concordance were actually
driven by subtype differences rather than MYC dosage itself, restricting to
a single subtype (holding it constant) should collapse the signal toward
chance. BRCA_LumB has the most balanced amp/non-amp split (32/33) of any
subtype, making it the best-powered within-subtype test available.

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
RESULTS_PATH = OUTPUTS_DIR / "experiment4_pam50_control_lumb.json"

_BASE_URL = "https://www.cbioportal.org/api"
_SSL_VERIFY = os.environ.get("CASCADE_SSL_NO_VERIFY", "0") != "1"
_STUDY = "brca_tcga_pan_can_atlas_2018"
TOP_N = 50
PROPAGATION_DEPTH = 2
ALPHA = 0.7  # CASCADE's own default (cascade_langgraph_workflow.py)
EMBEDDING_THRESHOLD = 0.1  # CASCADE's own default for the TCGA path
TARGET_SUBTYPE = "BRCA_LumB"


def batch_resolve_entrez(symbols: list[str]) -> dict[str, int]:
    resp = requests.post(f"{_BASE_URL}/genes/fetch", params={"geneIdType": "HUGO_GENE_SYMBOL"},
                          json=symbols, timeout=30, verify=_SSL_VERIFY)
    resp.raise_for_status()
    return {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in resp.json()}


def fetch_subtype_map() -> dict[str, str]:
    patients = [p["patientId"] for p in
                requests.get(f"{_BASE_URL}/studies/{_STUDY}/patients", timeout=20, verify=_SSL_VERIFY).json()]
    resp = requests.post(f"{_BASE_URL}/studies/{_STUDY}/clinical-data/fetch",
                          params={"clinicalDataType": "PATIENT"},
                          json={"attributeIds": ["SUBTYPE"], "ids": patients}, timeout=30, verify=_SSL_VERIFY)
    resp.raise_for_status()
    return {d["patientId"]: d["value"] for d in resp.json()}


def fetch_cna_per_sample(entrez_id: int) -> dict[str, int]:
    resp = requests.get(f"{_BASE_URL}/molecular-profiles/{_STUDY}_gistic/molecular-data",
                         params={"sampleListId": f"{_STUDY}_cna", "entrezGeneId": entrez_id, "projection": "SUMMARY"},
                         timeout=30, verify=_SSL_VERIFY)
    resp.raise_for_status()
    return {d["sampleId"]: d["value"] for d in resp.json()}


def batch_fetch_expression(entrez_ids: list[int]) -> dict[int, dict[str, float]]:
    resp = requests.post(f"{_BASE_URL}/molecular-profiles/{_STUDY}_rna_seq_v2_mrna_median_Zscores/molecular-data/fetch",
                          params={"projection": "SUMMARY"},
                          json={"entrezGeneIds": entrez_ids, "sampleListId": f"{_STUDY}_rna_seq_v2_mrna"},
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
    print("Getting CASCADE's MYC knockdown predictions (BRCA)...")
    network_df = load_tcga_network("brca")
    predicted = get_myc_predicted_targets(network_df, TOP_N)

    print("Fetching PAM50 subtype and MYC CNA status...")
    subtype_map = fetch_subtype_map()
    myc_entrez = batch_resolve_entrez(["MYC"])["MYC"]
    myc_cna = fetch_cna_per_sample(myc_entrez)

    def patient_of(sample_id: str) -> str:
        return sample_id.rsplit("-", 1)[0]

    lumb_amp = {s for s, v in myc_cna.items() if v == 2 and subtype_map.get(patient_of(s)) == TARGET_SUBTYPE}
    lumb_nonamp = {s for s, v in myc_cna.items() if v == 0 and subtype_map.get(patient_of(s)) == TARGET_SUBTYPE}
    print(f"  {TARGET_SUBTYPE}: {len(lumb_amp)} MYC-amplified, {len(lumb_nonamp)} non-amplified samples")

    pred_symbols = [g for g, _ in predicted]
    pred_entrez_map = batch_resolve_entrez(pred_symbols)
    resolved = [(g, d, pred_entrez_map[g]) for g, d in predicted if g in pred_entrez_map]

    print("Fetching per-sample expression for predicted targets...")
    expr = batch_fetch_expression([eid for _, _, eid in resolved])

    gene_results = []
    for gene, direction, eid in resolved:
        gene_expr = expr.get(eid, {})
        amp_vals = [v for s, v in gene_expr.items() if s in lumb_amp]
        nonamp_vals = [v for s, v in gene_expr.items() if s in lumb_nonamp]
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
    print(f"  {n_concordant}/{n_tested} concordant within {TARGET_SUBTYPE} alone")

    if n_tested > 0:
        binom_result = binomtest(n_concordant, n_tested, p=0.5, alternative="greater")
        print(f"  Binomial test (within-subtype): p={binom_result.pvalue:.4f}")
        p_value = float(binom_result.pvalue)
    else:
        p_value = None

    output = {
        "target_subtype": TARGET_SUBTYPE,
        "n_amp_samples": len(lumb_amp),
        "n_nonamp_samples": len(lumb_nonamp),
        "n_tested": n_tested,
        "n_concordant": n_concordant,
        "observed_concordance_rate": round(n_concordant / n_tested, 4) if n_tested else None,
        "binomial_p_value": p_value,
        "gene_results": gene_results,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
