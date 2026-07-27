"""
Experiment 5 (LINCS coverage feasibility check): can CASCADE's
lineage-identity generalization-panel failures (GATA3, FOXA1, SOX9, ESR1 --
0-38% concordance against the amplification-as-dosage-proxy test in
experiment4_tcga_myc_concordance.py, at or below the permutation baseline)
be checked against real experimental knockdown data instead of the
copy-number proxy?

This script answers only the feasibility question, not the concordance
question itself. It checks, using data already downloaded locally for the
issue #19 raw-LINCS investigation (data/lincs_raw/GSE106127_CGS_*, shRNA
consensus gene signatures, 978 L1000 landmark genes x 33,839 signatures
across 15 cell lines):

1. Does an MCF7 knockdown signature exist for each focal gene at all, and
   how many independent shRNA reagents / how internally consistent is it
   (distil_nsample, distil_cc_q75)?
2. Of CASCADE's own top-N predicted knockdown targets for that gene (same
   panel used throughout the paper's concordance tests), how many are even
   measured on the 978-gene L1000 landmark platform?

A full concordance test (pulling per-target z-scores and comparing
direction against CASCADE's predictions, as in experiment4) is only
statistically meaningful if enough predicted targets clear the platform's
landmark-gene coverage. No z-scores are read from the .gctx matrix here --
only the meta/gene_info index files -- so this step is cheap to re-run.

No core CASCADE server code is modified -- read-only queries against local
TCGA network files and the local GSE106127 index files. Results cached to
outputs/.
"""

import asyncio
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loader import load_tcga_network
from tools.perturb import _build_adjacency, _propagate_effect
from cascade_langgraph_workflow import CascadeWorkflow

OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
RESULTS_PATH = OUTPUTS_DIR / "experiment5_lincs_coverage_check.json"

LINCS_RAW_DIR = ROOT / "data" / "lincs_raw"
META_PATH = LINCS_RAW_DIR / "GSE106127_CGS_meta.txt.gz"
GENE_INFO_PATH = LINCS_RAW_DIR / "GSE106127_gene_info.txt.gz"

# The four lineage-identity genes whose amplification-based concordance was
# null or negative in the generalization panel (RESEARCH_PAPER_PLAN.md).
FOCAL_GENES = ["GATA3", "FOXA1", "SOX9", "ESR1"]
CELL_LINE = "MCF7"
CASCADE_CANCER_TYPE = "brca"
TOP_N = 50  # matches the panel size used throughout the paper's concordance tests
PROPAGATION_DEPTH = 2  # CASCADE's own default
ALPHA = 0.7  # CASCADE's own default embedding/network blend weight
EMBEDDING_THRESHOLD = 0.1  # CASCADE's own default for the TCGA embedding path
METHOD = sys.argv[1] if len(sys.argv) > 1 else "network"  # matches experiment4's own default; "embedding"
# requires a live Ensembl API lookup for symbol resolution and silently returns an
# empty target list if that call fails (e.g. no network access / SSL issues), so it
# is opt-in here rather than default -- verify target counts are non-zero before
# trusting an "embedding" run's coverage numbers.
MIN_OVERLAP = 10  # coverage threshold below which a concordance test isn't well-powered


def load_mcf7_signature_index() -> tuple:
    """gene symbol -> {n_reagents, replicate_consistency} for MCF7 shRNA
    consensus signatures (empty if no MCF7 signature exists for that gene),
    plus a second gene -> count of signatures in ANY cell line, so a missing
    MCF7 entry can be distinguished from "never in this study's shRNA reagent
    library at all" (0 rows anywhere) vs. "profiled elsewhere but not MCF7"."""
    index = {}
    any_cell_line_counts = {gene: 0 for gene in FOCAL_GENES}
    with gzip.open(META_PATH, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["pert_iname"] not in FOCAL_GENES:
                continue
            any_cell_line_counts[row["pert_iname"]] += 1
            if row["cell_id"] != CELL_LINE:
                continue
            index[row["pert_iname"]] = {
                "n_reagents": int(row["distil_nsample"]),
                "replicate_consistency_cc_q75": float(row["distil_cc_q75"]),
            }
    return index, any_cell_line_counts


def load_landmark_genes() -> set:
    genes = set()
    with gzip.open(GENE_INFO_PATH, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            genes.add(row["pr_gene_symbol"])
    return genes


async def _run_workflow_knockdown(tcga_network: str, focal_gene: str, top_k: int) -> dict:
    workflow = CascadeWorkflow()
    return await workflow.run(
        gene=focal_gene,
        cell_type=tcga_network,
        perturbation_type="knockdown",
        analysis_depth="focused",
        top_k=top_k,
        network_source="tcga",
        tcga_network=tcga_network,
    )


def get_predicted_targets(network_df, focal_gene: str, top_n: int) -> list:
    """Same logic as experiment4_tcga_myc_concordance.py's get_predicted_targets
    (network-only or embedding-blended, per METHOD) -- returns gene symbols only,
    direction isn't needed for a coverage check."""
    if METHOD == "embedding":
        result = asyncio.run(_run_workflow_knockdown(CASCADE_CANCER_TYPE, focal_gene, top_n))
        affected = result.get("top_affected_genes", [])
        return [g["symbol"] for g in affected]

    adj = _build_adjacency(network_df)
    effects = _propagate_effect(adj, focal_gene, initial_effect=-1.0, depth=PROPAGATION_DEPTH)
    effects.pop(focal_gene, None)
    ranked = sorted(effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return [g for g, _ in ranked]


def main() -> None:
    mcf7_index, any_cell_line_counts = load_mcf7_signature_index()
    landmark_genes = load_landmark_genes()
    print(f"Loaded MCF7 signature index ({len(mcf7_index)}/{len(FOCAL_GENES)} focal genes have a signature) "
          f"and {len(landmark_genes)} L1000 landmark genes.")

    network_df = load_tcga_network(CASCADE_CANCER_TYPE)

    results = {}
    for gene in FOCAL_GENES:
        sig = mcf7_index.get(gene)
        if sig is None:
            n_any = any_cell_line_counts[gene]
            reason = ("never in this study's shRNA reagent library (0 signatures in any of the 15 "
                      "cell lines profiled)" if n_any == 0 else
                      f"profiled in {n_any} other cell line(s) but not MCF7")
            results[gene] = {
                "mcf7_signature_exists": False,
                "n_signatures_any_cell_line": n_any,
                "n_predicted_targets": None,
                "n_overlap_with_landmark": None,
                "coverage_sufficient": False,
            }
            print(f"{gene}: no MCF7 shRNA signature in GSE106127 ({reason}) -- excluded, not testable with this dataset.")
            continue

        print(f"Getting CASCADE's top-{TOP_N} {gene} knockdown predictions (method={METHOD})...")
        targets = get_predicted_targets(network_df, gene, TOP_N)
        overlap = [t for t in targets if t in landmark_genes]
        sufficient = len(overlap) >= MIN_OVERLAP

        results[gene] = {
            "mcf7_signature_exists": True,
            "n_reagents": sig["n_reagents"],
            "replicate_consistency_cc_q75": sig["replicate_consistency_cc_q75"],
            "n_predicted_targets": len(targets),
            "n_overlap_with_landmark": len(overlap),
            "coverage_sufficient": sufficient,
        }
        status = "SUFFICIENT" if sufficient else f"INSUFFICIENT (< {MIN_OVERLAP})"
        print(f"{gene}: MCF7 signature present ({sig['n_reagents']} reagents, "
              f"cc_q75={sig['replicate_consistency_cc_q75']:.3f}); "
              f"{len(overlap)}/{len(targets)} predicted targets on L1000 landmark panel -- {status}")

    n_testable = sum(1 for r in results.values() if r.get("coverage_sufficient"))
    print(f"\n{n_testable}/{len(FOCAL_GENES)} genes clear the coverage threshold for a LINCS-based "
          f"directional concordance test using this dataset (GSE106127 shRNA, MCF7, L1000 landmark-only).")

    output = {
        "method": METHOD,
        "top_n": TOP_N,
        "min_overlap_threshold": MIN_OVERLAP,
        "cell_line": CELL_LINE,
        "source": "GSE106127 (shRNA consensus gene signatures, 978 L1000 landmark genes)",
        "results": results,
        "n_testable": n_testable,
        "n_focal_genes": len(FOCAL_GENES),
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
