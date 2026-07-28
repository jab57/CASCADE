"""
Hallmark-baseline sanity check for the TCGA MYC concordance methodology
(experiment4_tcga_myc_concordance.py).

Question this answers: does CASCADE's specific network-propagation-derived
MYC-knockdown target list do any better than a naive, generic "these are
MYC target genes" list at predicting real amplified-vs-non-amplified
expression differences? If a public gene set with no CASCADE reasoning
behind it scores comparably, the concordance test would be validating
"MYC-amplified tumors are proliferative" in general, not CASCADE's specific
predictions.

Gene set: MSigDB Hallmark HALLMARK_MYC_TARGETS_V1 (Liberzon et al. 2015,
PMID 26771021; systematic name M5926; "a subgroup of genes regulated by
MYC - version 1"), 200 genes. Extracted directly from MSigDB's raw gene
member table (gsea-msigdb.org), not reconstructed from memory or an LLM
summary of the page -- a first extraction attempt via page-summarization
produced fabricated gene symbols not present in the actual table, so the
raw HTML was parsed directly instead.

Every gene in this set is treated as positively regulated by MYC (the
gene set's own definition), so the naive predicted direction for all 200
genes is uniformly "down" upon MYC knockdown -- no per-gene reasoning,
unlike CASCADE's individual signed network propagation. MYC itself is
excluded (it is the perturbed gene, not a downstream target).

Reuses experiment4_tcga_myc_concordance.py's cBioPortal fetch and
concordance/permutation logic unmodified, for direct comparability with
the existing CASCADE-prediction results already in outputs/.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loader import load_tcga_network
from scripts.experiment4_tcga_myc_concordance import (
    CASCADE_TO_CBIOPORTAL,
    batch_resolve_entrez,
    fetch_cna_per_sample,
    batch_fetch_expression,
    concordance_direction,
)

OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

CASCADE_CANCER_TYPE = sys.argv[1] if len(sys.argv) > 1 else "brca"
GENE_SET_NAME = sys.argv[2] if len(sys.argv) > 2 else "myc"  # "myc" or "e2f"
assert CASCADE_CANCER_TYPE in CASCADE_TO_CBIOPORTAL
RESULTS_PATH = OUTPUTS_DIR / f"experiment4_hallmark_baseline_{GENE_SET_NAME}_{CASCADE_CANCER_TYPE}.json"

BACKGROUND_POOL_SIZE = 200
N_PERMUTATIONS = 1000
RNG_SEED = 42

# HALLMARK_MYC_TARGETS_V1, MSigDB systematic name M5926, 200 genes.
# Source: https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_MYC_TARGETS_V1.html
# Extracted verbatim from the page's raw gene member table (see module docstring).
HALLMARK_MYC_TARGETS_V1 = [
    "ABCE1", "ACP1", "AIMP2", "AP3S1", "APEX1", "BUB3", "C1QBP", "CAD", "CANX", "CBX3",
    "CCNA2", "CCT2", "CCT3", "CCT4", "CCT5", "CCT7", "CDC20", "CDC45", "CDK2", "CDK4",
    "CLNS1A", "CNBP", "COPS5", "COX5A", "CSTF2", "CTPS", "CUL1", "CYC1", "DDX18", "DDX21",
    "DEK", "DHX15", "DUT", "EEF1B2", "EIF1AX", "EIF2S1", "EIF2S2", "EIF3B", "EIF3D", "EIF3J",
    "EIF4A1", "EIF4E", "EIF4G2", "EIF4H", "EPRS", "ERH", "ETF1", "EXOSC7", "FAM120A", "FBL",
    "G3BP1", "GLO1", "GNB2L1", "GNL3", "GOT2", "GSPT1", "H2AFZ", "HDAC2", "HDDC2", "HDGF",
    "HNRNPA1", "HNRNPA2B1", "HNRNPA3", "HNRNPC", "HNRNPD", "HNRNPR", "HNRNPU", "HPRT1", "HSP90AB1", "HSPD1",
    "HSPE1", "IARS", "IFRD1", "ILF2", "IMPDH2", "KARS", "KPNA2", "KPNB1", "LDHA", "LSM2",
    "LSM7", "MAD2L1", "MCM2", "MCM4", "MCM5", "MCM6", "MCM7", "MRPL23", "MRPL9", "MRPS18B",
    "MYC", "NAP1L1", "NCBP1", "NCBP2", "NDUFAB1", "NHP2", "NME1", "NOLC1", "NOP16", "NOP56",
    "NPM1", "ODC1", "ORC2", "PA2G4", "PABPC1", "PABPC4", "PCBP1", "PCNA", "PGK1", "PHB",
    "PHB2", "POLD2", "POLE3", "PPIA", "PPM1G", "PRDX3", "PRDX4", "PRPF31", "PRPS2", "PSMA1",
    "PSMA2", "PSMA4", "PSMA6", "PSMA7", "PSMB2", "PSMB3", "PSMC4", "PSMC6", "PSMD1", "PSMD14",
    "PSMD3", "PSMD7", "PSMD8", "PTGES3", "PWP1", "RAD23B", "RAN", "RANBP1", "RFC4", "RNPS1",
    "RPL14", "RPL18", "RPL22", "RPL34", "RPL6", "RPLP0", "RPS10", "RPS2", "RPS3", "RPS5",
    "RPS6", "RRM1", "RRP9", "RSL1D1", "RUVBL2", "SERBP1", "SET", "SF3A1", "SF3B3", "SLC25A3",
    "SMARCC1", "SNRPA", "SNRPA1", "SNRPB2", "SNRPD1", "SNRPD2", "SNRPD3", "SNRPG", "SRM", "SRPK1",
    "SRSF1", "SRSF2", "SRSF3", "SRSF7", "SSB", "SSBP1", "STARD7", "SYNCRIP", "TARDBP", "TCP1",
    "TFDP1", "TOMM70A", "TRA2B", "TRIM28", "TUFM", "TXNL4A", "TYMS", "U2AF1", "UBA2", "UBE2E1",
    "UBE2L3", "USP1", "VBP1", "VDAC1", "VDAC3", "XPO1", "XPOT", "XRCC6", "YWHAE", "YWHAQ",
]
assert len(HALLMARK_MYC_TARGETS_V1) == 200, f"expected 200 genes, got {len(HALLMARK_MYC_TARGETS_V1)}"

# HALLMARK_E2F_TARGETS, MSigDB systematic name M5925, 200 genes.
# Source: https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_E2F_TARGETS.html
# Extracted verbatim from the page's raw gene member table (same method as above).
# Note: substantially overlaps CASCADE's own tested proliferation-machinery panel
# (contains MYC, AURKA, CCNE1, TOP2A, CDK4, MYBL2, MCM2-7, among others).
HALLMARK_E2F_TARGETS = [
    "AK2", "ANP32E", "ASF1A", "ASF1B", "ATAD2", "AURKA", "AURKB", "BARD1", "BIRC5", "BRCA1",
    "BRCA2", "BRMS1L", "BUB1B", "CBX5", "CCNB2", "CCNE1", "CCP110", "CDC20", "CDC25A", "CDC25B",
    "CDCA3", "CDCA8", "CDK1", "CDK4", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2C", "CDKN3", "CENPE",
    "CENPM", "CHEK1", "CHEK2", "CIT", "CKS1B", "CKS2", "CSE1L", "CTCF", "CTPS", "DCK",
    "DCLRE1B", "DCTPP1", "DDX39A", "DEK", "DEPDC1", "DIAPH3", "DLGAP5", "DNMT1", "DONSON", "DSCC1",
    "DUT", "E2F8", "EED", "EIF2S1", "ESPL1", "EXOSC8", "EZH2", "GINS1", "GINS3", "GINS4",
    "GSPT1", "H2AFX", "H2AFZ", "HELLS", "HMGA1", "HMGB2", "HMGB3", "HMMR", "HN1", "HNRNPD",
    "HUS1", "ILF3", "ING3", "IPO7", "KIF18B", "KIF22", "KIF2C", "KIF4A", "KPNA2", "LBR",
    "LIG1", "LMNB1", "LUC7L3", "LYAR", "MAD2L1", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6",
    "MCM7", "MELK", "MKI67", "MLH1", "MMS22L", "MRE11A", "MSH2", "MTHFD2", "MXD3", "MYBL2",
    "MYC", "NAA38", "NAP1L1", "NASP", "NBN", "NCAPD2", "NME1", "NOLC1", "NOP56", "NUDT21",
    "NUP107", "NUP153", "NUP205", "ORC2", "ORC6", "PA2G4", "PAICS", "PAN2", "PCNA", "PDS5B",
    "PHF5A", "PLK1", "PLK4", "PMS2", "PNN", "POLA2", "POLD1", "POLD2", "POLD3", "POLE",
    "POLE4", "POP7", "PPM1D", "PPP1R8", "PRDX4", "PRIM2", "PRKDC", "PRPS1", "PSIP1", "PSMC3IP",
    "PTTG1", "RACGAP1", "RAD1", "RAD21", "RAD50", "RAD51AP1", "RAD51C", "RAN", "RANBP1", "RBBP7",
    "RFC1", "RFC2", "RFC3", "RNASEH2A", "RPA1", "RPA2", "RPA3", "RQCD1", "RRM2", "SHMT1",
    "SLBP", "SMC1A", "SMC3", "SMC4", "SMC6", "SNRPB", "SPAG5", "SPC24", "SPC25", "SRSF1",
    "SRSF2", "SSRP1", "STAG1", "STMN1", "SUV39H1", "SYNCRIP", "TACC3", "TBRG4", "TCF19", "TFRC",
    "TIMELESS", "TIPIN", "TK1", "TMPO", "TOP2A", "TP53", "TRA2B", "TRIP13", "TUBB", "TUBG1",
    "UBE2S", "UBE2T", "UBR7", "UNG", "USP1", "WDR90", "WEE1", "XPO1", "XRCC6", "ZW10",
]
assert len(HALLMARK_E2F_TARGETS) == 200, f"expected 200 genes, got {len(HALLMARK_E2F_TARGETS)}"

GENE_SETS = {
    "myc": ("MYC", "HALLMARK_MYC_TARGETS_V1", "MSigDB M5926", HALLMARK_MYC_TARGETS_V1),
    "e2f": ("E2F3", "HALLMARK_E2F_TARGETS", "MSigDB M5925", HALLMARK_E2F_TARGETS),
}
FOCAL_GENE, GENE_SET_LABEL, GENE_SET_SOURCE, GENE_SET_LIST = GENE_SETS[GENE_SET_NAME]


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    target_genes = [g for g in GENE_SET_LIST if g != FOCAL_GENE]
    print(f"{GENE_SET_LABEL} baseline ({CASCADE_CANCER_TYPE.upper()}): "
          f"{len(target_genes)} genes (excluding {FOCAL_GENE} itself), all assigned "
          f"naive predicted direction 'down' (no CASCADE propagation involved).")

    print(f"Fetching {FOCAL_GENE} copy-number status per patient ({CASCADE_CANCER_TYPE.upper()}, cBioPortal)...")
    focal_entrez = batch_resolve_entrez([FOCAL_GENE])[FOCAL_GENE]
    focal_cna = fetch_cna_per_sample(focal_entrez)
    amplified = {s for s, v in focal_cna.items() if v == 2}
    nonamplified = {s for s, v in focal_cna.items() if v == 0}
    print(f"  {len(amplified)} {FOCAL_GENE}-amplified samples, {len(nonamplified)} non-amplified samples")

    print("Resolving Hallmark gene symbols to Entrez IDs...")
    entrez_map = batch_resolve_entrez(target_genes)
    resolved = [(g, entrez_map[g]) for g in target_genes if g in entrez_map]
    print(f"  {len(resolved)}/{len(target_genes)} resolved")

    print("Fetching per-sample expression for Hallmark genes...")
    expr_by_gene = batch_fetch_expression([eid for _, eid in resolved])

    gene_results = []
    for gene, eid in resolved:
        expr = expr_by_gene.get(eid, {})
        amp_vals = [v for s, v in expr.items() if s in amplified]
        nonamp_vals = [v for s, v in expr.items() if s in nonamplified]
        if len(amp_vals) < 10 or len(nonamp_vals) < 10:
            gene_results.append({"gene": gene, "direction": "down", "tested": False,
                                  "reason": "insufficient_samples"})
            continue
        amp_mean = float(np.mean(amp_vals))
        nonamp_mean = float(np.mean(nonamp_vals))
        concordant = concordance_direction(amp_mean, nonamp_mean, "down")
        gene_results.append({
            "gene": gene, "direction": "down", "tested": True,
            "amp_mean_zscore": round(amp_mean, 4), "nonamp_mean_zscore": round(nonamp_mean, 4),
            "n_amp": len(amp_vals), "n_nonamp": len(nonamp_vals), "concordant": concordant,
        })

    tested = [r for r in gene_results if r["tested"]]
    n_concordant = sum(1 for r in tested if r["concordant"])
    n_tested = len(tested)
    print(f"  {n_concordant}/{n_tested} concordant")

    binom_result = binomtest(n_concordant, n_tested, p=0.5, alternative="greater")
    print(f"  Binomial test: p={binom_result.pvalue:.4f}")

    # Same background-permutation control as experiment4_tcga_myc_concordance.py,
    # but with frac_down=1.0 (every Hallmark gene predicted "down"), matching this
    # baseline's naive, uniform-direction assumption.
    print(f"\nBuilding background pool ({BACKGROUND_POOL_SIZE} random network genes) for permutation control...")
    network_df = load_tcga_network(CASCADE_CANCER_TYPE)
    all_genes = sorted(set(network_df["regulator"].unique()) | set(network_df["target"].unique()))
    hallmark_set = set(target_genes)
    candidate_bg = [g for g in all_genes if g not in hallmark_set and g != FOCAL_GENE]
    bg_sample = list(rng.choice(candidate_bg, size=min(BACKGROUND_POOL_SIZE, len(candidate_bg)), replace=False))

    bg_entrez_map = batch_resolve_entrez(bg_sample)
    bg_resolved = [(g, bg_entrez_map[g]) for g in bg_sample if g in bg_entrez_map]
    print(f"  {len(bg_resolved)}/{len(bg_sample)} background genes resolved")

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

    bg_directions_arr = np.array(bg_directions)
    perm_concordant_rates = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        sample_idx = rng.integers(0, len(bg_directions_arr), size=n_tested)
        sampled_amp_higher = bg_directions_arr[sample_idx]
        # frac_down = 1.0 for this baseline -> every draw is "predicted down" (expect amp higher)
        concordant = sampled_amp_higher == True
        perm_concordant_rates[i] = concordant.mean()

    observed_rate = n_concordant / n_tested
    empirical_p = float((perm_concordant_rates >= observed_rate).sum() / N_PERMUTATIONS)
    print(f"  Permutation empirical p = {empirical_p:.4f} (observed rate={observed_rate:.3f}, "
          f"permutation mean={perm_concordant_rates.mean():.3f})")

    output = {
        "focal_gene": FOCAL_GENE,
        "gene_set": GENE_SET_LABEL,
        "gene_set_source": GENE_SET_SOURCE,
        "cancer_type": CASCADE_CANCER_TYPE,
        "n_gene_set": len(target_genes),
        "n_resolved": len(resolved),
        "n_tested": n_tested,
        "n_concordant": n_concordant,
        "observed_concordance_rate": round(observed_rate, 4),
        "binomial_p_value": float(binom_result.pvalue),
        "permutation_empirical_p": empirical_p,
        "permutation_mean_rate": round(float(perm_concordant_rates.mean()), 4),
        "n_amplified_samples": len(amplified),
        "n_nonamplified_samples": len(nonamplified),
        "n_background_pool": len(bg_directions),
        "gene_results": gene_results,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
