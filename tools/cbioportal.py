"""
cBioPortal TCGA Primary Tumor Data Module

Queries the TCGA PanCancer Atlas 2018 via the cBioPortal REST API to provide
primary tumor mRNA expression and somatic alteration data — complementing
cell-line-based sources (DepMap, LINCS) with primary patient tissue evidence.

API: https://www.cbioportal.org/api
Data: TCGA PanCancer Atlas 2018 (~10,000 primary tumor samples, 32 cancer types)
"""

import os

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

_BASE_URL = "https://www.cbioportal.org/api"
_TIMEOUT = 15
_MAX_WORKERS = 8  # concurrent API calls per function call
_SSL_VERIFY = os.environ.get("CASCADE_SSL_NO_VERIFY", "0") != "1"

# TCGA PanCancer Atlas 2018 — 32 cancer types
# Keys: cBioPortal study prefix; Values: human-readable cancer type name
_TCGA_STUDIES = {
    "acc":      "Adrenocortical Carcinoma",
    "blca":     "Bladder Urothelial Carcinoma",
    "brca":     "Breast Invasive Carcinoma",
    "cesc":     "Cervical SCC & Adenocarcinoma",
    "chol":     "Cholangiocarcinoma",
    "coadread": "Colorectal Adenocarcinoma",
    "dlbc":     "Diffuse Large B-cell Lymphoma",
    "esca":     "Esophageal Carcinoma",
    "gbm":      "Glioblastoma Multiforme",
    "hnsc":     "Head and Neck SCC",
    "kich":     "Kidney Chromophobe",
    "kirc":     "Kidney Renal Clear Cell",
    "kirp":     "Kidney Renal Papillary Cell",
    "laml":     "Acute Myeloid Leukemia",
    "lgg":      "Brain Lower Grade Glioma",
    "lihc":     "Liver Hepatocellular Carcinoma",
    "luad":     "Lung Adenocarcinoma",
    "lusc":     "Lung Squamous Cell Carcinoma",
    "meso":     "Mesothelioma",
    "ov":       "Ovarian Serous Cystadenocarcinoma",
    "paad":     "Pancreatic Adenocarcinoma",
    "pcpg":     "Pheochromocytoma & Paraganglioma",
    "prad":     "Prostate Adenocarcinoma",
    "sarc":     "Sarcoma",
    "skcm":     "Skin Cutaneous Melanoma",
    "stad":     "Stomach Adenocarcinoma",
    "tgct":     "Testicular Germ Cell Tumors",
    "thca":     "Thyroid Carcinoma",
    "thym":     "Thymoma",
    "ucec":     "Uterine Corpus Endometrial",
    "ucs":      "Uterine Carcinosarcoma",
    "uvm":      "Uveal Melanoma",
}


def _get_entrez_id(gene_symbol: str) -> Optional[int]:
    """Resolve gene symbol to Entrez ID via cBioPortal API."""
    try:
        resp = requests.get(f"{_BASE_URL}/genes/{gene_symbol}", timeout=_TIMEOUT, verify=_SSL_VERIFY)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("entrezGeneId")
    except requests.RequestException:
        return None


def _fetch_expression_for_study(cancer_prefix: str, entrez_id: int) -> tuple:
    """Fetch mRNA z-scores for one TCGA PanCancer Atlas study.

    Returns (cancer_prefix, list_of_z_scores).
    Returns empty list on 404 (study/profile doesn't exist) or any error.
    """
    suffix = "_tcga_pan_can_atlas_2018"
    profile_id = f"{cancer_prefix}{suffix}_rna_seq_v2_mrna_median_Zscores"
    sample_list_id = f"{cancer_prefix}{suffix}_rna_seq_v2_mrna"

    try:
        resp = requests.get(
            f"{_BASE_URL}/molecular-profiles/{profile_id}/molecular-data",
            params={
                "sampleListId": sample_list_id,
                "entrezGeneId": entrez_id,
                "projection": "SUMMARY",
            },
            timeout=_TIMEOUT,
            verify=_SSL_VERIFY,
        )
        if resp.status_code == 404:
            return cancer_prefix, []
        resp.raise_for_status()
        data = resp.json()
        values = [d["value"] for d in data if d.get("value") is not None]
        return cancer_prefix, values
    except Exception:
        return cancer_prefix, []


def _fetch_alterations_for_study(cancer_prefix: str, entrez_id: int) -> dict:
    """Fetch mutation and GISTIC CNA data for one TCGA PanCancer Atlas study.

    Returns a dict with keys: cancer_prefix, mut_count, mut_total,
    amp_count, del_count, cna_total.
    """
    suffix = "_tcga_pan_can_atlas_2018"
    result = {
        "cancer_prefix": cancer_prefix,
        "mut_count": 0,
        "mut_total": 0,
        "amp_count": 0,
        "del_count": 0,
        "cna_total": 0,
    }

    # --- Mutations ---
    mut_profile = f"{cancer_prefix}{suffix}_mutations"
    seq_list = f"{cancer_prefix}{suffix}_sequenced"
    try:
        mut_resp = requests.get(
            f"{_BASE_URL}/molecular-profiles/{mut_profile}/mutations",
            params={
                "sampleListId": seq_list,
                "entrezGeneId": entrez_id,
                "projection": "SUMMARY",
            },
            timeout=_TIMEOUT,
            verify=_SSL_VERIFY,
        )
        if mut_resp.status_code == 200:
            mutations = mut_resp.json()
            result["mut_count"] = len(
                {m["sampleId"] for m in mutations if "sampleId" in m}
            )
            sl_resp = requests.get(
                f"{_BASE_URL}/sample-lists/{seq_list}", timeout=_TIMEOUT, verify=_SSL_VERIFY
            )
            if sl_resp.status_code == 200:
                result["mut_total"] = sl_resp.json().get("sampleCount", 0)
    except Exception:
        pass

    # --- GISTIC discrete CNA ---
    # value: 2=amplification, 1=gain, 0=diploid, -1=hetloss, -2=homdel
    cna_profile = f"{cancer_prefix}{suffix}_gistic"
    cna_list = f"{cancer_prefix}{suffix}_cna"
    try:
        cna_resp = requests.get(
            f"{_BASE_URL}/molecular-profiles/{cna_profile}/molecular-data",
            params={
                "sampleListId": cna_list,
                "entrezGeneId": entrez_id,
                "projection": "SUMMARY",
            },
            timeout=_TIMEOUT,
            verify=_SSL_VERIFY,
        )
        if cna_resp.status_code == 200:
            cna_data = cna_resp.json()
            result["amp_count"] = sum(1 for d in cna_data if d.get("value") == 2)
            result["del_count"] = sum(1 for d in cna_data if d.get("value") == -2)
            result["cna_total"] = len(cna_data)
    except Exception:
        pass

    return result


def get_gene_tumor_expression(gene_symbol: str, top_n: int = 10) -> dict:
    """
    mRNA expression z-scores for a gene across TCGA PanCancer Atlas cancer types.

    Queries all 32 TCGA PanCancer Atlas studies concurrently and returns mean
    mRNA z-scores per cancer type, ranked from most to least overexpressed.

    Args:
        gene_symbol: Gene symbol (e.g., "MYC", "TP53")
        top_n: Number of top overexpressed / underexpressed cancer types to return

    Returns:
        Dict with:
        - gene: str
        - top_overexpressed: list[dict] (cancer_type, mean_z_score, num_samples)
        - top_underexpressed: list[dict]
        - num_cancer_types_queried: int
        - pan_cancer_mean_z: float
        - data_source: str
        - error: str (only on failure)
    """
    entrez_id = _get_entrez_id(gene_symbol)
    if entrez_id is None:
        return {
            "gene": gene_symbol,
            "error": f"Gene '{gene_symbol}' not found in cBioPortal",
            "data_source": "TCGA PanCancer Atlas 2018 (cBioPortal)",
        }

    cancer_stats = []
    all_values = []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_expression_for_study, ct, entrez_id): ct
            for ct in _TCGA_STUDIES
        }
        for future in as_completed(futures):
            ct, values = future.result()
            if not values:
                continue
            mean_z = sum(values) / len(values)
            cancer_stats.append({
                "cancer_type": _TCGA_STUDIES[ct],
                "cancer_prefix": ct,
                "mean_z_score": round(mean_z, 3),
                "num_samples": len(values),
            })
            all_values.extend(values)

    if not cancer_stats:
        return {
            "gene": gene_symbol,
            "error": f"No expression data found for '{gene_symbol}' in TCGA PanCancer Atlas",
            "data_source": "TCGA PanCancer Atlas 2018 (cBioPortal)",
        }

    cancer_stats.sort(key=lambda x: x["mean_z_score"], reverse=True)
    pan_cancer_mean_z = round(sum(all_values) / len(all_values), 3)

    return {
        "gene": gene_symbol,
        "top_overexpressed": cancer_stats[:top_n],
        "top_underexpressed": cancer_stats[-top_n:],
        "num_cancer_types_queried": len(cancer_stats),
        "pan_cancer_mean_z": pan_cancer_mean_z,
        "data_source": "TCGA PanCancer Atlas 2018 (cBioPortal)",
    }


def get_gene_alteration_frequency(gene_symbol: str) -> dict:
    """
    Somatic mutation and copy-number alteration (CNA) frequency across TCGA PanCancer Atlas.

    Queries mutation and GISTIC CNA data for all 32 TCGA PanCancer Atlas cancer types
    concurrently and returns pan-cancer and per-cancer alteration rates.

    Args:
        gene_symbol: Gene symbol (e.g., "MYC", "TP53")

    Returns:
        Dict with:
        - gene: str
        - mutation_frequency_pct: float    (pan-cancer % samples mutated)
        - amplification_frequency_pct: float  (pan-cancer % samples amplified)
        - deletion_frequency_pct: float    (pan-cancer % samples homozygous deleted)
        - most_altered_cancer_type: str
        - alteration_by_cancer: list[dict]  (cancer_type, mutation_pct, amplification_pct, deletion_pct)
        - data_source: str
        - error: str (only on failure)
    """
    entrez_id = _get_entrez_id(gene_symbol)
    if entrez_id is None:
        return {
            "gene": gene_symbol,
            "error": f"Gene '{gene_symbol}' not found in cBioPortal",
            "data_source": "TCGA PanCancer Atlas 2018 (cBioPortal)",
        }

    alteration_results = {}

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_alterations_for_study, ct, entrez_id): ct
            for ct in _TCGA_STUDIES
        }
        for future in as_completed(futures):
            res = future.result()
            alteration_results[res["cancer_prefix"]] = res

    alteration_by_cancer = []
    total_mut = total_seq = total_amp = total_del = total_cna = 0

    for ct, cancer_name in _TCGA_STUDIES.items():
        res = alteration_results.get(ct, {})
        mut_count = res.get("mut_count", 0)
        mut_total = res.get("mut_total", 0)
        amp_count = res.get("amp_count", 0)
        del_count = res.get("del_count", 0)
        cna_total = res.get("cna_total", 0)

        if mut_total == 0 and cna_total == 0:
            continue

        mut_pct = round(100 * mut_count / mut_total, 2) if mut_total > 0 else 0.0
        amp_pct = round(100 * amp_count / cna_total, 2) if cna_total > 0 else 0.0
        del_pct = round(100 * del_count / cna_total, 2) if cna_total > 0 else 0.0

        alteration_by_cancer.append({
            "cancer_type": cancer_name,
            "cancer_prefix": ct,
            "mutation_pct": mut_pct,
            "amplification_pct": amp_pct,
            "deletion_pct": del_pct,
            "total_altered_pct": round(mut_pct + amp_pct + del_pct, 2),
        })

        total_mut += mut_count
        total_seq += mut_total
        total_amp += amp_count
        total_del += del_count
        total_cna += cna_total

    alteration_by_cancer.sort(key=lambda x: x["total_altered_pct"], reverse=True)
    most_altered = alteration_by_cancer[0]["cancer_type"] if alteration_by_cancer else None

    pan_mut_pct = round(100 * total_mut / total_seq, 2) if total_seq > 0 else 0.0
    pan_amp_pct = round(100 * total_amp / total_cna, 2) if total_cna > 0 else 0.0
    pan_del_pct = round(100 * total_del / total_cna, 2) if total_cna > 0 else 0.0

    return {
        "gene": gene_symbol,
        "mutation_frequency_pct": pan_mut_pct,
        "amplification_frequency_pct": pan_amp_pct,
        "deletion_frequency_pct": pan_del_pct,
        "most_altered_cancer_type": most_altered,
        "alteration_by_cancer": alteration_by_cancer,
        "data_source": "TCGA PanCancer Atlas 2018 (cBioPortal)",
    }


def get_cbioportal_stats() -> dict:
    """Return basic stats about TCGA PanCancer Atlas data via cBioPortal API."""
    try:
        resp = requests.get(f"{_BASE_URL}/info", timeout=_TIMEOUT, verify=_SSL_VERIFY)
        api_reachable = resp.status_code == 200
    except Exception:
        api_reachable = False

    return {
        "num_tcga_studies": len(_TCGA_STUDIES),
        "num_pan_cancer_samples_approx": 10967,
        "api_reachable": api_reachable,
        "data_source": "TCGA PanCancer Atlas 2018 (cBioPortal)",
    }
