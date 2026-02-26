"""
DepMap CRISPR Essentiality Data Module

Provides gene essentiality profiles derived from DepMap CRISPR screens
(Chronos gene effect scores) across 1,000+ cancer cell lines.

Negative Chronos scores indicate essentiality:
    score < -0.5  → essential in that cell line
    score < -1.0  → strongly essential

Data source: DepMap Public release
https://depmap.org/portal/download/
  - CRISPRGeneEffect.csv  (rows=cell lines, cols="GENE (entrez_id)")
  - Model.csv             (ModelID -> OncotreeLineage mapping)
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Paths to DepMap data files
DEPMAP_DIR = Path(__file__).parent.parent / "data" / "depmap"
GENE_EFFECT_PATH = DEPMAP_DIR / "CRISPRGeneEffect.csv"
MODEL_PATH = DEPMAP_DIR / "Model.csv"

# Module-level cache
_gene_scores: Optional[pd.DataFrame] = None   # cell_line x gene (Chronos scores)
_lineage_map: Optional[pd.Series] = None      # ModelID -> OncotreeLineage


def load_depmap_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load DepMap CRISPR gene effect and cell line lineage data.

    Returns:
        Tuple of:
        - scores DataFrame: index=cell_line (ACH-XXXXXX), columns=gene_symbol
          (stripped of Entrez IDs), values=Chronos gene effect scores
        - lineage_map Series: ModelID -> OncotreeLineage
    """
    global _gene_scores, _lineage_map

    if _gene_scores is not None and _lineage_map is not None:
        return _gene_scores, _lineage_map

    if not GENE_EFFECT_PATH.exists():
        raise FileNotFoundError(
            f"DepMap gene effect data not found at {GENE_EFFECT_PATH}. "
            "Download CRISPRGeneEffect.csv from https://depmap.org/portal/download/ "
            "(DepMap Public release). See data/depmap/README.md for instructions."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"DepMap model metadata not found at {MODEL_PATH}. "
            "Download Model.csv from https://depmap.org/portal/download/ "
            "(DepMap Public release). See data/depmap/README.md for instructions."
        )

    print(f"[DepMap] Loading gene effect scores from {GENE_EFFECT_PATH}")

    # Load gene effect matrix (rows=cell lines, cols="GENE (entrez_id)")
    scores = pd.read_csv(GENE_EFFECT_PATH, index_col=0)

    # Strip Entrez IDs from column names: "MYC (4609)" -> "MYC"
    scores.columns = scores.columns.str.replace(r'\s*\(\d+\)$', '', regex=True)

    print(f"[DepMap] Loaded {scores.shape[0]} cell lines x {scores.shape[1]} genes")

    # Load lineage mapping
    model_df = pd.read_csv(MODEL_PATH, usecols=lambda c: c in ("ModelID", "OncotreeLineage"))
    lineage_map = model_df.set_index("ModelID")["OncotreeLineage"]

    print(f"[DepMap] Loaded lineage map for {len(lineage_map)} cell lines")

    _gene_scores = scores
    _lineage_map = lineage_map
    return _gene_scores, _lineage_map


def get_gene_essentiality(gene: str) -> dict:
    """
    Return essentiality profile for a gene across cancer cell lines.

    Args:
        gene: Gene symbol (e.g., "MYC", "TP53")

    Returns:
        Dict with:
        - gene: str
        - mean_chronos_score: float  (negative = essential)
        - median_chronos_score: float
        - std_chronos_score: float
        - essential_fraction: float  (fraction of lines with score < -0.5)
        - strongly_essential_fraction: float  (fraction with score < -1.0)
        - pan_cancer_essential: bool  (essential in >50% of lines)
        - common_essential: bool  (essential in >90%, housekeeping-like)
        - cell_lines_tested: int
        - top_lineages: list of {lineage, mean_score, n_cell_lines}
        - data_source: str
        - not_found: bool
    """
    try:
        scores, lineage_map = load_depmap_data()
    except FileNotFoundError as e:
        return {
            "gene": gene,
            "not_found": True,
            "error": str(e),
            "data_source": "DepMap CRISPR (Chronos)"
        }

    # Case-insensitive gene lookup
    gene_upper = gene.upper()
    col_match = None
    for col in scores.columns:
        if col.upper() == gene_upper:
            col_match = col
            break

    if col_match is None:
        return {
            "gene": gene,
            "not_found": True,
            "data_source": "DepMap CRISPR (Chronos)"
        }

    gene_data = scores[col_match].dropna()

    if len(gene_data) == 0:
        return {
            "gene": gene,
            "not_found": True,
            "data_source": "DepMap CRISPR (Chronos)"
        }

    mean_score = float(gene_data.mean())
    median_score = float(gene_data.median())
    std_score = float(gene_data.std())
    n_lines = int(len(gene_data))
    essential_frac = float((gene_data < -0.5).sum() / n_lines)
    strongly_essential_frac = float((gene_data < -1.0).sum() / n_lines)

    # Compute top lineages by mean essentiality
    # Join gene scores with lineage info
    gene_series = gene_data.rename("score")
    lineage_series = lineage_map.reindex(gene_series.index)
    combined = pd.DataFrame({"score": gene_series, "lineage": lineage_series})
    combined = combined.dropna(subset=["lineage"])

    top_lineages = []
    if not combined.empty:
        lineage_stats = (
            combined.groupby("lineage")["score"]
            .agg(mean_score_lin="mean", n_cell_lines="count")
            .reset_index()
        )
        lineage_stats = lineage_stats.sort_values("mean_score_lin", ascending=True)
        for _, row in lineage_stats.head(5).iterrows():
            top_lineages.append({
                "lineage": row["lineage"],
                "mean_score": round(float(row["mean_score_lin"]), 4),
                "n_cell_lines": int(row["n_cell_lines"])
            })

    return {
        "gene": gene,
        "mean_chronos_score": round(mean_score, 4),
        "median_chronos_score": round(median_score, 4),
        "std_chronos_score": round(std_score, 4),
        "essential_fraction": round(essential_frac, 4),
        "strongly_essential_fraction": round(strongly_essential_frac, 4),
        "pan_cancer_essential": essential_frac > 0.5,
        "common_essential": essential_frac > 0.9,
        "cell_lines_tested": n_lines,
        "top_lineages": top_lineages,
        "data_source": "DepMap CRISPR (Chronos)",
        "not_found": False
    }
