"""
Super-Enhancer Annotation Module

Provides lookup for genes associated with super-enhancers, indicating
potential sensitivity to BRD4/BET inhibitors.

Data source: dbSUPER (http://asntech.org/dbsuper/)
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Path to super-enhancer data
SE_DATA_PATH = Path(__file__).parent.parent / "data" / "super_enhancers" / "dbSUPER_hg19.tsv"

# Module-level cache
_se_data: Optional[pd.DataFrame] = None
_se_genes: Optional[set] = None


def load_super_enhancer_data() -> pd.DataFrame:
    """Load super-enhancer data from dbSUPER."""
    global _se_data

    if _se_data is not None:
        return _se_data

    if not SE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Super-enhancer data not found at {SE_DATA_PATH}. "
            "Download from: https://asntech.org/dbsuper/data/dbSUPER_SuperEnhancers_hg19.tsv"
        )

    print(f"[SuperEnhancer] Loading data from {SE_DATA_PATH}")

    _se_data = pd.read_csv(SE_DATA_PATH, sep='\t')
    # Clean column names (remove leading spaces)
    _se_data.columns = _se_data.columns.str.strip()

    print(f"[SuperEnhancer] Loaded {len(_se_data):,} super-enhancer associations")
    print(f"[SuperEnhancer] Unique genes: {_se_data['gene_symbol'].nunique():,}")
    print(f"[SuperEnhancer] Cell types: {_se_data['cell_name'].nunique():,}")

    return _se_data


def get_se_genes() -> set:
    """Get set of all genes with super-enhancer associations."""
    global _se_genes

    if _se_genes is not None:
        return _se_genes

    df = load_super_enhancer_data()
    _se_genes = set(df['gene_symbol'].str.upper().unique())

    return _se_genes


def has_super_enhancer(gene: str) -> bool:
    """
    Check if a gene has super-enhancer associations.

    Genes with super-enhancers are often sensitive to BRD4/BET inhibitors.

    Args:
        gene: Gene symbol (e.g., "MYC")

    Returns:
        True if gene has super-enhancer associations in any cell type
    """
    se_genes = get_se_genes()
    return gene.upper() in se_genes


def get_super_enhancer_info(gene: str) -> dict:
    """
    Get detailed super-enhancer information for a gene.

    Args:
        gene: Gene symbol (e.g., "MYC")

    Returns:
        Dict with super-enhancer status and cell type details
    """
    df = load_super_enhancer_data()

    gene_upper = gene.upper()
    matches = df[df['gene_symbol'].str.upper() == gene_upper]

    if len(matches) == 0:
        return {
            "gene": gene,
            "has_super_enhancer": False,
            "cell_types": [],
            "brd4_sensitive": False,
            "note": "No super-enhancer associations found in dbSUPER"
        }

    cell_types = matches['cell_name'].unique().tolist()

    return {
        "gene": gene,
        "has_super_enhancer": True,
        "cell_type_count": len(cell_types),
        "cell_types": cell_types[:10],  # Limit to first 10
        "total_associations": len(matches),
        "brd4_sensitive": True,
        "therapeutic_implication": f"{gene} is associated with super-enhancers in {len(cell_types)} cell types. "
                                   f"May be sensitive to BRD4/BET inhibitors (e.g., JQ1, OTX015).",
        "data_source": "dbSUPER"
    }


def check_genes_for_super_enhancers(genes: list[str]) -> list[dict]:
    """
    Check multiple genes for super-enhancer associations.

    Args:
        genes: List of gene symbols

    Returns:
        List of dicts with super-enhancer status for each gene
    """
    results = []
    se_genes = get_se_genes()

    for gene in genes:
        gene_upper = gene.upper()
        has_se = gene_upper in se_genes

        results.append({
            "gene": gene,
            "has_super_enhancer": has_se,
            "brd4_sensitive": has_se
        })

    return results


def get_super_enhancer_stats() -> dict:
    """Get statistics about the super-enhancer dataset."""
    df = load_super_enhancer_data()

    return {
        "total_associations": len(df),
        "unique_genes": df['gene_symbol'].nunique(),
        "unique_cell_types": df['cell_name'].nunique(),
        "data_source": "dbSUPER (http://asntech.org/dbsuper/)",
        "genome_build": "hg19"
    }
