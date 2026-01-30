"""
LINCS L1000 Expression Perturbation Data Module

Provides access to gene expression changes following CRISPR knockouts,
enabling discovery of regulatory relationships not captured in transcriptional networks.

Data source: Harmonizome LINCS L1000 CMAP CRISPR Knockout Consensus Signatures
https://maayanlab.cloud/Harmonizome/dataset/LINCS+L1000+CMAP+CRISPR+Knockout+Consensus+Signatures
"""

import gzip
from pathlib import Path
from typing import Optional

import pandas as pd

# Path to LINCS data file
LINCS_DATA_PATH = Path(__file__).parent.parent / "data" / "lincs" / "gene_attribute_edges.txt.gz"

# Module-level cache
_lincs_data: Optional[pd.DataFrame] = None


def load_lincs_data() -> pd.DataFrame:
    """
    Load LINCS L1000 knockdown expression data.

    Returns:
        DataFrame with columns:
        - gene: Gene whose expression changed
        - gene_ko: Gene that was knocked out
        - effect: Standardized effect size (negative = downregulated)
        - direction: -1 (down), +1 (up)
    """
    global _lincs_data

    if _lincs_data is not None:
        return _lincs_data

    if not LINCS_DATA_PATH.exists():
        raise FileNotFoundError(
            f"LINCS data not found at {LINCS_DATA_PATH}. "
            "Download from: https://maayanlab.cloud/static/hdfs/harmonizome/data/l1000crispr/gene_attribute_edges.txt.gz"
        )

    print(f"[LINCS] Loading expression perturbation data from {LINCS_DATA_PATH}")

    # Load gzipped TSV
    df = pd.read_csv(
        LINCS_DATA_PATH,
        sep='\t',
        compression='gzip',
        usecols=[1, 3, 5, 6],  # Gene, Gene KO, Standardized Value, Threshold Value
        names=['gene', 'gene_ko', 'effect', 'direction'],
        skiprows=1
    )

    # Clean up gene_ko column (remove _KO suffix and ID)
    # Format is like "TP53" with separate column for ID_KO

    print(f"[LINCS] Loaded {len(df):,} gene-perturbation associations")
    print(f"[LINCS] Unique genes measured: {df['gene'].nunique():,}")
    print(f"[LINCS] Unique knockdowns: {df['gene_ko'].nunique():,}")

    _lincs_data = df
    return df


def find_expression_regulators(
    gene: str,
    direction: str = "any",
    top_k: int = 20
) -> list[dict]:
    """
    Find genes whose knockdown affects target gene expression.

    This identifies regulatory relationships from experimental perturbation data,
    capturing effects that transcriptional networks may miss (epigenetic, post-translational).

    Args:
        gene: Target gene symbol (e.g., "MYC", "CDKN1A")
        direction: Filter by effect direction:
            - "down": Knockdowns that decrease target expression
            - "up": Knockdowns that increase target expression
            - "any": Both directions (default)
        top_k: Number of top results to return

    Returns:
        List of dicts with:
        - gene_ko: The knocked-out gene
        - effect: Standardized effect size
        - direction: "down" (-1) or "up" (+1)
        - interpretation: Human-readable explanation

    Example:
        find_expression_regulators("CDKN1A", direction="down")
        → Returns TP53 (TP53 KO reduces CDKN1A expression)
    """
    df = load_lincs_data()

    # Filter for target gene
    gene_upper = gene.upper()
    mask = df['gene'].str.upper() == gene_upper

    if not mask.any():
        return []

    results = df[mask].copy()

    # Filter by direction
    if direction == "down":
        results = results[results['direction'] == -1]
    elif direction == "up":
        results = results[results['direction'] == 1]

    # Sort by absolute effect size
    results['abs_effect'] = results['effect'].abs()
    results = results.sort_values('abs_effect', ascending=False)

    # Take top_k
    results = results.head(top_k)

    # Format output
    output = []
    for _, row in results.iterrows():
        dir_str = "down" if row['direction'] == -1 else "up"

        if dir_str == "down":
            interp = f"Knocking out {row['gene_ko']} DECREASES {gene} expression"
        else:
            interp = f"Knocking out {row['gene_ko']} INCREASES {gene} expression"

        output.append({
            "gene_ko": row['gene_ko'],
            "effect": round(row['effect'], 4),
            "direction": dir_str,
            "interpretation": interp
        })

    return output


def get_knockdown_effects(
    gene_ko: str,
    direction: str = "any",
    top_k: int = 20
) -> list[dict]:
    """
    Find genes whose expression changes when a specific gene is knocked out.

    This is the inverse of find_expression_regulators - instead of asking
    "what regulates gene X?", it asks "what does gene X regulate?".

    Args:
        gene_ko: Gene that was knocked out (e.g., "BRD4", "TP53")
        direction: Filter by effect direction ("down", "up", "any")
        top_k: Number of top results to return

    Returns:
        List of dicts with affected genes and effect sizes

    Example:
        get_knockdown_effects("TP53")
        → Returns genes affected by TP53 knockdown (CDKN1A down, etc.)
    """
    df = load_lincs_data()

    # Filter for knocked-out gene
    gene_upper = gene_ko.upper()
    mask = df['gene_ko'].str.upper() == gene_upper

    if not mask.any():
        return []

    results = df[mask].copy()

    # Filter by direction
    if direction == "down":
        results = results[results['direction'] == -1]
    elif direction == "up":
        results = results[results['direction'] == 1]

    # Sort by absolute effect size
    results['abs_effect'] = results['effect'].abs()
    results = results.sort_values('abs_effect', ascending=False)

    # Take top_k
    results = results.head(top_k)

    # Format output
    output = []
    for _, row in results.iterrows():
        dir_str = "down" if row['direction'] == -1 else "up"

        output.append({
            "gene": row['gene'],
            "effect": round(row['effect'], 4),
            "direction": dir_str,
        })

    return output


def get_lincs_stats() -> dict:
    """Get statistics about the loaded LINCS data."""
    df = load_lincs_data()

    return {
        "total_associations": len(df),
        "unique_genes_measured": df['gene'].nunique(),
        "unique_knockdowns": df['gene_ko'].nunique(),
        "upregulated_associations": int((df['direction'] == 1).sum()),
        "downregulated_associations": int((df['direction'] == -1).sum()),
    }
