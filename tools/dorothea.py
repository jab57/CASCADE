"""
DoRothEA TF Regulon Validation Module

Provides access to curated transcription factor regulons from DoRothEA,
enabling validation of ARACNe-derived TF classifications against multi-evidence
curated data (literature, ChIP-seq, motifs, co-expression).

Data source: DoRothEA via decoupler-py
https://decoupler-py.readthedocs.io/
"""

from typing import Optional

import pandas as pd

# Module-level cache
_dorothea_data: Optional[pd.DataFrame] = None


def load_dorothea_regulons(levels: list[str] | None = None) -> pd.DataFrame:
    """
    Load DoRothEA TF regulons filtered by confidence level.

    Args:
        levels: Confidence levels to include (default: ["A", "B", "C"]).
            A = highest confidence (multiple evidence types)
            E = lowest confidence (co-expression only)

    Returns:
        DataFrame with columns: source (TF), target, confidence, mor (mode of regulation)
    """
    global _dorothea_data

    if levels is None:
        levels = ["A", "B", "C"]

    if _dorothea_data is not None:
        filtered = _dorothea_data[_dorothea_data["confidence"].isin(levels)]
        return filtered

    try:
        import decoupler as dc
    except ImportError:
        raise ImportError(
            "decoupler package not installed. "
            "Install with: pip install decoupler>=1.6"
        )

    print("[DoRothEA] Loading TF regulons from decoupler...")

    try:
        df = dc.get_dorothea(organism="human")
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch DoRothEA regulons: {e}. "
            "Check your internet connection or try again later."
        )

    # Standardize column names (decoupler returns source, target, confidence, mor)
    expected_cols = {"source", "target", "confidence"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Unexpected DoRothEA format. Expected columns {expected_cols}, "
            f"got {set(df.columns)}"
        )

    # Ensure mor column exists (mode of regulation: +1 activation, -1 repression)
    if "mor" not in df.columns:
        df["mor"] = 1.0

    print(f"[DoRothEA] Loaded {len(df):,} TF-target interactions")
    print(f"[DoRothEA] Unique TFs: {df['source'].nunique():,}")
    print(f"[DoRothEA] Confidence levels: {sorted(df['confidence'].unique())}")

    _dorothea_data = df

    filtered = _dorothea_data[_dorothea_data["confidence"].isin(levels)]
    return filtered


def get_tf_targets(
    gene: str,
    confidence_levels: list[str] | None = None,
    top_k: int = 50
) -> list[dict]:
    """
    Get DoRothEA regulon targets for a transcription factor.

    Args:
        gene: TF gene symbol (e.g., "TP53", "MYC")
        confidence_levels: Filter by confidence (default: ["A", "B", "C"])
        top_k: Maximum targets to return

    Returns:
        List of dicts with: target, mor, confidence
    """
    if confidence_levels is None:
        confidence_levels = ["A", "B", "C"]

    try:
        df = load_dorothea_regulons(levels=confidence_levels)
    except (ImportError, RuntimeError) as e:
        return [{"error": str(e)}]

    gene_upper = gene.upper()
    mask = df["source"].str.upper() == gene_upper

    if not mask.any():
        return []

    results = df[mask].copy()

    # Sort by confidence level (A first) then by absolute mor
    level_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    results["_level_rank"] = results["confidence"].map(level_order).fillna(5)
    results["_abs_mor"] = results["mor"].abs()
    results = results.sort_values(
        ["_level_rank", "_abs_mor"],
        ascending=[True, False]
    )

    results = results.head(top_k)

    output = []
    for _, row in results.iterrows():
        output.append({
            "target": row["target"],
            "mor": round(float(row["mor"]), 4),
            "confidence": row["confidence"]
        })

    return output


def validate_tf_classification(gene: str) -> dict:
    """
    Validate whether a gene is a known TF in DoRothEA.

    Args:
        gene: Gene symbol to validate

    Returns:
        Dict with: is_known_tf, best_confidence, num_targets_by_level, evidence_summary
    """
    try:
        df = load_dorothea_regulons(levels=["A", "B", "C", "D", "E"])
    except (ImportError, RuntimeError) as e:
        return {"error": str(e), "is_known_tf": False}

    gene_upper = gene.upper()
    mask = df["source"].str.upper() == gene_upper

    if not mask.any():
        return {
            "gene": gene,
            "is_known_tf": False,
            "best_confidence": None,
            "num_targets_by_level": {},
            "total_targets": 0,
            "evidence_summary": f"{gene} is not found in the DoRothEA TF regulon database."
        }

    results = df[mask]

    # Count targets per confidence level
    level_counts = results["confidence"].value_counts().to_dict()

    # Determine best confidence
    level_order = ["A", "B", "C", "D", "E"]
    best_confidence = None
    for level in level_order:
        if level in level_counts:
            best_confidence = level
            break

    total_targets = len(results)

    # Build evidence summary
    level_descriptions = {
        "A": "highest (literature + ChIP-seq + motifs)",
        "B": "high (literature + ChIP-seq or motifs)",
        "C": "moderate (ChIP-seq or motifs + co-expression)",
        "D": "low (co-expression + motifs)",
        "E": "predicted (co-expression only)"
    }

    summary_parts = []
    for level in level_order:
        if level in level_counts:
            desc = level_descriptions.get(level, "unknown")
            summary_parts.append(
                f"Level {level} ({desc}): {level_counts[level]} targets"
            )

    evidence_summary = (
        f"{gene} is a confirmed TF in DoRothEA with {total_targets} "
        f"total target(s). Best confidence: {best_confidence}. "
        + "; ".join(summary_parts)
    )

    return {
        "gene": gene,
        "is_known_tf": True,
        "best_confidence": best_confidence,
        "num_targets_by_level": level_counts,
        "total_targets": total_targets,
        "evidence_summary": evidence_summary
    }


def get_dorothea_stats() -> dict:
    """Get statistics about the loaded DoRothEA regulon data."""
    try:
        df = load_dorothea_regulons(levels=["A", "B", "C", "D", "E"])
    except (ImportError, RuntimeError) as e:
        return {"error": str(e)}

    level_counts = df["confidence"].value_counts().to_dict()

    return {
        "total_interactions": len(df),
        "unique_tfs": df["source"].nunique(),
        "unique_targets": df["target"].nunique(),
        "interactions_by_level": level_counts,
        "data_source": "DoRothEA via decoupler-py"
    }
