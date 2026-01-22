"""
Perturbation analysis tools for gene regulatory networks.

Implements network propagation to simulate gene knockdown/overexpression
and predict downstream effects on target gene expression.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tools.gene_id_mapper import get_mapper


def _build_adjacency(network_df: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """
    Build adjacency list from network dataframe.

    Returns:
        Dict mapping regulator -> [(target, weight), ...]
    """
    adj = defaultdict(list)
    for _, row in network_df.iterrows():
        regulator = row["regulator"]
        target = row["target"]
        # Use mutual information as edge weight
        weight = row.get("mi", 1.0)
        if pd.notna(weight) and weight > 0:
            adj[regulator].append((target, float(weight)))
    return dict(adj)


def _build_reverse_adjacency(network_df: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """
    Build reverse adjacency list (target -> regulators).

    Returns:
        Dict mapping target -> [(regulator, weight), ...]
    """
    adj = defaultdict(list)
    for _, row in network_df.iterrows():
        regulator = row["regulator"]
        target = row["target"]
        weight = row.get("mi", 1.0)
        if pd.notna(weight) and weight > 0:
            adj[target].append((regulator, float(weight)))
    return dict(adj)


def _propagate_effect(
    adj: dict[str, list[tuple[str, float]]],
    start_gene: str,
    initial_effect: float,
    depth: int,
    decay: float = 0.5
) -> dict[str, float]:
    """
    Propagate perturbation effect through the network using BFS.

    Args:
        adj: Adjacency list (regulator -> targets)
        start_gene: Gene being perturbed
        initial_effect: Initial perturbation strength (-1 for KD, log2(FC) for OE)
        depth: Maximum propagation depth
        decay: Effect decay per hop (0-1)

    Returns:
        Dict mapping gene -> cumulative effect
    """
    effects = defaultdict(float)
    effects[start_gene] = initial_effect

    # BFS propagation
    current_level = {start_gene: initial_effect}

    for _ in range(depth):
        next_level = {}
        for gene, gene_effect in current_level.items():
            if gene not in adj:
                continue
            for target, weight in adj[gene]:
                # Effect = parent_effect * edge_weight * decay
                propagated = gene_effect * weight * decay
                next_level[target] = next_level.get(target, 0) + propagated
                effects[target] += propagated

        current_level = next_level
        if not current_level:
            break

    return dict(effects)


def simulate_knockdown(
    network_df: pd.DataFrame,
    gene: str,
    depth: int = 2,
    top_k: int = 25
) -> dict:
    """
    Simulate knocking down a gene and predict downstream effects.

    Args:
        network_df: Gene regulatory network
        gene: Gene to knock down (Ensembl ID)
        depth: Propagation depth (1=direct, 2+=indirect)
        top_k: Number of top affected genes to return

    Returns:
        Dict with perturbation results
    """
    adj = _build_adjacency(network_df)

    # Check if gene exists as a regulator
    all_regulators = set(network_df["regulator"].unique())
    all_targets = set(network_df["target"].unique())
    all_genes = all_regulators | all_targets

    if gene not in all_genes:
        return {
            "status": "error",
            "error": f"Gene {gene} not found in network",
            "available_regulators_sample": list(all_regulators)[:20],
            "total_regulators": len(all_regulators),
            "total_genes": len(all_genes)
        }

    # Knockdown = complete loss of expression (-1.0 effect)
    effects = _propagate_effect(adj, gene, initial_effect=-1.0, depth=depth)

    # Remove the perturbed gene itself from results
    effects.pop(gene, None)

    if not effects:
        return {
            "status": "complete",
            "perturbed_gene": gene,
            "perturbation_type": "knockdown",
            "message": f"Gene {gene} has no downstream targets in this network",
            "affected_genes": []
        }

    # Sort by absolute effect magnitude
    sorted_effects = sorted(
        effects.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    # Get gene mapper for symbol lookups
    mapper = get_mapper()

    affected_genes = [
        {
            "ensembl_id": g,
            "symbol": mapper.ensembl_to_symbol(g) or g,
            "predicted_effect": round(e, 4),
            "direction": "down" if e < 0 else "up",
            "magnitude": round(abs(e), 4)
        }
        for g, e in sorted_effects
    ]

    return {
        "status": "complete",
        "perturbed_gene": gene,
        "perturbation_type": "knockdown",
        "propagation_depth": depth,
        "total_affected_genes": len(effects),
        "top_affected_genes": affected_genes
    }


def simulate_overexpression(
    network_df: pd.DataFrame,
    gene: str,
    fold_change: float = 2.0,
    depth: int = 2,
    top_k: int = 25
) -> dict:
    """
    Simulate overexpressing a gene and predict downstream effects.

    Args:
        network_df: Gene regulatory network
        gene: Gene to overexpress (Ensembl ID)
        fold_change: Expression multiplier (e.g., 2.0 = 2x expression)
        depth: Propagation depth
        top_k: Number of top affected genes to return

    Returns:
        Dict with perturbation results
    """
    adj = _build_adjacency(network_df)
    all_regulators = set(network_df["regulator"].unique())
    all_targets = set(network_df["target"].unique())
    all_genes = all_regulators | all_targets

    if gene not in all_genes:
        return {
            "status": "error",
            "error": f"Gene {gene} not found in network",
            "available_regulators_sample": list(all_regulators)[:20]
        }

    # Overexpression effect = log2(fold_change)
    initial_effect = np.log2(fold_change)
    effects = _propagate_effect(adj, gene, initial_effect=initial_effect, depth=depth)

    effects.pop(gene, None)

    if not effects:
        return {
            "status": "complete",
            "perturbed_gene": gene,
            "perturbation_type": "overexpression",
            "fold_change": fold_change,
            "message": f"Gene {gene} has no downstream targets in this network",
            "affected_genes": []
        }

    sorted_effects = sorted(
        effects.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    # Get gene mapper for symbol lookups
    mapper = get_mapper()

    affected_genes = [
        {
            "ensembl_id": g,
            "symbol": mapper.ensembl_to_symbol(g) or g,
            "predicted_effect": round(e, 4),
            "direction": "up" if e > 0 else "down",
            "magnitude": round(abs(e), 4)
        }
        for g, e in sorted_effects
    ]

    return {
        "status": "complete",
        "perturbed_gene": gene,
        "perturbation_type": "overexpression",
        "fold_change": fold_change,
        "propagation_depth": depth,
        "total_affected_genes": len(effects),
        "top_affected_genes": affected_genes
    }


def get_regulators(network_df: pd.DataFrame, target_gene: str) -> dict:
    """
    Find all regulators (transcription factors) that control a target gene.

    Args:
        network_df: Gene regulatory network
        target_gene: Gene to find regulators for

    Returns:
        Dict with list of regulators and their edge weights
    """
    reverse_adj = _build_reverse_adjacency(network_df)

    if target_gene not in reverse_adj:
        all_targets = set(network_df["target"].unique())
        return {
            "status": "not_found",
            "target_gene": target_gene,
            "message": f"Gene {target_gene} not found as a target in this network",
            "available_targets_sample": list(all_targets)[:20]
        }

    regulators = reverse_adj[target_gene]
    sorted_regs = sorted(regulators, key=lambda x: x[1], reverse=True)

    # Get gene mapper for symbol lookups
    mapper = get_mapper()

    return {
        "status": "complete",
        "target_gene": target_gene,
        "num_regulators": len(sorted_regs),
        "regulators": [
            {
                "ensembl_id": reg,
                "symbol": mapper.ensembl_to_symbol(reg) or reg,
                "edge_weight": round(w, 4)
            }
            for reg, w in sorted_regs
        ]
    }


def get_targets(network_df: pd.DataFrame, regulator_gene: str) -> dict:
    """
    Find all target genes controlled by a regulator.

    Args:
        network_df: Gene regulatory network
        regulator_gene: Regulator to find targets for

    Returns:
        Dict with list of targets and their edge weights
    """
    adj = _build_adjacency(network_df)

    if regulator_gene not in adj:
        all_regulators = set(network_df["regulator"].unique())
        return {
            "status": "not_found",
            "regulator_gene": regulator_gene,
            "message": f"Gene {regulator_gene} not found as a regulator in this network",
            "available_regulators_sample": list(all_regulators)[:20]
        }

    targets = adj[regulator_gene]
    sorted_targets = sorted(targets, key=lambda x: x[1], reverse=True)

    # Get gene mapper for symbol lookups
    mapper = get_mapper()

    return {
        "status": "complete",
        "regulator_gene": regulator_gene,
        "num_targets": len(sorted_targets),
        "targets": [
            {
                "ensembl_id": tgt,
                "symbol": mapper.ensembl_to_symbol(tgt) or tgt,
                "edge_weight": round(w, 4)
            }
            for tgt, w in sorted_targets
        ]
    }
