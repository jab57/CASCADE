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


def get_regulators(network_df: pd.DataFrame, target_gene: str, max_regulators: int = 50) -> dict:
    """
    Find all regulators (transcription factors) that control a target gene.

    Args:
        network_df: Gene regulatory network
        target_gene: Gene to find regulators for
        max_regulators: Maximum number of regulators to return (limits API calls)

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
    total_regulators = len(sorted_regs)

    # Limit regulators to avoid timeout from symbol lookups
    limited_regs = sorted_regs[:max_regulators]

    # Get gene mapper for symbol lookups (only for limited set)
    mapper = get_mapper()

    result = {
        "status": "complete",
        "target_gene": target_gene,
        "num_regulators": total_regulators,
        "regulators_returned": len(limited_regs),
        "regulators": [
            {
                "ensembl_id": reg,
                "symbol": mapper.ensembl_to_symbol(reg) or reg,
                "edge_weight": round(w, 4)
            }
            for reg, w in limited_regs
        ]
    }

    if total_regulators > max_regulators:
        result["note"] = f"Showing top {max_regulators} of {total_regulators} regulators (sorted by edge weight). Increase max_regulators to see more."

    return result


def get_targets(network_df: pd.DataFrame, regulator_gene: str, max_targets: int = 50) -> dict:
    """
    Find all target genes controlled by a regulator.

    Args:
        network_df: Gene regulatory network
        regulator_gene: Regulator to find targets for
        max_targets: Maximum number of targets to return (limits API calls)

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
    total_targets = len(sorted_targets)

    # Limit targets to avoid timeout from symbol lookups
    limited_targets = sorted_targets[:max_targets]

    # Get gene mapper for symbol lookups (only for limited set)
    mapper = get_mapper()

    result = {
        "status": "complete",
        "regulator_gene": regulator_gene,
        "num_targets": total_targets,
        "targets_returned": len(limited_targets),
        "targets": [
            {
                "ensembl_id": tgt,
                "symbol": mapper.ensembl_to_symbol(tgt) or tgt,
                "edge_weight": round(w, 4)
            }
            for tgt, w in limited_targets
        ]
    }

    if total_targets > max_targets:
        result["note"] = f"Showing top {max_targets} of {total_targets} targets (sorted by edge weight). Increase max_targets to see more."

    return result


def simulate_knockdown_with_embeddings(
    network_df: pd.DataFrame,
    gene: str,
    model: "CascadeModel",
    depth: int = 2,
    top_k: int = 25,
    alpha: float = 0.7,
    decay: float = 0.5,
    embedding_threshold: float = 0.3
) -> dict:
    """
    Simulate knockdown using network topology + embedding similarity.

    Combines two sources of information:
    1. Network-based propagation (existing BFS algorithm)
    2. Embedding similarity (learned functional relationships)

    The embedding similarity modulates network effects and can discover
    indirect effects not captured in the static network.

    Args:
        network_df: Gene regulatory network
        gene: Gene to knock down (Ensembl ID)
        model: CascadeModel with loaded embeddings
        depth: Network propagation depth (1=direct, 2+=indirect)
        top_k: Number of top affected genes to return
        alpha: Weight for network score (1-alpha for embedding). Default 0.7
        decay: Effect decay per hop in network propagation
        embedding_threshold: Minimum similarity to consider (0-1)

    Returns:
        Dict with perturbation results including combined scores
    """
    from tools.cache import get_embedding_cache

    adj = _build_adjacency(network_df)

    # Check if gene exists in model vocabulary
    if not model.is_gene_in_vocab(gene):
        return {
            "status": "error",
            "error": f"Gene {gene} not found in model vocabulary",
            "suggestion": "Check that the Ensembl ID is correct"
        }

    # 1. Get network-based effects (existing algorithm)
    network_effects = _propagate_effect(adj, gene, initial_effect=-1.0, depth=depth, decay=decay)
    network_effects.pop(gene, None)  # Remove perturbed gene

    # 2. Get embedding-based similarities
    cache = get_embedding_cache(model)
    similarities_df = cache.get_similarities(gene)

    if similarities_df is None:
        return {
            "status": "error",
            "error": f"Could not compute similarities for gene {gene}",
        }

    # Convert to dict for fast lookup
    similarity_map = dict(zip(
        similarities_df["ensembl_id"],
        similarities_df["similarity"]
    ))

    # 3. Combine scores
    combined_effects = {}

    # Process genes that are in the network
    for target, net_effect in network_effects.items():
        emb_sim = similarity_map.get(target, 0.0)
        # Combined score: weighted average, with embedding modulating the effect
        # Higher similarity = effect is more reliable/stronger
        if emb_sim >= embedding_threshold:
            combined = alpha * net_effect + (1 - alpha) * emb_sim * net_effect
        else:
            combined = alpha * net_effect
        combined_effects[target] = {
            "network_effect": net_effect,
            "embedding_similarity": emb_sim,
            "combined_effect": combined
        }

    # 4. Add genes with high embedding similarity but not in network
    # These are potential indirect effects
    for _, row in similarities_df.iterrows():
        target = row["ensembl_id"]
        emb_sim = row["similarity"]

        if target in combined_effects:
            continue  # Already processed
        if target == gene:
            continue
        if emb_sim < embedding_threshold:
            continue

        # Gene has high embedding similarity but no network connection
        # Predict effect based purely on similarity
        indirect_effect = emb_sim * -1.0  # knockdown direction

        combined_effects[target] = {
            "network_effect": 0.0,
            "embedding_similarity": emb_sim,
            "combined_effect": (1 - alpha) * indirect_effect,
            "source": "embedding_only"
        }

    if not combined_effects:
        return {
            "status": "complete",
            "perturbed_gene": gene,
            "perturbation_type": "knockdown_with_embeddings",
            "message": f"Gene {gene} has no predicted effects",
            "affected_genes": []
        }

    # Sort by absolute combined effect magnitude
    sorted_effects = sorted(
        combined_effects.items(),
        key=lambda x: abs(x[1]["combined_effect"]),
        reverse=True
    )[:top_k]

    # Get gene mapper for symbol lookups
    mapper = get_mapper()

    affected_genes = []
    for g, scores in sorted_effects:
        combined = scores["combined_effect"]
        affected_genes.append({
            "ensembl_id": g,
            "symbol": mapper.ensembl_to_symbol(g) or g,
            "combined_effect": round(combined, 4),
            "network_effect": round(scores["network_effect"], 4),
            "embedding_similarity": round(scores["embedding_similarity"], 4),
            "direction": "down" if combined < 0 else "up",
            "magnitude": round(abs(combined), 4),
            "source": scores.get("source", "network+embedding")
        })

    return {
        "status": "complete",
        "perturbed_gene": gene,
        "perturbation_type": "knockdown_with_embeddings",
        "propagation_depth": depth,
        "alpha": alpha,
        "embedding_threshold": embedding_threshold,
        "total_affected_genes": len(combined_effects),
        "top_affected_genes": affected_genes,
        "method": "Combined network propagation + GREmLN embeddings"
    }


def simulate_overexpression_with_embeddings(
    network_df: pd.DataFrame,
    gene: str,
    model: "CascadeModel",
    fold_change: float = 2.0,
    depth: int = 2,
    top_k: int = 25,
    alpha: float = 0.7,
    decay: float = 0.5,
    embedding_threshold: float = 0.3
) -> dict:
    """
    Simulate overexpression using network topology + embedding similarity.

    Args:
        network_df: Gene regulatory network
        gene: Gene to overexpress (Ensembl ID)
        model: CascadeModel with loaded embeddings
        fold_change: Expression multiplier (e.g., 2.0 = 2x expression)
        depth: Network propagation depth
        top_k: Number of top affected genes to return
        alpha: Weight for network score (1-alpha for embedding)
        decay: Effect decay per hop
        embedding_threshold: Minimum similarity to consider

    Returns:
        Dict with perturbation results including combined scores
    """
    from tools.cache import get_embedding_cache

    adj = _build_adjacency(network_df)

    if not model.is_gene_in_vocab(gene):
        return {
            "status": "error",
            "error": f"Gene {gene} not found in model vocabulary",
        }

    # Overexpression effect = log2(fold_change)
    initial_effect = np.log2(fold_change)

    # 1. Get network-based effects
    network_effects = _propagate_effect(adj, gene, initial_effect=initial_effect, depth=depth, decay=decay)
    network_effects.pop(gene, None)

    # 2. Get embedding-based similarities
    cache = get_embedding_cache(model)
    similarities_df = cache.get_similarities(gene)

    if similarities_df is None:
        return {
            "status": "error",
            "error": f"Could not compute similarities for gene {gene}",
        }

    similarity_map = dict(zip(
        similarities_df["ensembl_id"],
        similarities_df["similarity"]
    ))

    # 3. Combine scores
    combined_effects = {}

    for target, net_effect in network_effects.items():
        emb_sim = similarity_map.get(target, 0.0)
        if emb_sim >= embedding_threshold:
            combined = alpha * net_effect + (1 - alpha) * emb_sim * net_effect
        else:
            combined = alpha * net_effect
        combined_effects[target] = {
            "network_effect": net_effect,
            "embedding_similarity": emb_sim,
            "combined_effect": combined
        }

    # 4. Add embedding-only effects
    for _, row in similarities_df.iterrows():
        target = row["ensembl_id"]
        emb_sim = row["similarity"]

        if target in combined_effects or target == gene:
            continue
        if emb_sim < embedding_threshold:
            continue

        indirect_effect = emb_sim * initial_effect
        combined_effects[target] = {
            "network_effect": 0.0,
            "embedding_similarity": emb_sim,
            "combined_effect": (1 - alpha) * indirect_effect,
            "source": "embedding_only"
        }

    if not combined_effects:
        return {
            "status": "complete",
            "perturbed_gene": gene,
            "perturbation_type": "overexpression_with_embeddings",
            "fold_change": fold_change,
            "message": f"Gene {gene} has no predicted effects",
            "affected_genes": []
        }

    sorted_effects = sorted(
        combined_effects.items(),
        key=lambda x: abs(x[1]["combined_effect"]),
        reverse=True
    )[:top_k]

    mapper = get_mapper()

    affected_genes = []
    for g, scores in sorted_effects:
        combined = scores["combined_effect"]
        affected_genes.append({
            "ensembl_id": g,
            "symbol": mapper.ensembl_to_symbol(g) or g,
            "combined_effect": round(combined, 4),
            "network_effect": round(scores["network_effect"], 4),
            "embedding_similarity": round(scores["embedding_similarity"], 4),
            "direction": "up" if combined > 0 else "down",
            "magnitude": round(abs(combined), 4),
            "source": scores.get("source", "network+embedding")
        })

    return {
        "status": "complete",
        "perturbed_gene": gene,
        "perturbation_type": "overexpression_with_embeddings",
        "fold_change": fold_change,
        "propagation_depth": depth,
        "alpha": alpha,
        "embedding_threshold": embedding_threshold,
        "total_affected_genes": len(combined_effects),
        "top_affected_genes": affected_genes,
        "method": "Combined network propagation + GREmLN embeddings"
    }


# Type hint for documentation (not runtime import)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.model_inference import CascadeModel
