"""
GREmLN Perturbation Analysis MCP Server

Provides tools for in silico gene perturbation analysis using
pre-computed gene regulatory networks and GREmLN model embeddings.
"""

from pathlib import Path
from typing import Optional
from fastmcp import FastMCP

from tools.loader import load_network, get_available_cell_types, MODEL_PATH
from tools.perturb import (
    simulate_knockdown,
    simulate_overexpression,
    get_regulators,
    get_targets,
    simulate_knockdown_with_embeddings,
    simulate_overexpression_with_embeddings,
)
from tools.gene_id_mapper import GeneIDMapper
from tools.ppi.string_client import get_string_client

# Initialize MCP server
mcp = FastMCP("gremln_mcp_server")

BASE_DIR = Path(__file__).parent
NETWORKS_DIR = BASE_DIR / "data" / "networks"

# Initialize gene ID mapper (singleton)
gene_mapper = GeneIDMapper()

# Global model instance (lazy loaded)
_gremln_model = None


def get_model():
    """Get or create the singleton GREmLNModel instance."""
    global _gremln_model
    if _gremln_model is None:
        from tools.model_inference import GREmLNModel
        _gremln_model = GREmLNModel(MODEL_PATH)
        _gremln_model.load()
    return _gremln_model


@mcp.tool()
def list_cell_types() -> dict:
    """
    List all available cell types with pre-computed regulatory networks.

    Returns a list of cell type names that can be used in other perturbation
    analysis tools.
    """
    cell_types = get_available_cell_types(NETWORKS_DIR)
    return {
        "available_cell_types": cell_types,
        "count": len(cell_types),
        "note": "Use one of these cell types in perturbation analysis tools."
    }


@mcp.tool()
def analyze_gene_knockdown(
    gene: str,
    cell_type: str = "epithelial_cell",
    depth: int = 2,
    top_k: int = 25
) -> dict:
    """
    Simulate knocking down a gene and predict downstream effects on gene expression.

    This tool uses the gene regulatory network to propagate the effect of
    completely silencing a gene (e.g., via siRNA or CRISPR knockout) and
    predicts which downstream genes will be affected.

    Args:
        gene: Gene to knock down (gene symbol like MYC or Ensembl ID like ENSG00000136997)
        cell_type: Cell type context for the regulatory network (default: epithelial_cell)
        depth: How many network hops to propagate (1=direct targets only, 2=includes indirect effects)
        top_k: Number of top affected genes to return (default: 25)

    Returns:
        Predicted expression changes for downstream target genes, sorted by impact magnitude.
    """
    network_path = NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        available = get_available_cell_types(NETWORKS_DIR)
        return {
            "error": f"Network not found for cell type: {cell_type}",
            "available_cell_types": available
        }

    # Resolve gene symbol to Ensembl ID if needed
    ensembl_id = gene_mapper.symbol_to_ensembl(gene)
    if ensembl_id is None:
        return {
            "error": f"Could not resolve gene '{gene}' to Ensembl ID",
            "suggestion": "Use an Ensembl ID (ENSG...) or check the gene symbol spelling"
        }

    network_df = load_network(network_path)
    result = simulate_knockdown(network_df, ensembl_id, depth=depth, top_k=top_k)
    result["cell_type"] = cell_type
    result["input_gene"] = gene
    result["resolved_ensembl_id"] = ensembl_id
    return result


@mcp.tool()
def analyze_gene_overexpression(
    gene: str,
    cell_type: str = "epithelial_cell",
    fold_change: float = 2.0,
    depth: int = 2,
    top_k: int = 25
) -> dict:
    """
    Simulate overexpressing a gene and predict downstream effects on gene expression.

    This tool models what happens when a gene's expression is artificially increased
    (e.g., via transfection or CRISPRa) and predicts downstream expression changes.

    Args:
        gene: Gene to overexpress (gene symbol like MYC or Ensembl ID like ENSG00000136997)
        cell_type: Cell type context for the regulatory network
        fold_change: How much to increase expression (2.0 = 2x normal expression)
        depth: How many network hops to propagate effects
        top_k: Number of top affected genes to return

    Returns:
        Predicted expression changes for downstream target genes.
    """
    network_path = NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        available = get_available_cell_types(NETWORKS_DIR)
        return {
            "error": f"Network not found for cell type: {cell_type}",
            "available_cell_types": available
        }

    # Resolve gene symbol to Ensembl ID if needed
    ensembl_id = gene_mapper.symbol_to_ensembl(gene)
    if ensembl_id is None:
        return {
            "error": f"Could not resolve gene '{gene}' to Ensembl ID",
            "suggestion": "Use an Ensembl ID (ENSG...) or check the gene symbol spelling"
        }

    network_df = load_network(network_path)
    result = simulate_overexpression(
        network_df, ensembl_id, fold_change=fold_change, depth=depth, top_k=top_k
    )
    result["cell_type"] = cell_type
    result["input_gene"] = gene
    result["resolved_ensembl_id"] = ensembl_id
    return result


@mcp.tool()
def find_gene_regulators(
    target_gene: str,
    cell_type: str = "epithelial_cell",
    max_regulators: int = 50
) -> dict:
    """
    Find all transcription factors and regulators that control a target gene.

    This identifies upstream regulators in the gene regulatory network -
    useful for understanding what controls a gene of interest.

    Args:
        target_gene: Gene to find regulators for (gene symbol like TP53 or Ensembl ID)
        cell_type: Cell type context for the regulatory network
        max_regulators: Maximum number of regulators to return (default: 50, to avoid timeout)

    Returns:
        List of regulators with their regulatory edge weights (mutual information scores).
    """
    try:
        network_path = NETWORKS_DIR / cell_type / "network.tsv"
        if not network_path.exists():
            available = get_available_cell_types(NETWORKS_DIR)
            return {
                "error": f"Network not found for cell type: {cell_type}",
                "available_cell_types": available
            }

        # Resolve gene symbol to Ensembl ID if needed
        ensembl_id = gene_mapper.symbol_to_ensembl(target_gene)
        if ensembl_id is None:
            return {
                "error": f"Could not resolve gene '{target_gene}' to Ensembl ID",
                "suggestion": "Use an Ensembl ID (ENSG...) or check the gene symbol spelling"
            }

        network_df = load_network(network_path)
        result = get_regulators(network_df, ensembl_id, max_regulators=max_regulators)
        result["cell_type"] = cell_type
        result["input_gene"] = target_gene
        result["resolved_ensembl_id"] = ensembl_id
        return result
    except Exception as e:
        return {
            "error": f"Tool execution failed: {str(e)}",
            "error_type": type(e).__name__,
            "suggestion": "This may be a timeout issue. Try reducing max_regulators parameter."
        }


@mcp.tool()
def find_gene_targets(
    regulator_gene: str,
    cell_type: str = "epithelial_cell",
    max_targets: int = 50
) -> dict:
    """
    Find all target genes controlled by a regulator or transcription factor.

    This identifies downstream targets in the gene regulatory network -
    useful for understanding what a regulator controls.

    Args:
        regulator_gene: Regulator/TF to find targets for (gene symbol like MYC or Ensembl ID)
        cell_type: Cell type context for the regulatory network
        max_targets: Maximum number of targets to return (default: 50, to avoid timeout)

    Returns:
        List of target genes with their regulatory edge weights.
    """
    try:
        network_path = NETWORKS_DIR / cell_type / "network.tsv"
        if not network_path.exists():
            available = get_available_cell_types(NETWORKS_DIR)
            return {
                "error": f"Network not found for cell type: {cell_type}",
                "available_cell_types": available
            }

        # Resolve gene symbol to Ensembl ID if needed
        ensembl_id = gene_mapper.symbol_to_ensembl(regulator_gene)
        if ensembl_id is None:
            return {
                "error": f"Could not resolve gene '{regulator_gene}' to Ensembl ID",
                "suggestion": "Use an Ensembl ID (ENSG...) or check the gene symbol spelling"
            }

        network_df = load_network(network_path)
        result = get_targets(network_df, ensembl_id, max_targets=max_targets)
        result["cell_type"] = cell_type
        result["input_gene"] = regulator_gene
        result["resolved_ensembl_id"] = ensembl_id
        return result
    except Exception as e:
        return {
            "error": f"Tool execution failed: {str(e)}",
            "error_type": type(e).__name__,
            "suggestion": "This may be a timeout issue. Try reducing max_targets parameter."
        }


@mcp.tool()
def lookup_gene(gene: str) -> dict:
    """
    Look up gene information and convert between symbol and Ensembl ID.

    Args:
        gene: Gene symbol (e.g., MYC) or Ensembl ID (e.g., ENSG00000136997)

    Returns:
        Gene symbol and Ensembl ID mapping.
    """
    if gene.upper().startswith("ENSG"):
        # Input is Ensembl ID, look up symbol
        symbol = gene_mapper.ensembl_to_symbol(gene)
        return {
            "input": gene,
            "ensembl_id": gene.upper(),
            "gene_symbol": symbol,
            "status": "found" if symbol else "symbol_not_found"
        }
    else:
        # Input is symbol, look up Ensembl ID
        ensembl_id = gene_mapper.symbol_to_ensembl(gene)
        return {
            "input": gene,
            "gene_symbol": gene.upper(),
            "ensembl_id": ensembl_id,
            "status": "found" if ensembl_id else "ensembl_id_not_found"
        }


# =============================================================================
# Protein-Protein Interaction Tools (STRING database)
# =============================================================================


@mcp.tool()
def get_protein_interactions(
    gene: str,
    min_score: int = 400,
    limit: int = 25
) -> dict:
    """
    Get protein-protein interactions from STRING database.

    Retrieves physical and functional protein interactions for a gene,
    useful for understanding what happens at the protein level after
    perturbation (e.g., APC knockdown affects beta-catenin binding).

    Args:
        gene: Gene symbol (e.g., APC, TP53, CTNNB1)
        min_score: Minimum confidence score 0-1000 (150=low, 400=medium, 700=high, 900=highest)
        limit: Maximum number of interaction partners to return (default: 25)

    Returns:
        List of interaction partners with confidence scores and evidence types.
        Evidence types include: experimental, database, textmining, coexpression.
    """
    try:
        # Resolve Ensembl ID to symbol if needed
        if gene.upper().startswith("ENSG"):
            symbol = gene_mapper.ensembl_to_symbol(gene)
            if symbol is None:
                return {
                    "error": f"Could not resolve Ensembl ID '{gene}' to gene symbol",
                    "suggestion": "Use a gene symbol directly (e.g., APC, TP53)"
                }
            gene = symbol

        client = get_string_client()
        result = client.get_interactions(gene, min_score=min_score, limit=limit)

        # Add interpretation help
        if "interactions" in result and result["interactions"]:
            result["use_case"] = (
                "These proteins physically or functionally interact with your query gene. "
                "After perturbation, these interactions may be disrupted, affecting downstream signaling."
            )

        return result

    except Exception as e:
        return {
            "error": f"Failed to fetch protein interactions: {str(e)}",
            "error_type": type(e).__name__
        }


# =============================================================================
# Model-Based Tools (using GREmLN embeddings)
# =============================================================================


@mcp.tool()
def get_model_status() -> dict:
    """
    Check GREmLN model status and GPU availability.

    Returns information about whether the model is loaded, GPU availability,
    and the number of genes in the model vocabulary.
    """
    import torch

    try:
        model = get_model()
        stats = model.get_embedding_stats()
        return {
            "model_loaded": True,
            "gpu_available": torch.cuda.is_available(),
            "device": stats["device"],
            "num_genes": stats["num_actual_genes"],
            "embedding_dim": stats["embedding_dim"],
            "checkpoint_path": str(MODEL_PATH),
        }
    except Exception as e:
        return {
            "model_loaded": False,
            "gpu_available": torch.cuda.is_available(),
            "error": str(e),
            "checkpoint_path": str(MODEL_PATH),
        }


@mcp.tool()
def analyze_gene_knockdown_model(
    gene: str,
    cell_type: str = "epithelial_cell",
    depth: int = 2,
    top_k: int = 25,
    alpha: float = 0.7,
    embedding_threshold: float = 0.3
) -> dict:
    """
    Simulate gene knockdown using GREmLN model embeddings.

    This enhanced version combines:
    1. Network topology (regulatory connections from mutual information)
    2. GREmLN embeddings (learned gene relationships from 11M cells)

    The embeddings capture functional relationships that may not be present
    in the static network, potentially discovering indirect effects.

    Args:
        gene: Gene to knock down (gene symbol like MYC or Ensembl ID)
        cell_type: Cell type context for the regulatory network
        depth: Network propagation depth (1=direct, 2+=indirect)
        top_k: Number of top affected genes to return
        alpha: Weight for network vs embedding (0.0-1.0). Higher = more network weight
        embedding_threshold: Minimum embedding similarity to consider (0.0-1.0)

    Returns:
        Predicted expression changes with both network and embedding scores.
    """
    network_path = NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        available = get_available_cell_types(NETWORKS_DIR)
        return {
            "error": f"Network not found for cell type: {cell_type}",
            "available_cell_types": available
        }

    # Resolve gene symbol to Ensembl ID if needed
    ensembl_id = gene_mapper.symbol_to_ensembl(gene)
    if ensembl_id is None:
        return {
            "error": f"Could not resolve gene '{gene}' to Ensembl ID",
            "suggestion": "Use an Ensembl ID (ENSG...) or check the gene symbol spelling"
        }

    model = get_model()
    network_df = load_network(network_path)

    result = simulate_knockdown_with_embeddings(
        network_df,
        ensembl_id,
        model,
        depth=depth,
        top_k=top_k,
        alpha=alpha,
        embedding_threshold=embedding_threshold
    )

    result["cell_type"] = cell_type
    result["input_gene"] = gene
    result["resolved_ensembl_id"] = ensembl_id
    return result


@mcp.tool()
def analyze_gene_overexpression_model(
    gene: str,
    cell_type: str = "epithelial_cell",
    fold_change: float = 2.0,
    depth: int = 2,
    top_k: int = 25,
    alpha: float = 0.7,
    embedding_threshold: float = 0.3
) -> dict:
    """
    Simulate gene overexpression using GREmLN model embeddings.

    This enhanced version combines network topology with learned gene
    embeddings for improved predictions.

    Args:
        gene: Gene to overexpress (gene symbol like MYC or Ensembl ID)
        cell_type: Cell type context for the regulatory network
        fold_change: How much to increase expression (2.0 = 2x normal)
        depth: Network propagation depth
        top_k: Number of top affected genes to return
        alpha: Weight for network vs embedding (0.0-1.0)
        embedding_threshold: Minimum embedding similarity to consider

    Returns:
        Predicted expression changes with both network and embedding scores.
    """
    network_path = NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        available = get_available_cell_types(NETWORKS_DIR)
        return {
            "error": f"Network not found for cell type: {cell_type}",
            "available_cell_types": available
        }

    ensembl_id = gene_mapper.symbol_to_ensembl(gene)
    if ensembl_id is None:
        return {
            "error": f"Could not resolve gene '{gene}' to Ensembl ID",
            "suggestion": "Use an Ensembl ID (ENSG...) or check the gene symbol spelling"
        }

    model = get_model()
    network_df = load_network(network_path)

    result = simulate_overexpression_with_embeddings(
        network_df,
        ensembl_id,
        model,
        fold_change=fold_change,
        depth=depth,
        top_k=top_k,
        alpha=alpha,
        embedding_threshold=embedding_threshold
    )

    result["cell_type"] = cell_type
    result["input_gene"] = gene
    result["resolved_ensembl_id"] = ensembl_id
    return result


@mcp.tool()
def get_gene_similarity(gene1: str, gene2: str) -> dict:
    """
    Get embedding similarity between two genes.

    Uses the GREmLN model's learned gene representations to compute
    cosine similarity. Higher similarity indicates genes that behave
    similarly across cell types.

    Args:
        gene1: First gene (symbol or Ensembl ID)
        gene2: Second gene (symbol or Ensembl ID)

    Returns:
        Similarity score (-1 to 1) where 1 = identical behavior.
    """
    # Resolve gene symbols to Ensembl IDs
    ensembl1 = gene_mapper.symbol_to_ensembl(gene1)
    ensembl2 = gene_mapper.symbol_to_ensembl(gene2)

    if ensembl1 is None:
        return {"error": f"Could not resolve gene '{gene1}' to Ensembl ID"}
    if ensembl2 is None:
        return {"error": f"Could not resolve gene '{gene2}' to Ensembl ID"}

    model = get_model()

    # Check if genes are in vocabulary
    if not model.is_gene_in_vocab(ensembl1):
        return {"error": f"Gene {gene1} ({ensembl1}) not in model vocabulary"}
    if not model.is_gene_in_vocab(ensembl2):
        return {"error": f"Gene {gene2} ({ensembl2}) not in model vocabulary"}

    similarity = model.compute_similarity(ensembl1, ensembl2)

    return {
        "gene1": gene1,
        "gene1_ensembl": ensembl1,
        "gene2": gene2,
        "gene2_ensembl": ensembl2,
        "similarity": round(similarity, 4) if similarity is not None else None,
        "interpretation": _interpret_similarity(similarity)
    }


def _interpret_similarity(sim: Optional[float]) -> str:
    """Interpret a similarity score."""
    if sim is None:
        return "Could not compute similarity"
    if sim >= 0.8:
        return "Very high similarity - genes likely have similar functions"
    if sim >= 0.5:
        return "High similarity - genes may be functionally related"
    if sim >= 0.3:
        return "Moderate similarity - some functional overlap possible"
    if sim >= 0.0:
        return "Low similarity - likely unrelated functions"
    return "Negative similarity - potentially opposite functions"


@mcp.tool()
def find_similar_genes(
    gene: str,
    top_k: int = 20
) -> dict:
    """
    Find genes most similar to a query gene based on GREmLN embeddings.

    Uses learned gene representations to find functionally similar genes,
    even if they don't share direct network connections.

    Args:
        gene: Query gene (symbol or Ensembl ID)
        top_k: Number of similar genes to return (default: 20, max recommended: 50)

    Returns:
        List of most similar genes with similarity scores.
    """
    try:
        ensembl_id = gene_mapper.symbol_to_ensembl(gene)
        if ensembl_id is None:
            return {"error": f"Could not resolve gene '{gene}' to Ensembl ID"}

        model = get_model()

        if not model.is_gene_in_vocab(ensembl_id):
            return {"error": f"Gene {gene} ({ensembl_id}) not in model vocabulary"}

        similar_df = model.get_top_similar_genes(ensembl_id, top_k=top_k)

        if similar_df is None:
            return {"error": f"Could not compute similarities for {gene}"}

        similar_genes = []
        for _, row in similar_df.iterrows():
            target_ensembl = row["ensembl_id"]
            # Use cached lookup only to avoid API timeouts
            symbol = gene_mapper.ensembl_to_symbol(target_ensembl) or target_ensembl
            similar_genes.append({
                "gene_symbol": symbol,
                "ensembl_id": target_ensembl,
                "similarity": round(row["similarity"], 4)
            })

        return {
            "query_gene": gene,
            "query_ensembl": ensembl_id,
            "top_similar_genes": similar_genes,
            "note": "Similarity based on GREmLN embeddings learned from 11M cells"
        }
    except Exception as e:
        return {
            "error": f"Tool execution failed: {str(e)}",
            "error_type": type(e).__name__,
            "suggestion": "This may be a timeout issue. Try reducing top_k parameter or check model status with get_model_status."
        }


@mcp.tool()
def analyze_network_vulnerability(
    cell_type: str = "epithelial_cell",
    top_k: int = 20,
    include_cascade: bool = True
) -> dict:
    """
    Identify critical hub genes and network vulnerabilities for drug target discovery.

    Analyzes network topology to find genes whose disruption would cause
    maximum downstream impact - potential high-value drug targets.

    Args:
        cell_type: Cell type context for the regulatory network
        top_k: Number of top vulnerable genes to return (default: 20)
        include_cascade: Whether to simulate knockdown cascade (slower but more accurate)

    Returns:
        Ranked list of genes by vulnerability score with:
        - Hub score (number of direct targets)
        - Regulator count (number of upstream controllers)
        - Cascade impact (predicted downstream disruption)
        - Vulnerability score (combined metric)
    """
    try:
        network_path = NETWORKS_DIR / cell_type / "network.tsv"
        if not network_path.exists():
            available = get_available_cell_types(NETWORKS_DIR)
            return {
                "error": f"Network not found for cell type: {cell_type}",
                "available_cell_types": available
            }

        network_df = load_network(network_path)

        # Build adjacency structures
        from collections import defaultdict

        # Forward: regulator -> targets
        forward_adj = defaultdict(list)
        # Reverse: target -> regulators
        reverse_adj = defaultdict(list)

        for _, row in network_df.iterrows():
            reg = row["regulator"]
            tgt = row["target"]
            weight = row.get("mi", 1.0)
            if weight > 0:
                forward_adj[reg].append((tgt, float(weight)))
                reverse_adj[tgt].append((reg, float(weight)))

        # Calculate vulnerability metrics for all regulators
        vulnerability_scores = []
        all_regulators = set(network_df["regulator"].unique())

        for gene in all_regulators:
            targets = forward_adj.get(gene, [])
            regulators = reverse_adj.get(gene, [])

            hub_score = len(targets)
            regulator_count = len(regulators)
            avg_target_weight = sum(w for _, w in targets) / len(targets) if targets else 0

            # Cascade impact: count 2nd-order targets
            cascade_targets = set()
            if include_cascade and hub_score > 0:
                for tgt, _ in targets[:50]:  # Limit to avoid timeout
                    second_order = forward_adj.get(tgt, [])
                    cascade_targets.update(t for t, _ in second_order[:20])

            cascade_size = len(cascade_targets)

            # Combined vulnerability score
            # Higher = more critical to network (better drug target)
            vulnerability = (
                hub_score * 1.0 +           # Direct targets
                cascade_size * 0.3 +         # Indirect reach
                avg_target_weight * 10 +     # Connection strength
                (1 / (regulator_count + 1)) * 5  # Less regulated = harder to compensate
            )

            vulnerability_scores.append({
                "ensembl_id": gene,
                "hub_score": hub_score,
                "regulator_count": regulator_count,
                "cascade_reach": cascade_size,
                "avg_edge_weight": round(avg_target_weight, 4),
                "vulnerability_score": round(vulnerability, 2)
            })

        # Sort by vulnerability score
        vulnerability_scores.sort(key=lambda x: x["vulnerability_score"], reverse=True)
        top_genes = vulnerability_scores[:top_k]

        # Add gene symbols for top genes
        for gene_data in top_genes:
            symbol = gene_mapper.ensembl_to_symbol(gene_data["ensembl_id"])
            gene_data["symbol"] = symbol or gene_data["ensembl_id"]

        return {
            "status": "complete",
            "cell_type": cell_type,
            "total_regulators_analyzed": len(all_regulators),
            "top_vulnerable_genes": top_genes,
            "interpretation": {
                "hub_score": "Number of direct target genes (higher = more influence)",
                "regulator_count": "Number of upstream controllers (lower = harder to compensate)",
                "cascade_reach": "2nd-order targets affected by knockdown",
                "vulnerability_score": "Combined metric for drug target prioritization"
            },
            "use_case": "High vulnerability genes are potential drug targets - their disruption causes maximum network damage"
        }

    except Exception as e:
        return {
            "error": f"Tool execution failed: {str(e)}",
            "error_type": type(e).__name__,
            "suggestion": "Try reducing top_k or setting include_cascade=false"
        }


@mcp.tool()
def compare_gene_vulnerability(
    genes: list[str],
    cell_type: str = "epithelial_cell"
) -> dict:
    """
    Compare vulnerability scores for specific genes of interest.

    Use this to evaluate candidate drug targets or compare known disease genes.

    Args:
        genes: List of gene symbols or Ensembl IDs to compare
        cell_type: Cell type context for the regulatory network

    Returns:
        Vulnerability metrics for each gene, ranked by score.
    """
    try:
        network_path = NETWORKS_DIR / cell_type / "network.tsv"
        if not network_path.exists():
            available = get_available_cell_types(NETWORKS_DIR)
            return {
                "error": f"Network not found for cell type: {cell_type}",
                "available_cell_types": available
            }

        network_df = load_network(network_path)

        # Build adjacency structures
        from collections import defaultdict

        forward_adj = defaultdict(list)
        reverse_adj = defaultdict(list)

        for _, row in network_df.iterrows():
            reg = row["regulator"]
            tgt = row["target"]
            weight = row.get("mi", 1.0)
            if weight > 0:
                forward_adj[reg].append((tgt, float(weight)))
                reverse_adj[tgt].append((reg, float(weight)))

        results = []

        for gene in genes:
            # Resolve to Ensembl ID
            ensembl_id = gene_mapper.symbol_to_ensembl(gene)
            if ensembl_id is None:
                results.append({
                    "input": gene,
                    "error": "Could not resolve gene"
                })
                continue

            targets = forward_adj.get(ensembl_id, [])
            regulators = reverse_adj.get(ensembl_id, [])

            hub_score = len(targets)
            regulator_count = len(regulators)
            avg_weight = sum(w for _, w in targets) / len(targets) if targets else 0

            # Cascade calculation
            cascade_targets = set()
            for tgt, _ in targets[:50]:
                second_order = forward_adj.get(tgt, [])
                cascade_targets.update(t for t, _ in second_order[:20])

            vulnerability = (
                hub_score * 1.0 +
                len(cascade_targets) * 0.3 +
                avg_weight * 10 +
                (1 / (regulator_count + 1)) * 5
            )

            # Determine role
            if hub_score > 50 and regulator_count < 10:
                role = "Master regulator (high-value target)"
            elif hub_score > 20:
                role = "Hub gene"
            elif regulator_count > hub_score * 2:
                role = "Downstream effector"
            else:
                role = "Intermediate node"

            results.append({
                "input": gene,
                "ensembl_id": ensembl_id,
                "symbol": gene_mapper.ensembl_to_symbol(ensembl_id) or gene,
                "hub_score": hub_score,
                "regulator_count": regulator_count,
                "cascade_reach": len(cascade_targets),
                "vulnerability_score": round(vulnerability, 2),
                "network_role": role
            })

        # Sort by vulnerability
        results.sort(key=lambda x: x.get("vulnerability_score", 0), reverse=True)

        return {
            "status": "complete",
            "cell_type": cell_type,
            "genes_analyzed": len(genes),
            "comparison": results,
            "recommendation": results[0]["input"] if results and "vulnerability_score" in results[0] else None
        }

    except Exception as e:
        return {
            "error": f"Tool execution failed: {str(e)}",
            "error_type": type(e).__name__
        }


@mcp.tool()
def get_embedding_cache_stats() -> dict:
    """
    Get statistics about the embedding similarity cache.

    Returns cache hit rate and size information, useful for monitoring
    performance of repeated queries.
    """
    try:
        from tools.cache import _embedding_cache
        if _embedding_cache is None:
            return {
                "cache_initialized": False,
                "note": "Cache is initialized on first model-based query"
            }
        return {
            "cache_initialized": True,
            **_embedding_cache.get_stats()
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
