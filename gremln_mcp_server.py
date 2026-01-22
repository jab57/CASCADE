"""
GREmLN Perturbation Analysis MCP Server

Provides tools for in silico gene perturbation analysis using
pre-computed gene regulatory networks.
"""

from pathlib import Path
from fastmcp import FastMCP

from tools.loader import load_network, get_available_cell_types
from tools.perturb import (
    simulate_knockdown,
    simulate_overexpression,
    get_regulators,
    get_targets,
)
from tools.gene_id_mapper import GeneIDMapper

# Initialize MCP server
mcp = FastMCP("gremln_mcp_server")

BASE_DIR = Path(__file__).parent
NETWORKS_DIR = BASE_DIR / "data" / "networks"

# Initialize gene ID mapper (singleton)
gene_mapper = GeneIDMapper()


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
    cell_type: str = "epithelial_cell"
) -> dict:
    """
    Find all transcription factors and regulators that control a target gene.

    This identifies upstream regulators in the gene regulatory network -
    useful for understanding what controls a gene of interest.

    Args:
        target_gene: Gene to find regulators for (gene symbol like TP53 or Ensembl ID)
        cell_type: Cell type context for the regulatory network

    Returns:
        List of regulators with their regulatory edge weights (mutual information scores).
    """
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
    result = get_regulators(network_df, ensembl_id)
    result["cell_type"] = cell_type
    result["input_gene"] = target_gene
    result["resolved_ensembl_id"] = ensembl_id
    return result


@mcp.tool()
def find_gene_targets(
    regulator_gene: str,
    cell_type: str = "epithelial_cell"
) -> dict:
    """
    Find all target genes controlled by a regulator or transcription factor.

    This identifies downstream targets in the gene regulatory network -
    useful for understanding what a regulator controls.

    Args:
        regulator_gene: Regulator/TF to find targets for (gene symbol like MYC or Ensembl ID)
        cell_type: Cell type context for the regulatory network

    Returns:
        List of target genes with their regulatory edge weights.
    """
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
    result = get_targets(network_df, ensembl_id)
    result["cell_type"] = cell_type
    result["input_gene"] = regulator_gene
    result["resolved_ensembl_id"] = ensembl_id
    return result


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


if __name__ == "__main__":
    mcp.run()
