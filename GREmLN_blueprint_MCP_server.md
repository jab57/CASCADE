# Blueprint: GREmLN Perturbation Analysis MCP Server

## 1. Project Objective

Build a Model Context Protocol (MCP) server that enables Claude Desktop to perform **in silico gene perturbation analysis** using the GREmLN gene regulatory network. Given a perturbed gene, the server computes predicted impact on gene expression across target genes in the regulatory network.

**Core Capabilities:**
- Load pre-computed gene regulatory networks for different cell types
- Perform knockdown/overexpression perturbation simulations
- Identify downstream target genes and predict expression changes
- Support biomarker discovery through network propagation analysis

## 2. Technical Requirements

| Component | Requirement |
|-----------|-------------|
| Language | Python 3.10+ |
| Inference | scGraphLLM (GREmLN core from CZI) |
| Framework | FastMCP for Model Context Protocol |
| Compute | CPU sufficient for network analysis; GPU for model embeddings |

## 3. Environment Setup

```bash
# Create and activate virtual environment
python -m venv env
.\env\Scripts\activate  # Windows
# source env/bin/activate  # Linux/Mac

# Install core dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric lightning numpy pandas scipy

# Install GREmLN from official CZI repository
pip install git+https://github.com/czi-ai/GREmLN.git

# Install MCP framework
pip install fastmcp
```
## 4. Project Structure

```
GREmLN/
├── gremln_mcp_server.py   # MCP server entry point
├── tools/
│   ├── __init__.py
│   ├── loader.py          # Network and model loading utilities
│   └── perturb.py         # Perturbation analysis logic
├── data/
│   └── networks/          # Pre-computed regulatory networks per cell type
│       ├── epithelial_cell/network.tsv
│       ├── cd4_t_cells/network.tsv
│       └── ...
├── models/
│   └── model.ckpt         # GREmLN model checkpoint
└── requirements.txt
```

## 5. Network Data Format

The regulatory networks are stored as TSV files with columns:
- `regulator.values`: Ensembl gene ID of the regulator (transcription factor)
- `target.values`: Ensembl gene ID of the target gene
- `mi.values`: Mutual information score (edge weight)
- `scc.values`: Spearman correlation coefficient
- `count.values`: Number of observations
- `log.p.values`: Statistical significance
## 6. MCP Server Implementation (gremln_mcp_server.py)

```python
import os
from pathlib import Path
from fastmcp import FastMCP
from tools.loader import load_network, get_available_cell_types
from tools.perturb import (
    simulate_knockdown,
    simulate_overexpression,
    get_regulators,
    get_targets,
)

# Initialize MCP server
mcp = FastMCP("gremln_mcp_server")

BASE_DIR = Path(__file__).parent
NETWORKS_DIR = BASE_DIR / "data" / "networks"


@mcp.tool()
def list_cell_types() -> dict:
    """List all available cell types with pre-computed regulatory networks."""
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
    Simulate knocking down a gene and predict downstream effects.

    Args:
        gene: Gene to knock down (Ensembl ID or gene symbol)
        cell_type: Cell type for network context
        depth: How many hops to propagate effects (1=direct targets, 2=includes indirect)
        top_k: Number of top affected genes to return

    Returns:
        Predicted expression changes for downstream target genes
    """
    network_path = NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        return {"error": f"Network not found for cell type: {cell_type}"}

    network_df = load_network(network_path)
    result = simulate_knockdown(network_df, gene, depth=depth, top_k=top_k)
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
    Simulate overexpressing a gene and predict downstream effects.

    Args:
        gene: Gene to overexpress (Ensembl ID or gene symbol)
        cell_type: Cell type for network context
        fold_change: Expression multiplier (e.g., 2.0 = 2x expression)
        depth: How many hops to propagate effects
        top_k: Number of top affected genes to return
    """
    network_path = NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        return {"error": f"Network not found for cell type: {cell_type}"}

    network_df = load_network(network_path)
    result = simulate_overexpression(
        network_df, gene, fold_change=fold_change, depth=depth, top_k=top_k
    )
    return result


@mcp.tool()
def find_regulators(
    target_gene: str,
    cell_type: str = "epithelial_cell"
) -> dict:
    """
    Find all transcription factors/regulators that control a target gene.

    Args:
        target_gene: Gene to find regulators for
        cell_type: Cell type for network context
    """
    network_path = NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        return {"error": f"Network not found for cell type: {cell_type}"}

    network_df = load_network(network_path)
    return get_regulators(network_df, target_gene)


@mcp.tool()
def find_targets(
    regulator_gene: str,
    cell_type: str = "epithelial_cell"
) -> dict:
    """
    Find all target genes controlled by a regulator.

    Args:
        regulator_gene: Regulator/TF to find targets for
        cell_type: Cell type for network context
    """
    network_path = NETWORKS_DIR / cell_type / "network.tsv"
    if not network_path.exists():
        return {"error": f"Network not found for cell type: {cell_type}"}

    network_df = load_network(network_path)
    return get_targets(network_df, regulator_gene)


if __name__ == "__main__":
    mcp.run()
```
## 7. Claude Desktop Configuration

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "gremln_mcp_server": {
      "command": "C:/Dev/GREmLN/env/Scripts/python.exe",
      "args": ["C:/Dev/GREmLN/gremln_mcp_server.py"],
      "env": {
        "PYTHONPATH": "C:/Dev/GREmLN"
      }
    }
  }
}
```

## 8. Usage Examples

**List available cell types:**
> "What cell types are available for perturbation analysis?"

**Gene knockdown analysis:**
> "Simulate knocking down ENSG00000248333 in epithelial cells and show me the top 20 affected genes."

**Find regulators:**
> "What transcription factors regulate ENSG00000141510 (TP53) in CD4 T cells?"

**Find targets:**
> "What genes are regulated by MYC in epithelial cells?"

**Overexpression analysis:**
> "What happens if we overexpress HNF4A by 3-fold in epithelial cells?"

## 9. Perturbation Analysis Algorithm

The perturbation analysis uses network propagation:

1. **Direct Effects (depth=1):** Find all edges where the perturbed gene is a regulator
2. **Indirect Effects (depth=2+):** Recursively find targets of affected genes
3. **Impact Scoring:** Compute propagated effect using edge weights (MI scores)
4. **Ranking:** Return top-k most affected genes sorted by impact magnitude

```
Impact(target) = Σ (edge_weight × parent_impact × perturbation_strength)
```

For knockdown: perturbation_strength = -1.0 (complete loss)
For overexpression: perturbation_strength = log2(fold_change)