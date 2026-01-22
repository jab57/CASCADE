# GREmLN MCP Server

A Model Context Protocol (MCP) server for **in silico gene perturbation analysis** using pre-computed gene regulatory networks from the GREmLN model.

## Features

- **Gene Knockdown Simulation**: Predict downstream effects of silencing a gene
- **Gene Overexpression Simulation**: Predict effects of increased gene expression
- **Regulator Discovery**: Find transcription factors controlling a target gene
- **Target Discovery**: Find genes controlled by a regulator
- **Gene ID Mapping**: Convert between gene symbols (MYC) and Ensembl IDs (ENSG...)

## Supported Cell Types

- Epithelial cells
- CD4/CD8 T cells
- CD14/CD16 Monocytes
- CD20 B cells
- NK cells, NKT cells
- Erythrocytes
- Monocyte-derived dendritic cells

## Installation

```bash
# Create virtual environment
python -m venv env

# Activate (Windows)
.\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the MCP Server

```bash
python gremln_mcp_server.py
```

### Claude Desktop Configuration

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "GREmLN": {
      "command": "C:/Dev/GREmLN/env/Scripts/python.exe",
      "args": ["C:/Dev/GREmLN/gremln_mcp_server.py"]
    }
  }
}
```

### Example Prompts

- "Simulate knocking down MYC in epithelial cells"
- "What genes does TP53 regulate in CD4 T cells?"
- "Find all regulators of BRCA1 in epithelial cells"
- "What happens if we overexpress HNF4A 3-fold?"

## Project Structure

```
GREmLN/
├── gremln_mcp_server.py    # MCP server entry point
├── tools/
│   ├── loader.py           # Network loading utilities
│   ├── perturb.py          # Perturbation analysis logic
│   └── gene_id_mapper.py   # Gene symbol/Ensembl ID conversion
├── data/
│   └── networks/           # Pre-computed regulatory networks
└── models/
    └── model.ckpt          # GREmLN model checkpoint
```

## Requirements

- Python 3.10+
- FastMCP
- pandas, numpy
- requests (for gene ID lookups)

## License

MIT
