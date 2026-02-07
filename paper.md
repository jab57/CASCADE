---
title: 'CASCADE: An MCP Server for Automated Gene Perturbation Analysis'
tags:
  - Python
  - gene perturbation
  - regulatory networks
  - Model Context Protocol
  - LangGraph
authors:
  - name: Jose Bird
    orcid: 0009-0006-2744-0606
    affiliation: 1
affiliations:
  - name: Bird AI Solutions
    index: 1
date: 6 February 2026
bibliography: paper.bib
---

# Summary

CASCADE (Computational Analysis of Simulated Cell And Drug Effects) is a Python server that exposes *in silico* gene perturbation analysis as structured tools via the Model Context Protocol (MCP) [@mcp]. Given a gene and cell type, CASCADE simulates knockdown or overexpression effects by propagating signals through pre-computed regulatory networks, queries external databases for corroborating evidence, and returns a structured report. A LangGraph-based workflow [@langgraph] automates the full analysis pipeline---gene resolution, role classification, parallel data retrieval, and report generation---so that a single tool call replaces what would otherwise require manual orchestration of multiple databases and analysis scripts.

CASCADE supports 10 immune and epithelial cell types, integrates protein-protein interactions from STRING [@szklarczyk2023], experimental knockdown signatures from LINCS L1000 [@subramanian2017], and super-enhancer annotations from dbSUPER [@khan2016], and optionally incorporates pre-trained gene embeddings from the GREmLN model [@gremln] to enhance predictions beyond static network topology.

# Statement of Need

Simulating the downstream effects of gene perturbation---whether through CRISPR knockout, RNAi, or pharmacological inhibition---requires combining regulatory network analysis, protein interaction data, and experimental perturbation signatures from separate tools and databases. Researchers must manually query each source, reconcile gene identifiers across databases, and synthesize cross-database results into a coherent interpretation.

Existing tools address individual aspects of this workflow: network inference packages reconstruct regulatory networks [@aibar2017], perturbation models forecast expression changes [@roohani2024], and interaction databases catalog physical associations [@szklarczyk2023]. However, no tool unifies these capabilities behind a single programmatic interface that automates the full analysis pipeline.

CASCADE fills this gap by providing an MCP server that any compatible client can call. A single request to `comprehensive_perturbation_analysis` triggers the complete workflow: resolve gene identifiers, classify the gene's regulatory role, select and execute appropriate analyses in parallel, and return a structured multi-source report. This eliminates manual orchestration and provides reproducible, deterministic results for any supported gene and cell type.

# Architecture

CASCADE follows a layered design (\autoref{fig:architecture}). The MCP server exposes 22 tools organized into six categories: workflow orchestration, perturbation simulation, gene similarity, network vulnerability, experimental data, and druggability assessment.

![CASCADE architecture. An MCP client sends a request to the CASCADE server, which routes it through a LangGraph workflow. The workflow classifies the gene, selects analyses based on gene role and depth, and executes independent batches in parallel. Analysis tools operate on pre-computed regulatory networks and gene embeddings, while external modules query STRING, LINCS, and dbSUPER.\label{fig:architecture}](figure_architecture.png)

The LangGraph workflow operates as a directed acyclic graph with conditional routing:

1. **Resolve** the gene identifier (symbol or Ensembl ID) via the Ensembl REST API with local caching.
2. **Classify** the gene's regulatory role by counting its targets and regulators in the cell-type-specific network.
3. **Route** to appropriate analysis batches based on gene role and requested depth (basic, focused, or comprehensive).
4. **Execute** up to three parallel batches: core analysis (perturbation propagation, regulators, targets), external data (STRING PPI, LINCS effects, super-enhancers), and insights (embedding similarity, vulnerability, cross-cell comparison).
5. **Generate** a structured report aggregating all results.

Network perturbation effects are computed via breadth-first propagation through directed regulatory edges weighted by mutual information. When gene embeddings are available, network scores are combined with embedding-based similarity to capture functional relationships beyond static network topology. Full algorithmic details are provided in the repository documentation.

# Functionality

CASCADE can be installed and used as follows:

```bash
pip install -r requirements.txt
python cascade_langgraph_mcp_server.py
```

Once running, any MCP-compatible client can call CASCADE tools. For example, a request to `comprehensive_perturbation_analysis` with `gene="TP53"` and `cell_type="cd8_t_cells"` returns a structured JSON report containing predicted downstream effects, protein interaction partners, experimental knockdown corroboration from LINCS, super-enhancer status, and similar genes by embedding.

The workflow is deterministic: the same gene, cell type, and depth parameters always produce identical results, as all analysis steps use fixed pre-computed networks and embeddings with no stochastic components.

# Software Availability

CASCADE is available at [https://github.com/jab57/CASCADE](https://github.com/jab57/CASCADE) under the MIT license. The repository includes 141 automated tests achieving 76% code coverage, continuous integration via GitHub Actions, and documentation for installation, usage, and contributing.

# Acknowledgements

CASCADE uses pre-trained gene embeddings from the GREmLN model developed by the Chan Zuckerberg Initiative AI team. We acknowledge the STRING Consortium, the LINCS Program, and dbSUPER for providing the external datasets integrated into CASCADE. CASCADE uses PyTorch [@pytorch] for model inference and LangGraph for workflow orchestration. Development was assisted by AI coding tools (Claude Code).

# References
