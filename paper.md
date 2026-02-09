---
title: 'CASCADE: Agentic In Silico Gene Perturbation Analysis via the Model Context Protocol'
tags:
  - Python
  - gene perturbation
  - regulatory networks
  - agentic AI
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

Simulating the downstream effects of gene perturbation---whether through CRISPR knockout, RNAi, or pharmacological inhibition---requires combining regulatory network analysis, protein interaction data, and experimental perturbation signatures from separate tools and databases. Researchers must manually query each source, reconcile gene identifiers across naming conventions (HUGO symbols, Ensembl IDs), and synthesize cross-database results into a coherent interpretation.

Existing tools address individual aspects of this workflow: SCENIC reconstructs regulatory networks but does not simulate perturbation effects [@aibar2017]; GEARS forecasts expression changes but requires Perturb-seq training data and single-cell RNA-seq input [@roohani2024]; and interaction databases catalog physical associations but leave interpretation to the user [@szklarczyk2023]. No existing tool unifies perturbation simulation, multi-database evidence gathering, and structured reporting behind a single programmatic interface.

CASCADE fills this gap by exposing the full analysis pipeline as an MCP server [@mcp]---an open protocol that allows AI assistants to call external tools. This means a researcher can ask an AI assistant *"What happens if I knock down MYC in CD8+ T cells?"* and receive a structured multi-source report without writing code, reconciling identifiers, or querying databases manually. The agentic workflow autonomously resolves gene identifiers, classifies the gene's regulatory role, selects and executes appropriate analyses in parallel, and returns deterministic, reproducible results. CASCADE requires no single-cell RNA-seq input and no model training, making it accessible to researchers who lack computational infrastructure for tools like GEARS or CellOracle [@kamimoto2023].

# Architecture

CASCADE follows a layered design (\autoref{fig:architecture}). The MCP server exposes 22 tools organized into six categories: workflow orchestration, perturbation simulation, gene similarity, network vulnerability, experimental data, and druggability assessment.

![CASCADE architecture. An MCP client sends a request to the CASCADE server, which routes it through a LangGraph workflow. The workflow classifies the gene, selects analyses based on gene role and depth, and executes independent batches in parallel. Analysis tools operate on pre-computed regulatory networks and gene embeddings, while external modules query STRING, LINCS, and dbSUPER.\label{fig:architecture}](figure_architecture.png)

The agentic workflow, built on LangGraph [@langgraph], autonomously coordinates the analysis pipeline:

1. **Resolve** the gene identifier (symbol or Ensembl ID) via the Ensembl REST API with local caching.
2. **Classify** the gene's regulatory role (master regulator, transcription factor, effector, or isolated) from the cell-type-specific network.
3. **Route** to appropriate analyses based on gene role and requested depth---for example, a transcription factor triggers full downstream propagation, while an effector gene prioritizes protein interaction and upstream regulator analysis.
4. **Execute** independent analyses in parallel: perturbation simulation, STRING protein interactions, LINCS experimental knockdown signatures, super-enhancer status, embedding-based gene similarity, and cross-cell-type comparison.
5. **Report** structured JSON results aggregating all analyses into a single response.
6. **Synthesize** a narrative biological interpretation by passing the structured results to a configurable LLM (local or cloud Ollama), combining mechanism summaries, therapeutic implications, and suggested follow-up experiments into a coherent report. When no LLM is available, the structured results from step 5 remain fully usable by the calling AI assistant.

Network perturbation effects are computed via breadth-first propagation through directed regulatory edges weighted by mutual information. When pre-trained gene embeddings are available, network-derived scores are combined with embedding-based similarity to capture functional relationships beyond static topology.

# Functionality

CASCADE can be installed and used as follows:

```bash
pip install -r requirements.txt
python cascade_langgraph_mcp_server.py
```

Once running, any MCP-compatible client can call CASCADE tools. For example, a request to `comprehensive_perturbation_analysis` with `gene="TP53"` and `cell_type="cd8_t_cells"` returns a structured JSON report containing predicted downstream effects, protein interaction partners, experimental knockdown corroboration from LINCS, super-enhancer status, and similar genes by embedding.

The core workflow is deterministic: the same gene, cell type, and depth parameters always produce identical analytical results, as all analysis steps use fixed pre-computed networks and embeddings with no stochastic components. Because CASCADE communicates via MCP, the structured results are directly available to the AI assistant that initiated the request, enabling conversational follow-up---a researcher can refine the analysis, compare cell types, or explore related genes without leaving the conversation.

# Software Availability

CASCADE is available at [https://github.com/jab57/CASCADE](https://github.com/jab57/CASCADE) under the MIT license. The repository includes automated tests covering network propagation, embedding similarity, LINCS queries, gene identifier resolution, and MCP tool registration, with continuous integration via GitHub Actions and documentation for installation, usage, and contributing.

# Acknowledgements

CASCADE uses pre-trained gene embeddings from the GREmLN model developed by the Chan Zuckerberg Initiative AI team. We acknowledge the STRING Consortium, the LINCS Program, and dbSUPER for providing the external datasets integrated into CASCADE. CASCADE uses PyTorch [@pytorch] for model inference and LangGraph for workflow orchestration. Development was assisted by AI coding tools (Claude Code).

# References
