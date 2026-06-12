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
  - name: Jose A. Bird
    orcid: 0009-0006-2744-0606
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 29 April 2026
archive_doi: 10.5281/zenodo.20631772
bibliography: paper.bib
---

# Summary

CASCADE (Computational Analysis of Simulated Cell And Drug Effects) is a Python server that exposes *in silico* gene perturbation analysis as structured tools via the Model Context Protocol (MCP) [@mcp]. Given a gene and cell type, CASCADE simulates knockdown or overexpression effects by propagating signals through pre-computed regulatory networks, queries external databases for corroborating evidence, and returns a structured report. A LangGraph-based workflow [@langgraph] automates the full analysis pipeline---gene resolution, role classification, parallel data retrieval, and report generation---so that a single tool call replaces what would otherwise require manual orchestration of multiple databases and analysis scripts.

CASCADE ships with two classes of pre-computed ARACNe [@margolin2006] regulatory networks: population-averaged networks for 10 immune and epithelial cell types [@zhang2026gremln] and tumor-state networks for 8 TCGA cancer types [@aracne_networks; @lim2018], enabling perturbation analysis in both reference and tumor-specific regulatory contexts. External evidence is drawn from STRING [@szklarczyk2023], LINCS L1000 [@subramanian2017], dbSUPER [@khan2016], DoRothEA [@garcia-alonso2019] via decoupler [@badia2022], DepMap [@tsherniak2017], and cBioPortal [@cerami2012; @gao2013]. When pre-trained gene embeddings from the GREmLN model are available, CASCADE combines embedding-based similarity with network propagation to capture functional relationships beyond static topology.

# Statement of Need

Simulating the downstream effects of gene perturbation requires combining regulatory network analysis, protein interaction data, and experimental perturbation signatures from separate tools and databases. Researchers must manually query each source, reconcile gene identifiers across naming conventions, and synthesize cross-database results into a coherent interpretation. To our knowledge, no existing tool unifies perturbation simulation, multi-database evidence gathering, and structured reporting behind a single programmatic interface.

CASCADE fills this gap by exposing the full analysis pipeline as an MCP server---an open protocol that allows AI assistants to call external tools. A researcher can ask an AI assistant *"What happens if I knock down MYC in CD8+ T cells?"* and receive a structured multi-source report without writing code, reconciling identifiers, or querying databases manually. CASCADE requires no single-cell RNA-seq input and no model training, making it accessible to researchers without computational infrastructure for deep learning-based perturbation tools. CASCADE is designed to generate hypotheses by surfacing convergent evidence across independent data sources; experimental validation of predicted effects remains the researcher's responsibility.

# State of the Field

Several tools address individual components of gene perturbation analysis. VIPER [@alvarez2016] infers protein activity from ARACNe regulatory networks and is the closest methodological relative to CASCADE, but operates on user-supplied expression matrices rather than pre-computed networks, does not integrate external databases, and exposes no programmatic interface for AI-assisted workflows. SCENIC [@aibar2017] reconstructs gene regulatory networks from single-cell data but does not simulate perturbation effects or integrate external databases. CellOracle [@kamimoto2023] performs *in silico* perturbation using GRN inference with dynamical systems modeling, but requires single-cell RNA-seq input and custom GRN construction. GEARS [@roohani2024] predicts transcriptional outcomes of multigene perturbations but requires Perturb-seq training data and is limited to cell types with available perturbation screens. Interaction databases such as STRING [@szklarczyk2023] and regulon resources such as DoRothEA [@garcia-alonso2019] catalog associations but leave synthesis to the user.

CASCADE differs from these tools in three respects: it requires no single-cell input or model training, it integrates multiple external databases automatically, and it exposes the full pipeline as MCP tools that AI assistants can call directly.

# Architecture

CASCADE follows a layered design (\autoref{fig:architecture}). The MCP server exposes 28 tools organized into nine categories, five MCP Resources for browsable network and model metadata, and five prompt templates for common workflows. The agentic workflow, built on LangGraph [@langgraph], classifies each gene's regulatory role from the network, routes to appropriate analyses based on that role and requested depth, and executes independent analyses---perturbation simulation, STRING interactions, LINCS signatures, DoRothEA regulon validation, DepMap essentiality, and cBioPortal tumor data---concurrently in parallel batches before assembling a structured JSON report. An optional LLM synthesis node adds narrative interpretation without altering the structured output.

![CASCADE architecture. An MCP client sends a request to the CASCADE server, which routes it through a LangGraph workflow. The workflow classifies the gene, selects analyses based on gene role and depth, and executes independent batches in parallel. Analysis tools operate on pre-computed regulatory networks and gene embeddings, while external modules query STRING, LINCS, dbSUPER, DoRothEA, DepMap, and cBioPortal.\label{fig:architecture}](figure_architecture.png)

# Functionality

CASCADE can be installed and used as follows:

```bash
pip install -r requirements.txt
python verify_installation.py   # smoke-test: networks, model, perturbation
python cascade_langgraph_mcp_server.py
```

Once running, any MCP-compatible client can call CASCADE tools. Gene-role-aware routing shapes the analysis automatically based on each gene's position in the regulatory network.

For a transcription factor, querying `gene="MYC"` in `epithelial_cell` with `perturbation_type="knockdown"` classifies MYC as a master regulator, propagates effects through its network targets, validates its TF classification against DoRothEA regulons, retrieves STRING interactions and DepMap essentiality scores, and synthesizes convergent evidence into a structured report.

For an effector gene, querying `gene="APC"` in epithelial cells classifies APC as an effector with no transcriptional targets---network propagation returns zero downstream effects and the report routes the user to protein-level evidence. STRING PPI returns the Wnt/β-catenin destruction complex, and upstream regulator analysis surfaces a CTNNB1→APC negative feedback loop consistent with APC's role as a gatekeeper tumor suppressor.

For an immune cell context, querying `gene="JUNB"` in `cd8_t_cells` predicts 56 downstream effects from knockdown, including downregulation of ZFP36, DUSP1, DUSP2, FOS, and JUN, consistent with JUNB's established role in cytotoxic T cell activation.

For a tumor-state network, supplying `network_source="tcga"` and `tcga_network="brca"` and querying `gene="ESR1"` classifies ESR1 as a master regulator whose top predicted knockdown targets---GATA3, FOXA1, GREB1, XBP1, and PGR---constitute the canonical ER-positive luminal identity program, consistent with ESR1's established role in ER+ breast cancer.

The core perturbation simulation is deterministic: the same gene, cell type, and depth parameters always produce identical results from fixed pre-computed networks with no stochastic components.

# Software Availability

CASCADE is available at [https://github.com/jab57/CASCADE](https://github.com/jab57/CASCADE) under the MIT license. The repository includes 334 automated tests covering network propagation, embedding similarity, LINCS queries, gene identifier resolution, DoRothEA regulon validation, MCP tool registration, workflow orchestration, and cell-type input validation, with continuous integration via GitHub Actions.

# AI Usage Disclosure

Development of CASCADE was assisted by Claude Code (Anthropic), an AI coding tool. The AI assistant was used for code generation, refactoring, test writing, and documentation drafting. All AI-generated code and text were reviewed, tested, and validated by the human author. This paper was drafted collaboratively with AI assistance and reviewed for accuracy by the author.

# Acknowledgements

CASCADE uses pre-trained gene embeddings from the GREmLN model developed by the Chan Zuckerberg Initiative AI team. We acknowledge the STRING Consortium, the LINCS Program, dbSUPER, the DoRothEA/decoupler projects, and the Broad Institute DepMap team for providing the external datasets and tools integrated into CASCADE. CASCADE uses PyTorch [@pytorch] for model inference and LangGraph for workflow orchestration.

# References
