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

CASCADE ships with pre-computed ARACNe regulatory networks for 10 immune and epithelial cell types and is extensible to additional cell types by supplying new networks in the same format. It integrates protein-protein interactions from STRING [@szklarczyk2023], experimental knockdown signatures from LINCS L1000 [@subramanian2017], super-enhancer annotations from dbSUPER [@khan2016], curated transcription factor regulons from DoRothEA [@garcia-alonso2019] via decoupler [@badia2022], CRISPR gene effect scores from DepMap [@tsherniak2017] as an empirical phenotypic validation layer, and primary tumor mRNA expression and somatic alteration data from the TCGA PanCancer Atlas 2018 [@cerami2012] via cBioPortal [@gao2013] to bridge the gap between cell-line-derived findings and patient tissue evidence. When pre-trained gene embeddings from the GREmLN model [@gremln] are available, CASCADE combines embedding-based similarity with network propagation to capture functional relationships beyond static topology; when the checkpoint is unavailable, CASCADE falls back to network-only propagation with no loss of core functionality.

# Statement of Need

Simulating the downstream effects of gene perturbation---whether through CRISPR knockout, RNAi, or pharmacological inhibition---requires combining regulatory network analysis, protein interaction data, and experimental perturbation signatures from separate tools and databases. Researchers must manually query each source, reconcile gene identifiers across naming conventions (HUGO symbols, Ensembl IDs), and synthesize cross-database results into a coherent interpretation. No existing tool unifies perturbation simulation, multi-database evidence gathering, and structured reporting behind a single programmatic interface.

CASCADE fills this gap by exposing the full analysis pipeline as an MCP server [@mcp]---an open protocol that allows AI assistants to call external tools. This means a researcher can ask an AI assistant *"What happens if I knock down MYC in CD8+ T cells?"* and receive a structured multi-source report without writing code, reconciling identifiers, or querying databases manually. The agentic workflow autonomously resolves gene identifiers, classifies the gene's regulatory role, selects and executes appropriate analyses in parallel, and returns structured, reproducible results. CASCADE requires no single-cell RNA-seq input and no model training, making it accessible to researchers who lack computational infrastructure for deep learning-based perturbation tools.

# State of the Field

Several tools address individual components of gene perturbation analysis. SCENIC [@aibar2017] reconstructs gene regulatory networks from single-cell data but does not simulate perturbation effects or integrate external databases. CellOracle [@kamimoto2023] performs *in silico* perturbation by combining GRN inference with dynamical systems modeling, but requires single-cell RNA-seq input and custom GRN construction for each experiment. GEARS [@roohani2024] uses graph neural networks to predict transcriptional outcomes of multigene perturbations, but requires Perturb-seq training data and is limited to cell types with available perturbation screens. Interaction databases such as STRING [@szklarczyk2023] and regulon resources such as DoRothEA [@garcia-alonso2019] catalog associations but leave synthesis and interpretation to the user.

CASCADE differs from these tools in three respects: it requires no single-cell input or model training (using pre-computed networks and pre-trained embeddings), it integrates multiple external databases automatically (STRING, LINCS, dbSUPER, DoRothEA), and it exposes the full pipeline as structured MCP tools that AI assistants can call directly.

# Architecture

CASCADE follows a layered design (\autoref{fig:architecture}). The MCP server exposes 30 tools organized into nine categories: workflow orchestration, perturbation simulation, gene similarity, network vulnerability, experimental data, druggability assessment, TF regulon validation, CRISPR essentiality, and primary tumor data. Alongside tools, the server provides five **MCP Resources**---browsable read-only endpoints (`cascade://cell-types`, `cascade://lincs/summary`, `cascade://model/status`, and URI templates for per-cell-type network summaries and per-gene metadata)---that allow clients to discover available data and model status without invoking a full analysis.

![CASCADE architecture. An MCP client sends a request to the CASCADE server, which routes it through a LangGraph workflow. The workflow classifies the gene, selects analyses based on gene role and depth, and executes independent batches in parallel. Analysis tools operate on pre-computed regulatory networks and gene embeddings, while external modules query STRING, LINCS, dbSUPER, DoRothEA, DepMap, and cBioPortal.\label{fig:architecture}](figure_architecture.png)

The agentic workflow, built on LangGraph [@langgraph], autonomously coordinates the analysis pipeline:

1. **Resolve** the gene identifier (symbol or Ensembl ID) via the Ensembl REST API with local caching.
2. **Classify** the gene's regulatory role (master regulator, transcription factor, effector, or isolated) from the cell-type-specific network.
3. **Route** to appropriate analyses based on gene role and requested depth---for example, a transcription factor triggers full downstream propagation, while an effector gene prioritizes protein interaction and upstream regulator analysis; isolated genes (not present in the network) skip regulators analysis entirely and receive an explanatory note directing users to protein interaction and embedding evidence.
4. **Execute** independent analyses in parallel: perturbation simulation, STRING protein interactions, LINCS experimental knockdown signatures, super-enhancer status, DoRothEA TF regulon validation [@garcia-alonso2019], DepMap CRISPR essentiality scores [@tsherniak2017], TCGA primary tumor expression and alteration data via cBioPortal [@cerami2012; @gao2013], embedding-based gene similarity, and cross-cell-type comparison.
5. **Report** structured JSON results aggregating all analyses into a single response.
6. **Synthesize** a narrative biological interpretation by passing the structured results to a configurable LLM (Ollama local or cloud by default), combining mechanism summaries, therapeutic implications, and suggested follow-up experiments into a coherent report. Because the synthesis node is decoupled from the MCP client, it can be configured with domain-specific models (e.g., biomedical LLMs fine-tuned on gene regulation literature) that may outperform general-purpose assistants at interpreting perturbation biology. When no LLM is configured, the structured results from step 5 remain fully usable by the calling AI assistant.

Network perturbation effects are computed via breadth-first propagation through directed regulatory edges weighted by mutual information. When pre-trained gene embeddings are available, network-derived scores are combined with embedding-based similarity to capture functional relationships beyond static topology.

CASCADE uses GREmLN embeddings and pre-computed regulatory networks---rather than LINCS experimental signatures---as the primary perturbation engine. This design reflects three practical constraints: GREmLN embeddings provide complete coverage of ~19,247 genes across all supported cell types with deterministic results, whereas LINCS L1000 signatures are sparse (covering a subset of genes and cell lines), biased toward transformed cell lines rather than primary immune populations, and subject to batch variability across experimental conditions. LINCS data remain integrated as an independent corroboration layer: when experimental knockdown signatures are available for a queried gene, the workflow reports directional agreement or disagreement with network predictions, giving users an empirical check without limiting the analysis to genes with LINCS coverage.

The pre-computed regulatory networks represent population-averaged regulatory relationships and do not capture cell-state-specific dynamics; analysis of state-dependent regulation (e.g., resting versus activated T cells) requires cell-state-specific networks generated from appropriately resolved single-cell data and supplied in the same format as the bundled networks.

# Functionality

CASCADE can be installed and used as follows:

```bash
pip install -r requirements.txt
python verify_installation.py   # smoke-test: networks, model, perturbation
python cascade_langgraph_mcp_server.py
```

Once running, any MCP-compatible client can call CASCADE tools. Gene-role-aware routing shapes the analysis automatically based on each gene's position in the regulatory network.

For a transcription factor, a request to `comprehensive_perturbation_analysis` with `gene="MYC"`, `cell_type="epithelial_cell"`, and `perturbation_type="knockdown"` identifies MYC as a master regulator with 427 direct network targets and 4,358 downstream affected genes, validates its TF classification against DoRothEA curated regulons (confidence A, 384 high-confidence targets including TERT and HDAC2), reports super-enhancer association in 32 cell types (BRD4-sensitive), returns 25 STRING protein interaction partners, reports DepMap CRISPR essentiality: MYC is essential in 95% of cancer cell lines (Chronos mean −1.98), and retrieves TCGA primary tumor data via cBioPortal: MYC is overexpressed (pan-cancer mean z=1.8) with highest expression in liver and gastric tumors. The evidence synthesis flags convergent cell-line and primary tumor evidence — DepMap pan-cancer essentiality combined with TCGA overexpression — as the strongest multi-layer therapeutic target signal. The evidence synthesis identifies 24 genes supported by multiple independent sources, with STAT3 appearing across the network, STRING, and DoRothEA layers simultaneously.

For an effector gene, the workflow routes differently. Querying `gene="APC"` in epithelial cells classifies APC as an effector regulated by 23 upstream transcription factors but with no transcriptional targets of its own---network propagation returns zero downstream effects, and the report explicitly directs users to protein-level evidence. STRING PPI returns the full Wnt/β-catenin destruction complex (CTNNB1, AXIN1, AXIN2, GSK3B, CSNK1A1, BTRC) at 0.999 combined confidence, and the upstream regulator analysis surfaces a CTNNB1→APC negative feedback loop: β-catenin transcriptionally suppresses its own negative regulator, creating a self-reinforcing Wnt activation cycle upon APC loss. CASCADE also flags that APC is present only in the epithelial cell network---absent from all immune cell types---consistent with its role as the gatekeeper tumor suppressor of colorectal cancer.

The core perturbation simulation is deterministic: the same gene, cell type, and depth parameters always produce identical propagation and embedding results from fixed pre-computed networks with no stochastic components. External data sources (STRING protein interactions, LINCS experimental signatures) may vary across queries as upstream databases are updated, but the simulation itself is fully reproducible. Because CASCADE communicates via MCP, the structured results are directly available to the AI assistant that initiated the request, enabling conversational follow-up---a researcher can refine the analysis, compare cell types, or explore related genes without leaving the conversation.

# Research Impact Statement

CASCADE enables non-computational researchers to perform multi-source gene perturbation analysis through conversational AI interfaces, eliminating the need to write analysis scripts, install bioinformatics pipelines, or manually reconcile identifiers across databases. Its deterministic simulation engine produces fully reproducible results from pre-computed networks, and its structured JSON output provides a machine-readable audit trail for each analysis. The MCP-based architecture allows any compatible AI assistant to invoke CASCADE tools, making the analysis pipeline available to the growing ecosystem of AI-augmented research workflows. The extensible design---supporting new cell types through additional network files in a standard format---positions CASCADE for adoption across diverse research contexts beyond its initial immuno-oncology focus.

# Software Availability

CASCADE is available at [https://github.com/jab57/CASCADE](https://github.com/jab57/CASCADE) under the MIT license. The repository includes automated tests covering network propagation, embedding similarity, LINCS queries, gene identifier resolution, DoRothEA regulon validation, MCP tool registration, and workflow orchestration logic, with continuous integration via GitHub Actions and documentation for installation, usage, and contributing.

# AI Usage Disclosure

Development of CASCADE was assisted by Claude Code (Anthropic), an AI coding tool. The AI assistant was used for code generation, refactoring, test writing, and documentation drafting. All AI-generated code and text were reviewed, tested, and validated by the human author. This paper was drafted collaboratively with AI assistance and reviewed for accuracy by the author.

# Acknowledgements

CASCADE uses pre-trained gene embeddings from the GREmLN model developed by the Chan Zuckerberg Initiative AI team. We acknowledge the STRING Consortium, the LINCS Program, dbSUPER, the DoRothEA/decoupler projects, and the Broad Institute DepMap team for providing the external datasets and tools integrated into CASCADE. CASCADE uses PyTorch [@pytorch] for model inference and LangGraph for workflow orchestration.

# References
