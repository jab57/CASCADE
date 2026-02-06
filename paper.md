---
title: 'CASCADE: An MCP Server for In Silico Gene Perturbation Analysis in Immuno-Oncology'
tags:
  - Python
  - gene perturbation
  - regulatory networks
  - immuno-oncology
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

CASCADE (Computational Analysis of Simulated Cell And Drug Effects) is a Python-based Model Context Protocol (MCP) server that enables *in silico* gene perturbation analysis for cancer and immuno-oncology research. It provides 22 tools for simulating gene knockdowns and overexpression across 10 immune and epithelial cell types, using pre-computed gene regulatory networks derived from single-cell RNA sequencing data and 256-dimensional gene embeddings learned from 11 million cells via the GREmLN model [@gremln]. CASCADE integrates external data sources including protein-protein interactions from STRING [@szklarczyk2023], experimental CRISPR knockdown signatures from LINCS L1000 [@subramanian2017], and super-enhancer annotations from dbSUPER [@khan2016]. A LangGraph-based workflow orchestration layer [@langgraph] automatically classifies genes by network role, routes analyses accordingly, and executes independent analysis steps in parallel, producing comprehensive reports with therapeutic targeting suggestions.

# Statement of Need

Computational biologists studying gene regulatory networks in cancer and immunology face a fragmented tooling landscape. Simulating the downstream consequences of perturbing a gene---whether through CRISPR knockout, RNAi knockdown, or pharmacological inhibition---requires combining network topology analysis, gene embedding similarity, protein interaction data, and experimental perturbation signatures from separate tools and databases. Researchers must manually orchestrate these analyses, interpret cross-database results, and synthesize findings into actionable biological insights.

Existing tools address individual aspects of this problem: network inference packages reconstruct regulatory networks [@aibar2017], perturbation prediction models forecast expression changes [@roohani2024], and protein interaction databases catalog physical associations [@szklarczyk2023]. However, no tool combines these capabilities into a unified, conversational interface that an LLM-powered research assistant can invoke programmatically.

CASCADE addresses this gap by exposing perturbation analysis as MCP tools [@mcp] that any MCP-compatible client can call. The LangGraph orchestration layer eliminates manual workflow management: a single call to `comprehensive_perturbation_analysis` automatically resolves gene identifiers, classifies the gene's network role (master regulator, transcription factor, or effector), selects appropriate analyses, executes them in parallel, and synthesizes a report with therapeutic recommendations. This enables researchers to ask natural-language questions---such as "What happens if we knock down TP53 in CD8 T cells?"---and receive structured, multi-source analyses within seconds.

# Architecture and Design

CASCADE follows a layered architecture (\autoref{fig:architecture}). The MCP server layer exposes 22 tools organized into six categories: workflow orchestration, perturbation analysis, gene similarity, network vulnerability, experimental data (LINCS), and super-enhancer/druggability assessment.

![CASCADE architecture. An MCP client sends requests to the CASCADE server, which routes them through a LangGraph StateGraph workflow. The workflow classifies the gene, selects analyses based on gene role and requested depth, and executes independent analysis batches in parallel. Core analysis tools operate on pre-computed regulatory networks and GREmLN embeddings, while external data modules query STRING, LINCS, and dbSUPER.\label{fig:architecture}](figure_architecture.png)

The LangGraph workflow operates as a state machine with conditional routing. Upon receiving a gene and cell type, the workflow:

1. **Resolves** the gene identifier (symbol or Ensembl ID) via Ensembl REST API with local caching.
2. **Classifies** the gene's role by counting its targets and regulators in the cell-type-specific network: master regulators (>50 targets), transcription factors (10--50), minor regulators (1--10), effectors (regulated but non-regulating), or isolated.
3. **Routes** to appropriate analysis batches based on gene role and requested depth (basic, focused, or comprehensive).
4. **Executes** up to three parallel batches: core analysis (perturbation, regulators, targets), external data (STRING PPI, LINCS knockdown effects, super-enhancers), and insight generation (embedding similarity, vulnerability scoring, cross-cell comparison).
5. **Synthesizes** results into a structured report with therapeutic targeting suggestions.

## Perturbation Scoring

Network effects are computed via breadth-first search (BFS) propagation through directed regulatory edges weighted by mutual information. For embedding-enhanced analysis, network and embedding signals are combined:

$$\text{effect}_\text{combined} = \alpha \cdot \text{effect}_\text{network} + (1 - \alpha) \cdot s_{ij} \cdot \text{effect}_\text{network}$$

where $\alpha = 0.7$ by default and $s_{ij}$ is the cosine similarity between gene embeddings. Genes with high embedding similarity ($s_{ij} \geq 0.3$) but no direct network connection are reported as potential indirect effects, enabling discovery of relationships absent from the static network.

## Network Vulnerability Scoring

For drug target prioritization, CASCADE computes a vulnerability score:

$$V_g = h_g + 0.3 \cdot c_g + 10 \cdot \bar{w}_g + \frac{5}{r_g + 1}$$

where $h_g$ is the hub score (direct target count), $c_g$ is the cascade reach (second-order targets), $\bar{w}_g$ is mean outgoing edge weight, and $r_g$ is the upstream regulator count. Genes with high vulnerability and few regulators represent high-value therapeutic targets because the network cannot easily compensate for their loss.

# Key Features

- **10 cell-type-specific regulatory networks** covering epithelial cells, CD4/CD8 T cells, B cells, NK cells, NKT cells, monocytes (CD14/CD16), dendritic cells, and erythrocytes.
- **Embedding-enhanced predictions** using 256-dimensional gene representations learned from 11 million single cells, capturing functional relationships beyond static network topology.
- **Multi-source integration** combining network propagation, STRING protein interactions, LINCS L1000 experimental knockdown data, and dbSUPER super-enhancer annotations for druggability assessment.
- **Intelligent routing** that automatically adapts analysis strategy based on gene classification---transcription factors receive knockdown/target analysis while effector genes receive PPI-focused analysis.
- **Optional LLM synthesis** via Ollama integration for generating narrative biological interpretations of analysis results.

# Acknowledgements

CASCADE builds upon the GREmLN model and gene embeddings developed by the Chan Zuckerberg Initiative AI team. We acknowledge the STRING Consortium, the LINCS Program, and dbSUPER for providing the external datasets integrated into CASCADE. CASCADE uses PyTorch [@pytorch] for model inference and LangGraph for workflow orchestration.

# References
