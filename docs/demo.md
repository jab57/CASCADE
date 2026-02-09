# CASCADE Demo: Immunotherapy Target Discovery

A scripted walkthrough demonstrating CASCADE's agentic analysis pipeline through Claude Desktop. This demo tells a complete biological story: identifying and evaluating therapeutic targets in CD8+ T cells for cancer immunotherapy.

**Time:** ~10 minutes
**Prerequisites:** CASCADE MCP server running with Ollama (`USE_LLM_INSIGHTS=true`)

## Setup

### 1. Start Ollama with a model

```bash
ollama pull llama3.1:8b
ollama serve
```

### 2. Start CASCADE with LLM insights

```bash
cd C:\Dev\CASCADE
.\env\Scripts\activate
set USE_LLM_INSIGHTS=true
python cascade_langgraph_mcp_server.py
```

### 3. Confirm Claude Desktop configuration

In `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "CASCADE": {
      "command": "C:/Dev/CASCADE/env/Scripts/python.exe",
      "args": ["C:/Dev/CASCADE/cascade_langgraph_mcp_server.py"],
      "env": {
        "USE_LLM_INSIGHTS": "true"
      }
    }
  }
}
```

Restart Claude Desktop after editing.

---

## The Demo

### Scene: A researcher is exploring immunotherapy targets

You are a computational biologist at a biotech company evaluating targets for a next-generation checkpoint inhibitor program. You want to understand what happens at the network level when key immune genes are perturbed in CD8+ T cells.

---

### Step 1 — The Opening Question

**Type in Claude Desktop:**

> What happens if I knock down TP53 in epithelial cells? Give me the full analysis with LLM insights.

**What CASCADE does behind the scenes:**
- Calls `comprehensive_perturbation_analysis` with `gene="TP53"`, `cell_type="epithelial_cell"`, `include_llm_insights=true`
- The LangGraph workflow: resolves TP53 to ENSG00000141510, classifies it as a transcription factor, routes to full analysis
- Parallel batches fire: perturbation propagation, STRING PPI, LINCS experimental data, super-enhancer check, embedding similarity, cross-cell comparison
- Ollama generates a narrative biological interpretation

**What to highlight:**
- The single natural language question triggered 6+ parallel analyses across multiple databases
- The LLM narrative explains the biology in plain English — mechanism, therapeutic implications, suggested follow-ups
- The structured data is all there underneath for anyone who wants the numbers

---

### Step 2 — Cross-Cell Comparison

**Type:**

> How does TP53 knockdown compare in CD8 T cells vs epithelial cells?

**What CASCADE does:**
- Calls `comprehensive_perturbation_analysis` again for CD8 T cells
- Claude compares the two structured reports side by side

**What to highlight:**
- Same gene, completely different biology in different cell types
- The regulatory network is cell-type-specific — TP53 may be a master regulator in one context and an effector in another
- This is the kind of comparison that would take a researcher hours to do manually across multiple databases

---

### Step 3 — Drug Target Discovery

**Type:**

> What are the most critical drug targets in the CD8 T cell network? Show me the top master regulators.

**What CASCADE does:**
- Calls `analyze_network_vulnerability` for CD8 T cells
- Returns ranked list of hub genes with vulnerability scores

**What to highlight:**
- Vulnerability scoring combines hub connectivity, cascade reach, edge weight, and isolation factor
- Master regulators (high targets, few upstream regulators) are the highest-value drug targets
- This prioritization would otherwise require custom network analysis scripts

---

### Step 4 — Head-to-Head Comparison

**Type:**

> Compare STAT3, MYC, and IRF4 as potential drug targets in CD8 T cells.

**What CASCADE does:**
- Calls `compare_gene_vulnerability` with all three genes
- Returns comparative vulnerability scores, network positions, and downstream impact

**What to highlight:**
- Direct comparison of candidate targets with quantified metrics
- Each gene's role in the network is classified automatically
- This is the kind of analysis a medicinal chemistry team needs to prioritize their program

---

### Step 5 — Deep Dive with Experimental Corroboration

**Type:**

> What experimental evidence does LINCS have for STAT3 knockdown? Do the experimental results match CASCADE's network predictions?

**What CASCADE does:**
- Calls `get_knockdown_effects` for STAT3 (LINCS experimental data)
- Claude compares LINCS experimental targets with CASCADE's predicted targets from the network

**What to highlight:**
- CASCADE integrates computational predictions with real experimental knockdown data
- Overlap between predicted and experimental targets validates the network
- LINCS also captures post-transcriptional effects that the network can't predict — the tools are complementary

---

### Step 6 — Protein-Level Mechanism

**Type:**

> What proteins does STAT3 interact with? How do those interactions explain the knockdown effects we just saw?

**What CASCADE does:**
- Calls `get_protein_interactions` for STAT3 (STRING database)
- Claude synthesizes PPI data with the perturbation and LINCS results

**What to highlight:**
- The AI assistant connects the dots across three data sources (network, LINCS, STRING) in one conversation
- This is the agentic loop: each follow-up question builds on previous context
- A researcher gets a coherent biological narrative, not disconnected database outputs

---

### Step 7 — The Closing

**Type:**

> Based on everything we've analyzed, summarize the case for STAT3 as a therapeutic target in CD8 T cells. Include the network evidence, experimental validation, and protein interactions.

**What to highlight:**
- Claude synthesizes the entire conversation into a target assessment
- This is the deliverable: a structured, evidence-based target evaluation that would take days to compile manually
- The entire analysis took minutes, is reproducible, and is grounded in data

---

## Key Messages for Partners

When walking someone through this demo, emphasize:

1. **One question, six databases.** A single natural language query triggers parallel analysis across regulatory networks, gene embeddings, STRING, LINCS, dbSUPER, and Ensembl. No scripting, no identifier reconciliation.

2. **Cell-type-specific biology.** The same gene behaves differently in different immune cell types. CASCADE has pre-computed networks for 10 cell types relevant to immuno-oncology.

3. **AI interprets, not just retrieves.** The optional LLM synthesis turns structured JSON into biological narratives with mechanism summaries, therapeutic implications, and follow-up suggestions.

4. **Conversational follow-up.** Because CASCADE is an MCP server, the AI assistant maintains context across the conversation. Each question builds on the last.

5. **Reproducible and deterministic.** The core analysis always produces identical results for the same inputs. No stochastic variation between runs.

## Adapting the Demo

**For oncology audiences:** Use TP53, MYC, or BRCA1 in epithelial cells. Focus on tumor suppressor loss and oncogenic activation.

**For immunology audiences:** Use PDCD1 (PD-1), CD274 (PD-L1), or CTLA4 in CD8 T cells or NK cells. Focus on checkpoint biology and T cell exhaustion.

**For drug discovery audiences:** Lead with the vulnerability analysis (Step 3-4). They care about target prioritization and competitive positioning.

**For AI/tech audiences:** Emphasize the MCP architecture, parallel execution, and the agentic loop. Show that the same server could be called by any MCP-compatible client.
