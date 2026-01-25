# Evaluation: Enhanced Pathway Integration Proposal

## Summary

**The enhancement is partially worth implementing, but with modifications.** The document proposes 4 tools, but RegNetAgents already has Reactome pathway integration, making 2 of them redundant if added to GREmLN.

---

## Current State Comparison

| Capability | GREmLN | RegNetAgents |
|------------|--------|--------------|
| Gene perturbation simulation | Yes (14 tools) | No |
| GREmLN embeddings (19K genes) | Yes | No |
| Network vulnerability analysis | Yes | No |
| Reactome pathway enrichment | **No** | **Yes** |
| Protein-protein interactions | No | No |
| LLM domain agents | No | Yes (4 agents) |
| Multi-gene orchestration | No | Yes |

---

## Evaluation of Each Proposed Tool

### 1. `get_protein_interactions` (STRING API)
**Verdict: WORTH ADDING - genuinely new capability**

- Neither codebase has PPI data
- Fills the documented gap: "APC → β-catenin destruction"
- Complements gene regulatory networks with protein-level mechanisms

### 2. `get_gene_pathways` (Reactome API)
**Verdict: REDUNDANT - RegNetAgents already has this**

- RegNetAgents' `pathway_focused_analysis` already queries Reactome
- Includes statistical p-values and FDR correction
- Adding to GREmLN would duplicate functionality

### 3. `get_pathway_details` (Reactome API)
**Verdict: POTENTIALLY REDUNDANT**

- Could be useful as standalone deep-dive tool
- But RegNetAgents pathway analysis likely covers this use case

### 4. `analyze_gene_context` (Combined analysis)
**Verdict: REDESIGN NEEDED**

- The concept is valuable (unified view)
- But should orchestrate across both servers, not duplicate data fetching

---

## Recommendation

### Option A: Add STRING PPI to GREmLN only (Recommended)

**Add to GREmLN:**
- `get_protein_interactions` - STRING API for PPI data

**Skip for GREmLN:**
- Pathway tools (already in RegNetAgents)

**Rationale:**
- Keeps GREmLN focused on its core: perturbation + mechanisms
- Avoids duplication with RegNetAgents
- PPI data directly explains "what happens at protein level" after perturbation
- Users can combine both servers: GREmLN for perturbation → RegNetAgents for pathway context

### Option B: Add PPI to RegNetAgents instead

**Alternative if you prefer single-server analysis:**
- Add STRING PPI to RegNetAgents alongside existing Reactome
- Create unified `analyze_gene_context` there that combines PPI + pathways + regulatory networks

**Trade-off:** RegNetAgents becomes heavier; GREmLN stays limited

### Option C: Full implementation in GREmLN (as doc proposes)

**Not recommended because:**
- Duplicates RegNetAgents' Reactome integration
- Creates maintenance burden for same functionality in two places
- Users would have pathway data in both servers (confusing)

---

## Architecture Fit Analysis

```
GREmLN (Low-level)          RegNetAgents (High-level)
─────────────────           ─────────────────────────
Perturbation simulation  →  Multi-agent analysis
Embedding similarity     →  LLM domain insights
Network vulnerability    →  Pathway enrichment (Reactome)
+ PPI data (STRING)      →  Cross-cell comparison
                         →  Report generation
```

**STRING PPI fits better in GREmLN** because:
1. It answers "what does this protein DO" (mechanism level)
2. Complements perturbation results with functional context
3. GREmLN users asking "what happens when APC is knocked down" get both regulatory cascade AND protein interactions

---

## Implementation Scope (if Option A chosen)

**Files to modify:**
- `gremln_mcp_server.py` - Add 1 new tool
- New: `tools/ppi/string_client.py` - STRING API wrapper

**Effort:** ~4-6 hours (just STRING integration, no pathway duplication)

**Testing:**
```python
# Verify STRING returns expected APC interactions
interactions = get_protein_interactions("APC")
assert "CTNNB1" in [i["partner"] for i in interactions]
```

---

## Decision Points

1. Do you use GREmLN standalone, or always with RegNetAgents?
2. Is the pathway duplication concern valid for your workflow?
3. Do you want PPI in GREmLN (mechanism context) or RegNetAgents (unified analysis)?

---

## STRING Implementation Reference

### STRING Client

```python
import requests
from typing import List
from dataclasses import dataclass

@dataclass
class ProteinInteraction:
    protein_a: str
    protein_b: str
    score: float
    interaction_types: List[str]

class StringClient:
    BASE_URL = "https://string-db.org/api"

    def __init__(self, species: int = 9606):
        self.species = species

    def get_interactions(
        self,
        gene: str,
        score_threshold: float = 0.7,
        max_results: int = 25
    ) -> List[ProteinInteraction]:
        """Get protein-protein interactions from STRING."""

        url = f"{self.BASE_URL}/json/network"
        params = {
            "identifiers": gene,
            "species": self.species,
            "required_score": int(score_threshold * 1000),
            "limit": max_results
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        interactions = []
        for item in response.json():
            interactions.append(ProteinInteraction(
                protein_a=item["preferredName_A"],
                protein_b=item["preferredName_B"],
                score=item["score"],
                interaction_types=self._parse_types(item)
            ))

        return interactions

    def _parse_types(self, item: dict) -> List[str]:
        types = []
        if item.get("nscore", 0) > 0: types.append("neighborhood")
        if item.get("fscore", 0) > 0: types.append("fusion")
        if item.get("pscore", 0) > 0: types.append("phylogenetic")
        if item.get("ascore", 0) > 0: types.append("coexpression")
        if item.get("escore", 0) > 0: types.append("experimental")
        if item.get("dscore", 0) > 0: types.append("database")
        if item.get("tscore", 0) > 0: types.append("textmining")
        return types
```

### MCP Tool Registration

```python
@server.tool()
async def get_protein_interactions(
    gene: str,
    score_threshold: float = 0.7,
    max_interactions: int = 25
) -> dict:
    """
    Get protein-protein interactions for a gene from STRING database.

    Shows what proteins physically interact with the query gene.
    Useful for understanding protein complexes and signaling.

    Args:
        gene: Gene symbol (e.g., APC, TP53, BRCA1)
        score_threshold: Minimum confidence score 0-1 (default: 0.7)
        max_interactions: Maximum interactions to return (default: 25)

    Returns:
        List of interacting proteins with confidence scores
    """
    client = StringClient()
    interactions = client.get_interactions(gene, score_threshold, max_interactions)

    return {
        "gene": gene,
        "total_interactions": len(interactions),
        "interactions": [
            {
                "partner": i.protein_b if i.protein_a == gene else i.protein_a,
                "score": i.score,
                "evidence": i.interaction_types
            }
            for i in interactions
        ]
    }
```

### API Rate Limit

| API | Rate Limit | Notes |
|-----|------------|-------|
| STRING | 1 request/second | No API key needed |

---

## References

- STRING Database API: https://string-db.org/help/api/
- RegNetAgents pathway implementation: `c:\Dev\RegNetAgents\regnetagents_langgraph_mcp_server.py`
