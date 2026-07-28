"""
Agent tool-call grounding benchmark (Section 3.6 / Results, this paper).

Question this answers: when an LLM-based agent translates a natural-language
request into a call to CASCADE's comprehensive_perturbation_analysis MCP
tool, does it correctly ground the request into the right parameters (gene,
perturbation_type, network_source, tcga_network/cell_type)? This is a
different claim from the rest of this paper: every other experiment calls
CascadeWorkflow.run() directly with hand-specified parameters. This script
tests the step upstream of that -- whether an agent gets those parameters
right in the first place.

35 natural-language queries across 7 categories (5 each), hand-labeled with
ground truth, sent to a local model via Ollama's tool-calling API using
CASCADE's real tool schema (verbatim from cascade_langgraph_mcp_server.py,
comprehensive_perturbation_analysis). Ground truth accounts for CASCADE's
documented parameter defaults: a field the model omits is scored against the
default the server would actually apply, not against a bare None.

Categories:
  1. baseline_tcga         - straightforward gene + TCGA cancer type
  2. baseline_immune       - straightforward gene + immune cell type
  3. gene_alias            - gene synonyms; a deliberate mix of ones CASCADE
                             now resolves (HER2, p53 -- see
                             tools/gene_id_mapper.py GENE_SYMBOL_ALIASES) and
                             ones it still does NOT (PD-L1, HER3), to make
                             this an honest test of what remains unresolved
  4. cancer_type_name      - informal/full cancer-type names -> enum abbreviation
  5. perturbation_phrasing - non-canonical wording for knockdown/overexpression
  6. implicit_params       - queries that omit parameters, relying on defaults
  7. multi_entity          - a distractor gene/cancer type mentioned but not
                             the one actually requested

Two models tested: llama3.1:8b (CASCADE's own documented OLLAMA_MODEL
default) as the primary/representative result, and qwen2.5:72b-instruct-q4_0
as a secondary, larger-model comparison. Ollama sampling is not seeded, so
a rerun may reproduce slightly different exact figures; the qualitative
per-category pattern reported in the paper was stable across runs.

Usage: python scripts/experiment6_agent_tool_grounding.py
Requires a local Ollama server (http://localhost:11434) with both models
pulled.
"""

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gene_id_mapper import get_mapper

OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
RESULTS_PATH = OUTPUTS_DIR / "experiment6_agent_tool_grounding.json"

OLLAMA_HOST = "http://localhost:11434"
MODELS = ["llama3.1:8b", "qwen2.5:72b-instruct-q4_0"]

CELL_TYPE_ENUM = [
    "epithelial_cell", "cd4_t_cells", "cd8_t_cells", "cd14_monocytes",
    "cd16_monocytes", "cd20_b_cells", "nk_cells", "nkt_cells",
    "erythrocytes", "monocyte-derived_dendritic_cells",
]
TCGA_NETWORK_ENUM = ["blca", "brca", "cesc", "coad", "hnsc", "kirc", "lihc",
                      "luad", "lusc", "ov", "paad", "prad", "stad", "ucec"]

# Verbatim from cascade_langgraph_mcp_server.py's comprehensive_perturbation_analysis
# tool definition (lines ~450-514), translated to Ollama's tool-schema format.
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "comprehensive_perturbation_analysis",
        "description": (
            "Analyze what happens when a gene is knocked down or overexpressed. "
            "This is the RECOMMENDED tool for most analyses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gene": {
                    "type": "string",
                    "description": "Gene symbol (e.g., TP53, MYC, APC) or Ensembl ID",
                },
                "cell_type": {
                    "type": "string",
                    "enum": CELL_TYPE_ENUM,
                    "description": "Cell type for network analysis",
                    "default": "epithelial_cell",
                },
                "perturbation_type": {
                    "type": "string",
                    "enum": ["knockdown", "overexpression"],
                    "description": "Type of perturbation to simulate",
                    "default": "knockdown",
                },
                "analysis_depth": {
                    "type": "string",
                    "enum": ["basic", "comprehensive", "focused"],
                    "description": "How deep to analyze (basic=fast, comprehensive=full, focused=gene-type-specific)",
                    "default": "comprehensive",
                },
                "network_source": {
                    "type": "string",
                    "enum": ["cell_type", "tcga"],
                    "description": "Use 'tcga' for tumor-state regulatory network; requires tcga_network param",
                    "default": "cell_type",
                },
                "tcga_network": {
                    "type": "string",
                    "enum": TCGA_NETWORK_ENUM,
                    "description": "TCGA cancer type network (required when network_source=tcga). Epithelial-origin only.",
                },
            },
            "required": ["gene"],
        },
    },
}

DEFAULTS = {
    "cell_type": "epithelial_cell",
    "perturbation_type": "knockdown",
    "analysis_depth": "comprehensive",
    "network_source": "cell_type",
}

QUERIES = [
    {"category": "baseline_tcga", "query": "What happens if we knock down TP53 in liver cancer?",
     "expected": {"gene": "TP53", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "lihc"}},
    {"category": "baseline_tcga", "query": "Simulate knockdown of PTEN in kidney cancer",
     "expected": {"gene": "PTEN", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "kirc"}},
    {"category": "baseline_tcga", "query": "What are the downstream effects of MYC overexpression in lung adenocarcinoma?",
     "expected": {"gene": "MYC", "perturbation_type": "overexpression", "network_source": "tcga", "tcga_network": "luad"}},
    {"category": "baseline_tcga", "query": "Predict what happens when EGFR is knocked down in cervical cancer",
     "expected": {"gene": "EGFR", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "cesc"}},
    {"category": "baseline_tcga", "query": "Overexpress BRCA1 in ovarian cancer and show effects",
     "expected": {"gene": "BRCA1", "perturbation_type": "overexpression", "network_source": "tcga", "tcga_network": "ov"}},

    {"category": "baseline_immune", "query": "What if we knock down STAT3 in CD4 T cells?",
     "expected": {"gene": "STAT3", "perturbation_type": "knockdown", "cell_type": "cd4_t_cells", "network_source": "cell_type"}},
    {"category": "baseline_immune", "query": "Show effects of overexpressing IL2 in NK cells",
     "expected": {"gene": "IL2", "perturbation_type": "overexpression", "cell_type": "nk_cells", "network_source": "cell_type"}},
    {"category": "baseline_immune", "query": "Knock down FOXP3 in CD8 T cells",
     "expected": {"gene": "FOXP3", "perturbation_type": "knockdown", "cell_type": "cd8_t_cells", "network_source": "cell_type"}},
    {"category": "baseline_immune", "query": "What happens if BCL2 is overexpressed in B cells?",
     "expected": {"gene": "BCL2", "perturbation_type": "overexpression", "cell_type": "cd20_b_cells", "network_source": "cell_type"}},
    {"category": "baseline_immune", "query": "Simulate knockdown of TLR4 in CD14 monocytes",
     "expected": {"gene": "TLR4", "perturbation_type": "knockdown", "cell_type": "cd14_monocytes", "network_source": "cell_type"}},

    {"category": "gene_alias", "query": "Knock down HER2 in breast cancer",
     "expected": {"gene": "ERBB2", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "brca"}},
    {"category": "gene_alias", "query": "What if p53 is overexpressed in lung squamous cell carcinoma?",
     "expected": {"gene": "TP53", "perturbation_type": "overexpression", "network_source": "tcga", "tcga_network": "lusc"}},
    {"category": "gene_alias", "query": "Simulate knockdown of the estrogen receptor in breast cancer",
     "expected": {"gene": "ESR1", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "brca"}},
    {"category": "gene_alias", "query": "What happens when we knock down PD-L1 in NK cells?",
     "expected": {"gene": "CD274", "perturbation_type": "knockdown", "cell_type": "nk_cells", "network_source": "cell_type"}},
    {"category": "gene_alias", "query": "Overexpress HER3 in stomach cancer",
     "expected": {"gene": "ERBB3", "perturbation_type": "overexpression", "network_source": "tcga", "tcga_network": "stad"}},

    {"category": "cancer_type_name", "query": "Knock down APC in head and neck cancer",
     "expected": {"gene": "APC", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "hnsc"}},
    {"category": "cancer_type_name", "query": "What if TP53 is overexpressed in uterine cancer?",
     "expected": {"gene": "TP53", "perturbation_type": "overexpression", "network_source": "tcga", "tcga_network": "ucec"}},
    {"category": "cancer_type_name", "query": "Simulate MYC knockdown in prostate cancer",
     "expected": {"gene": "MYC", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "prad"}},
    {"category": "cancer_type_name", "query": "Overexpress KRAS in bladder cancer",
     "expected": {"gene": "KRAS", "perturbation_type": "overexpression", "network_source": "tcga", "tcga_network": "blca"}},
    {"category": "cancer_type_name", "query": "Knock down SMAD4 in pancreatic cancer",
     "expected": {"gene": "SMAD4", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "paad"}},

    {"category": "perturbation_phrasing", "query": "Silence MYC in breast cancer",
     "expected": {"gene": "MYC", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "brca"}},
    {"category": "perturbation_phrasing", "query": "Boost expression of PTEN in colon cancer",
     "expected": {"gene": "PTEN", "perturbation_type": "overexpression", "network_source": "tcga", "tcga_network": "coad"}},
    {"category": "perturbation_phrasing", "query": "Suppress TP53 activity in stomach cancer",
     "expected": {"gene": "TP53", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "stad"}},
    {"category": "perturbation_phrasing", "query": "Amplify GATA3 signaling in breast tumors",
     "expected": {"gene": "GATA3", "perturbation_type": "overexpression", "network_source": "tcga", "tcga_network": "brca"}},
    {"category": "perturbation_phrasing", "query": "Delete AURKA function in bladder cancer",
     "expected": {"gene": "AURKA", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "blca"}},

    {"category": "implicit_params", "query": "What does MYC do?",
     "expected": {"gene": "MYC", "perturbation_type": "knockdown", "cell_type": "epithelial_cell", "network_source": "cell_type"}},
    {"category": "implicit_params", "query": "Tell me about knocking down TP53",
     "expected": {"gene": "TP53", "perturbation_type": "knockdown", "cell_type": "epithelial_cell", "network_source": "cell_type"}},
    {"category": "implicit_params", "query": "Is APC a good drug target?",
     "expected": {"gene": "APC", "perturbation_type": "knockdown", "cell_type": "epithelial_cell", "network_source": "cell_type"}},
    {"category": "implicit_params", "query": "Overexpress ESR1",
     "expected": {"gene": "ESR1", "perturbation_type": "overexpression", "cell_type": "epithelial_cell", "network_source": "cell_type"}},
    {"category": "implicit_params", "query": "What genes are affected by GATA3?",
     "expected": {"gene": "GATA3", "perturbation_type": "knockdown", "cell_type": "epithelial_cell", "network_source": "cell_type"}},

    {"category": "multi_entity", "query": "We're studying MYC and TP53 together, but for now just knock down MYC in breast cancer",
     "expected": {"gene": "MYC", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "brca"}},
    {"category": "multi_entity", "query": "Compared to breast cancer, I actually want the APC knockdown result in colon cancer",
     "expected": {"gene": "APC", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "coad"}},
    {"category": "multi_entity", "query": "I know ERBB2 is important in breast cancer, but right now show me AURKA overexpression in breast cancer instead",
     "expected": {"gene": "AURKA", "perturbation_type": "overexpression", "network_source": "tcga", "tcga_network": "brca"}},
    {"category": "multi_entity", "query": "Previously we knocked down GATA3, now let's overexpress FOXA1 in breast tumors",
     "expected": {"gene": "FOXA1", "perturbation_type": "overexpression", "network_source": "tcga", "tcga_network": "brca"}},
    {"category": "multi_entity", "query": "Skip the stomach cancer analysis for now, instead simulate MYC knockdown in colon cancer",
     "expected": {"gene": "MYC", "perturbation_type": "knockdown", "network_source": "tcga", "tcga_network": "coad"}},
]


def run_model(model: str) -> list[dict]:
    results = []
    for item in QUERIES:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": item["query"]}],
                "tools": [TOOL_SCHEMA],
                "stream": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        tool_calls = data.get("message", {}).get("tool_calls", [])
        args = tool_calls[0]["function"]["arguments"] if tool_calls else None
        results.append({
            "category": item["category"],
            "query": item["query"],
            "expected": item["expected"],
            "actual": args,
        })
    return results


def resolved_value(actual, key: str):
    """What the server would actually use for `key`, applying documented
    defaults if the model omitted the field entirely."""
    if actual is None:
        return None
    v = actual.get(key)
    if v is None or v == "":
        return DEFAULTS.get(key)
    return v


def score(results: list[dict]) -> dict:
    """Score each query two ways: `exact_match` against the model's raw
    tool-call output (what the benchmark has always measured), and
    `server_exact_match`, which additionally runs the model's `gene` value
    through CASCADE's real `resolve_alias()` before comparing -- the same
    step CASCADE's own MCP server runs before any network lookup. This
    distinguishes gene-alias cases the model gets wrong but CASCADE's
    server would silently fix (e.g. a model saying "HER2", which
    resolve_alias() maps to ERBB2) from cases neither the model nor
    CASCADE's alias table can resolve (e.g. an alias not yet in the table).
    """
    mapper = get_mapper()
    per_category: dict = {}
    field_mismatches: dict = {}
    server_field_mismatches: dict = {}
    total_exact = 0
    total_server_exact = 0
    scored_results = []

    for r in results:
        cat = r["category"]
        per_category.setdefault(cat, {"n": 0, "exact": 0, "server_exact": 0})
        per_category[cat]["n"] += 1

        mismatches = []
        server_mismatches = []
        if r["actual"] is None:
            mismatches = ["no_tool_call"]
            server_mismatches = ["no_tool_call"]
        else:
            for k, expected_v in r["expected"].items():
                actual_v = resolved_value(r["actual"], k)
                if str(actual_v).strip().lower() != str(expected_v).strip().lower():
                    mismatches.append(f"{k}: expected={expected_v!r} got={actual_v!r}")
                    field_mismatches[k] = field_mismatches.get(k, 0) + 1

                server_v = mapper.resolve_alias(actual_v) if k == "gene" and actual_v is not None else actual_v
                if str(server_v).strip().lower() != str(expected_v).strip().lower():
                    server_mismatches.append(f"{k}: expected={expected_v!r} got={server_v!r}")
                    server_field_mismatches[k] = server_field_mismatches.get(k, 0) + 1

        exact = not mismatches
        server_exact = not server_mismatches
        if exact:
            total_exact += 1
            per_category[cat]["exact"] += 1
        if server_exact:
            total_server_exact += 1
            per_category[cat]["server_exact"] += 1

        scored_results.append({
            **r,
            "exact_match": exact,
            "mismatches": mismatches,
            "server_exact_match": server_exact,
            "server_mismatches": server_mismatches,
        })

    return {
        "n": len(results),
        "overall_exact": total_exact,
        "overall_exact_pct": round(100 * total_exact / len(results), 1),
        "overall_server_exact": total_server_exact,
        "overall_server_exact_pct": round(100 * total_server_exact / len(results), 1),
        "per_category": per_category,
        "field_mismatches": field_mismatches,
        "server_field_mismatches": server_field_mismatches,
        "results": scored_results,
    }


def main() -> None:
    all_summaries = {}
    for model in MODELS:
        print(f"\n=== {model} ===")
        results = run_model(model)
        summary = score(results)
        all_summaries[model] = summary
        print(f"  raw:    {summary['overall_exact']}/{summary['n']} ({summary['overall_exact_pct']}%)")
        print(f"  server: {summary['overall_server_exact']}/{summary['n']} ({summary['overall_server_exact_pct']}%)")
        for cat, v in summary["per_category"].items():
            print(f"    {cat}: raw {v['exact']}/{v['n']}, server {v['server_exact']}/{v['n']}")

    RESULTS_PATH.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nWrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
