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
  3. gene_alias            - gene synonyms, all of which CASCADE now resolves
                             server-side via tools/gene_id_mapper.py
                             GENE_SYMBOL_ALIASES (HER2, p53, PD-L1, HER3);
                             scored both against the model's raw output and
                             against CASCADE's server-side resolution
  4. cancer_type_name      - informal/full cancer-type names -> enum abbreviation
  5. perturbation_phrasing - non-canonical wording for knockdown/overexpression
  6. implicit_params       - queries that omit parameters, relying on defaults
  7. multi_entity          - a distractor gene/cancer type mentioned but not
                             the one actually requested

Perturbation-type ambiguity fix (Section 3.6): three implicit_params queries
("What does MYC do?", "Is APC a good drug target?", "What genes are affected
by GATA3?") give no directional cue at all -- CASCADE's MCP server used to
silently default perturbation_type to "knockdown" for these. The server now
takes an optional "query" argument (the caller's original NL request) and,
when perturbation_type is omitted, scans it for directional cues before
defaulting; if none (or conflicting cues) are found, it returns
clarification_needed instead of guessing. This benchmark tests that fix
end-to-end: the tool schema advertises the new "query" field, and whether the
model actually populates it is itself measured (a model can omit
perturbation_type but not comply with sending "query", in which case the old
silent default still applies -- see score_ambiguity_handling()).

Two models are run by default and are the ones reported in the paper:
llama3.1:8b (CASCADE's own documented OLLAMA_MODEL default) as the
primary/representative result, and qwen2.5:72b-instruct-q4_0 as a
secondary, larger local-model comparison. Ollama sampling is not seeded,
so a rerun may reproduce slightly different exact figures; the qualitative
per-category pattern reported in the paper was stable across runs.

Optional cloud models (e.g. gpt-oss:120b via Ollama Cloud) can be run with
--models but are not part of the paper's reported results.

Usage: python scripts/experiment6_agent_tool_grounding.py [--models m1,m2,...]
Local models require a local Ollama server (http://localhost:11434) with
the models pulled. Any model not in LOCAL_MODELS is routed to
https://ollama.com and requires OLLAMA_API_KEY to be set in the
environment (see .env.example). Results merge into the existing output
file rather than overwrite it, so re-running a subset of models leaves the
others' previously recorded results untouched.
"""

import argparse
import json
import os
import re
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

LOCAL_OLLAMA_HOST = "http://localhost:11434"
CLOUD_OLLAMA_HOST = "https://ollama.com"
LOCAL_MODELS = {"llama3.1:8b", "qwen2.5:72b-instruct-q4_0"}
MODELS = ["llama3.1:8b", "qwen2.5:72b-instruct-q4_0"]

_SSL_VERIFY = os.environ.get("CASCADE_SSL_NO_VERIFY", "0") != "1"

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
                "query": {
                    "type": "string",
                    "description": (
                        "Optional: the user's original natural-language request, verbatim. "
                        "If perturbation_type is not given, this text is scanned for "
                        "directional cues (e.g. 'knock down'/'silence' vs 'overexpress'/"
                        "'boost') instead of silently defaulting to knockdown. If omitted "
                        "or the cues are absent/conflicting, behavior is unchanged."
                    ),
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

# Mirrored from cascade_langgraph_mcp_server.py's _detect_perturbation_direction
# and _KNOCKDOWN_CUES/_OVEREXPRESSION_CUES (Section 3.6 fix). Kept as a literal
# copy rather than imported so this benchmark doesn't pull in the full server
# module's heavy dependencies (torch, mcp.server, langgraph) just to score
# tool-call args; keep in sync with the server if the cue lists change.
_KNOCKDOWN_CUES = [
    r"knock(?:ed|ing)?[\s-]?down",
    r"silenc\w*",
    r"suppress\w*",
    r"inhibit\w*",
    r"delet\w*",
    r"disrupt\w*",
    r"deplet\w*",
]
_OVEREXPRESSION_CUES = [
    r"overexpress\w*",
    r"over[\s-]express\w*",
    r"boost\w*",
    r"increas\w*",
    r"activat\w*",
    r"upregulat\w*",
    r"up[\s-]regulat\w*",
    r"amplif\w*",
    r"enhanc\w*",
]


def _classify_perturbation_cues(query_text: str) -> tuple[str | None, str]:
    """Same detection logic as cascade_langgraph_mcp_server.py's
    _detect_perturbation_direction (that one is the actual server-side gate;
    this mirrors it for benchmark simulation), but also reports *why* it
    returned no single direction --
      'no_cues'          - no directional cue at all (the original gap this
                            fix targets, e.g. the 3 known-ambiguous queries).
      'conflicting_cues' - cues for BOTH directions are present, typically
                            because a multi-entity query mentions two genes
                            with different directions ("knocked down GATA3,
                            now overexpress FOXA1"). The keyword scanner has
                            no way to attribute each cue to the right gene,
                            so this is a distinct, disclosed limitation of
                            the fix's naive text-matching -- not the same
                            failure mode as the original no-cue gap, and not
                            a generic unexplained regression either.
    The server's real gate doesn't need this distinction (it only needs
    direction-or-None to decide default vs. clarification); it exists here
    purely so the benchmark can report *why* a query landed where it did.
    """
    text = query_text.lower()
    directions = set()
    if any(re.search(p, text) for p in _KNOCKDOWN_CUES):
        directions.add("knockdown")
    if any(re.search(p, text) for p in _OVEREXPRESSION_CUES):
        directions.add("overexpression")
    if len(directions) == 1:
        return directions.pop(), "single_cue"
    if len(directions) == 0:
        return None, "no_cues"
    return None, "conflicting_cues"


def simulate_server_perturbation_handling(actual: dict | None) -> dict:
    """Mirrors the gate in cascade_langgraph_mcp_server.py's
    _comprehensive_analysis: given the model's raw tool-call args, determines
    what the real (fixed) server would do for perturbation_type, and which
    of the three ambiguity-fix outcome cases applies:

      explicit_guess   - model supplied perturbation_type itself; the fix
                          never engages (out of scope by design -- this fix
                          only replaces the *default*, not an explicit value).
      omitted_complied  - model omitted perturbation_type but did supply the
                          new "query" field, so the server can run detection.
      omitted_no_query  - model omitted perturbation_type and did not supply
                          "query"; the server silently falls back to the old
                          "knockdown" default, unchanged from before the fix.
      no_tool_call      - model did not call the tool at all.

    Also returns "cue_reason" (see _classify_perturbation_cues) whenever
    detection actually ran, so callers can distinguish a true no-cue result
    from a multi-entity conflicting-cue false positive.
    """
    if actual is None:
        return {"case": "no_tool_call", "status": "no_tool_call", "resolved_perturbation_type": None, "cue_reason": None}

    perturbation_type = actual.get("perturbation_type")
    if perturbation_type:
        return {"case": "explicit_guess", "status": "resolved", "resolved_perturbation_type": perturbation_type, "cue_reason": None}

    query_text = actual.get("query")
    if query_text:
        direction, reason = _classify_perturbation_cues(query_text)
        if direction is None:
            return {"case": "omitted_complied", "status": "clarification_needed", "resolved_perturbation_type": None, "cue_reason": reason}
        return {"case": "omitted_complied", "status": "resolved", "resolved_perturbation_type": direction, "cue_reason": reason}

    return {"case": "omitted_no_query", "status": "resolved", "resolved_perturbation_type": DEFAULTS["perturbation_type"], "cue_reason": None}


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

    {"category": "implicit_params", "query": "What does MYC do?", "ambiguous": True,
     "expected": {"gene": "MYC", "perturbation_type": "knockdown", "cell_type": "epithelial_cell", "network_source": "cell_type"}},
    {"category": "implicit_params", "query": "Tell me about knocking down TP53",
     "expected": {"gene": "TP53", "perturbation_type": "knockdown", "cell_type": "epithelial_cell", "network_source": "cell_type"}},
    {"category": "implicit_params", "query": "Is APC a good drug target?", "ambiguous": True,
     "expected": {"gene": "APC", "perturbation_type": "knockdown", "cell_type": "epithelial_cell", "network_source": "cell_type"}},
    {"category": "implicit_params", "query": "Overexpress ESR1",
     "expected": {"gene": "ESR1", "perturbation_type": "overexpression", "cell_type": "epithelial_cell", "network_source": "cell_type"}},
    {"category": "implicit_params", "query": "What genes are affected by GATA3?", "ambiguous": True,
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
    is_local = model in LOCAL_MODELS
    api_key = os.environ.get("OLLAMA_API_KEY")
    headers = {}
    if is_local:
        host = LOCAL_OLLAMA_HOST
    else:
        if not api_key:
            raise RuntimeError(
                f"{model} is a cloud model but OLLAMA_API_KEY is not set "
                "(see .env.example, 'Ollama Cloud Mode')"
            )
        host = CLOUD_OLLAMA_HOST
        headers = {"Authorization": f"Bearer {api_key}"}

    results = []
    for item in QUERIES:
        resp = requests.post(
            f"{host}/api/chat",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": item["query"]}],
                "tools": [TOOL_SCHEMA],
                "stream": False,
            },
            timeout=180,
            verify=_SSL_VERIFY if not is_local else True,
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
            "ambiguous": item.get("ambiguous", False),
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


def score_ambiguity_handling(scored_results: list[dict]) -> dict:
    """New scoring category for the perturbation-type ambiguity fix (Section
    3.6). Independent of score() above -- exact_match/server_exact_match are
    unchanged and still compare the model's raw args against the old
    "knockdown"-default ground truth. This instead asks: for the 3 queries
    with no directional cue at all, does the (simulated) fixed server return
    clarification_needed rather than silently guessing? Takes score()'s
    `results` list (post-scoring, so "mismatches" is available for the
    network_source regression check below), not the raw run_model() output.

    Also reports, across all 35 queries:
      - query_field_compliance: how often the model populated the new
        optional "query" field at all (a prerequisite for the fix to engage
        when perturbation_type is omitted -- see simulate_server_perturbation_handling).
      - round_trip_triggers: how many queries would get a clarification
        response instead of a direct result under the fix (the added-latency
        cost of the fix, as a fraction of the 35-query set).
      - non_ambiguous_regressions: any non-ambiguous query that unexpectedly
        triggers clarification_needed. Each is tagged with a failure_category
        so a distinct, known limitation (multi-entity queries where the
        keyword scanner sees conflicting cues from two different genes, e.g.
        "knocked down GATA3, now overexpress FOXA1") is reported separately
        from a generic, unexplained regression -- these are different
        findings and conflating them would muddy the before/after comparison
        for the perturbation-type fix itself.
    """
    ambiguous_detail = []
    query_field_compliance = 0
    round_trip_triggers = 0
    non_ambiguous_regressions = []

    for r in scored_results:
        sim = simulate_server_perturbation_handling(r["actual"])
        query_supplied = bool(r["actual"].get("query")) if r["actual"] else False
        if query_supplied:
            query_field_compliance += 1

        if sim["status"] == "clarification_needed":
            round_trip_triggers += 1
            if not r["ambiguous"]:
                if r["category"] == "multi_entity" and sim.get("cue_reason") == "conflicting_cues":
                    failure_category = "multi_entity_keyword_conflict"
                else:
                    failure_category = "unexpected_non_ambiguous_regression"
                non_ambiguous_regressions.append({
                    "category": r["category"], "query": r["query"], "case": sim["case"],
                    "cue_reason": sim.get("cue_reason"), "failure_category": failure_category,
                })

        if r["ambiguous"]:
            # Independent of the perturbation_type outcome: does this query
            # *also* still exhibit the (unfixed, out-of-scope) network_source
            # misroute? Reported separately, not folded into one pass/fail.
            network_source_mismatch = any(
                m.startswith("network_source:") for m in r.get("mismatches", [])
            )
            ambiguous_detail.append({
                "query": r["query"],
                "case": sim["case"],
                "status": sim["status"],
                "correctly_flagged": sim["status"] == "clarification_needed",
                "network_source_mismatch": network_source_mismatch,
            })

    n = len(scored_results)
    n_ambiguous = len(ambiguous_detail)
    n_correctly_flagged = sum(1 for d in ambiguous_detail if d["correctly_flagged"])

    return {
        "n_ambiguous": n_ambiguous,
        "n_correctly_flagged": n_correctly_flagged,
        "ambiguous_detail": ambiguous_detail,
        "query_field_compliance": query_field_compliance,
        "query_field_compliance_pct": round(100 * query_field_compliance / n, 1),
        "round_trip_triggers": round_trip_triggers,
        "round_trip_triggers_pct": round(100 * round_trip_triggers / n, 1),
        "non_ambiguous_regressions": non_ambiguous_regressions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated subset of MODELS to run (default: all). "
             "Results merge into the existing output file; models not "
             "rerun keep their previously recorded results.",
    )
    args = parser.parse_args()
    models_to_run = args.models.split(",") if args.models else MODELS

    all_summaries = {}
    if RESULTS_PATH.exists():
        all_summaries = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    for model in models_to_run:
        print(f"\n=== {model} ===")
        old_summary = all_summaries.get(model)  # pre-rerun, for before/after reporting
        results = run_model(model)
        summary = score(results)
        ambiguity = score_ambiguity_handling(summary["results"])
        summary["ambiguity_handling"] = ambiguity
        all_summaries[model] = summary

        print(f"  raw:    {summary['overall_exact']}/{summary['n']} ({summary['overall_exact_pct']}%)")
        print(f"  server: {summary['overall_server_exact']}/{summary['n']} ({summary['overall_server_exact_pct']}%)")
        for cat, v in summary["per_category"].items():
            print(f"    {cat}: raw {v['exact']}/{v['n']}, server {v['server_exact']}/{v['n']}")

        print(f"\n  --- perturbation-type ambiguity fix (Section 3.6) ---")
        print(f"  correctly flagged as ambiguous: {ambiguity['n_correctly_flagged']}/{ambiguity['n_ambiguous']}")
        for d in ambiguity["ambiguous_detail"]:
            before = None
            if old_summary:
                old_r = next((x for x in old_summary.get("results", []) if x["query"] == d["query"]), None)
                if old_r is not None:
                    before = "matched knockdown default" if old_r.get("exact_match") else "mismatched"
            before_str = f", before={before}" if before is not None else ""
            extra = " [network_source misroute persists, unfixed by design]" if d["network_source_mismatch"] else ""
            print(f"    [{d['case']}] {d['query']!r} -> {d['status']}{before_str}{extra}")

        print(f"  query-field compliance (all {summary['n']}): "
              f"{ambiguity['query_field_compliance']}/{summary['n']} ({ambiguity['query_field_compliance_pct']}%)")
        print(f"  added round-trip cost (all {summary['n']}): "
              f"{ambiguity['round_trip_triggers']}/{summary['n']} ({ambiguity['round_trip_triggers_pct']}%)")
        if ambiguity["non_ambiguous_regressions"]:
            multi_entity_conflicts = [r for r in ambiguity["non_ambiguous_regressions"]
                                       if r["failure_category"] == "multi_entity_keyword_conflict"]
            other_regressions = [r for r in ambiguity["non_ambiguous_regressions"]
                                  if r["failure_category"] != "multi_entity_keyword_conflict"]
            if multi_entity_conflicts:
                print(f"  KNOWN LIMITATION -- multi-entity keyword conflict "
                      f"(scanner can't attribute cues to the right gene): {multi_entity_conflicts}")
            if other_regressions:
                print(f"  WARNING -- unexplained non-ambiguous regression(s): {other_regressions}")
        else:
            print(f"  regression check: no previously-non-ambiguous query triggered clarification")

    RESULTS_PATH.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nWrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
