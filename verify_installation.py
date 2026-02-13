#!/usr/bin/env python
"""CASCADE installation verification script.

Runs a series of checks to confirm that the CASCADE MCP server
is correctly installed and functional:
  1. Core dependency imports
  2. Network file loading (all 10 cell types)
  3. Model checkpoint loading
  4. Gene ID mapping (requires internet; skip with --offline)
  5. Perturbation analysis
  6. Embedding similarity search

Usage:
    python verify_installation.py            # full check
    python verify_installation.py --offline  # skip gene ID mapping
"""

import argparse
import importlib
import sys

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_USE_COLOR = True


def _init_color():
    """Enable ANSI colors if the terminal supports them."""
    global _USE_COLOR
    if sys.platform == "win32":
        try:
            import colorama
            colorama.init()
        except ImportError:
            # Enable ANSI on Windows 10+ via VT100 mode
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                _USE_COLOR = False
    if not sys.stdout.isatty():
        _USE_COLOR = False


def _pass(msg: str):
    tag = "\033[92m[PASS]\033[0m" if _USE_COLOR else "[PASS]"
    print(f"{tag} {msg}")


def _fail(msg: str):
    tag = "\033[91m[FAIL]\033[0m" if _USE_COLOR else "[FAIL]"
    print(f"{tag} {msg}")


def _skip(msg: str):
    tag = "\033[93m[SKIP]\033[0m" if _USE_COLOR else "[SKIP]"
    print(f"{tag} {msg}")


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

def check_dependencies() -> tuple[int, int]:
    """Import core dependencies and report results."""
    modules = [
        ("torch", "torch"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("fastmcp", "fastmcp"),
        ("langgraph", "langgraph"),
        ("decoupler", "decoupler"),
        ("scGraphLLM", "scGraphLLM"),
    ]
    failed = []
    for display_name, import_name in modules:
        try:
            importlib.import_module(import_name)
        except ImportError:
            failed.append(display_name)

    if not failed:
        _pass("Core dependencies imported")
        return 1, 0
    else:
        _fail(f"Missing dependencies: {', '.join(failed)}")
        return 0, 1


def check_networks() -> tuple[int, int]:
    """Load every cell-type network and validate columns."""
    try:
        from tools.loader import NETWORKS_DIR, load_network
    except Exception as exc:
        _fail(f"Network loading — import error: {exc}")
        return 0, 1

    passed = failed = 0
    required_cols = {"regulator", "target", "mi"}

    if not NETWORKS_DIR.exists():
        _fail(f"Networks directory not found: {NETWORKS_DIR}")
        return 0, 1

    cell_types = sorted(p.name for p in NETWORKS_DIR.iterdir() if p.is_dir())
    if not cell_types:
        _fail("No cell-type directories found")
        return 0, 1

    for ct in cell_types:
        tsv = NETWORKS_DIR / ct / "network.tsv"
        try:
            df = load_network(tsv)
            missing = required_cols - set(df.columns)
            if missing:
                _fail(f"Network: {ct} — missing columns {missing}")
                failed += 1
            else:
                _pass(f"Network: {ct} ({len(df):,} edges)")
                passed += 1
        except Exception as exc:
            _fail(f"Network: {ct} — {exc}")
            failed += 1

    return passed, failed


def check_model() -> tuple[int, int]:
    """Load the model checkpoint and verify embedding stats."""
    try:
        from tools.loader import MODEL_PATH
        from tools.model_inference import CascadeModel
    except Exception as exc:
        _fail(f"Model checkpoint — import error: {exc}")
        return 0, 1

    try:
        model = CascadeModel(MODEL_PATH).load()
        stats = model.get_embedding_stats()
        num_genes = stats.get("num_genes", 0)
        dim = stats.get("embedding_dim", "?")
        if num_genes == 0:
            _fail("Model loaded but gene count is 0")
            return 0, 1
        _pass(f"Model checkpoint loaded ({num_genes:,} genes, dim={dim})")
        return 1, 0
    except Exception as exc:
        _fail(f"Model checkpoint — {exc}")
        return 0, 1


def check_gene_id_mapping() -> tuple[int, int]:
    """Round-trip TP53 <-> ENSG00000141510 via Ensembl API."""
    try:
        from tools.gene_id_mapper import get_mapper
    except Exception as exc:
        _fail(f"Gene ID mapping — import error: {exc}")
        return 0, 1

    mapper = get_mapper()
    expected_id = "ENSG00000141510"
    try:
        ensembl = mapper.symbol_to_ensembl("TP53")
        if ensembl is None:
            _fail("Gene ID mapping: TP53 -> None")
            return 0, 1
        symbol = mapper.ensembl_to_symbol(ensembl)
        if symbol is None:
            _fail(f"Gene ID mapping: {ensembl} -> None")
            return 0, 1
        if ensembl != expected_id:
            _fail(f"Gene ID mapping: TP53 -> {ensembl} (expected {expected_id})")
            return 0, 1
        _pass(f"Gene ID mapping: TP53 <-> {ensembl}")
        return 1, 0
    except Exception as exc:
        _fail(f"Gene ID mapping — {exc}")
        return 0, 1


def check_perturbation() -> tuple[int, int]:
    """Run a knockdown simulation on the epithelial_cell network."""
    try:
        from tools.loader import NETWORKS_DIR, load_network
        from tools.perturb import simulate_knockdown
    except Exception as exc:
        _fail(f"Perturbation — import error: {exc}")
        return 0, 1

    tsv = NETWORKS_DIR / "epithelial_cell" / "network.tsv"
    try:
        df = load_network(tsv)
    except Exception as exc:
        _fail(f"Perturbation — could not load network: {exc}")
        return 0, 1

    regulators = df["regulator"].unique()
    if len(regulators) == 0:
        _fail("Perturbation — no regulators in epithelial_cell network")
        return 0, 1

    gene = regulators[0]
    try:
        result = simulate_knockdown(df, gene, depth=2, top_k=10)
        status = result.get("status")
        if status == "error":
            _fail(f"Perturbation — {result.get('error', 'unknown error')}")
            return 0, 1
        affected = result.get("total_affected_genes", 0)
        _pass(f"Perturbation: {gene} knockdown -> {affected} affected genes")
        return 1, 0
    except Exception as exc:
        _fail(f"Perturbation — {exc}")
        return 0, 1


def check_embedding_similarity() -> tuple[int, int]:
    """Find top-5 similar genes for a known gene in the vocabulary."""
    try:
        from tools.loader import MODEL_PATH
        from tools.model_inference import CascadeModel
    except Exception as exc:
        _fail(f"Embedding similarity — import error: {exc}")
        return 0, 1

    try:
        model = CascadeModel(MODEL_PATH).load()
    except Exception as exc:
        _fail(f"Embedding similarity — model load failed: {exc}")
        return 0, 1

    # Use TP53 Ensembl ID; fall back to any gene in vocab if missing
    test_gene = "ENSG00000141510"
    if not model.is_gene_in_vocab(test_gene):
        stats = model.get_embedding_stats()
        num_special = stats.get("num_special_tokens", 3)
        test_gene = None
        for token, idx in model.vocab.vocabulary.items():
            if idx >= num_special and token.startswith("ENSG"):
                test_gene = token
                break
        if test_gene is None:
            _fail("Embedding similarity — no valid gene found in vocab")
            return 0, 1

    try:
        similar = model.get_top_similar_genes(test_gene, top_k=5)
        if similar is None or len(similar) == 0:
            _fail(f"Embedding similarity — no results for {test_gene}")
            return 0, 1
        _pass(f"Embedding similarity: top-5 genes for {test_gene}")
        return 1, 0
    except Exception as exc:
        _fail(f"Embedding similarity — {exc}")
        return 0, 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify CASCADE installation"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip checks that require internet (gene ID mapping)",
    )
    # Legacy alias
    parser.add_argument("--skip-network", dest="offline", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    _init_color()

    print("CASCADE Installation Verification")
    print("=" * 38)

    total_passed = 0
    total_failed = 0

    # 1. Dependencies
    p, f = check_dependencies()
    total_passed += p
    total_failed += f

    # Bail early if core deps are missing — subsequent checks will all fail
    if f > 0:
        print("=" * 38)
        print("Fix missing dependencies before continuing.")
        return 1

    # 2. Networks
    p, f = check_networks()
    total_passed += p
    total_failed += f

    # 3. Model checkpoint
    p, f = check_model()
    total_passed += p
    total_failed += f

    # 4. Gene ID mapping (optional)
    if args.offline:
        _skip("Gene ID mapping (--offline)")
    else:
        p, f = check_gene_id_mapping()
        total_passed += p
        total_failed += f

    # 5. Perturbation
    p, f = check_perturbation()
    total_passed += p
    total_failed += f

    # 6. Embedding similarity
    p, f = check_embedding_similarity()
    total_passed += p
    total_failed += f

    # Summary
    total = total_passed + total_failed
    print("=" * 38)
    if total_failed == 0:
        print(f"All {total} checks passed!")
        return 0
    else:
        print(f"{total_passed}/{total} checks passed, {total_failed} failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
