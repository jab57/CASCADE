#!/usr/bin/env python3
"""
Gene ID Mapper for CASCADE MCP Server
Converts between gene symbols and Ensembl IDs

Consistent with RegNetAgents gene_id_mapper API.
"""

import logging
import requests
import json
import os
import threading
import time
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Semaphore limiting concurrent Ensembl REST API calls across all threads.
# Shared across all GeneIDMapper instances (module-level singleton pattern).
_ensembl_semaphore = threading.Semaphore(int(os.getenv('API_RATE_LIMIT', '3')))

# TLS verification toggle — matches every other CASCADE HTTP client
# (tools/cbioportal.py, tools/ppi/string_client.py). Set CASCADE_SSL_NO_VERIFY=1
# on networks with corporate SSL inspection so Ensembl calls fail fast / succeed
# via the bypass instead of burning a full 10s timeout per gene.
_SSL_VERIFY = os.environ.get("CASCADE_SSL_NO_VERIFY", "0") != "1"

# Per-request timeout for Ensembl REST calls. Ensembl is best-effort metadata
# enrichment, never on the critical path — a lookup is not worth 10s. Keep it
# short so failures are cheap.
_ENSEMBL_TIMEOUT = float(os.getenv("ENSEMBL_TIMEOUT", "3"))

# Circuit breaker: once Ensembl is clearly unreachable (corporate SSL block,
# outage, throttling), stop calling it so every lookup short-circuits to None
# for _CB_COOLDOWN seconds. This bounds a comprehensive analysis that would
# otherwise pay one timeout per affected gene. Two independent trip conditions:
#   * _CB_FAIL_THRESHOLD *consecutive* transport failures — fast detection of a
#     total outage.
#   * _CB_TOTAL_FAIL_THRESHOLD cumulative failures — catches an intermittently
#     flaky endpoint where the occasional success would otherwise keep resetting
#     the consecutive counter. A success only decrements this by one, so sustained
#     health clears it but one lucky call does not.
# State is process-wide and thread-safe.
_CB_FAIL_THRESHOLD = int(os.getenv("ENSEMBL_CB_THRESHOLD", "3"))
_CB_TOTAL_FAIL_THRESHOLD = int(os.getenv("ENSEMBL_CB_TOTAL_THRESHOLD", "8"))
_CB_COOLDOWN = float(os.getenv("ENSEMBL_CB_COOLDOWN", "60"))
_cb_lock = threading.Lock()
_cb_consecutive_failures = 0
_cb_total_failures = 0
_cb_open_until = 0.0
_cb_tripped_ever = False


def _circuit_open() -> bool:
    """True while the Ensembl circuit breaker is tripped (skip all lookups)."""
    with _cb_lock:
        return time.monotonic() < _cb_open_until


def _record_ensembl_result(reachable: bool) -> None:
    """Feed the circuit breaker. ``reachable`` = got any HTTP response (even 404);
    False only for transport errors (timeout, SSL, connection refused)."""
    global _cb_consecutive_failures, _cb_total_failures, _cb_open_until, _cb_tripped_ever
    with _cb_lock:
        if reachable:
            _cb_consecutive_failures = 0
            _cb_total_failures = max(0, _cb_total_failures - 1)
            if _cb_total_failures == 0:
                _cb_open_until = 0.0
        else:
            _cb_consecutive_failures += 1
            _cb_total_failures += 1
            if (_cb_consecutive_failures >= _CB_FAIL_THRESHOLD
                    or _cb_total_failures >= _CB_TOTAL_FAIL_THRESHOLD):
                _cb_open_until = time.monotonic() + _CB_COOLDOWN
                _cb_tripped_ever = True


def ensembl_unreachable() -> bool:
    """True if the Ensembl circuit breaker is currently open OR has tripped at
    least once this process. Used by the workflow to flag degraded enrichment
    in the report."""
    with _cb_lock:
        return _cb_tripped_ever or time.monotonic() < _cb_open_until


def reset_circuit_breaker() -> None:
    """Reset circuit-breaker state (test helper / manual recovery)."""
    global _cb_consecutive_failures, _cb_total_failures, _cb_open_until, _cb_tripped_ever
    with _cb_lock:
        _cb_consecutive_failures = 0
        _cb_total_failures = 0
        _cb_open_until = 0.0
        _cb_tripped_ever = False


# Common informal/clinical gene names that do not match the official HGNC
# symbol CASCADE's networks and Ensembl lookups expect (e.g. a TCGA network
# contains "ERBB2", not "HER2", so an unresolved alias fails exact-match
# lookup even though the gene is the correct one).
GENE_SYMBOL_ALIASES = {
    "HER2": "ERBB2",
    "HER-2": "ERBB2",
    "HER2/NEU": "ERBB2",
    "NEU": "ERBB2",
    "P53": "TP53",
    "ER": "ESR1",
    "ER-ALPHA": "ESR1",
    "ERALPHA": "ESR1",
    "HDM2": "MDM2",
    "C-MYC": "MYC",
    "CMYC": "MYC",
    "PD-L1": "CD274",
    "PDL1": "CD274",
    "HER3": "ERBB3",
}


class GeneIDMapper:
    """Maps between gene symbols and Ensembl IDs using Ensembl REST API"""

    def __init__(self, cache_file: str = None):
        if cache_file is None:
            cache_dir = Path(__file__).parent.parent / "cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = str(cache_dir / "gene_id_cache.json")
        self.cache_file = cache_file
        self.cache = self._load_cache()
        logger.info("Gene mapping initialized: %d genes cached", len(self.cache['symbol_to_ensembl']))

    def _load_cache(self) -> Dict:
        """Load cached mappings from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"symbol_to_ensembl": {}, "ensembl_to_symbol": {}}

    def _save_cache(self):
        """Save cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning("Warning: Could not save cache: %s", e)

    def resolve_alias(self, gene_symbol: str) -> str:
        """Map a common informal/clinical gene name to its official HGNC symbol.

        Returns the input uppercased and unchanged if it is not a known alias.
        """
        gene_upper = gene_symbol.upper()
        return GENE_SYMBOL_ALIASES.get(gene_upper, gene_upper)

    def symbol_to_ensembl(self, gene_symbol: str) -> Optional[str]:
        """Convert gene symbol to Ensembl ID"""
        # If already an Ensembl ID, return as-is
        if gene_symbol.upper().startswith("ENSG"):
            return gene_symbol.upper()

        gene_symbol = self.resolve_alias(gene_symbol)

        # Check cache first
        gene_upper = gene_symbol.upper()
        if gene_upper in self.cache["symbol_to_ensembl"]:
            return self.cache["symbol_to_ensembl"][gene_upper]

        # Skip the call entirely if Ensembl is known-unreachable this process.
        if _circuit_open():
            logger.debug("Ensembl circuit breaker open — skipping lookup for %s", gene_symbol)
            return None

        # Query Ensembl API (rate-limited)
        try:
            url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}"
            headers = {"Content-Type": "application/json"}
            with _ensembl_semaphore:
                response = requests.get(url, headers=headers, timeout=_ENSEMBL_TIMEOUT, verify=_SSL_VERIFY)

            # Any HTTP response (incl. 404 "gene not found") means Ensembl is reachable.
            _record_ensembl_result(reachable=True)

            if response.status_code == 200:
                data = response.json()
                ensembl_id = data.get("id")
                if ensembl_id:
                    # Cache the result
                    self.cache["symbol_to_ensembl"][gene_upper] = ensembl_id
                    self.cache["ensembl_to_symbol"][ensembl_id] = gene_upper
                    self._save_cache()
                    return ensembl_id
        except Exception as e:
            _record_ensembl_result(reachable=False)
            logger.warning("Error querying Ensembl API for %s: %s", gene_symbol, e)

        return None

    def ensembl_to_symbol(self, ensembl_id: str) -> Optional[str]:
        """Convert Ensembl ID to gene symbol"""
        # Check cache first
        if ensembl_id in self.cache["ensembl_to_symbol"]:
            return self.cache["ensembl_to_symbol"][ensembl_id]

        # Skip the call entirely if Ensembl is known-unreachable this process.
        if _circuit_open():
            logger.debug("Ensembl circuit breaker open — skipping lookup for %s", ensembl_id)
            return None

        # Query Ensembl API (rate-limited)
        try:
            url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}"
            headers = {"Content-Type": "application/json"}
            with _ensembl_semaphore:
                response = requests.get(url, headers=headers, timeout=_ENSEMBL_TIMEOUT, verify=_SSL_VERIFY)

            # Any HTTP response (incl. 404) means Ensembl is reachable.
            _record_ensembl_result(reachable=True)

            if response.status_code == 200:
                data = response.json()
                gene_symbol = data.get("display_name")
                if gene_symbol:
                    # Cache the result
                    self.cache["ensembl_to_symbol"][ensembl_id] = gene_symbol.upper()
                    self.cache["symbol_to_ensembl"][gene_symbol.upper()] = ensembl_id
                    self._save_cache()
                    return gene_symbol.upper()
        except Exception as e:
            _record_ensembl_result(reachable=False)
            logger.warning("Error querying Ensembl API for %s: %s", ensembl_id, e)

        return None

    def batch_symbol_to_ensembl(self, gene_symbols: List[str]) -> Dict[str, str]:
        """Convert multiple gene symbols to Ensembl IDs"""
        result = {}
        for symbol in gene_symbols:
            ensembl_id = self.symbol_to_ensembl(symbol)
            if ensembl_id:
                result[symbol.upper()] = ensembl_id
        return result

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cached_symbols": len(self.cache["symbol_to_ensembl"]),
            "cached_ensembls": len(self.cache["ensembl_to_symbol"]),
            "cache_file": self.cache_file
        }


# Module-level singleton
_mapper: Optional[GeneIDMapper] = None


def get_mapper() -> GeneIDMapper:
    """Get or create the singleton GeneIDMapper instance"""
    global _mapper
    if _mapper is None:
        _mapper = GeneIDMapper()
    return _mapper


# Test common genes
def test_mapper():
    mapper = GeneIDMapper()

    test_genes = ["APC", "TP53", "BRCA1", "MYC", "GAPDH"]
    print("Testing gene symbol to Ensembl ID conversion:")

    for gene in test_genes:
        ensembl_id = mapper.symbol_to_ensembl(gene)
        print(f"  {gene} -> {ensembl_id}")

    print(f"\nCache stats: {mapper.get_cache_stats()}")


if __name__ == "__main__":
    test_mapper()
