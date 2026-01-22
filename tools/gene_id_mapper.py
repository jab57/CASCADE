#!/usr/bin/env python3
"""
Gene ID Mapper for GREmLN MCP Server
Converts between gene symbols and Ensembl IDs

Consistent with RegNetAgents gene_id_mapper API.
"""

import requests
import pickle
import os
from typing import Dict, List, Optional
from pathlib import Path


class GeneIDMapper:
    """Maps between gene symbols and Ensembl IDs using Ensembl REST API"""

    def __init__(self, cache_file: str = None):
        if cache_file is None:
            cache_dir = Path(__file__).parent.parent / "cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = str(cache_dir / "gene_id_cache.pkl")
        self.cache_file = cache_file
        self.cache = self._load_cache()
        print(f"Gene mapping initialized: {len(self.cache['symbol_to_ensembl'])} genes cached")

    def _load_cache(self) -> Dict:
        """Load cached mappings from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {"symbol_to_ensembl": {}, "ensembl_to_symbol": {}}

    def _save_cache(self):
        """Save cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def symbol_to_ensembl(self, gene_symbol: str) -> Optional[str]:
        """Convert gene symbol to Ensembl ID"""
        # If already an Ensembl ID, return as-is
        if gene_symbol.upper().startswith("ENSG"):
            return gene_symbol.upper()

        # Check cache first
        gene_upper = gene_symbol.upper()
        if gene_upper in self.cache["symbol_to_ensembl"]:
            return self.cache["symbol_to_ensembl"][gene_upper]

        # Query Ensembl API
        try:
            url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}"
            headers = {"Content-Type": "application/json"}
            response = requests.get(url, headers=headers, timeout=10)

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
            print(f"Error querying Ensembl API for {gene_symbol}: {e}")

        return None

    def ensembl_to_symbol(self, ensembl_id: str) -> Optional[str]:
        """Convert Ensembl ID to gene symbol"""
        # Check cache first
        if ensembl_id in self.cache["ensembl_to_symbol"]:
            return self.cache["ensembl_to_symbol"][ensembl_id]

        # Query Ensembl API
        try:
            url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}"
            headers = {"Content-Type": "application/json"}
            response = requests.get(url, headers=headers, timeout=10)

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
            print(f"Error querying Ensembl API for {ensembl_id}: {e}")

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
