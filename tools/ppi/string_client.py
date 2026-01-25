"""
STRING API Client for Protein-Protein Interaction Data

Queries the STRING database (https://string-db.org) to retrieve
protein-protein interaction networks for human genes.
"""

import requests
from typing import Optional
from dataclasses import dataclass


@dataclass
class ProteinInteraction:
    """A protein-protein interaction from STRING."""
    partner: str  # Gene symbol of interaction partner
    partner_ensembl: str  # Ensembl protein ID
    combined_score: int  # Overall confidence (0-1000)
    experimental: int  # Experimental evidence score
    database: int  # Database annotation score
    textmining: int  # Text mining score
    coexpression: int  # Co-expression score


class STRINGClient:
    """Client for querying STRING protein-protein interaction database."""

    BASE_URL = "https://string-db.org/api"
    SPECIES_HUMAN = 9606  # NCBI taxonomy ID for Homo sapiens

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def get_interactions(
        self,
        gene_symbol: str,
        min_score: int = 400,
        limit: int = 50
    ) -> dict:
        """
        Get protein-protein interactions for a gene from STRING.

        Args:
            gene_symbol: Gene symbol (e.g., "APC", "TP53")
            min_score: Minimum combined score (0-1000). Default 400 = medium confidence.
                      150 = low, 400 = medium, 700 = high, 900 = highest
            limit: Maximum number of interactions to return

        Returns:
            Dict with interaction data and metadata
        """
        # First, resolve gene symbol to STRING protein ID
        protein_id = self._resolve_to_string_id(gene_symbol)
        if protein_id is None:
            return {
                "error": f"Could not find protein '{gene_symbol}' in STRING database",
                "suggestion": "Check gene symbol spelling or try an alias"
            }

        # Get interaction partners
        url = f"{self.BASE_URL}/json/network"
        params = {
            "identifiers": protein_id,
            "species": self.SPECIES_HUMAN,
            "required_score": min_score,
            "limit": limit,
        }

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            return {"error": f"STRING API request failed: {str(e)}"}

        if not data:
            return {
                "query_gene": gene_symbol,
                "string_id": protein_id,
                "interactions": [],
                "count": 0,
                "note": f"No interactions found above score threshold {min_score}"
            }

        # Parse interactions
        interactions = []
        seen_partners = set()

        for item in data:
            # STRING returns edges - need to identify the partner
            if item.get("preferredName_A") == gene_symbol.upper():
                partner_name = item.get("preferredName_B", "")
                partner_id = item.get("stringId_B", "")
            else:
                partner_name = item.get("preferredName_A", "")
                partner_id = item.get("stringId_A", "")

            # Skip self-interactions and duplicates
            if partner_name.upper() == gene_symbol.upper():
                continue
            if partner_name in seen_partners:
                continue
            seen_partners.add(partner_name)

            interactions.append({
                "partner": partner_name,
                "partner_string_id": partner_id,
                "combined_score": item.get("score", 0),
                "evidence": self._get_evidence_summary(item)
            })

        # Sort by score descending
        interactions.sort(key=lambda x: x["combined_score"], reverse=True)

        return {
            "query_gene": gene_symbol,
            "string_id": protein_id,
            "interactions": interactions[:limit],
            "count": len(interactions),
            "min_score_used": min_score,
            "score_interpretation": self._score_interpretation(min_score)
        }

    def _resolve_to_string_id(self, gene_symbol: str) -> Optional[str]:
        """Resolve a gene symbol to STRING protein ID."""
        url = f"{self.BASE_URL}/json/get_string_ids"
        params = {
            "identifiers": gene_symbol,
            "species": self.SPECIES_HUMAN,
            "limit": 1,
        }

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if data and len(data) > 0:
                return data[0].get("stringId")
        except requests.RequestException:
            pass

        return None

    def _get_evidence_summary(self, item: dict) -> dict:
        """Extract evidence type scores from STRING response."""
        return {
            "experimental": item.get("escore", 0),
            "database": item.get("dscore", 0),
            "textmining": item.get("tscore", 0),
            "coexpression": item.get("ascore", 0),
            "neighborhood": item.get("nscore", 0),
        }

    def _score_interpretation(self, score: int) -> str:
        """Interpret the confidence score threshold."""
        if score >= 900:
            return "highest confidence only"
        elif score >= 700:
            return "high confidence"
        elif score >= 400:
            return "medium confidence"
        elif score >= 150:
            return "low confidence"
        else:
            return "all interactions"


# Module-level singleton
_client: Optional[STRINGClient] = None


def get_string_client() -> STRINGClient:
    """Get or create singleton STRING client."""
    global _client
    if _client is None:
        _client = STRINGClient()
    return _client
