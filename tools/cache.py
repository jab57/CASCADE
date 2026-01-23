"""
Embedding Cache Module

Provides caching for precomputed embedding similarities to improve
performance for frequently queried genes.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Optional
import pandas as pd

if TYPE_CHECKING:
    from tools.model_inference import GREmLNModel


class EmbeddingCache:
    """
    LRU cache for precomputed embedding similarities.

    Stores similarity DataFrames for frequently queried genes to avoid
    recomputing cosine similarities on every request.
    """

    def __init__(self, model: "GREmLNModel", cache_size: int = 1000):
        """
        Initialize the embedding cache.

        Args:
            model: GREmLNModel instance for computing similarities
            cache_size: Maximum number of genes to cache similarities for
        """
        self.model = model
        self.cache_size = cache_size
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get_similarities(self, gene: str) -> Optional[pd.DataFrame]:
        """
        Get cached similarities or compute and cache them.

        Args:
            gene: Ensembl ID of the query gene

        Returns:
            DataFrame with columns ['ensembl_id', 'similarity'] or None
        """
        if gene in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(gene)
            self._hits += 1
            return self._cache[gene]

        # Cache miss - compute similarities
        self._misses += 1
        similarities = self.model.get_all_similarities(gene)

        if similarities is not None:
            # Evict oldest entry if at capacity
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)

            self._cache[gene] = similarities

        return similarities

    def get_top_similar(
        self,
        gene: str,
        top_k: int = 50
    ) -> Optional[pd.DataFrame]:
        """
        Get top-k similar genes (uses cache).

        Args:
            gene: Ensembl ID of the query gene
            top_k: Number of similar genes to return

        Returns:
            DataFrame with top-k similar genes
        """
        all_sims = self.get_similarities(gene)
        if all_sims is None:
            return None
        return all_sims.head(top_k)

    def preload(self, genes: list[str]) -> int:
        """
        Preload cache with similarities for a list of genes.

        Args:
            genes: List of Ensembl IDs to preload

        Returns:
            Number of genes successfully loaded
        """
        loaded = 0
        for gene in genes:
            if gene not in self._cache:
                sims = self.model.get_all_similarities(gene)
                if sims is not None:
                    if len(self._cache) >= self.cache_size:
                        self._cache.popitem(last=False)
                    self._cache[gene] = sims
                    loaded += 1
        return loaded

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "cached_genes": len(self._cache),
            "cache_size_limit": self.cache_size,
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 4),
        }

    def contains(self, gene: str) -> bool:
        """Check if a gene is in the cache."""
        return gene in self._cache

    def invalidate(self, gene: str) -> bool:
        """
        Remove a specific gene from cache.

        Args:
            gene: Ensembl ID to remove

        Returns:
            True if gene was in cache and removed
        """
        if gene in self._cache:
            del self._cache[gene]
            return True
        return False


# Module-level singleton
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(model: "GREmLNModel" = None) -> EmbeddingCache:
    """
    Get or create the singleton EmbeddingCache instance.

    Args:
        model: GREmLNModel instance (required on first call)

    Returns:
        EmbeddingCache instance
    """
    global _embedding_cache

    if _embedding_cache is None:
        if model is None:
            raise ValueError("model is required when creating cache for first time")
        _embedding_cache = EmbeddingCache(model)

    return _embedding_cache


def reset_embedding_cache():
    """Reset the singleton cache instance."""
    global _embedding_cache
    _embedding_cache = None
