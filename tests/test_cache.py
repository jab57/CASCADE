"""Tests for tools/cache.py — embedding similarity LRU cache."""

import pytest
import pandas as pd

from tools.cache import EmbeddingCache, reset_embedding_cache


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before each test."""
    reset_embedding_cache()
    yield
    reset_embedding_cache()


class TestEmbeddingCache:
    def test_cache_miss_then_hit(self, mock_cascade_model):
        cache = EmbeddingCache(mock_cascade_model, cache_size=100)

        # First call: miss
        result1 = cache.get_similarities("ENSG_TF1")
        assert result1 is not None
        assert cache._misses == 1
        assert cache._hits == 0

        # Second call: hit
        result2 = cache.get_similarities("ENSG_TF1")
        assert result2 is not None
        assert cache._hits == 1

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_lru_eviction(self, mock_cascade_model):
        cache = EmbeddingCache(mock_cascade_model, cache_size=2)

        cache.get_similarities("ENSG_TF1")
        cache.get_similarities("ENSG_TF2")
        assert cache.contains("ENSG_TF1")
        assert cache.contains("ENSG_TF2")

        # Adding a third should evict the oldest (TF1)
        cache.get_similarities("ENSG_TARGET1")
        assert not cache.contains("ENSG_TF1")
        assert cache.contains("ENSG_TF2")
        assert cache.contains("ENSG_TARGET1")

    def test_lru_access_refreshes(self, mock_cascade_model):
        cache = EmbeddingCache(mock_cascade_model, cache_size=2)

        cache.get_similarities("ENSG_TF1")
        cache.get_similarities("ENSG_TF2")

        # Access TF1 again to refresh it
        cache.get_similarities("ENSG_TF1")

        # Now TF2 should be evicted (oldest), not TF1
        cache.get_similarities("ENSG_TARGET1")
        assert cache.contains("ENSG_TF1")
        assert not cache.contains("ENSG_TF2")

    def test_get_top_similar(self, mock_cascade_model):
        cache = EmbeddingCache(mock_cascade_model, cache_size=100)
        result = cache.get_top_similar("ENSG_TF1", top_k=2)
        assert result is not None
        assert len(result) <= 2

    def test_unknown_gene_returns_none(self, mock_cascade_model):
        cache = EmbeddingCache(mock_cascade_model, cache_size=100)
        result = cache.get_similarities("ENSG_NONEXISTENT")
        assert result is None

    def test_invalidate(self, mock_cascade_model):
        cache = EmbeddingCache(mock_cascade_model, cache_size=100)
        cache.get_similarities("ENSG_TF1")
        assert cache.contains("ENSG_TF1")

        removed = cache.invalidate("ENSG_TF1")
        assert removed is True
        assert not cache.contains("ENSG_TF1")

    def test_invalidate_missing_gene(self, mock_cascade_model):
        cache = EmbeddingCache(mock_cascade_model, cache_size=100)
        assert cache.invalidate("ENSG_NONEXISTENT") is False

    def test_clear(self, mock_cascade_model):
        cache = EmbeddingCache(mock_cascade_model, cache_size=100)
        cache.get_similarities("ENSG_TF1")
        cache.get_similarities("ENSG_TF2")

        cache.clear()
        assert not cache.contains("ENSG_TF1")
        assert not cache.contains("ENSG_TF2")
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["cached_genes"] == 0

    def test_stats(self, mock_cascade_model):
        cache = EmbeddingCache(mock_cascade_model, cache_size=100)
        cache.get_similarities("ENSG_TF1")  # miss
        cache.get_similarities("ENSG_TF1")  # hit

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["cached_genes"] == 1

    def test_preload(self, mock_cascade_model):
        cache = EmbeddingCache(mock_cascade_model, cache_size=100)
        loaded = cache.preload(["ENSG_TF1", "ENSG_TF2", "ENSG_TARGET1"])
        assert loaded == 3
        assert cache.contains("ENSG_TF1")
        assert cache.contains("ENSG_TF2")
        assert cache.contains("ENSG_TARGET1")
