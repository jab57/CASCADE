"""Tests for tools/model_inference.py — embedding loading and similarity.

These tests use the mock_cascade_model fixture to avoid needing
the actual GREmLN checkpoint (~120MB) or GPU.
"""

import pytest
from unittest.mock import patch, MagicMock

from tools.model_inference import CascadeModel, reset_model


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before each test."""
    reset_model()
    yield
    reset_model()


class TestCascadeModelVocab:
    def test_is_gene_in_vocab(self, mock_cascade_model):
        assert mock_cascade_model.is_gene_in_vocab("ENSG_TF1") is True
        assert mock_cascade_model.is_gene_in_vocab("ENSG_NONEXISTENT") is False

    def test_special_tokens_in_vocab(self, mock_cascade_model):
        assert mock_cascade_model.is_gene_in_vocab("<PAD>") is True


class TestSimilarityComputation:
    def test_compute_similarity_returns_float(self, mock_cascade_model):
        sim = mock_cascade_model.compute_similarity("ENSG_TF1", "ENSG_TF2")
        assert isinstance(sim, float)
        assert -1.0 <= sim <= 1.0

    def test_self_similarity_is_one(self, mock_cascade_model):
        sim = mock_cascade_model.compute_similarity("ENSG_TF1", "ENSG_TF1")
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_unknown_gene_returns_none(self, mock_cascade_model):
        sim = mock_cascade_model.compute_similarity("ENSG_TF1", "ENSG_FAKE")
        assert sim is None

    def test_get_all_similarities(self, mock_cascade_model):
        df = mock_cascade_model.get_all_similarities("ENSG_TF1")
        assert df is not None
        assert "ensembl_id" in df.columns
        assert "similarity" in df.columns
        # Should not contain the query gene or special tokens
        assert "ENSG_TF1" not in df["ensembl_id"].values
        assert "<PAD>" not in df["ensembl_id"].values

    def test_get_all_similarities_unknown_gene(self, mock_cascade_model):
        df = mock_cascade_model.get_all_similarities("ENSG_NONEXISTENT")
        assert df is None


class TestCascadeModelInit:
    @patch("tools.model_inference.torch")
    @patch("tools.model_inference.GeneVocab")
    def test_model_init_no_checkpoint(self, mock_vocab, mock_torch):
        """CascadeModel should raise FileNotFoundError for missing checkpoint."""
        mock_vocab.load_default.return_value = MagicMock()
        model = CascadeModel("nonexistent_path.ckpt")
        with pytest.raises(FileNotFoundError):
            model.load()

    def test_device_fallback_to_cpu(self):
        """When no GPU available, device should fall back to cpu."""
        with patch("tools.model_inference.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_vocab = MagicMock()
            with patch("tools.model_inference.GeneVocab") as MockVocab:
                MockVocab.load_default.return_value = mock_vocab
                model = CascadeModel("fake.ckpt", device=None)
                assert model.device == "cpu"
