"""Unit tests for tools/depmap.py gene co-dependency analysis.

Uses synthetic Chronos score data (via a patched load_depmap_data) so these
tests do not depend on the real DepMap CSVs being present, and do not touch
the existing get_gene_essentiality code path.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from tools.depmap import get_gene_codependency


def _make_scores(n=200, seed=0):
    """Synthetic cell-line x gene Chronos score matrix."""
    rng = np.random.default_rng(seed)
    base = rng.normal(-0.3, 0.5, n)

    # GENE_A / GENE_B: strongly co-dependent (shared signal + small noise)
    gene_a = base + rng.normal(0, 0.05, n)
    gene_b = base + rng.normal(0, 0.05, n)

    # GENE_C: anti-correlated with GENE_A
    gene_c = -base + rng.normal(0, 0.05, n)

    # GENE_D / GENE_E: independent random noise, no relationship
    gene_d = rng.normal(-0.2, 0.4, n)
    gene_e = rng.normal(-0.2, 0.4, n)

    index = [f"ACH-{i:06d}" for i in range(n)]
    scores = pd.DataFrame(
        {
            "GENE_A": gene_a,
            "GENE_B": gene_b,
            "GENE_C": gene_c,
            "GENE_D": gene_d,
            "GENE_E": gene_e,
        },
        index=index,
    )
    # Introduce a few NaNs to exercise the matched-cell-line alignment
    scores.loc[index[0], "GENE_B"] = np.nan
    scores.loc[index[1], "GENE_A"] = np.nan

    lineage = pd.Series(["lung"] * n, index=index)
    return scores, lineage


class TestGeneCodependency:

    def test_strong_positive_codependency(self):
        scores, lineage = _make_scores()
        with patch("tools.depmap.load_depmap_data", return_value=(scores, lineage)):
            result = get_gene_codependency("GENE_A", "GENE_B")

        assert result["not_found"] is False
        assert result["gene_a"] == "GENE_A"
        assert result["gene_b"] == "GENE_B"
        assert result["pearson_r"] > 0.8
        assert result["p_value"] < 0.05
        assert "co-dependency" in result["interpretation"].lower()

    def test_negative_correlation(self):
        scores, lineage = _make_scores()
        with patch("tools.depmap.load_depmap_data", return_value=(scores, lineage)):
            result = get_gene_codependency("GENE_A", "GENE_C")

        assert result["not_found"] is False
        assert result["pearson_r"] < -0.8
        assert result["p_value"] < 0.05

    def test_no_relationship_not_significant(self):
        scores, lineage = _make_scores()
        with patch("tools.depmap.load_depmap_data", return_value=(scores, lineage)):
            result = get_gene_codependency("GENE_D", "GENE_E")

        assert result["not_found"] is False
        assert result["p_value"] >= 0.05
        assert "no significant" in result["interpretation"].lower()

    def test_matched_cell_line_count_excludes_nans(self):
        scores, lineage = _make_scores(n=200)
        with patch("tools.depmap.load_depmap_data", return_value=(scores, lineage)):
            result = get_gene_codependency("GENE_A", "GENE_B")

        # Two rows have a NaN in either GENE_A or GENE_B
        assert result["n_cell_lines"] == 198

    def test_gene_lookup_is_case_insensitive(self):
        scores, lineage = _make_scores()
        with patch("tools.depmap.load_depmap_data", return_value=(scores, lineage)):
            result = get_gene_codependency("gene_a", "gene_b")

        assert result["not_found"] is False
        assert result["pearson_r"] > 0.8

    def test_gene_not_found(self):
        scores, lineage = _make_scores()
        with patch("tools.depmap.load_depmap_data", return_value=(scores, lineage)):
            result = get_gene_codependency("GENE_A", "NOT_A_REAL_GENE")

        assert result["not_found"] is True
        assert "NOT_A_REAL_GENE" in result["error"]

    def test_both_genes_not_found(self):
        scores, lineage = _make_scores()
        with patch("tools.depmap.load_depmap_data", return_value=(scores, lineage)):
            result = get_gene_codependency("FAKE_1", "FAKE_2")

        assert result["not_found"] is True
        assert "FAKE_1" in result["error"]
        assert "FAKE_2" in result["error"]

    def test_insufficient_matched_cell_lines(self):
        index = ["ACH-000001", "ACH-000002"]
        scores = pd.DataFrame(
            {"GENE_A": [-0.5, -0.4], "GENE_B": [-0.6, np.nan]}, index=index
        )
        lineage = pd.Series(["lung"] * 2, index=index)
        with patch("tools.depmap.load_depmap_data", return_value=(scores, lineage)):
            result = get_gene_codependency("GENE_A", "GENE_B")

        assert result["not_found"] is True
        assert "Insufficient" in result["error"]

    def test_file_not_found_propagates(self):
        with patch("tools.depmap.load_depmap_data", side_effect=FileNotFoundError("missing")):
            result = get_gene_codependency("GENE_A", "GENE_B")

        assert result["not_found"] is True
        assert "error" in result

    def test_data_source_label(self):
        scores, lineage = _make_scores()
        with patch("tools.depmap.load_depmap_data", return_value=(scores, lineage)):
            result = get_gene_codependency("GENE_A", "GENE_B")

        assert result["data_source"] == "DepMap CRISPR (Chronos)"


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/depmap/CRISPRGeneEffect.csv").exists(),
    reason="Real DepMap data not present in this environment",
)
class TestGeneCodependencyReferenceValues:
    """Sanity checks against known reference co-dependency relationships."""

    def test_myc_max_obligate_codependency(self):
        result = get_gene_codependency("MYC", "MAX")
        assert result["not_found"] is False
        assert result["pearson_r"] == pytest.approx(0.32, abs=0.02)
        assert result["p_value"] < 1e-20

    def test_brd4_myc_known_regulatory_pair(self):
        result = get_gene_codependency("BRD4", "MYC")
        assert result["not_found"] is False
        assert result["pearson_r"] == pytest.approx(0.12, abs=0.02)
        assert result["p_value"] < 1e-3

    def test_ctnnb1_tp53_no_known_link(self):
        result = get_gene_codependency("CTNNB1", "TP53")
        assert result["not_found"] is False
        assert abs(result["pearson_r"]) < 0.05
        assert result["p_value"] >= 0.05
