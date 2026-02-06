"""Tests for tools/perturb.py — network perturbation and BFS propagation."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from tools.perturb import (
    _build_adjacency,
    _build_reverse_adjacency,
    _propagate_effect,
    simulate_knockdown,
    simulate_overexpression,
    get_regulators,
    get_targets,
)


class TestBuildAdjacency:
    def test_forward_adjacency(self, mock_network_df):
        adj = _build_adjacency(mock_network_df)
        assert "ENSG_TF1" in adj
        assert len(adj["ENSG_TF1"]) == 2
        targets = [t for t, _ in adj["ENSG_TF1"]]
        assert "ENSG_TARGET1" in targets
        assert "ENSG_TARGET2" in targets

    def test_weights_are_floats(self, mock_network_df):
        adj = _build_adjacency(mock_network_df)
        for targets in adj.values():
            for _, weight in targets:
                assert isinstance(weight, float)

    def test_reverse_adjacency(self, mock_network_df):
        rev = _build_reverse_adjacency(mock_network_df)
        # TARGET2 is regulated by both TF1 and TF2
        assert "ENSG_TARGET2" in rev
        regulators = [r for r, _ in rev["ENSG_TARGET2"]]
        assert "ENSG_TF1" in regulators
        assert "ENSG_TF2" in regulators


class TestPropagateEffect:
    def test_depth_1_direct_only(self, mock_network_df):
        adj = _build_adjacency(mock_network_df)
        effects = _propagate_effect(adj, "ENSG_TF1", initial_effect=-1.0, depth=1)
        # Direct targets: TARGET1, TARGET2
        assert "ENSG_TARGET1" in effects
        assert "ENSG_TARGET2" in effects
        # Should NOT reach DOWNSTREAM1 at depth 1
        assert "ENSG_DOWNSTREAM1" not in effects

    def test_depth_2_indirect(self, mock_network_df):
        adj = _build_adjacency(mock_network_df)
        effects = _propagate_effect(adj, "ENSG_TF1", initial_effect=-1.0, depth=2)
        # At depth 2, should reach DOWNSTREAM1 via TARGET1
        assert "ENSG_DOWNSTREAM1" in effects

    def test_effect_direction(self, mock_network_df):
        adj = _build_adjacency(mock_network_df)
        effects = _propagate_effect(adj, "ENSG_TF1", initial_effect=-1.0, depth=1)
        # Knockdown produces negative effects (decay * weight * -1)
        assert effects["ENSG_TARGET1"] < 0

    def test_effect_magnitude_scales_with_weight(self, mock_network_df):
        adj = _build_adjacency(mock_network_df)
        effects = _propagate_effect(adj, "ENSG_TF1", initial_effect=-1.0, depth=1)
        # TARGET1 has weight 0.8, TARGET2 has weight 0.6
        assert abs(effects["ENSG_TARGET1"]) > abs(effects["ENSG_TARGET2"])

    def test_decay_reduces_effect(self, mock_network_df):
        adj = _build_adjacency(mock_network_df)
        effects_low_decay = _propagate_effect(adj, "ENSG_TF1", -1.0, depth=1, decay=0.2)
        effects_high_decay = _propagate_effect(adj, "ENSG_TF1", -1.0, depth=1, decay=0.8)
        assert abs(effects_low_decay["ENSG_TARGET1"]) < abs(effects_high_decay["ENSG_TARGET1"])

    def test_no_targets_returns_only_start(self, mock_network_df):
        adj = _build_adjacency(mock_network_df)
        # DOWNSTREAM1 has no outgoing edges
        effects = _propagate_effect(adj, "ENSG_DOWNSTREAM1", -1.0, depth=2)
        assert len(effects) == 1
        assert "ENSG_DOWNSTREAM1" in effects

    def test_overexpression_positive(self, mock_network_df):
        adj = _build_adjacency(mock_network_df)
        effects = _propagate_effect(adj, "ENSG_TF1", initial_effect=1.0, depth=1)
        assert effects["ENSG_TARGET1"] > 0


class TestSimulateKnockdown:
    @patch("tools.perturb.get_mapper")
    def test_basic_knockdown(self, mock_mapper_fn, mock_network_df):
        mapper = MagicMock()
        mapper.ensembl_to_symbol.side_effect = lambda x: x.replace("ENSG_", "")
        mock_mapper_fn.return_value = mapper

        result = simulate_knockdown(mock_network_df, "ENSG_TF1", depth=2)
        assert result["status"] == "complete"
        assert result["perturbation_type"] == "knockdown"
        assert result["total_affected_genes"] > 0
        assert len(result["top_affected_genes"]) > 0

    @patch("tools.perturb.get_mapper")
    def test_knockdown_gene_not_in_network(self, mock_mapper_fn, mock_network_df):
        result = simulate_knockdown(mock_network_df, "ENSG_FAKE_GENE")
        assert result["status"] == "error"
        assert "not found" in result["error"]

    @patch("tools.perturb.get_mapper")
    def test_knockdown_effector_no_targets(self, mock_mapper_fn, mock_network_df):
        mapper = MagicMock()
        mapper.ensembl_to_symbol.side_effect = lambda x: x
        mock_mapper_fn.return_value = mapper

        # DOWNSTREAM1 has no outgoing edges
        result = simulate_knockdown(mock_network_df, "ENSG_DOWNSTREAM1", depth=1)
        assert result["status"] == "complete"
        assert result["affected_genes"] == []

    @patch("tools.perturb.get_mapper")
    def test_affected_genes_have_required_fields(self, mock_mapper_fn, mock_network_df):
        mapper = MagicMock()
        mapper.ensembl_to_symbol.side_effect = lambda x: x.replace("ENSG_", "")
        mock_mapper_fn.return_value = mapper

        result = simulate_knockdown(mock_network_df, "ENSG_TF1")
        for gene in result["top_affected_genes"]:
            assert "ensembl_id" in gene
            assert "symbol" in gene
            assert "predicted_effect" in gene
            assert "direction" in gene
            assert "magnitude" in gene

    @patch("tools.perturb.get_mapper")
    def test_top_k_limits_results(self, mock_mapper_fn, mock_network_df):
        mapper = MagicMock()
        mapper.ensembl_to_symbol.side_effect = lambda x: x
        mock_mapper_fn.return_value = mapper

        result = simulate_knockdown(mock_network_df, "ENSG_TF1", top_k=1)
        assert len(result["top_affected_genes"]) == 1


class TestSimulateOverexpression:
    @patch("tools.perturb.get_mapper")
    def test_basic_overexpression(self, mock_mapper_fn, mock_network_df):
        mapper = MagicMock()
        mapper.ensembl_to_symbol.side_effect = lambda x: x.replace("ENSG_", "")
        mock_mapper_fn.return_value = mapper

        result = simulate_overexpression(mock_network_df, "ENSG_TF1", fold_change=2.0)
        assert result["status"] == "complete"
        assert result["perturbation_type"] == "overexpression"
        assert result["fold_change"] == 2.0

    @patch("tools.perturb.get_mapper")
    def test_overexpression_direction_is_up(self, mock_mapper_fn, mock_network_df):
        mapper = MagicMock()
        mapper.ensembl_to_symbol.side_effect = lambda x: x
        mock_mapper_fn.return_value = mapper

        result = simulate_overexpression(mock_network_df, "ENSG_TF1", fold_change=2.0)
        for gene in result["top_affected_genes"]:
            # Overexpression should produce positive (up) effects
            assert gene["direction"] == "up"

    @patch("tools.perturb.get_mapper")
    def test_fold_change_affects_magnitude(self, mock_mapper_fn, mock_network_df):
        mapper = MagicMock()
        mapper.ensembl_to_symbol.side_effect = lambda x: x
        mock_mapper_fn.return_value = mapper

        result_2x = simulate_overexpression(mock_network_df, "ENSG_TF1", fold_change=2.0)
        result_4x = simulate_overexpression(mock_network_df, "ENSG_TF1", fold_change=4.0)
        mag_2x = result_2x["top_affected_genes"][0]["magnitude"]
        mag_4x = result_4x["top_affected_genes"][0]["magnitude"]
        assert mag_4x > mag_2x


class TestGetRegulators:
    @patch("tools.perturb.get_mapper")
    def test_find_regulators(self, mock_mapper_fn, mock_network_df):
        mapper = MagicMock()
        mapper.ensembl_to_symbol.side_effect = lambda x: x.replace("ENSG_", "")
        mock_mapper_fn.return_value = mapper

        result = get_regulators(mock_network_df, "ENSG_TARGET2")
        assert result["status"] == "complete"
        assert result["num_regulators"] == 2
        reg_ids = [r["ensembl_id"] for r in result["regulators"]]
        assert "ENSG_TF1" in reg_ids
        assert "ENSG_TF2" in reg_ids

    @patch("tools.perturb.get_mapper")
    def test_no_regulators(self, mock_mapper_fn, mock_network_df):
        result = get_regulators(mock_network_df, "ENSG_TF1")
        # TF1 has no upstream regulators in our mock network
        assert result["status"] == "not_found"


class TestGetTargets:
    @patch("tools.perturb.get_mapper")
    def test_find_targets(self, mock_mapper_fn, mock_network_df):
        mapper = MagicMock()
        mapper.ensembl_to_symbol.side_effect = lambda x: x.replace("ENSG_", "")
        mock_mapper_fn.return_value = mapper

        result = get_targets(mock_network_df, "ENSG_TF1")
        assert result["status"] == "complete"
        assert result["num_targets"] == 2

    @patch("tools.perturb.get_mapper")
    def test_no_targets(self, mock_mapper_fn, mock_network_df):
        result = get_targets(mock_network_df, "ENSG_DOWNSTREAM1")
        assert result["status"] == "not_found"
