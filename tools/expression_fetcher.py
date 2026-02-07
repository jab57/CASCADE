"""
Expression Data Fetcher (Future Implementation)

This module will provide utilities to automatically fetch baseline expression
data for different cell types from public databases.

STATUS: PLANNED - Not yet implemented
"""

from pathlib import Path
from typing import Optional
import pandas as pd

# Mapping of our cell types to standardized ontology terms
CELL_TYPE_MAPPING = {
    "epithelial_cell": {
        "cellxgene": "epithelial cell",
        "hpa_tissue": "epithelium",
        "cl_ontology": "CL:0000066",
    },
    "cd4_t_cells": {
        "cellxgene": "CD4-positive, alpha-beta T cell",
        "hpa_tissue": "T-cells",
        "cl_ontology": "CL:0000624",
    },
    "cd8_t_cells": {
        "cellxgene": "CD8-positive, alpha-beta T cell",
        "hpa_tissue": "T-cells",
        "cl_ontology": "CL:0000625",
    },
    "cd14_monocytes": {
        "cellxgene": "CD14-positive monocyte",
        "hpa_tissue": "monocyte",
        "cl_ontology": "CL:0002057",
    },
    "cd16_monocytes": {
        "cellxgene": "CD16-positive monocyte",
        "hpa_tissue": "monocyte",
        "cl_ontology": "CL:0002396",
    },
    "cd20_b_cells": {
        "cellxgene": "B cell",
        "hpa_tissue": "B-cells",
        "cl_ontology": "CL:0000236",
    },
    "nk_cells": {
        "cellxgene": "natural killer cell",
        "hpa_tissue": "NK-cells",
        "cl_ontology": "CL:0000623",
    },
    "nkt_cells": {
        "cellxgene": "natural killer T cell",
        "hpa_tissue": "NKT-cells",
        "cl_ontology": "CL:0000814",
    },
    "erythrocytes": {
        "cellxgene": "erythrocyte",
        "hpa_tissue": "erythrocyte",
        "cl_ontology": "CL:0000232",
    },
    "monocyte-derived_dendritic_cells": {
        "cellxgene": "monocyte-derived dendritic cell",
        "hpa_tissue": "dendritic cells",
        "cl_ontology": "CL:0011031",
    },
}


class ExpressionFetcher:
    """
    Fetches baseline expression data for cell types from public databases.

    Potential data sources:
    1. CellxGene Census - Single-cell RNA-seq data (requires tiledbsoma)
    2. Human Protein Atlas - Bulk RNA expression by cell type
    3. Tabula Sapiens - Human cell atlas
    4. PanglaoDB - Cell type markers

    The fetched data will be cached locally for reuse.
    """

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent / "cache" / "expression"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_baseline_expression(
        self,
        cell_type: str,
        source: str = "auto"
    ) -> Optional[pd.DataFrame]:
        """
        Get baseline expression profile for a cell type.

        Args:
            cell_type: One of our supported cell types (e.g., "epithelial_cell")
            source: Data source ("cellxgene", "hpa", "auto")

        Returns:
            DataFrame with columns ['gene_id', 'expression'] where:
            - gene_id is Ensembl ID
            - expression is normalized expression value (TPM or similar)

        TODO: Implement actual data fetching
        """
        raise NotImplementedError(
            "Expression fetching not yet implemented."
        )

    def _fetch_from_cellxgene(self, cell_type: str) -> pd.DataFrame:
        """
        Fetch from CellxGene Census API.

        Approach:
        1. pip install cellxgene-census (requires tiledbsoma)
        2. Query cells of specified type
        3. Compute mean expression across cells
        4. Return as gene_id -> expression mapping

        Example (pseudocode):
        ```python
        import cellxgene_census

        with cellxgene_census.open_soma() as census:
            adata = cellxgene_census.get_anndata(
                census=census,
                organism="Homo sapiens",
                obs_value_filter=f"cell_type == '{CELL_TYPE_MAPPING[cell_type]['cellxgene']}'",
                var_value_filter="feature_biotype == 'gene'",
            )
            # Compute mean expression per gene
            mean_expr = adata.X.mean(axis=0)
            return pd.DataFrame({
                'gene_id': adata.var_names,
                'expression': mean_expr
            })
        ```
        """
        raise NotImplementedError("CellxGene fetching not implemented")

    def _fetch_from_hpa(self, cell_type: str) -> pd.DataFrame:
        """
        Fetch from Human Protein Atlas API.

        Approach:
        1. Use HPA's downloadable RNA expression data
        2. Filter by cell type
        3. Return as gene_id -> expression mapping

        Data URL: https://www.proteinatlas.org/download/rna_single_cell_type.tsv.zip
        """
        raise NotImplementedError("HPA fetching not implemented")

    def _fetch_from_tabula_sapiens(self, cell_type: str) -> pd.DataFrame:
        """
        Fetch from Tabula Sapiens dataset.

        Approach:
        1. Download pre-computed cell type averages
        2. Map to our cell types
        3. Return expression profile

        Data: https://tabula-sapiens-portal.ds.czbiohub.org/
        """
        raise NotImplementedError("Tabula Sapiens fetching not implemented")


# =============================================================================
# FUTURE IMPLEMENTATION NOTES
# =============================================================================
"""
IMPLEMENTATION PLAN:

1. SHORT-TERM (Human Protein Atlas):
   - Download: https://www.proteinatlas.org/download/rna_single_cell_type.tsv.zip
   - Parse TSV file, filter by cell type
   - Map gene names to Ensembl IDs
   - Cache locally
   - Pros: Simple, pre-computed averages, no dependencies
   - Cons: Limited cell types, bulk-like data

2. MEDIUM-TERM (CellxGene Census):
   - Requires: pip install cellxgene-census tiledbsoma
   - Query API for specific cell types
   - Compute mean expression across cells
   - Pros: Most comprehensive, true single-cell
   - Cons: Large downloads, complex dependencies

3. INTEGRATION WITH MODEL:
   Once we have expression data, we can:
   a) Use it as input to get context-aware gene embeddings
   b) Use it as baseline for perturbation predictions
   c) Rank genes and use rank embeddings from model

EXPRESSION FORMAT FOR MODEL:
- The GDTransformer expects expression as "ranks" (0-99 bins)
- Need to convert TPM/counts to rank bins
- See scGraphLLM/tokenizer.py for binning logic
"""
