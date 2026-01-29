"""
GREmLN Model Inference Module

Provides access to learned gene embeddings from the GREmLN model checkpoint
for enhanced perturbation analysis.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch

from scGraphLLM import GeneVocab


class GREmLNModel:
    """
    Wrapper for GREmLN model with gene embedding extraction.

    Loads the model checkpoint and provides methods to:
    - Extract gene embeddings by Ensembl ID
    - Compute cosine similarity between genes
    - Get similarity rankings for all genes
    """

    def __init__(self, checkpoint_path: Path | str, device: str = None):
        """
        Initialize the GREmLN model wrapper.

        Args:
            checkpoint_path: Path to the model.ckpt file
            device: Device to load tensors on ('cuda' or 'cpu')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = GeneVocab.load_default()
        self.gene_embeddings: Optional[torch.Tensor] = None
        self._normalized_embeddings: Optional[torch.Tensor] = None
        self._loaded = False

    def load(self) -> "GREmLNModel":
        """Load checkpoint and extract gene embeddings."""
        if self._loaded:
            return self

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        print(f"[GREmLN] Loading model checkpoint from {self.checkpoint_path}")
        print(f"[GREmLN] Using device: {self.device}")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Extract gene embeddings from state dict
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Try different possible key names for the gene embedding
        embedding_keys = [
            "gene_embedding.weight",
            "model.gene_embedding.weight",
            "encoder.gene_embedding.weight",
        ]

        for key in embedding_keys:
            if key in state_dict:
                self.gene_embeddings = state_dict[key].to(self.device)
                print(f"[GREmLN] Found gene embeddings at '{key}'")
                break

        if self.gene_embeddings is None:
            # List available keys for debugging
            available_keys = [k for k in state_dict.keys() if "embed" in k.lower()]
            raise KeyError(
                f"Could not find gene embeddings in checkpoint. "
                f"Embedding-related keys found: {available_keys}"
            )

        # Pre-compute normalized embeddings for fast cosine similarity
        self._normalized_embeddings = torch.nn.functional.normalize(
            self.gene_embeddings, p=2, dim=1
        )

        self._loaded = True
        print(f"[GREmLN] Loaded embeddings for {self.gene_embeddings.shape[0]} genes")
        print(f"[GREmLN] Embedding dimension: {self.gene_embeddings.shape[1]}")

        return self

    def _ensure_loaded(self):
        """Ensure the model is loaded before use."""
        if not self._loaded:
            self.load()

    def get_vocab_index(self, ensembl_id: str) -> Optional[int]:
        """
        Get the vocabulary index for an Ensembl ID.

        Args:
            ensembl_id: Ensembl gene ID (e.g., 'ENSG00000136997')

        Returns:
            Vocabulary index or None if not found
        """
        return self.vocab.gene_to_node.get(ensembl_id)

    def get_gene_embedding(self, ensembl_id: str) -> Optional[torch.Tensor]:
        """
        Get the embedding vector for a gene by Ensembl ID.

        Args:
            ensembl_id: Ensembl gene ID

        Returns:
            Embedding tensor of shape [embedding_dim] or None if not found
        """
        self._ensure_loaded()

        idx = self.get_vocab_index(ensembl_id)
        if idx is None:
            return None

        return self.gene_embeddings[idx]

    def compute_similarity(self, gene1: str, gene2: str) -> Optional[float]:
        """
        Compute cosine similarity between two genes.

        Args:
            gene1: First gene Ensembl ID
            gene2: Second gene Ensembl ID

        Returns:
            Cosine similarity (-1 to 1) or None if either gene not found
        """
        self._ensure_loaded()

        idx1 = self.get_vocab_index(gene1)
        idx2 = self.get_vocab_index(gene2)

        if idx1 is None or idx2 is None:
            return None

        # Use pre-normalized embeddings for efficiency
        sim = torch.dot(
            self._normalized_embeddings[idx1],
            self._normalized_embeddings[idx2]
        ).item()

        # Monitor for potential artifacts
        if abs(sim) > 0.99:
            print(f"[GREmLN] Warning: extreme similarity {sim:.4f} between {gene1} and {gene2}")

        return sim

    def get_all_similarities(self, gene: str) -> Optional[pd.DataFrame]:
        """
        Get similarity of one gene to all other genes in vocabulary.

        Args:
            gene: Ensembl ID of the query gene

        Returns:
            DataFrame with columns ['ensembl_id', 'similarity'] sorted by similarity,
            or None if gene not found
        """
        self._ensure_loaded()

        idx = self.get_vocab_index(gene)
        if idx is None:
            return None

        # Batch cosine similarity using normalized embeddings
        query_emb = self._normalized_embeddings[idx]
        similarities = torch.mv(self._normalized_embeddings, query_emb)

        # Build DataFrame with all genes
        result = pd.DataFrame({
            "ensembl_id": self.vocab.genes,
            "similarity": similarities.cpu().numpy()
        })

        # Filter out special tokens and the query gene itself
        special_tokens = {"<PAD>", "<MASK>", "<CLS>"}
        result = result[~result["ensembl_id"].isin(special_tokens)]
        result = result[result["ensembl_id"] != gene]

        # Sort by similarity descending
        result = result.sort_values("similarity", ascending=False).reset_index(drop=True)

        return result

    def get_top_similar_genes(
        self,
        gene: str,
        top_k: int = 50
    ) -> Optional[pd.DataFrame]:
        """
        Get the top-k most similar genes to a query gene.

        Args:
            gene: Ensembl ID of the query gene
            top_k: Number of similar genes to return

        Returns:
            DataFrame with columns ['ensembl_id', 'similarity']
        """
        all_sims = self.get_all_similarities(gene)
        if all_sims is None:
            return None

        return all_sims.head(top_k)

    def get_embedding_stats(self) -> dict:
        """Get statistics about the loaded embeddings."""
        self._ensure_loaded()

        return {
            "num_genes": self.gene_embeddings.shape[0],
            "embedding_dim": self.gene_embeddings.shape[1],
            "device": str(self.device),
            "dtype": str(self.gene_embeddings.dtype),
            "num_special_tokens": 3,  # PAD, MASK, CLS
            "num_actual_genes": self.gene_embeddings.shape[0] - 3,
        }

    def is_gene_in_vocab(self, ensembl_id: str) -> bool:
        """Check if a gene is in the vocabulary."""
        return ensembl_id in self.vocab.gene_to_node

    def batch_similarities(
        self,
        query_genes: list[str],
        target_genes: list[str]
    ) -> pd.DataFrame:
        """
        Compute pairwise similarities between query and target gene sets.

        Args:
            query_genes: List of query Ensembl IDs
            target_genes: List of target Ensembl IDs

        Returns:
            DataFrame with similarities (query x target matrix)
        """
        self._ensure_loaded()

        # Get indices for valid genes
        query_indices = []
        valid_query_genes = []
        for g in query_genes:
            idx = self.get_vocab_index(g)
            if idx is not None:
                query_indices.append(idx)
                valid_query_genes.append(g)

        target_indices = []
        valid_target_genes = []
        for g in target_genes:
            idx = self.get_vocab_index(g)
            if idx is not None:
                target_indices.append(idx)
                valid_target_genes.append(g)

        if not query_indices or not target_indices:
            return pd.DataFrame()

        # Extract embedding matrices
        query_embs = self._normalized_embeddings[query_indices]  # [Q, D]
        target_embs = self._normalized_embeddings[target_indices]  # [T, D]

        # Compute all pairwise similarities
        sim_matrix = torch.mm(query_embs, target_embs.T)  # [Q, T]

        # Convert to DataFrame
        result = pd.DataFrame(
            sim_matrix.cpu().numpy(),
            index=valid_query_genes,
            columns=valid_target_genes
        )

        return result


# Module-level singleton for lazy loading
_model_instance: Optional[GREmLNModel] = None


def get_model(checkpoint_path: Path | str = None) -> GREmLNModel:
    """
    Get or create the singleton GREmLNModel instance.

    Args:
        checkpoint_path: Path to model checkpoint (only used on first call)

    Returns:
        Loaded GREmLNModel instance
    """
    global _model_instance

    if _model_instance is None:
        if checkpoint_path is None:
            from tools.loader import MODEL_PATH
            checkpoint_path = MODEL_PATH

        _model_instance = GREmLNModel(checkpoint_path)

    _model_instance._ensure_loaded()
    return _model_instance


def reset_model():
    """Reset the singleton model instance (for testing)."""
    global _model_instance
    _model_instance = None
