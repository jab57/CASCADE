"""Quick test script for the CASCADE MCP server."""

import time

# Test network-based tools
print("=" * 60)
print("TESTING NETWORK-BASED TOOLS")
print("=" * 60)

from tools.loader import load_network, get_available_cell_types
from tools.perturb import simulate_knockdown
from tools.gene_id_mapper import GeneIDMapper

# List available cell types
print("\nAvailable cell types:")
print(get_available_cell_types("data/networks"))

# Test gene ID mapper
print("\nTesting gene ID mapper...")
mapper = GeneIDMapper()

test_genes = ["MYC", "TP53", "BRCA1"]
for gene in test_genes:
    ensembl_id = mapper.symbol_to_ensembl(gene)
    print(f"  {gene} -> {ensembl_id}")

# Test knockdown with Ensembl ID
print("\nTesting knockdown of MYC in epithelial cells...")
myc_ensembl = mapper.symbol_to_ensembl("MYC")
network = load_network("data/networks/epithelial_cell/network.tsv")
result = simulate_knockdown(network, myc_ensembl, depth=2, top_k=5)

print(f"Status: {result.get('status')}")
print(f"Perturbed gene: {result.get('perturbed_gene')}")
print(f"Total affected genes: {result.get('total_affected_genes')}")
print("\nTop affected genes:")
for gene in result.get("top_affected_genes", []):
    print(f"  {gene['symbol']} ({gene['ensembl_id']}): {gene['direction']} ({gene['magnitude']})")

# Test model-based tools
print("\n" + "=" * 60)
print("TESTING MODEL-BASED TOOLS")
print("=" * 60)

from tools.model_inference import CascadeModel, get_model
from tools.perturb import simulate_knockdown_with_embeddings

# Load model
print("\nLoading GREmLN model...")
start = time.time()
model = get_model("models/model.ckpt")
print(f"Model loaded in {time.time() - start:.2f}s")
print(f"Device: {model.device}")
print(f"Embeddings shape: {model.gene_embeddings.shape}")

# Test similarity
print("\nTesting gene similarity...")
tp53_ensembl = mapper.symbol_to_ensembl("TP53")
sim = model.compute_similarity(myc_ensembl, tp53_ensembl)
print(f"MYC-TP53 similarity: {sim:.4f}")

# Test similar genes
print("\nTop 5 genes similar to MYC:")
similar = model.get_top_similar_genes(myc_ensembl, top_k=5)
for _, row in similar.iterrows():
    symbol = mapper.ensembl_to_symbol(row["ensembl_id"]) or row["ensembl_id"]
    print(f"  {symbol}: {row['similarity']:.4f}")

# Test knockdown with embeddings
print("\nTesting embedding-enhanced knockdown of MYC...")
start = time.time()
result = simulate_knockdown_with_embeddings(
    network, myc_ensembl, model, depth=2, top_k=5, alpha=0.7
)
print(f"Completed in {time.time() - start:.2f}s")
print(f"Status: {result.get('status')}")
print(f"Total affected genes: {result.get('total_affected_genes')}")
print("\nTop affected genes:")
for gene in result.get("top_affected_genes", []):
    print(f"  {gene['symbol']}: combined={gene['combined_effect']}, "
          f"net={gene['network_effect']}, emb={gene['embedding_similarity']}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
