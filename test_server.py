"""Quick test script for the GREmLN MCP server."""

from tools.loader import load_network, get_available_cell_types
from tools.perturb import simulate_knockdown
from tools.gene_id_mapper import GeneIDMapper

# List available cell types
print("Available cell types:")
print(get_available_cell_types("data/networks"))
print()

# Test gene ID mapper
print("Testing gene ID mapper...")
mapper = GeneIDMapper()
print()

test_genes = ["MYC", "TP53", "BRCA1"]
for gene in test_genes:
    ensembl_id = mapper.symbol_to_ensembl(gene)
    print(f"  {gene} -> {ensembl_id}")
print()

# Test knockdown with Ensembl ID
print("Testing knockdown of ENSG00000248333 in epithelial cells...")
network = load_network("data/networks/epithelial_cell/network.tsv")
result = simulate_knockdown(network, "ENSG00000248333", depth=2, top_k=5)

print(f"Status: {result.get('status')}")
print(f"Perturbed gene: {result.get('perturbed_gene')}")
print(f"Total affected genes: {result.get('total_affected_genes')}")
print()
print("Top affected genes:")
for gene in result.get("top_affected_genes", []):
    print(f"  {gene['symbol']} ({gene['ensembl_id']}): {gene['direction']} ({gene['magnitude']})")

print()
print(f"Cache stats: {mapper.get_cache_stats()}")
