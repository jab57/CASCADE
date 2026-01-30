# LINCS L1000 Data

This directory contains LINCS L1000 CRISPR knockout expression data.

## Data File

`gene_attribute_edges.txt.gz` (~35MB)

**Not included in git** - download manually:

```bash
curl -L "https://maayanlab.cloud/static/hdfs/harmonizome/data/l1000crispr/gene_attribute_edges.txt.gz" -o gene_attribute_edges.txt.gz
```

## Source

- **Dataset**: LINCS L1000 CMAP CRISPR Knockout Consensus Signatures
- **Provider**: Harmonizome (Ma'ayan Lab)
- **URL**: https://maayanlab.cloud/Harmonizome/dataset/LINCS+L1000+CMAP+CRISPR+Knockout+Consensus+Signatures

## Contents

- 2.5M gene-perturbation associations
- 9,551 genes measured
- 5,049 gene knockdowns
- Effect direction: +1 (upregulated), -1 (downregulated)

## Known Limitations

Harmonizome pre-filters the raw LINCS data, which removes some biologically validated relationships:

- **BRD4 → MYC**: Well-established drug target relationship (BRD4 inhibitors reduce MYC), but not present in this filtered dataset
- **Validation case**: TP53 → CDKN1A works (rank #3)

For complete coverage of known regulatory relationships, consider using raw LINCS data from [clue.io](https://clue.io/data/CMap2020#LINCS2020) (future enhancement).
