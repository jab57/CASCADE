# DepMap CRISPR Essentiality Data

This directory holds DepMap CRISPR screen data used by CASCADE to validate
predicted gene essentiality with empirical phenotypic evidence.

## Files Required

| File | Size (approx.) | Description |
|------|----------------|-------------|
| `CRISPRGeneEffect.csv` | ~200 MB | Chronos gene effect scores (rows = cell lines, cols = genes) |
| `Model.csv` | ~1 MB | Cell line metadata including OncotreeLineage |

## Download Instructions

1. Go to **https://depmap.org/portal/download/**
2. Select the latest **DepMap Public** release (e.g., "DepMap Public 24Q4")
3. Download the following files:

### CRISPRGeneEffect.csv
- Find under **CRISPR (DepMap Internal 24Q4+Score, Chronos)**
  or search for "CRISPRGeneEffect"
- Column format: `"GENE (entrez_id)"` — CASCADE strips the Entrez suffix automatically

### Model.csv
- Find under **Model** or **Cell Line Metadata**
- Must contain at minimum the columns: `ModelID`, `OncotreeLineage`

4. Place both files in this directory (`data/depmap/`)

## Score Interpretation

| Chronos Score | Interpretation |
|---------------|----------------|
| 0             | Non-essential (no fitness effect) |
| < -0.5        | Essential in that cell line |
| < -1.0        | Strongly essential |
| > 0           | Slightly anti-proliferative when knocked out |

## Thresholds Used by CASCADE

- **Pan-cancer essential**: essential fraction > 50% of tested cell lines
- **Common essential**: essential fraction > 90% (housekeeping-like genes, e.g., RPL genes)

## Citation

DepMap, Broad (2024). DepMap 24Q4 Public. Figshare+.
https://doi.org/10.25452/figshare.plus.27993248
