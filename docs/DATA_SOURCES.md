# Data Sources for CASCADE

This document provides comprehensive information about the data sources used by CASCADE (Computational Analysis of Simulated Cell And Drug Effects), including download instructions, file formats, confidence thresholds, and citations.

---

## Overview of Data Sources

CASCADE integrates eight data sources, divided into two categories:

**Local files (bundled in the repository — no download required):**

| Source | Purpose | Location | Size |
|--------|---------|----------|------|
| GREmLN regulatory networks | Population-averaged cell-type regulatory network topology | `data/networks/` | ~2 MB total |
| TCGA ARACNe networks | Tumor-state regulatory networks (14 cancer types) | `data/networks/tcga/` | ~155 MB total |
| LINCS L1000 | Experimental CRISPR knockout expression effects | `data/lincs/` | ~35 MB |
| dbSUPER | Super-enhancer annotations (BRD4/BET sensitivity) | `data/super_enhancers/` | ~3.7 MB |
| DoRothEA disk cache | TF regulon parquet cache for fast server startup | `data/dorothea/` | ~1 MB |
| DepMap model metadata | Cell line annotations (lineage, disease) | `data/depmap/Model.csv` | ~1 MB |

**Files requiring a one-time download after cloning:**

| Source | Purpose | Location | Size |
|--------|---------|----------|------|
| GREmLN model checkpoint | Gene embeddings (256-dim, 19,247 genes) | `models/model.ckpt` | ~120 MB |
| DepMap CRISPR scores | Chronos gene essentiality (1000+ cancer lines) | `data/depmap/CRISPRGeneEffect.csv` | ~413 MB |

**Python package + live APIs (no local files required):**

| Source | Purpose | Access |
|--------|---------|--------|
| DoRothEA | TF regulon validation | Python package (`decoupler`) + disk cache |
| STRING | Protein-protein interactions | REST API, queried at runtime |
| Ensembl | Gene symbol ↔ Ensembl ID mapping | REST API, queried at runtime |
| cBioPortal | TCGA primary tumor expression & somatic alterations | REST API, queried at runtime |

---

## 1. GREmLN Regulatory Networks

### Overview

**Primary Source**: [GREmLN Foundation Model](https://github.com/czi-ai/GREmLN) (Chan Zuckerberg Initiative / CZ Biohub NY)
**Download Location**: [GREmLN Quickstart Tutorial](https://virtualcellmodels.cziscience.com/quickstart/gremln-quickstart) (networks available via Google Drive)
**Underlying Data**: [CellxGene Data Portal](https://cellxgene.cziscience.com/) (11M scRNA-seq profiles, 162 cell types from Census release 2024-07-01)
**Processing Method**: ARACNe algorithm
**Development Team**: Zhang et al. (2026), Califano Lab (Columbia University / CZ Biohub NY)
**Format**: Pre-computed networks as TSV files (`network.tsv`)
**Publication**: ICLR 2026

**Note on disease status**: The CellxGene Census corpus used by GREmLN includes both healthy and disease/cancer-infiltrating cells — disease status was not filtered to normal cells only (Zhang et al. 2026). The pre-computed networks therefore represent population-averaged regulatory relationships across heterogeneous cell states. These networks are appropriate for hypothesis generation and regulatory network analysis but should not be assumed to represent exclusively normal cell regulatory wiring.

### Supported Cell Types

CASCADE currently supports **10 human cell types**:

#### Immune & Blood Cell Types (9 types)

| Cell Type | Directory Key | Network Edges | Cell Type Markers | Research Applications |
|-----------|--------------|---------------|-------------------|----------------------|
| CD14 Monocytes | `cd14_monocytes` | 2,009 | CD14 | Inflammation, innate immunity |
| CD16 Monocytes | `cd16_monocytes` | 1,236 | FCGR3A (CD16) | Vascular patrolling, tissue repair |
| CD20 B Cells | `cd20_b_cells` | 1,128 | MS4A1 (CD20), CD19 | Adaptive immunity, B cell malignancies |
| CD4 T Cells | `cd4_t_cells` | 1,371 | CD4 | Helper T cell responses, autoimmunity |
| CD8 T Cells | `cd8_t_cells` | 3,154 | CD8A, CD8B | Cancer immunotherapy, viral immunity |
| Erythrocytes | `erythrocytes` | 19,398 | HBA1, HBA2, HBB | Anemia, hemoglobinopathies |
| NK Cells | `nk_cells` | 404 | NCAM1 (CD56), NCR1 | Innate immunity, cancer immunosurveillance |
| NKT Cells | `nkt_cells` | 2,509 | CD3E, KLRB1 | Immune regulation, tumor immunity |
| Monocyte-Derived Dendritic Cells | `monocyte-derived_dendritic_cells` | 5,317 | CD1C, ITGAX (CD11c) | Antigen presentation, vaccine development |

#### Epithelial Cell Type (1 type)

| Cell Type | Directory Key | Network Edges | Cell Type Markers | Research Applications |
|-----------|--------------|---------------|-------------------|----------------------|
| Epithelial Cells | `epithelial_cell` | 183,247 | EPCAM, KRT8, KRT18 | Cancer biology (carcinomas), barrier function |

### Download Instructions

> **Note for standard users:** The pre-computed network files (`network.tsv`) for all 10 cell types are committed directly to this repository. Cloning the repo is sufficient — no separate download is required for the networks.
>
> The instructions below document how these networks were originally obtained and are provided for reproducibility. They also apply if you need to regenerate or update the network files.

```bash
# Install gdown
pip install gdown

# Download the GREmLN tutorial folder (includes networks + model checkpoint)
gdown --folder https://drive.google.com/drive/folders/1cMR9HoAC22i6sKSWgfQUEQRf0UP_w3_m?usp=sharing

# Copy network files to CASCADE
cd GREmLN_tutorial
cp -r networks/* /path/to/CASCADE/data/networks/
```

The downloaded folder contains:
```
GREmLN_tutorial/
  data/
    human_immune_cells.h5ad      # Source expression data (not used by CASCADE directly)
    epithelial_cells.h5ad
  networks/
    cd14_monocytes/network.tsv
    cd16_monocytes/network.tsv
    cd20_b_cells/network.tsv
    cd4_t_cells/network.tsv
    cd8_t_cells/network.tsv
    erythrocytes/network.tsv
    nk_cells/network.tsv
    nkt_cells/network.tsv
    monocyte-derived_dendritic_cells/network.tsv
    epithelial_cell/network.tsv
  model.ckpt                     # GREmLN model weights (see Section 2)
  vocab.csv
```

### Network File Format

Each `network.tsv` is tab-separated with ARACNe-inferred regulatory edges:

```
regulator.values    target.values    mi.values    scc.values    count.values    log.p.values
ENSG00000213626    ENSG00000233927    0.120847    0.123325    1    -0.674163
```

| Column | Description | Range |
|--------|-------------|-------|
| `regulator.values` | Ensembl gene ID of the transcription factor/regulator | `ENSG00000*` |
| `target.values` | Ensembl gene ID of the target gene | `ENSG00000*` |
| `mi.values` | Mutual information (strength of regulatory relationship) | 0.0 – 0.5 |
| `scc.values` | Spearman correlation coefficient (+ = activation, - = repression) | -1.0 to +1.0 |
| `count.values` | Bootstrap iterations where edge appeared (robustness) | 1 – 100 |
| `log.p.values` | Log-transformed p-value (statistical significance) | Negative values |

### ARACNe Processing Parameters

Networks were generated with:

```bash
./aracne3_app_release \
    --input metacells_{cell_type}.txt \
    --output {cell_type}_network.tsv \
    --pvalue 1e-8 \
    --dpi 1.0 \
    --threads 16 \
    --bootstrap 100
```

Key parameters: mutual information adaptive partitioning, DPI tolerance 1.0, p-value threshold 1e-8, 100 bootstrap iterations. Processing time: 12–14 hours per cell type on HPC (64–128 GB RAM).

---

## 2. GREmLN Model Checkpoint

### Overview

**File**: `models/model.ckpt` (~120 MB)
**Source**: GREmLN Quickstart Tutorial (Chan Zuckerberg Initiative / CZ Biohub NY)
**Contents**: Pre-trained gene embedding weights covering ~19,247 human genes
**Embedding dimension**: 256-dimensional vectors, trained on 11M scRNA-seq profiles
**Framework**: PyTorch Lightning checkpoint

> **Note for standard users:** `model.ckpt` is not committed to the repository due to its size (~120 MB). Download it with the provided script — this is the only file that requires a separate download after cloning.

CASCADE uses these embeddings to:
- Compute functional gene similarity (`find_similar_genes`)
- Enhance perturbation predictions with learned representations (`embedding_enhanced_knockdown`, `embedding_enhanced_overexpression`)
- Identify functionally related genes without direct network connections

### Download

```bash
# One-time setup after cloning
python scripts/download_model.py
```

This script downloads the model checkpoint from the GREmLN Quickstart Tutorial Google Drive folder provided by CZI. It requires `gdown`:

```bash
pip install gdown
```

To download manually (e.g. if gdown fails):

```bash
# Download the GREmLN tutorial folder, then copy model.ckpt
gdown --folder https://drive.google.com/drive/folders/1cMR9HoAC22i6sKSWgfQUEQRf0UP_w3_m?usp=sharing
cp GREmLN_tutorial/model.ckpt /path/to/CASCADE/models/model.ckpt
```

### Graceful Degradation

If `models/model.ckpt` is missing or fails to load, CASCADE automatically falls back to network-only analysis. Embedding-dependent tools (`find_similar_genes`, `embedding_enhanced_*`) will return a graceful error message indicating that the model checkpoint is unavailable.

---

## 3. LINCS L1000 CRISPR Knockout Data

### Overview

**Dataset**: LINCS L1000 CMAP CRISPR Knockout Consensus Signatures
**Provider**: Harmonizome (Ma'ayan Lab, Icahn School of Medicine at Mount Sinai)
**URL**: https://maayanlab.cloud/Harmonizome/dataset/LINCS+L1000+CMAP+CRISPR+Knockout+Consensus+Signatures
**File**: `data/lincs/gene_attribute_edges.txt.gz` (~35 MB)
**Included in the repository** — no download required after cloning.

### Contents

- 2.5M gene-perturbation associations
- 9,551 genes measured
- 5,049 gene knockdowns
- Effect direction: +1 (upregulated), -1 (downregulated)

### Download

```bash
curl -L "https://maayanlab.cloud/static/hdfs/harmonizome/data/l1000crispr/gene_attribute_edges.txt.gz" \
     -o data/lincs/gene_attribute_edges.txt.gz
```

### Usage in CASCADE

LINCS data is used by the `get_lincs_knockdown_effects` tool to cross-validate network-predicted perturbation effects with experimental CRISPR knockout measurements. When a gene is knocked out in LINCS screens, CASCADE reports which other genes changed expression and in which direction.

### Known Limitations

Harmonizome pre-filters the raw LINCS data to high-confidence associations, which removes some biologically validated relationships:

- **BRD4 → MYC**: Well-established (BRD4 inhibitors reduce MYC expression), but absent from this filtered dataset.
- **Validated example**: TP53 → CDKN1A is present (ranks #3).

For complete coverage, consider using raw LINCS data from [clue.io](https://clue.io/data/CMap2020#LINCS2020) (future enhancement).

### Citation

Rouillard, A.D., et al. (2016). The harmonizome: a collection of processed datasets gathered to serve and support the scientific community. *Database*, baw100. https://doi.org/10.1093/database/baw100

---

## 4. dbSUPER Super-Enhancer Database

### Overview

**Database**: dbSUPER
**URL**: https://asntech.org/dbsuper/
**Genome Build**: hg19
**File**: `data/super_enhancers/dbSUPER_hg19.tsv` (~3.7 MB)
**Included in the repository** — no download required after cloning.

### Contents

- 69,205 super-enhancer associations
- 10,548 unique genes
- 102 cell/tissue types

### Download

```bash
curl -L "https://asntech.org/dbsuper/data/dbSUPER_SuperEnhancers_hg19.tsv" \
     -o data/super_enhancers/dbSUPER_hg19.tsv
```

### Usage in CASCADE

The `get_super_enhancer_status` tool queries dbSUPER to determine whether a gene is driven by a super-enhancer. Genes with super-enhancers are often sensitive to BRD4/BET inhibitors (e.g., JQ1, OTX015) — a key therapeutic implication for "undruggable" oncogenes.

**Examples:**
- MYC: Super-enhancers in 32 cell types → BRD4-sensitive
- BCL2: Super-enhancers present → BRD4-sensitive
- TP53: No super-enhancers → Not BRD4-sensitive

### Citation

Khan, A. & Zhang, X. (2016). dbSUPER: a database of super-enhancers in mouse and human genome. *Nucleic Acids Research*, 44(D1), D164–D171. https://doi.org/10.1093/nar/gkv1002

---

## 5. DepMap CRISPR Essentiality Data

### Overview

**Source**: Broad Institute DepMap Portal
**URL**: https://depmap.org/portal/download/
**Release used**: DepMap Public 24Q4 (recommended; use latest available)
**Files:**

| File | Size | In repo? | Description |
|------|------|----------|-------------|
| `data/depmap/CRISPRGeneEffect.csv` | ~413 MB | No — download from depmap.org | Chronos gene effect scores (rows = cell lines, cols = genes) |
| `data/depmap/Model.csv` | ~1 MB | Yes | Cell line metadata including `OncotreeLineage` |

### Download Instructions

`Model.csv` is included in the repository. Only `CRISPRGeneEffect.csv` needs to be downloaded manually (413 MB, versioned quarterly by DepMap — users should pull from source to ensure they have the current release):

1. Go to **https://depmap.org/portal/download/**
2. Select the latest **DepMap Public** release (e.g., "DepMap Public 24Q4")
3. Download **CRISPRGeneEffect.csv** — listed under "CRISPR (DepMap Internal 24Q4+Score, Chronos)" or search "CRISPRGeneEffect"
4. Place the file in `data/depmap/`

### Score Interpretation

| Chronos Score | Interpretation |
|---------------|----------------|
| 0 | Non-essential (no fitness effect) |
| < -0.5 | Essential in that cell line |
| < -1.0 | Strongly essential |
| > 0 | Slightly anti-proliferative when knocked out |

### Thresholds Used by CASCADE

- **Pan-cancer essential**: essential in > 50% of tested cell lines
- **Common essential**: essential in > 90% of tested cell lines (housekeeping genes, e.g., RPL genes)

### Column Format

`CRISPRGeneEffect.csv` uses the format `"GENE (entrez_id)"` for column headers (e.g., `"TP53 (7157)"`). CASCADE strips the Entrez suffix automatically.

### Citation

DepMap, Broad (2024). DepMap 24Q4 Public. Figshare+. https://doi.org/10.25452/figshare.plus.27993248

---

## 6. STRING Protein-Protein Interactions

### Overview

**Database**: STRING (Search Tool for the Retrieval of Interacting Genes/Proteins)
**URL**: https://string-db.org
**Access**: Live REST API — no local file required
**Species**: Homo sapiens (NCBI taxonomy ID: 9606)
**Implementation**: `tools/ppi/string_client.py`

### Usage in CASCADE

The `get_protein_interactions` tool queries STRING to retrieve physical and functional protein interactions. This is especially valuable for effector and scaffold proteins that have few or no regulatory edges in the ARACNe networks.

### Confidence Score Tiers

| Score Range | Confidence Level |
|-------------|-----------------|
| 900 – 1000 | Highest confidence |
| 700 – 899 | High confidence |
| 400 – 699 | Medium confidence (CASCADE default) |
| 150 – 399 | Low confidence |
| 0 – 149 | All interactions |

CASCADE uses a default threshold of **400 (medium confidence)** and returns up to 50 interaction partners per query.

### Evidence Channels

Each STRING interaction reports scores across five independent evidence channels:

| Channel | Description |
|---------|-------------|
| `experimental` | Experimental binding/co-IP evidence |
| `database` | Curated pathway/interaction databases |
| `textmining` | Co-mention in literature |
| `coexpression` | Co-expression across conditions |
| `neighborhood` | Genomic co-localization |

### API Endpoints Used

- `GET /api/json/get_string_ids` — resolve gene symbol to STRING protein ID
- `GET /api/json/network` — retrieve interaction network

**Timeout**: 10 seconds per request. Results are cached in-memory for the server session.

### Citation

Szklarczyk, D., et al. (2023). The STRING database in 2023: protein–protein association networks and functional enrichment analyses for any of 14,094 organisms. *Nucleic Acids Research*, 51(D1), D638–D646. https://doi.org/10.1093/nar/gkac1000

---

## 7. DoRothEA TF Regulons

### Overview

**Database**: DoRothEA (Discriminant Regulon Expression Analysis)
**Access**: Python package (`decoupler`) + local disk cache
**Disk cache**: `data/dorothea/dorothea_cache.parquet`
**Implementation**: `tools/dorothea.py`

DoRothEA provides curated transcription factor regulons compiled from four evidence types: literature curation, ChIP-seq binding data, TF binding motifs, and co-expression.

### Usage in CASCADE

The `get_dorothea_regulons` tool validates ARACNe-derived TF classifications against DoRothEA. It confirms whether a gene classified as a transcription factor in the network has experimentally supported regulon evidence.

### Confidence Levels

DoRothEA assigns each TF–target interaction a confidence grade:

| Level | Evidence Basis | Used by CASCADE |
|-------|---------------|-----------------|
| A | Multiple independent evidence types | Yes (default) |
| B | Two evidence types | Yes (default) |
| C | Single strong evidence type | Yes (default) |
| D | Single weaker evidence type | No (default) |
| E | Co-expression only | No (default) |

CASCADE defaults to levels **A, B, C** for a balance of recall and precision.

### Disk Cache

The DoRothEA cache (`data/dorothea/dorothea_cache.parquet`) is committed to the repository so server restarts load instantly (~0.13 seconds) without re-downloading from `decoupler`. On a fresh checkout the cache is already present — no action required.

To refresh the cache (e.g., after a DoRothEA update):

```bash
rm data/dorothea/dorothea_cache.parquet
# Restart the CASCADE server — it will re-download and rebuild the cache
python cascade_langgraph_mcp_server.py
```

### Citation

Garcia-Alonso, L., et al. (2019). Benchmark and integration of resources for the estimation of human transcription factor activities. *Genome Research*, 29(8), 1363–1375. https://doi.org/10.1101/gr.240663.118

Badia-i-Mompel, P., et al. (2022). decoupleR: ensemble of computational methods to infer biological activities from omics data. *Bioinformatics Advances*, 2(1), vbac016. https://doi.org/10.1093/bioadv/vbac016

---

## 8. Ensembl Gene ID Mapping

### Overview

**Database**: Ensembl REST API
**URL**: https://rest.ensembl.org
**Access**: Live REST API with persistent local cache
**Cache file**: `cache/gene_id_cache.json`
**Implementation**: `tools/gene_id_mapper.py`

### Usage in CASCADE

All CASCADE tools accept either gene symbols (`MYC`) or Ensembl IDs (`ENSG00000136997`). The `GeneIDMapper` class resolves inputs to the Ensembl ID used internally by the ARACNe networks and GREmLN model.

### API Endpoints Used

- `GET /lookup/symbol/homo_sapiens/{symbol}` — symbol to Ensembl ID
- `GET /lookup/id/{ensembl_id}` — Ensembl ID to symbol

**Timeout**: 10 seconds per request.

### Persistent Cache

Resolved mappings are persisted to `cache/gene_id_cache.json` so that repeated lookups for the same genes do not require API calls. The cache grows automatically as new genes are queried.

### Citation

Martin, F.J., et al. (2023). Ensembl 2023. *Nucleic Acids Research*, 51(D1), D933–D941. https://doi.org/10.1093/nar/gkac958

---

## TCGA Tumor-State ARACNe Networks

CASCADE supports **14 TCGA cancer-type-specific ARACNe networks** derived from The Cancer Genome Atlas (TCGA) tumor expression data. These complement the GREmLN population-averaged cell-type networks with tumor-state regulatory wiring and include **Mode of Action (MoA)** annotations (activation vs. repression) not present in the GREmLN networks.

> **Note for standard users:** Pre-built network CSVs for all 14 cancer types are committed to this repository at `data/networks/tcga/`. Cloning the repo is sufficient — no separate download or build step is required. The instructions below document how to regenerate the CSVs from the Bioconductor source tarball if you need to reproduce or update them.

### Supported Cancer Types

| Key    | Cancer Type                              |
|--------|------------------------------------------|
| `blca` | Bladder Urothelial Carcinoma             |
| `brca` | Breast Invasive Carcinoma                |
| `cesc` | Cervical Squamous Cell Carcinoma         |
| `coad` | Colon Adenocarcinoma                     |
| `hnsc` | Head/Neck Squamous Cell Carcinoma        |
| `kirc` | Kidney Renal Clear Cell Carcinoma        |
| `lihc` | Liver Hepatocellular Carcinoma           |
| `luad` | Lung Adenocarcinoma                      |
| `lusc` | Lung Squamous Cell Carcinoma             |
| `ov`   | Ovarian Carcinoma                        |
| `paad` | Pancreatic Adenocarcinoma                |
| `prad` | Prostate Adenocarcinoma                  |
| `stad` | Stomach Adenocarcinoma                   |
| `ucec` | Uterine Corpus Endometrial Carcinoma     |

GBM and LAML are intentionally excluded — no reference network of the appropriate cell lineage exists in CASCADE for these cancer types.

### Data Source

**Package**: Bioconductor `aracne.networks` (Lim & Califano, 2018)
**Source paper**: Lim, W.K. & Califano, A. (2018). "Mapping the hallmarks of lung adenocarcinoma with massively parallel sequencing." *Cell Syst.* 6(4):446–460. doi:10.1016/j.cels.2018.02.011
**Download URL**: `https://bioconductor.org/packages/release/data/experiment/src/contrib/aracne.networks_1.38.0.tar.gz`

Networks are derived from TCGA tumor RNA-seq data processed through the ARACNe-AP algorithm at the Califano Lab (Columbia University).

### File Format

Each TCGA cancer type will provide a CSV at `data/networks/tcga/{ct}/network.csv`:

```
Regulator,Target,MoA,Likelihood
TP53,CDKN1A,1.0,0.312
TP53,MDM2,-1.0,0.251
...
```

| Column | Description |
|--------|-------------|
| `Regulator` | Gene symbol of the regulatory TF |
| `Target` | Gene symbol of the target gene |
| `MoA` | Mode of Action: +1 activation, -1 repression, 0 unknown |
| `Likelihood` | Edge confidence score (0–1) |

### How to Regenerate from Source

The pre-built CSVs are already in the repository. These instructions apply only if you need to regenerate them (e.g., after a new `aracne.networks` release).

#### Step 1: Download the Bioconductor tarball (~213 MB)

```bash
curl -o /tmp/aracne.networks.tar.gz \
  https://bioconductor.org/packages/release/data/experiment/src/contrib/aracne.networks_1.38.0.tar.gz
```

#### Step 2: Install required Python packages

```bash
pip install rdata
```

#### Step 3: Extract network CSVs

```bash
# Converts Entrez IDs → gene symbols via MyGene.info (~5 min, requires internet)
python scripts/extract_tcga_networks.py \
    --tarball /tmp/aracne.networks.tar.gz \
    --output-dir data/networks/tcga
```

The script reads each `.rda` file from the tarball, batch-converts all Entrez IDs to gene symbols via MyGene.info (resolves >99.9% of IDs), and writes symbol-keyed CSVs. A single cancer type can be extracted with `--cancer-type brca` for faster testing.

### Network Statistics

| Cancer Type | Genes  | Edges   | Regulons |
|-------------|--------|---------|----------|
| brca        | 19,514 | 331,644 | 6,052    |
| coad        | 19,795 | 413,481 | 6,054    |
| hnsc        | 19,763 | 422,855 | 6,053    |
| luad        | 19,742 | 399,216 | 6,053    |
| lusc        | 19,752 | 454,680 | 6,052    |
| ov          | 19,154 | 647,002 | 6,005    |
| prad        | 19,797 | 330,709 | 6,051    |
| ucec        | 19,735 | 469,523 | 6,053    |

### Citation

Lim, W.K. & Califano, A. (2018). "ARACNe-AP: gene network reverse engineering through adaptive partitioning inference of mutual information." *Cell Systems*, 6(4):446–460. https://doi.org/10.1016/j.cels.2018.02.011

---

## Data Citation and Attribution

When publishing results obtained with CASCADE, please cite the relevant underlying data sources:

### Required Citations

1. **GREmLN Foundation Model (networks + embeddings)**
   Zhang, M., Swamy, V., Cassius, R., Dupire, L., Karaletsos, T., & Califano, A. (2026). "GREmLN: A Cellular Graph Structure Aware Transcriptomics Foundation Model." *ICLR 2026*. https://openreview.net/forum?id=HdvI8bkdDG

2. **CellxGene Data Portal (underlying scRNA-seq data)**
   Megill, C., et al. (2021). "cellxgene: a performant, scalable exploration platform for high dimensional sparse matrices." *bioRxiv*. https://doi.org/10.1101/2021.04.05.438318

3. **ARACNe Algorithm**
   Lachmann, A., et al. (2016). "ARACNe-AP: gene network reverse engineering through adaptive partitioning inference of mutual information." *Bioinformatics*, 32(14), 2233–2235. https://doi.org/10.1093/bioinformatics/btw216

### Conditional Citations (use if corresponding data source contributed to your results)

4. **LINCS L1000** — Rouillard et al. (2016), *Database*, baw100. https://doi.org/10.1093/database/baw100

5. **dbSUPER** — Khan & Zhang (2016), *Nucleic Acids Research*, 44(D1), D164–D171. https://doi.org/10.1093/nar/gkv1002

6. **DepMap** — DepMap, Broad (2024). DepMap 24Q4 Public. Figshare+. https://doi.org/10.25452/figshare.plus.27993248

7. **STRING** — Szklarczyk et al. (2023), *Nucleic Acids Research*, 51(D1), D638–D646. https://doi.org/10.1093/nar/gkac1000

8. **DoRothEA** — Garcia-Alonso et al. (2019), *Genome Research*, 29(8), 1363–1375. https://doi.org/10.1101/gr.240663.118

9. **Ensembl** — Martin et al. (2023), *Nucleic Acids Research*, 51(D1), D933–D941. https://doi.org/10.1093/nar/gkac958

10. **TCGA ARACNe networks** — Lim, W.K. & Califano, A. (2018). *Cell Systems*, 6(4):446–460. https://doi.org/10.1016/j.cels.2018.02.011

---

## Quick Reference: What to Do After Cloning

Most data is bundled in the repository. Only two steps are required after `git clone`:

```bash
# Step 1: Download the GREmLN model checkpoint (~120 MB)
pip install gdown          # if not already installed
python scripts/download_model.py

# Step 2: Download DepMap CRISPR gene effect scores (~413 MB)
#   - Go to: https://depmap.org/portal/download/
#   - Download: CRISPRGeneEffect.csv (latest DepMap Public release)
#   - Place at: data/depmap/CRISPRGeneEffect.csv
```

Everything else is already in the repository: GREmLN cell-type networks, TCGA ARACNe networks (14 cancer types), LINCS L1000, dbSUPER, DoRothEA cache, and DepMap Model.csv.

Verify the full installation after both steps:

```bash
python verify_installation.py --offline
```

---

## Contact and Support

- **CASCADE issues**: https://github.com/your-org/CASCADE/issues
- **GREmLN / network data**: opensource@chanzuckerberg.com
- **Harmonizome / LINCS**: https://maayanlab.cloud/Harmonizome/
- **DepMap**: https://depmap.org/portal/contact/
- **STRING**: https://string-db.org/cgi/help

---

**Last Updated**: 2026-03-16
**Maintained by**: CASCADE Development Team
