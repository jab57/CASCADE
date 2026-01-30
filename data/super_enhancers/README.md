# Super-Enhancer Data

This directory contains super-enhancer annotations for identifying BRD4/BET inhibitor sensitive genes.

## Data File

`dbSUPER_hg19.tsv` (~3.7MB)

**Download:**
```bash
curl -L "https://asntech.org/dbsuper/data/dbSUPER_SuperEnhancers_hg19.tsv" -o dbSUPER_hg19.tsv
```

## Source

- **Database**: dbSUPER
- **URL**: http://asntech.org/dbsuper/
- **Genome Build**: hg19
- **Citation**: Khan A, Zhang X. dbSUPER: a database of super-enhancers in mouse and human genome. Nucleic Acids Res. 2016

## Contents

- 69,205 super-enhancer associations
- 10,548 unique genes
- 102 cell/tissue types

## Usage

Genes with super-enhancers are often sensitive to BRD4/BET inhibitors (e.g., JQ1, OTX015).
This is useful for drug discovery when a target gene cannot be directly drugged.

**Examples:**
- MYC: Has super-enhancers in 32 cell types → BRD4-sensitive
- BCL2: Has super-enhancers → BRD4-sensitive
- TP53: No super-enhancers → Not BRD4-sensitive

## Therapeutic Implications

Many "undruggable" transcription factors (MYC, MYCN) become targetable via their
chromatin dependencies. BRD4 inhibitors exist precisely because they disrupt the
super-enhancer machinery that drives oncogene expression.
