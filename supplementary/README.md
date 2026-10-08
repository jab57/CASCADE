# Supplementary results

Result files written by the validation scripts in `scripts/` for the CASCADE paper
(patient-data concordance tests, baseline comparisons, LINCS coverage check, agent tool-grounding
benchmark). They are copies of the scripts' `outputs/*.json` files; the scripts write their output
to `outputs/`, which is not tracked.

| Files | Written by |
|---|---|
| `experiment4_<gene>_<cancer>_concordance*.json`, `experiment4_myc_*_concordance*.json` | `scripts/experiment4_tcga_myc_concordance.py` |
| `experiment4_myc_metabric_replication.json` | `scripts/experiment4_metabric_replication.py` |
| `experiment4_pam50_control_lumb.json` | `scripts/experiment4_pam50_control.py` |
| `experiment4_hallmark_baseline_*.json` | `scripts/experiment4_hallmark_baseline.py` |
| `experiment4_baseline_comparison_stats.json` | `scripts/experiment4_baseline_comparison_stats.py` |
| `experiment4_generalization_bh_fdr.json` | `scripts/experiment4_generalization_bh_fdr.py` (BH-FDR across the 27 generalization combinations) |
| `experiment5_lincs_coverage_check.json` | `scripts/experiment5_lincs_coverage_check.py` |
| `experiment6_agent_tool_grounding.json` | `scripts/experiment6_agent_tool_grounding.py` |

## Data licensing

These files are not covered by the MIT license of the CASCADE source code.

- **TCGA network values** (which genes are predicted targets of a focal gene, and their predicted
  direction) are derived from the TCGA ARACNe networks of `aracne.networks` (Giorgi FM, Alvarez MJ;
  Zenodo doi:10.5281/zenodo.22918956), licensed
  [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/): attribution, non-commercial
  use only, no distribution of modified versions. Those values remain under that license.
- **GREmLN values** (the embedding-similarity component of the predictions) come from the GREmLN
  model and its pre-computed ARACNe networks (Zhang et al., 2025), obtained from the CZI Virtual Cells
  Platform for scientific research use and derived from CELLxGENE Census data
  ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); CZI Cell Science Program et al.,
  *Nucleic Acids Research* 2025, doi:10.1093/nar/gkae1142).
- **Patient-level summaries** (mean expression z-scores and sample counts per gene) are aggregates
  computed from public cBioPortal data (TCGA and METABRIC); no patient identifiers are included.
- **Gene sets** used in the baseline comparison are MSigDB Hallmark sets (Liberzon et al., 2015) and
  keep their upstream terms.

See [`docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md) for the full list of data sources and citations.
