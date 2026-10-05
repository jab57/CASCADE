# Network files

## GREmLN cell-type networks (bundled)

The 10 cell-type directories here (`cd14_monocytes/` … `nkt_cells/`, `epithelial_cell/`) hold
pre-computed ARACNe networks from the GREmLN project:

- **Source:** GREmLN Quickstart tutorial data
  (https://virtualcellmodels.cziscience.com/quickstart/gremln-quickstart), Chan Zuckerberg
  Initiative Virtual Cells Platform. Each `network.tsv` is the file as downloaded.
- **Terms:** obtained under the
  [Virtual Cells Platform Terms of Use](https://virtualcellmodels.cziscience.com/terms-of-use)
  (scientific research use). The GREmLN model is MIT-licensed; no separate license is stated
  for the tutorial networks, and they are **not** covered by the MIT license of the CASCADE
  source code. They are included for convenience and research use; they will be removed on
  request of the rights holder.
- **Cite:** Zhang M, et al. (2025) GREmLN, bioRxiv doi:10.1101/2025.07.03.663009; and the
  underlying CELLxGENE Census data
  ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)): CZI Cell Science Program et al.
  (2025) *Nucleic Acids Research* 53(D1):D886–D900, doi:10.1093/nar/gkae1142.

Details: `docs/DATA_SOURCES.md`.

## TCGA tumor networks (not bundled)

`tcga/` is filled locally; see the TCGA section of `docs/DATA_SOURCES.md` for how to generate
the files and for their license.
