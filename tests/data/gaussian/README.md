# Gaussian fixtures

Each case directory holds a Gaussian 16 Opt+Freq log (`stdout`, unmodified), the
`input.json` that stands in for the run, and the `expected.json` the codec must produce.
`tests/test_gaussian.py` checks the values against the log text and recomputes the
frequencies from each Hessian.

The logs are members of two public datasets, both released under CC-BY 4.0. Cite them
if you reuse the logs:

| case | source | member |
|---|---|---|
| `rgd1_nh3_minimum` | RGD1 (Zhao et al., *Sci. Data* **10**, 145 (2023); Zenodo doi:10.5281/zenodo.7860446), B3LYP-D3/TZVP | `raw_output/RP/mol83138.out` |
| `rgd1_minimum_mol12021` | RGD1 | `raw_output/RP/mol12021.out` |
| `rgd1_ts_MR_299320_1` | RGD1 | `raw_output/final-TS/MR_299320_1-TS.out` |
| `stjohn_radical_um062x` | St. John et al., *Sci. Data* **7**, 244 (2020); figshare doi:10.6084/m9.figshare.c.4944855, UM06-2X/def2-TZVP | `20200411_scidata_logs/4321_0.log.gz` |
| `stjohn_radical_stable_opt` | same, with a `stable=opt` step before the Opt+Freq | `20200411_scidata_logs/195768_5529c4.log.gz` |
| `stjohn_closed_shell_rm062x` | same, a closed shell (RM06-2X) | `20200411_scidata_logs/447885_07daac.log.gz` |

`tests/data/near_linear/` holds three more St. John logs (`1323_0`, `510622_507d51`,
`12405_0`, same source, gzipped as in the tarball): linear to the precision of the log, but
Gaussian prints 3N-6 modes for them.
