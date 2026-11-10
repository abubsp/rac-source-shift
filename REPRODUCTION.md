# Reproduction of the main manuscript analyses

`reproduce_analyses.py` re-implements Sections 3.5-3.8 / 4.1-4.7 of the manuscript from the methods text:

```bash
export RAC_DATA_DIR=/path/to/folder/with/the/two/xlsx/files
python reproduce_analyses.py table3 loso perm joint conformal paired pd     # results_<name>.json
```
Run with scikit-learn 1.9.1, numpy 2.5.4, pandas 3.0.6 (single thread). Runtime: about 30 minutes on one core.

| Analysis (manuscript) | Output | Agreement with the manuscript |
|---|---|---|
| Table 3 | `results_table3.json` | Manuscript values are those of this script (two repetitions, seeds 0 and 1). |
| Leave-one-source-out, Section 4.2 | `results_loso.json` | Identical (R2 0.56 / 0.19; per-source RMSE and bias as reported). |
| Paired within-source analysis, Section 4.7 | `results_paired.json` | Identical (840 pairs, 78 sources, ratios 0.99 / 0.92 / 0.92 with the reported intervals). |
| Joint vs separate modelling, Table 6 | `results_joint.json` | Identical. R2 and RMSE are means of per-fold values (5 folds x 4 repetitions). Spearman rho = 0.04 on all 937 matched mixtures, 0.10 on the 885 unflagged ones. |
| Conformal schemes A-C, Table 5 | `results_conformal.json` | Coverage within 0.01-0.02 and width within 5% of the manuscript (random seeds of the original run differ); for example slump scheme C width 175 mm here, 184 mm in Table 5. Section 4.10 re-runs these schemes with the seeds of this script. |
| Permutation importance, Section 4.7 | `results_perm.json` | Within about 0.5 MPa (CS) and 0.5 mm (slump) of Fig. 5; the ranking is identical. |
| Partial dependence, Fig. 7b | `results_pd.json` | Within 0.5 MPa (e.g. CV-RCA = 20%: 36.9 to 28.3 MPa here, 36.4 to 28.7 in the figure). |

Sensitivity of the perceptron: with the documented settings (two layers of 64 units, alpha = 0.01, Adam, early stopping,
standardised inputs and targets) the random-CV R2 varied between seeds from 0.82 to 0.84 (CS) and 0.74 to 0.80 (slump)
and the grouped R2 from 0.14 to 0.31 (CS) and -0.24 to -0.38 (slump); the MLP row of Table 3 is the two-seed mean
of this script. Results of other libraries' versions may differ slightly.
