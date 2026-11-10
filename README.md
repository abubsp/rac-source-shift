# Source-shift validation of machine-learning models for recycled aggregate concrete (RAC)

Code accompanying the manuscript "A source-shift validation framework for machine-learning models of recycled
aggregate concrete: diagnostics, learning curves and shift-aware conformal intervals" (Abu Bakor Siddique Patwary).

## Data (not redistributed here)
Open database of Zhang et al., Mendeley Data, doi:10.17632/3xvfyycp8d.1, licence CC BY 4.0.
Download `compressive_strength_database.xlsx` and `slump_database.xlsx` and place them in one folder.

## Files (all in the repository root)
| File(s) | Content | Manuscript section |
|---|---|---|
| `tune.py` | Nested source-grouped tuning of extra-trees, random forest and hist. gradient boosting; engineered features | 3.2, 3.4, 4.3, 4.4 |
| `shap_mc.py` | Monte-Carlo (permutation-sampling) Shapley values of extra-trees | 3.8, 4.7 |
| `xgb_lgbm_shap.py` | Nested tuning of XGBoost and LightGBM and exact TreeSHAP | 3.4, 4.9 |
| `exp1.py`, `ana1.py` | 20 repetitions of random vs source-grouped CV; source-bootstrap intervals | 4.10 |
| `exp2.py` | log(1 + slump) target | 4.10 |
| `exp3.py` | Conformal schemes A-F | 3.6, 3.10, 4.5, 4.10 |
| `exp4_icc.py` | ICC, design effect, source-level error share | 3.9, 4.11 |
| `exp5_curve.py`, `exp10_curve2.py` | Learning curves over the number of training sources | 3.9, 4.12 |
| `exp6_adaptive.py`, `exp8_conformal_more.py` | Novelty-normalised, error-model and grouped CV+ conformal intervals | 3.10, 4.13 |
| `exp7_learners.py`, `ana7.py`, `exp9_other_conformal.py` | Other learners (RF, HGB, XGBoost, LightGBM) | 4.14 |
| `reproduce_analyses.py`, `REPRODUCTION.md`, `results_*.json` | Re-implementation of Table 3, LOSO, permutation importance, joint modelling (Table 6), conformal schemes (Table 5), paired analysis and partial dependence; agreement with the manuscript | 3.5-3.8, 4.1-4.7 |
| `*.json`, `*.csv`, `*.png` | Result files (XGBoost/LightGBM/TreeSHAP JSON; CSV/JSON outputs of the analysis scripts; Figure 8) | Tables 7-14, Figure 8 |

## Running
```bash
pip install -r requirements.txt
export RAC_DATA_DIR=/path/to/folder/with/the/two/xlsx/files   # used by the exp*.py scripts
python exp1.py && python ana1.py   # etc.
```
The three scripts `tune.py`, `shap_mc.py` and `xgb_lgbm_shap.py` are provided exactly as used for the manuscript. They contain hard-coded
paths (`/mnt/project/` for the data and `/tmp/ms/` for outputs); edit these or create the folders before running.
`xgb_lgbm_shap.py` expects `tune.py` in the same folder and needs `DATA_DIR` set at the top.
All random seeds are fixed in the scripts; runs were single-threaded.

## Citation
Please cite the manuscript and the data set. This repository is archived at Zenodo: https://doi.org/10.5281/zenodo.24104984
