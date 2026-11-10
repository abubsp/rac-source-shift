# Source-shift validation of machine-learning models for recycled aggregate concrete (RAC)

Code accompanying the manuscript "A source-shift validation framework for machine-learning models of recycled
aggregate concrete: diagnostics, learning curves and shift-aware conformal intervals" (Abu Bakor Siddique Patwary).

## Data (not redistributed here)
Open database of Zhang et al., Mendeley Data, doi:10.17632/3xvfyycp8d.1, licence CC BY 4.0.
Download `compressive_strength_database.xlsx` and `slump_database.xlsx` and place them in one folder.

## Layout
| Path | Content | Manuscript section |
|---|---|---|
| `original_scripts/tune.py` | Nested source-grouped tuning of extra-trees, random forest and hist. gradient boosting; engineered features | 3.2, 3.4, 4.3, 4.4 |
| `original_scripts/shap_mc.py` | Monte-Carlo (permutation-sampling) Shapley values of extra-trees | 3.8, 4.7 |
| `original_scripts/xgb_lgbm_shap.py` | Nested tuning of XGBoost and LightGBM and exact TreeSHAP | 3.4, 4.9 |
| `analysis/exp1.py`, `ana1.py` | 20 repetitions of random vs source-grouped CV; source-bootstrap intervals | 4.10 |
| `analysis/exp2.py` | log(1 + slump) target | 4.10 |
| `analysis/exp3.py` | Conformal schemes A-F | 3.6, 3.10, 4.5, 4.10 |
| `analysis/exp4_icc.py` | ICC, design effect, source-level error share | 3.9, 4.11 |
| `analysis/exp5_curve.py`, `exp10_curve2.py` | Learning curves over the number of training sources | 3.9, 4.12 |
| `analysis/exp6_adaptive.py`, `exp8_conformal_more.py` | Novelty-normalised, error-model and grouped CV+ conformal intervals | 3.10, 4.13 |
| `analysis/exp7_learners.py`, `ana7.py`, `exp9_other_conformal.py` | Other learners (RF, HGB, XGBoost, LightGBM) | 4.14 |
| `results/` | Result files (XGBoost/LightGBM/TreeSHAP JSON; CSV/JSON outputs of the analysis scripts; Figure 8) | Tables 7-14, Figure 8 |

## Running
```bash
pip install -r requirements.txt
export RAC_DATA_DIR=/path/to/folder/with/the/two/xlsx/files   # used by analysis/*.py
cd analysis && python exp1.py && python ana1.py   # etc.
```
The three scripts in `original_scripts/` are provided exactly as used for the manuscript. They contain hard-coded
paths (`/mnt/project/` for the data and `/tmp/ms/` for outputs); edit these or create the folders before running.
`xgb_lgbm_shap.py` expects `tune.py` in the same folder and needs `DATA_DIR` set at the top.
All random seeds are fixed in the scripts; runs were single-threaded.

## Citation
Please cite the manuscript and the data set. Add the DOI of this repository after deposition.
