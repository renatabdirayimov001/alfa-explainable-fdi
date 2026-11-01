# Leakage-Aware Evaluation of Explainable Fault Detection for Fixed-Wing UAVs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23860263.svg)](https://doi.org/10.5281/zenodo.23860263)

MATLAB code and results for the paper submitted to the International Workshop on Applied AI & Mathematical Methods for Real-world Applications (A2MRA 2027), held with ANT 2027, Delft, the Netherlands.

## Summary
This repository accompanies a leakage-aware evaluation of explainable fault detection and isolation (FDI) on the CMU ALFA fixed-wing dataset.

**Leakage finding.** Two logged signals expose the injected faults directly:
- `mavros/vfr_hud` throttle drops to exactly 0 at an injected engine failure (a 1 s threshold rule flags 22/23 engine failures 0.7-1.5 s after onset);
- `mavros/rc-out` servo outputs of a failed surface become constant (a 3 s zero-variance rule flags 13/13 control-surface failures 2.4-3.2 s after onset).

Both are excluded from all learned models. `code/leakage_audit.py` reproduces the rule-based audit.

## What the MATLAB code does
- Reads the processed ALFA flight logs (46 flights with ground truth).
- Builds command-minus-measurement residuals (RES), autopilot commands (CMD) and windowed control-effectiveness features (EFF) on 3 s causal windows.
- Trains a bagged tree ensemble for joint fault detection and isolation (Normal, Engine, Aileron, Rudder, Elevator); isolation uses the K windows ending at the alarm.
- Evaluates with leave-one-flight-out validation, nested selection of the alarm threshold (tau, K), Clopper-Pearson and Poisson 95% intervals and an exact McNemar test.
- Audits held-out SHAP explanations (each window explained by its own fold model, background from that fold's training flights) against a priori fault signatures.

## Requirements
MATLAB R2021a or newer with the Statistics and Machine Learning Toolbox (tested in MATLAB Online). Python 3 with pandas for the leakage audit.

## Data
Download the processed ALFA dataset (Keipour et al., 2021): https://doi.org/10.1184/R1/12707963
Unzip it in the repository root so that the folder `processed` (one sub-folder per flight) sits next to `code/`, or change `cfg.dataDir` in `alfa_fdi_main.m`.

## How to run
```matlab
cd code
alfa_fdi_main   % LOFO evaluation of 4 feature sets, held-out SHAP audit, timing (about 20 min)
alfa_nested     % nested (tau, K) selection, confidence intervals, McNemar test (same session, about 20 min)
```
```bash
python code/leakage_audit.py processed
```

## Main results (nested LOFO, no leaking signals)
| Feature set | Detection rate [95% CI] | Isolation acc. | False alarms/h [95% CI] | Median delay (s) |
|---|---|---|---|---|
| RAW | 0.486 [0.31, 0.66] | 0.486 | 1.89 [0.23, 6.84] | 6.6 |
| RES+CMD+EFF | 0.543 [0.37, 0.71] | 0.514 | 0.95 [0.02, 5.27] | 9.5 |

Exact McNemar test between the two sets: p = 0.69. With the leaking throttle included, the same pipeline reached 0.686 (RAW) and 0.771 (RES+CMD+EFF); these earlier results are kept in `results/v3_with_throttle_leak/` for transparency.

## Citation
If you use this code, please cite the paper (see `CITATION.cff`) and the ALFA dataset:
A. Keipour, M. Mousaei, S. Scherer, "ALFA: A dataset for UAV fault and anomaly detection," The International Journal of Robotics Research, 40(2-3), 515-520, 2021.

## License
MIT License (see `LICENSE`).
