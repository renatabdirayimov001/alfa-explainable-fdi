# Explainable Fault Detection and Isolation for Fixed-Wing UAVs Using Physics-Informed Residuals

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23860263.svg)](https://doi.org/10.5281/zenodo.23860263)

MATLAB code and results for the paper submitted to the International Workshop on Applied AI & Mathematical Methods for Real-world Applications (A2MRA 2027), held with ANT 2027, Delft, the Netherlands.

## What the code does
- Reads the processed ALFA flight logs (46 flights with ground truth).
- Builds command-minus-measurement residuals (RES), autopilot commands (CMD) and windowed control-effectiveness features (EFF) on 3 s windows.
- Trains a bagged tree ensemble for joint fault detection and isolation (Normal, Engine, Aileron, Rudder, Elevator).
- Evaluates with leave-one-flight-out validation: detection rate, isolation accuracy, detection delay and false alarms per hour.
- Audits SHAP explanations against a priori fault signatures (top-1 hit rate and expected mass).

Servo outputs (`rc-out`) are deliberately not used: they directly expose the injected faults.

## Requirements
MATLAB R2021a or newer with the Statistics and Machine Learning Toolbox (tested in MATLAB Online).

## Data
Download the processed ALFA dataset (Keipour et al., 2021): https://doi.org/10.1184/R1/12707963
Unzip it in the repository root so that the folder `processed` (one sub-folder per flight) sits next to `code/`, or change `cfg.dataDir` in `alfa_fdi_main.m`.

## How to run
```matlab
cd code
alfa_fdi_main   % LOFO evaluation of 4 feature sets, SHAP audit, figures (about 30 min)
alfa_post       % operating-point sweep for all sets, tables at tau = 0.75, K = 7 (run in the same session)
```
Outputs are written to `results/` and `results_post/`.

## Main results (tau = 0.75, K = 7)
| Feature set | Detection rate | Isolation acc. | False alarms/h | Median delay (s) |
|---|---|---|---|---|
| RAW | 0.686 | 0.686 | 0.0 | 3.66 |
| RES | 0.771 | 0.771 | 22.7 | 3.85 |
| RES+CMD+EFF (proposed) | 0.857 | 0.857 | 1.9 | 4.49 |
| RAW+RES+CMD+EFF | 0.800 | 0.800 | 0.0 | 3.75 |

The `results/` folder contains the full operating-point sweep, per-class and per-flight results and the SHAP consistency scores.

## Citation
If you use this code, please cite the paper (see `CITATION.cff`) and the ALFA dataset:
A. Keipour, M. Mousaei, S. Scherer, "ALFA: A dataset for UAV fault and anomaly detection," The International Journal of Robotics Research, 40(2-3), 515-520, 2021.

## License
MIT License (see `LICENSE`).
