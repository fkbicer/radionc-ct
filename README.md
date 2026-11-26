# radionc-ct

Radiomics from treatment-planning CT, combined with clinical variables, to predict outcome in lung cancer. The pipeline is built and tested on the public TCIA Lung1 (NSCLC-Radiomics) stage III cohort (286 patients analysed, target: death within 2 years). The same pipeline is meant for hospital CT data, where the target becomes a three-class problem: no metastasis, oligometastatic, polymetastatic.

Raw patient images never enter git. Folders carry a study ID only (`P001`, `LUNG1-001`); no name, protocol number or birth date. Derived, pseudonymized feature tables may be versioned.

## Pipeline

1. **Extract** (`rad extract`): the largest CT series and its RTSTRUCT are read, HU converted, masks built for the tumor (GTV) and a 5 mm ring around it (ring excludes the GTV), then PyRadiomics computes 107 features per region (shape, firstorder, GLCM, GLRLM, GLSZM, NGTDM, GLDM; original image, 2 mm resampling, `binWidth=25`). Total: 214 features per patient. Settings live in `configs/default.yaml`.
2. **Dataset** (`rad dataset`): joins radiomics, 5 clinical variables (age, sex, T stage, N stage, histology; 9 columns) and the label `label_2yr`. Survival time, event and label are never used as features (`LeakageError`).
3. **Selection** (inside every training fold): missing-rate filter, near-constant filter, correlation clusters (|rho| > 0.9, keep one per cluster), then elastic net keeps at most 15 radiomic features. Clinical columns are imputed and scaled only.
4. **Models and validation** (`rad run`): logistic regression, linear SVM, random forest and XGBoost on five feature sets (M0 baseline, M1 clinical, M2 GTV, M5 GTV + ring, M6 M5 + clinical). Repeated stratified nested cross-validation (5 folds x 10 repeats, inner 3 folds), split by patient. Bootstrap confidence intervals, paired bootstrap with Holm correction for set comparisons, optional permutation test (`--permutations N`).

## Results (Lung1, stage III)

Mean out-of-fold AUC over 5 x 10 nested CV:

| Set | Logistic | SVM | Random forest | XGBoost |
|---|---|---|---|---|
| M0 baseline | 0.491 | | | |
| M1 clinical | 0.491 | 0.475 | 0.527 | 0.569 |
| M2 GTV | 0.579 | 0.581 | 0.583 | 0.574 |
| M5 GTV + ring | 0.595 | 0.596 | **0.606** | 0.582 |
| M6 M5 + clinical | 0.591 | 0.588 | **0.606** | 0.593 |

- Best: random forest on M5, AUC 0.606 (95% CI 0.550-0.686). Clinical data alone is near chance.
- Adding the ring (M5 - M2, +0.021) or clinical data (M6 - M5, +0.001) gave no demonstrable gain; both intervals include zero.
- Most selected features: ring NGTDM Strength (98% of folds), ring first-order Maximum (90%), GTV GLDM Large Dependence Low Gray Level (86%).
- Caveats: the best value is the maximum over 17 configurations (optimistic), there is no independent validation, and the permutation test was not run. Lung1 contains CT only (no PET).

## Where things are

```text
data/
  raw/<id>/inbox/        incoming DICOM, untouched (git-ignored)
  features/              pseudonymized outputs (versioned)
    radiomics_wide.csv   one row per patient, 214 radiomic columns
    dataset.csv          radiomics + clinical + label (model input)
    audit.csv            status and problems of every patient
    per_patient/         per-patient feature rows
outputs/<timestamp>/     run results, git-ignored: metrics.csv, comparisons.csv,
                         selected_features.csv, oof_predictions.csv, report.html, run.json
lung1_stage3_cohort.csv  clinical table of the stage III cohort
```

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "numpy>=1.23,<2" wheel
pip install -r requirements.txt --no-build-isolation   # pyradiomics needs it
pip install -e .

python -m rad extract --workers 4      # DICOM -> features (resumable); check data/features/audit.csv
python -m rad dataset                  # clinical + label + radiomics -> data/features/dataset.csv
python -m rad select                   # what the selection chain keeps on all data (inspection only)
python -m rad run                      # model ladder x models, nested CV -> outputs/<timestamp>/
python -m rad explain --set M6 --model logreg   # coefficients / SHAP
python -m rad report                   # rebuild report.html of the last run
python -m rad quality --run outputs/<timestamp> # calibration tables, contour review images
pytest
```

`quality` writes calibration tables and contour overlays (GTV red, ring teal) into the run folder; images are derived from DICOM and stay out of git. `reviewed_visually` in `review.csv` stays `False` until an expert has checked them.

## Notes

- Contours are rasterised by pixel centre (the old PIL polygon fill inflated volumes by about 9%).
- Missing CT slices are interpolated from neighbours (`dicom.missing_slice`) and counted in `audit.csv`.
- Wavelet and LoG features are off by default (enable in `configs/default.yaml`) and were not part of the results above.
- Tests cover HU conversion, geometry, phantom volume, ring thickness, ID audit, ROI matching, label rules, leakage guard and table joins. PyRadiomics phantom tests need `pyradiomics` and `SimpleITK`. DICOM reading should still be checked on real data.
