"""Whole path on synthetic files: clinical table + radiomic matrix -> ladder -> statistics.

Uses the real configs/default.yaml so a broken YAML shows up here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rad.config import load_config
from rad.experiment import load_inputs, run_experiment

REPO = Path(__file__).resolve().parents[1]


def _project(tmp_path, with_radiomics=True, n=140):
    rng = np.random.default_rng(11)
    died = rng.integers(0, 2, n)
    clinical = pd.DataFrame({
        "PatientID": [f"LUNG1-{i:03d}" for i in range(n)],
        "age": rng.normal(65, 9, n) + 2 * died,
        "clinical.T.Stage": rng.integers(1, 5, n).astype(float),
        "Clinical.N.Stage": rng.integers(0, 4, n),
        "gender": rng.choice(["male", "female"], n),
        "Histology": rng.choice(["large cell", "nos", "adenocarcinoma", None], n),
        "Survival.time": np.where(died == 1, rng.integers(20, 700, n), rng.integers(740, 3000, n)),
        "deadstatus.event": np.where(died == 1, 1, rng.integers(0, 2, n)),
    })
    clinical["label_2yr"] = died
    clinical.to_csv(tmp_path / "clinical.csv", index=False)

    features = tmp_path / "data" / "features"
    features.mkdir(parents=True)
    if with_radiomics:
        cols = {"patient_id": clinical["PatientID"]}
        for i in range(4):
            cols[f"gtv__original_firstorder_signal{i}"] = rng.normal(size=n) + 1.0 * died
        for roi in ("gtv", "periphery"):
            for i in range(60):
                cols[f"{roi}__original_glcm_noise{i}"] = rng.normal(size=n)
        pd.DataFrame(cols).to_csv(features / "radiomics_wide.csv", index=False)
        names = [c for c in cols if c != "patient_id"]
        icc = pd.Series(np.where(rng.random(len(names)) < 0.8, 0.95, 0.2), index=names, name="icc")
        icc.rename_axis("feature").to_csv(features / "stability_icc.csv")

    cfg = load_config(REPO / "configs" / "default.yaml")
    cfg["_root"] = str(tmp_path)
    cfg["paths"]["clinical_csv"] = "clinical.csv"
    cfg["models"] = ["logreg"]
    cfg["validation"].update(outer_folds=4, repeats=2, inner_folds=3, bootstrap=200, n_jobs=1)
    return cfg


def test_full_ladder_with_statistics(tmp_path):
    cfg = _project(tmp_path)
    X, y, icc, notes = load_inputs(cfg)
    assert icc is not None and X.shape[0] == 140
    assert not any("survival" in c.lower() or "label" in c.lower() or "dead" in c.lower() for c in X.columns)

    out = run_experiment(cfg, X, y, icc, out_dir=tmp_path / "out", log=lambda *_: None)
    metrics = pd.read_csv(out / "metrics.csv")
    assert set(metrics["set"]) == {"M0", "M1", "M2", "M5", "M6"}
    assert set(metrics.loc[metrics["set"] != "M0", "model"]) == {"logreg"}
    auc = metrics.set_index(["set", "model"])["auc"]
    assert auc[("M0", "prior")] == np.float64(auc[("M0", "prior")]) and abs(auc[("M0", "prior")] - 0.5) < 0.03
    assert auc[("M2", "logreg")] > 0.7                      # the planted signal is in the GTV block
    assert (metrics["auc_ci_low"] <= metrics["auc"]).all() and (metrics["auc"] <= metrics["auc_ci_high"]).all()

    comparisons = pd.read_csv(out / "comparisons.csv")
    assert {"M2", "M6"} <= set(comparisons["a"]) and (comparisons["p_holm"] >= comparisons["p"]).all()

    assert (out / "run.json").exists() and (out / "oof_predictions.csv").exists()


def test_clinical_only_run_works_before_radiomics_exist(tmp_path):
    cfg = _project(tmp_path, with_radiomics=False)
    X, y, icc, notes = load_inputs(cfg)
    assert icc is None and any("radiomics_wide.csv not found" in n for n in notes)
    out = run_experiment(cfg, X, y, icc, out_dir=tmp_path / "out", log=lambda *_: None)
    metrics = pd.read_csv(out / "metrics.csv")
    assert set(metrics["set"]) == {"M0", "M1"}
