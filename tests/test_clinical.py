from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rad.cli import main
from rad.clinical import (
    LeakageError,
    build_clinical,
    build_dataset,
    columns_for,
    join_matrix,
    make_label,
)
from rad.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, df, **over):
    path = tmp_path / "clin.csv"
    df.to_csv(path, index=False)
    clinical = dict(
        id_column="PatientID", label_column="label_2yr", time_column="Survival.time",
        event_column="deadstatus.event", horizon_days=730,
        numeric=["age", "clinical.T.Stage"], categorical=["gender", "Histology"],
    )
    clinical.update(over)
    return {"_root": str(tmp_path), "paths": {"clinical_csv": str(path), "features_dir": "features"}, "clinical": clinical}


@pytest.fixture
def frame():
    return pd.DataFrame({
        "PatientID": ["A", "B", "C", "D", "E"],
        "age": [60.0, np.nan, 70, 55, 65],
        "clinical.T.Stage": [1, 2, np.nan, 4, 3],
        "gender": ["male", "female", "male", "male", "female"],
        "Histology": ["large cell", None, "squamous cell carcinoma", "nos", "large cell"],
        "Survival.time": [100, 900, 2000, 300, 500],
        "deadstatus.event": [1, 1, 0, 1, 0],
        "label_2yr": [1, 0, 0, 1, np.nan],        # E: censored at 500 days, cannot be labelled
    })


def test_label_rule_and_censoring(frame):
    c = dict(label_column=None, time_column="Survival.time", event_column="deadstatus.event", horizon_days=730)
    y, info = make_label(frame, c)
    assert y.tolist()[:4] == [1, 0, 0, 1] and np.isnan(y[4])
    assert info["censored_before_horizon"] == 1


def test_features_exclude_followup_and_keep_missing(tmp_path, frame):
    X, y, report = build_clinical(_cfg(tmp_path, frame))
    assert list(y.index) == ["A", "B", "C", "D"]            # E dropped, unlabelled
    assert not any("survival" in c or "dead" in c or "label" in c for c in X.columns)
    assert X.loc["B", "clin__age"] != X.loc["B", "clin__age"]  # still NaN, imputed later in the fold
    assert "clin__gender_male" in X.columns and "clin__gender_female" not in X.columns
    assert "clin__histology_unknown" in X.columns
    assert report["class_counts"] == {0: 2, 1: 2}


def test_followup_columns_are_refused_as_features(tmp_path, frame):
    with pytest.raises(LeakageError):
        build_clinical(_cfg(tmp_path, frame, numeric=["age", "Survival.time"]))


def test_inconsistent_given_label_is_refused(tmp_path, frame):
    frame.loc[0, "label_2yr"] = 0                            # died at 100 days but labelled 0
    with pytest.raises(ValueError):
        build_clinical(_cfg(tmp_path, frame))


def test_join_reports_unmatched_and_groups_columns(tmp_path, frame):
    Xc, y, _ = build_clinical(_cfg(tmp_path, frame))
    rad = pd.DataFrame({"patient_id": ["A", "B", "Z"], "gtv__f1": [1.0, 2, 3], "periphery__f1": [4.0, 5, 6]})
    X, yy, rep = join_matrix(rad, Xc, y)
    assert list(X.index) == ["A", "B"] and len(yy) == 2
    assert rep["labelled_without_radiomics"] == 2 and rep["radiomics_without_label"] == 1
    assert columns_for(X, ["gtv"]) == ["gtv__f1"]
    assert set(columns_for(X, ["gtv", "periphery"])) == {"gtv__f1", "periphery__f1"}
    assert all(c.startswith("clin__") for c in columns_for(X, ["clin"]))


def test_dataset_is_clinical_only_until_radiomics_exist(tmp_path, frame):
    cfg = _cfg(tmp_path, frame)
    X, y, notes = build_dataset(cfg)
    assert all(c.startswith("clin__") for c in X.columns) and y.name == "label"
    assert any("no radiomic features yet" in n for n in notes)


def test_dataset_joins_radiomics_when_present(tmp_path, frame):
    cfg = _cfg(tmp_path, frame)
    (tmp_path / "features").mkdir()
    pd.DataFrame({"patient_id": ["A", "B", "C"], "gtv__f": [1.0, 2.0, 3.0]}).to_csv(
        tmp_path / "features" / "radiomics_wide.csv", index=False)
    X, y, notes = build_dataset(cfg)
    assert list(X.index) == ["A", "B", "C"] and "gtv__f" in X.columns
    assert y.tolist() == [1, 0, 0]
    assert any("join:" in n and "'labelled_without_radiomics': 1" in n for n in notes)


def test_dataset_command_writes_the_table(tmp_path, frame, capsys):
    cfg = _cfg(tmp_path, frame)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    import yaml
    (config_dir / "t.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))
    main(["--config", str(config_dir / "t.yaml"), "dataset"])
    out = pd.read_csv(tmp_path / "features" / "dataset.csv", index_col=0)
    assert list(out.index) == ["A", "B", "C", "D"] and "label" in out.columns
    assert "4 patients" in capsys.readouterr().out


@pytest.mark.skipif(not (REPO / "lung1_stage3_cohort.csv").exists(), reason="Lung1 table not in the repo")
def test_real_lung1_table_passes_all_checks():
    cfg = load_config(REPO / "configs" / "default.yaml")
    X, y, report = build_clinical(cfg)
    assert report["n_labelled"] == 287 and report["class_counts"] == {0: 114, 1: 173}
    assert report["censored_before_horizon"] == 0
    assert not any(k in c for c in X.columns for k in ("survival", "dead", "label", "overall_stage", "m_stage"))
