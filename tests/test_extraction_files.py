"""Driver logic that does not need DICOM: patient discovery and combining per-patient files."""

import json

import pandas as pd

from rad.extraction import combine, list_patients


def _cfg(tmp_path):
    return {
        "_root": str(tmp_path),
        "paths": {"raw_root": "data/raw", "features_dir": "data/features"},
        "dicom": {"inbox_subdir": "inbox"},
    }


def test_only_folders_with_an_inbox_are_patients(tmp_path):
    for name in ("P002", "P001"):
        (tmp_path / "data" / "raw" / name / "inbox").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "P003").mkdir()          # no inbox
    (tmp_path / "data" / "raw" / "notes.txt").write_text("x")
    assert list_patients(_cfg(tmp_path)) == ["P001", "P002"]
    assert list_patients({**_cfg(tmp_path), "paths": {"raw_root": "nope", "features_dir": "f"}}) == []


def test_combine_keeps_ok_patients_and_reports_the_rest(tmp_path):
    cfg = _cfg(tmp_path)
    per = tmp_path / "data" / "features" / "per_patient"
    per.mkdir(parents=True)
    for pid, status, problems in [("P001", "ok", []), ("P002", "excluded", ["gtv: empty mask"]), ("P003", "ok", [])]:
        (per / f"{pid}.json").write_text(json.dumps({"patient_id": pid, "status": status, "problems": problems}))
    pd.DataFrame([{"patient_id": "P001", "gtv__a": 1.0}]).to_csv(per / "P001.csv", index=False)
    pd.DataFrame([{"patient_id": "P003", "gtv__a": 3.0}]).to_csv(per / "P003.csv", index=False)

    wide = combine(cfg)
    assert list(wide["patient_id"]) == ["P001", "P003"]
    audit = pd.read_csv(tmp_path / "data" / "features" / "audit.csv")
    assert audit.set_index("patient_id").loc["P002", "problems"] == "gtv: empty mask"
    assert (tmp_path / "data" / "features" / "radiomics_wide.csv").exists()
