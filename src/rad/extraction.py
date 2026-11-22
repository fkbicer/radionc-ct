"""Per-patient extraction driver: DICOM -> validated masks -> feature rows.

Output under ``paths.features_dir``:
  per_patient/<id>.csv   one row of features (lets a long run resume)
  per_patient/<id>.json  audit record for the patient
  radiomics_wide.csv     one row per patient, columns ``<roi>__<feature>``
  audit.csv              every patient, status and problems
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .config import resolve


def _dirs(cfg):
    base = resolve(cfg, cfg["paths"]["features_dir"])
    return base, base / "per_patient"


def list_patients(cfg):
    """Every ``data/raw/<id>/inbox`` folder, sorted by id."""
    raw = resolve(cfg, cfg["paths"]["raw_root"])
    inbox = cfg["dicom"]["inbox_subdir"]
    if not raw.is_dir():
        return []
    return sorted(p.name for p in raw.iterdir() if (p / inbox).is_dir())


def extract_patient(pid, cfg, force=False):
    import pydicom

    from . import dicom_io, preprocess, radiomics

    _, per_patient = _dirs(cfg)
    per_patient.mkdir(parents=True, exist_ok=True)
    audit_path = per_patient / f"{pid}.json"
    if audit_path.exists() and not force:
        return json.loads(audit_path.read_text())

    record = {"patient_id": pid, "status": "ok", "problems": []}
    try:
        inbox = resolve(cfg, cfg["paths"]["raw_root"]) / pid / cfg["dicom"]["inbox_subdir"]
        scan = dicom_io.scan_inbox(inbox)
        slices, failed = dicom_io.load_ct_slices(scan.ct_paths)
        pattern = cfg["dicom"]["patient_id_pattern"]
        tags = cfg["dicom"].get("must_be_empty_tags", [])
        dicom_io.audit_identity(slices[0], pattern, tags)
        rtstruct = pydicom.dcmread(scan.rtstruct_path, stop_before_pixels=True)
        dicom_io.audit_identity(rtstruct, pattern, tags)

        volume = preprocess.build_volume(slices, cfg["dicom"]["missing_slice"])
        masks, info = preprocess.build_masks(
            rtstruct, volume, cfg["rois"], dicom_io.find_roi, dicom_io.contour_polygons
        )
        stats, problems = preprocess.audit_masks(masks, info, volume, cfg["audit"])
        record.update(stats)
        record.update(
            n_ct_slices=len(slices),
            unreadable_ct_slices=failed,
            spacing_xyz="x".join(f"{v:.4g}" for v in volume.spacing),
            problems=problems,
            status="excluded" if problems else "ok",
        )
        if not problems:
            extractor = radiomics.build_extractor(cfg["radiomics"]["params"])
            row = {"patient_id": pid}
            for roi, mask in masks.items():
                row.update(radiomics.extract_features(extractor, volume, mask, roi))
            pd.DataFrame([row]).to_csv(per_patient / f"{pid}.csv", index=False)
    except Exception as exc:  # one bad patient must not stop a batch; it is reported instead
        record.update(status="failed", problems=[f"{type(exc).__name__}: {exc}"])
    audit_path.write_text(json.dumps(record, indent=2, default=str))
    return record


def run_extraction(cfg, patient_ids=None, workers=1, force=False):
    base, _ = _dirs(cfg)
    base.mkdir(parents=True, exist_ok=True)
    if patient_ids is None:
        patient_ids = list_patients(cfg)
    print(f"extracting {len(patient_ids)} patients, workers={workers}", flush=True)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(extract_patient, p, cfg, force) for p in patient_ids]
            for i, future in enumerate(futures, 1):
                record = future.result()
                print(f"  {i}/{len(futures)} {record['patient_id']} {record['status']}", flush=True)
    else:
        for i, pid in enumerate(patient_ids, 1):
            record = extract_patient(pid, cfg, force)
            print(f"  {i}/{len(patient_ids)} {pid} {record['status']}", flush=True)
    return combine(cfg)


def combine(cfg):
    """Merge the per-patient files into the wide matrix and the audit table."""
    base, per_patient = _dirs(cfg)
    base.mkdir(parents=True, exist_ok=True)
    audits = [json.loads(p.read_text()) for p in sorted(per_patient.glob("*.json"))]
    audit = pd.DataFrame(audits)
    if not audit.empty:
        audit["problems"] = audit["problems"].apply(lambda v: "; ".join(v) if isinstance(v, list) else v)
        audit.to_csv(base / "audit.csv", index=False)
    ok = [a["patient_id"] for a in audits if a.get("status") == "ok"]
    frames = [pd.read_csv(per_patient / f"{pid}.csv") for pid in ok if (per_patient / f"{pid}.csv").exists()]
    wide = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["patient_id"])
    wide.to_csv(base / "radiomics_wide.csv", index=False)
    print(f"radiomics_wide.csv: {wide.shape[0]} patients x {wide.shape[1] - 1} features; "
          f"{len(audits) - len(ok)} patients not usable (see audit.csv)")
    return wide
