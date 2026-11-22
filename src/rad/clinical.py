"""Clinical table, label, and the join with the radiomic matrix (report 4.5, FR7).

Only information known before treatment becomes a feature. Follow-up columns
(survival time, event) generate the label and are refused as features.
Missing values are left as NaN here; they are imputed inside each training fold.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import resolve

CLIN = "clin"


class LeakageError(RuntimeError):
    """A follow-up column was configured as a model input."""


def _read(cfg):
    return pd.read_csv(resolve(cfg, cfg["paths"]["clinical_csv"]))


def read_ids(cfg):
    return _read(cfg)[cfg["clinical"]["id_column"]].astype(str).tolist()


def _column_name(source):
    return f"{CLIN}__" + re.sub(r"[^0-9a-z]+", "_", source.lower()).strip("_")


def make_label(df, c):
    """2-year (horizon) mortality label, checked against the follow-up columns.

    1 = died before the horizon, 0 = followed beyond it. A patient censored
    before the horizon cannot be labelled and gets NaN.
    """
    time = pd.to_numeric(df[c["time_column"]], errors="coerce")
    event = pd.to_numeric(df[c["event_column"]], errors="coerce")
    horizon = float(c["horizon_days"])
    derived = pd.Series(np.nan, index=df.index)
    derived[(event == 1) & (time < horizon)] = 1.0
    derived[time >= horizon] = 0.0
    info = {"censored_before_horizon": int(((event == 0) & (time < horizon)).sum())}

    given_name = c.get("label_column")
    if given_name and given_name in df:
        given = pd.to_numeric(df[given_name], errors="coerce")
        known = derived.notna()
        mismatches = int((given[known] != derived[known]).sum())
        unexplained = int((given.notna() & ~known).sum())
        if mismatches or unexplained:
            raise ValueError(
                f"{given_name} disagrees with the follow-up columns "
                f"({mismatches} mismatches, {unexplained} labelled but not derivable)"
            )
        info["label_source"] = given_name
        return given, info
    info["label_source"] = "derived"
    return derived, info


def build_clinical(cfg):
    """(X, y, report): X indexed by patient id with ``clin__*`` columns, y a 0/1 Series."""
    c = cfg["clinical"]
    used = list(c.get("numeric", [])) + list(c.get("categorical", []))
    forbidden = {c["time_column"], c["event_column"], c.get("label_column")} - {None}
    clash = forbidden & set(used)
    if clash:
        raise LeakageError(f"Follow-up columns cannot be features: {sorted(clash)}")

    df = _read(cfg)
    if not df[c["id_column"]].is_unique:
        raise ValueError("Patient ids are not unique")
    df = df.set_index(df[c["id_column"]].astype(str))
    df.index.name = "patient_id"

    y, info = make_label(df, c)
    columns = {}
    for source in c.get("numeric", []):
        columns[_column_name(source)] = pd.to_numeric(df[source], errors="coerce")
    for source in c.get("categorical", []):
        values = df[source].astype("string").str.strip().str.lower().fillna("unknown")
        levels = sorted(values.unique())
        if len(levels) == 2:
            levels = levels[1:]          # one indicator for a two-level variable
        for level in levels:
            columns[f"{_column_name(source)}_{re.sub(r'[^0-9a-z]+', '_', level).strip('_')}"] = (
                (values == level).astype(float)
            )
    X = pd.DataFrame(columns, index=df.index)

    keep = y.notna()
    report = {
        **info,
        "n_rows": int(len(df)),
        "n_labelled": int(keep.sum()),
        "class_counts": {int(k): int(v) for k, v in y[keep].value_counts().sort_index().items()},
        "missing_values": {k: int(v) for k, v in X.isna().sum().items() if v},
    }
    return X[keep], y[keep].astype(int), report


def join_matrix(radiomics_wide, clinical_X, y):
    """One patient, one label, one feature row (report 6.1). Reports unmatched ids."""
    rad = radiomics_wide.copy()
    rad["patient_id"] = rad["patient_id"].astype(str)
    rad = rad.set_index("patient_id")
    ids = [i for i in y.index if i in rad.index]
    report = {
        "n_joined": len(ids),
        "labelled_without_radiomics": int(len(y) - len(ids)),
        "radiomics_without_label": int(len(set(rad.index) - set(y.index))),
    }
    X = pd.concat([rad.loc[ids], clinical_X.loc[ids]], axis=1)
    return X, y.loc[ids].to_numpy(), report


def build_dataset(cfg):
    """(X, y, notes): clinical features joined with radiomics when ``radiomics_wide.csv`` exists.

    Without the radiomic matrix only the clinical columns are returned, so the
    clinical side can be checked before any DICOM has been processed.
    """
    clinical_X, y, report = build_clinical(cfg)
    notes = [f"clinical: {report}"]
    path = resolve(cfg, cfg["paths"]["features_dir"]) / "radiomics_wide.csv"
    radiomics = pd.read_csv(path) if path.exists() else None
    if radiomics is not None and len(radiomics):
        X, y_array, join_report = join_matrix(radiomics, clinical_X, y)
        notes.append(f"join: {join_report}")
        return X, pd.Series(y_array, index=X.index, name="label"), notes
    notes.append("no radiomic features yet (run `python -m rad extract`): clinical columns only")
    return clinical_X, y.rename("label"), notes


def columns_for(X, groups):
    """Columns of one ladder input set; a group is ``clin`` or an ROI name."""
    prefixes = tuple(f"{g}__" for g in groups)
    return [col for col in X.columns if col.startswith(prefixes)]
