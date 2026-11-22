"""DICOM discovery, identity audit and RTSTRUCT contour access.

The inbox may be flat or nested and contains CT slices and contour files
mixed together; nothing is assumed from file names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class AnonymityError(RuntimeError):
    """A header carries identity-like values; processing must stop."""


class DataError(RuntimeError):
    """The inbox does not contain what the pipeline needs."""


def normalize_name(name):
    """'GTV-1' -> 'gtv1'. Matching is exact on this form, so 'Lung-GTV' != 'gtv'."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


@dataclass
class Scan:
    ct_paths: list
    ct_series_uid: str
    frame_uid: str
    rtstruct_path: Path
    n_files: int
    n_unreadable: int
    n_other: int


def scan_inbox(inbox):
    """Find the planning CT series (the largest one) and its RTSTRUCT."""
    import pydicom

    inbox = Path(inbox)
    ct = {}
    rtstructs = []
    n_files = n_unreadable = n_other = 0
    for path in sorted(p for p in inbox.rglob("*") if p.is_file() and p.name != ".gitkeep"):
        n_files += 1
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            n_unreadable += 1
            continue
        modality = getattr(ds, "Modality", None)
        if modality == "CT":
            uid = str(ds.SeriesInstanceUID)
            ct.setdefault(uid, []).append((path, str(getattr(ds, "FrameOfReferenceUID", ""))))
        elif modality == "RTSTRUCT":
            rtstructs.append((path, ds))
        else:
            n_other += 1
    if not ct:
        raise DataError(f"No CT slices in {inbox}")
    uid = max(ct, key=lambda u: len(ct[u]))
    frame_uid = ct[uid][0][1]

    chosen = None
    for path, ds in rtstructs:
        refs = {str(f.FrameOfReferenceUID) for f in getattr(ds, "ReferencedFrameOfReferenceSequence", [])}
        if frame_uid and frame_uid in refs:
            chosen = path
            break
    if chosen is None and rtstructs:
        chosen = rtstructs[0][0]
    if chosen is None:
        raise DataError(f"No RTSTRUCT in {inbox}")
    return Scan(
        ct_paths=[p for p, _ in ct[uid]],
        ct_series_uid=uid,
        frame_uid=frame_uid,
        rtstruct_path=chosen,
        n_files=n_files,
        n_unreadable=n_unreadable,
        n_other=n_other,
    )


def load_ct_slices(paths):
    """Read CT slices sorted bottom to top. Unreadable slices are counted and skipped."""
    import pydicom

    slices = []
    failed = 0
    for path in paths:
        try:
            ds = pydicom.dcmread(path)
            _ = ds.pixel_array
        except Exception:
            failed += 1
            continue
        slices.append(ds)
    if not slices:
        raise DataError("No readable CT slices")
    slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    unique = [slices[0]]
    for ds in slices[1:]:
        if abs(float(ds.ImagePositionPatient[2]) - float(unique[-1].ImagePositionPatient[2])) > 1e-3:
            unique.append(ds)
    return unique, failed


def audit_identity(ds, id_pattern, must_be_empty_tags=()):
    """Stop (AnonymityError) if the header looks like it identifies a person."""
    pid = str(getattr(ds, "PatientID", ""))
    if not re.fullmatch(id_pattern, pid):
        raise AnonymityError(f"PatientID {pid!r} does not match the allowed pseudonym pattern")
    name = str(getattr(ds, "PatientName", "") or "")
    if name and name != pid and not re.fullmatch(id_pattern, name):
        raise AnonymityError("PatientName holds a value other than the pseudonym")
    for tag in must_be_empty_tags:
        if str(getattr(ds, tag, "") or "").strip():
            raise AnonymityError(f"{tag} is not empty")
    return pid


def find_roi(rtstruct, aliases):
    """(roi_number, roi_name) of the first ROI whose normalized name matches an alias."""
    wanted = {normalize_name(a) for a in aliases}
    for roi in rtstruct.StructureSetROISequence:
        if normalize_name(roi.ROIName) in wanted:
            return int(roi.ROINumber), str(roi.ROIName)
    return None, None


def roi_names(rtstruct):
    return [str(roi.ROIName) for roi in rtstruct.StructureSetROISequence]


def contour_polygons(rtstruct, roi_number):
    """List of (n, 3) arrays in patient millimetres, one per contour plane."""
    for seq in getattr(rtstruct, "ROIContourSequence", []):
        if int(seq.ReferencedROINumber) != roi_number:
            continue
        return [
            np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
            for c in (getattr(seq, "ContourSequence", None) or [])
        ]
    return []
