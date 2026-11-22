"""CT volume assembly, contour rasterization and mask audit.

Resampling to isotropic voxels is done by PyRadiomics (``resampledPixelSpacing``
in the config, report Appendix B), so the volume here stays on the native grid.
Assumes axial slices with identity orientation; anything else raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


class GeometryError(RuntimeError):
    pass


# --- physical meaning of DICOM values (report 4.2) --------------------------

def raw_to_hu(raw, slope, intercept):
    """HU = m * raw + b (report Equation 1)."""
    return np.asarray(raw, dtype=np.float32) * float(slope) + float(intercept)


def pixel_to_mm(row, col, origin, spacing):
    """(row i, column j) -> (x, y) in mm (report Equation 2, axial identity orientation)."""
    return origin[0] + col * spacing[0], origin[1] + row * spacing[1]


def mm_to_pixel(x, y, origin, spacing):
    return (y - origin[1]) / spacing[1], (x - origin[0]) / spacing[0]


# --- volume ----------------------------------------------------------------

@dataclass
class Volume:
    array: np.ndarray           # (z, y, x) in HU
    origin: np.ndarray          # mm, position of the first voxel
    spacing: tuple              # (x, y, z) in mm
    missing: list = field(default_factory=list)       # slice indices without data
    interpolated: bool = False


def build_volume(slices, missing_slice="interpolate"):
    ds0 = slices[0]
    iop = np.asarray(ds0.ImageOrientationPatient, dtype=float)
    if not np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=1e-3):
        raise GeometryError(f"Non-axial or rotated orientation: {iop.tolist()}")

    row_spacing, col_spacing = (float(v) for v in ds0.PixelSpacing)
    origin = np.asarray(ds0.ImagePositionPatient, dtype=float)
    zs = np.asarray([float(ds.ImagePositionPatient[2]) for ds in slices])
    step = float(np.median(np.diff(zs))) if len(zs) > 1 else float(getattr(ds0, "SliceThickness", 1.0))
    n_slices = int(round((zs[-1] - zs[0]) / step)) + 1 if len(zs) > 1 else 1

    volume = np.full((n_slices, ds0.Rows, ds0.Columns), -1024, dtype=np.float32)
    present = set()
    for ds, z in zip(slices, zs):
        k = int(round((z - origin[2]) / step))
        raw = ds.pixel_array.astype(np.int32)
        if int(getattr(ds, "PixelRepresentation", 0)) == 0:
            bits = int(getattr(ds, "BitsStored", 16))
            raw = raw & ((1 << bits) - 1)
        volume[k] = raw_to_hu(raw, getattr(ds, "RescaleSlope", 1.0), getattr(ds, "RescaleIntercept", 0.0))
        present.add(k)

    missing = [k for k in range(n_slices) if k not in present]
    interpolated = False
    if missing and missing_slice == "interpolate":
        have = sorted(present)
        for k in missing:
            lower = [p for p in have if p < k]
            upper = [p for p in have if p > k]
            if lower and upper:
                a, b = lower[-1], upper[0]
                w = (k - a) / (b - a)
                volume[k] = (1 - w) * volume[a] + w * volume[b]
            elif lower:
                volume[k] = volume[lower[-1]]
            elif upper:
                volume[k] = volume[upper[0]]
        interpolated = True
    return Volume(volume, origin, (col_spacing, row_spacing, abs(step)), missing, interpolated)


# --- masks ------------------------------------------------------------------

def fill_polygon(height, width, rows, cols):
    """Pixels whose centre lies inside the polygon (even-odd rule).

    Pixel (i, j) has its centre at integer (row, col). Drawing the outline too
    (as PIL's polygon fill does) adds about half a voxel along the whole border
    and inflates volumes, so it is not used.
    """
    rows = np.asarray(rows, dtype=float)
    cols = np.asarray(cols, dtype=float)
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(rows) < 3:
        return mask
    r0, r1 = max(int(np.floor(rows.min())), 0), min(int(np.ceil(rows.max())), height - 1)
    c0, c1 = max(int(np.floor(cols.min())), 0), min(int(np.ceil(cols.max())), width - 1)
    if r0 > r1 or c0 > c1:
        return mask
    rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    inside = np.zeros(rr.shape, dtype=bool)
    for ya, xa, yb, xb in zip(rows, cols, np.roll(rows, -1), np.roll(cols, -1)):
        if ya == yb:
            continue
        crosses = (ya > rr) != (yb > rr)
        x_at_row = xa + (rr - ya) * (xb - xa) / (yb - ya)
        inside ^= crosses & (cc < x_at_row)
    mask[r0:r1 + 1, c0:c1 + 1] = inside
    return mask


def rasterize(polys, shape, origin, spacing, tol=0.2):
    """Rasterize contour planes onto the CT grid.

    Polygons on the same plane are combined with XOR (even-odd rule), so a nested
    polygon becomes a hole. Returns (mask, number of polygons that missed every slice).
    """
    mask = np.zeros(shape, dtype=np.uint8)
    off_plane = 0
    for poly in polys:
        cols = (poly[:, 0] - origin[0]) / spacing[0]
        rows = (poly[:, 1] - origin[1]) / spacing[1]
        k = float(np.median((poly[:, 2] - origin[2]) / spacing[2]))
        kk = int(round(k))
        if abs(k - kk) > tol or not (0 <= kk < shape[0]):
            off_plane += 1
            continue
        mask[kk] ^= fill_polygon(shape[1], shape[2], rows, cols)
    return mask, off_plane


def expand_ring(mask, margin_mm, spacing):
    """Voxels within margin_mm of the mask, excluding the mask itself."""
    sampling = (spacing[2], spacing[1], spacing[0])
    distance = ndimage.distance_transform_edt(mask == 0, sampling=sampling)
    return ((distance <= margin_mm) & (mask == 0)).astype(np.uint8)


def build_masks(rtstruct, volume, rois_cfg, find_roi, contour_polygons):
    """Masks for every configured ROI, plus a record of where each came from."""
    masks, info = {}, {}
    for roi, spec in rois_cfg.items():
        number, name = find_roi(rtstruct, spec["aliases"])
        record = {"source": None, "roi_name": name, "off_plane": 0, "n_polygons": 0}
        mask = None
        if number is not None:
            polys = contour_polygons(rtstruct, number)
            mask, off = rasterize(polys, volume.array.shape, volume.origin, volume.spacing)
            record.update(source="contour", off_plane=off, n_polygons=len(polys))
        elif spec.get("derive_from") in masks:
            mask = expand_ring(masks[spec["derive_from"]], float(spec["margin_mm"]), volume.spacing)
            record["source"] = "derived"
        if mask is not None and spec.get("exclude") in masks:
            mask = (mask.astype(bool) & ~masks[spec["exclude"]].astype(bool)).astype(np.uint8)
        if mask is not None:
            masks[roi] = mask
        info[roi] = record
    return masks, info


def audit_masks(masks, info, volume, audit_cfg):
    """Report problems (FR2): empty, implausible volume, many off-plane polygons, overlap."""
    voxel_cm3 = float(np.prod(volume.spacing)) / 1000.0
    problems, stats = [], {}
    for roi, record in info.items():
        mask = masks.get(roi)
        if mask is None:
            problems.append(f"{roi}: not found")
            continue
        voxels = int(mask.sum())
        cm3 = voxels * voxel_cm3
        stats[f"{roi}_cm3"] = round(cm3, 3)
        stats[f"{roi}_source"] = record["source"]
        if voxels == 0:
            problems.append(f"{roi}: empty mask")
            continue
        if not (audit_cfg["min_volume_cm3"] <= cm3 <= audit_cfg["max_volume_cm3"]):
            problems.append(f"{roi}: volume {cm3:.1f} cm3 outside the expected range")
        n = max(record["n_polygons"], 1)
        if record["off_plane"] / n > audit_cfg["max_off_plane_fraction"]:
            problems.append(f"{roi}: {record['off_plane']}/{record['n_polygons']} polygons missed the slice grid")
    names = [r for r in masks if masks[r].any()]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.any(masks[a].astype(bool) & masks[b].astype(bool)):
                problems.append(f"{a} and {b} overlap")
    stats["n_missing_slices"] = len(volume.missing)
    return stats, problems
