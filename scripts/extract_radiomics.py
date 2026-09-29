"""Extract a first radiomics table for one pretreatment planning CT.

Reads data/raw/P001/inbox, rasterizes GTV and the peritumoral contour onto
the CT grid, and writes shape, first-order, and GLCM features. No names or
other DICOM identifiers are stored.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw
from radiomics import featureextractor

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data" / "raw" / "P001" / "inbox"
OUT = ROOT / "data" / "metadata" / "P001_radiomics.csv"
PATIENT_ID = "P001"

# DICOM ROI name -> column label. Lung-GTV contains "gtv", so match exactly.
ROIS = {
    "gtv": "gtv",
    "peritumoralspace": "peritumoral_5mm",
}

SHOW = [
    "original_shape_VoxelVolume",
    "original_shape_Sphericity",
    "original_shape_Elongation",
    "original_firstorder_Mean",
    "original_firstorder_Median",
    "original_firstorder_Entropy",
    "original_glcm_Contrast",
    "original_glcm_Correlation",
    "original_glcm_Idm",
]


def load_ct(inbox):
    slices = []
    for path in sorted(inbox.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(path)
        except Exception:
            continue
        if getattr(ds, "Modality", None) != "CT" or "PixelData" not in ds:
            continue
        slices.append(ds)
    if not slices:
        raise SystemExit(f"No CT slices in {inbox}")
    slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    return slices


def stack_volume(slices):
    slope = float(slices[0].RescaleSlope)
    intercept = float(slices[0].RescaleIntercept)
    stored_bits = int(getattr(slices[0], "BitsStored", 16))
    pixel_mask = (1 << stored_bits) - 1
    row_spacing, col_spacing = (float(v) for v in slices[0].PixelSpacing)
    origin = np.array(slices[0].ImagePositionPatient, dtype=float)
    zs = [float(ds.ImagePositionPatient[2]) for ds in slices]
    step = float(np.median(np.diff(zs)))
    n_slices = int(round((zs[-1] - zs[0]) / step)) + 1
    volume = np.full((n_slices, slices[0].Rows, slices[0].Columns), -1024, dtype=np.float32)
    present = set()
    for ds, z in zip(slices, zs):
        k = int(round((z - origin[2]) / step))
        raw = ds.pixel_array.astype(np.int32) & pixel_mask
        volume[k] = raw.astype(np.float32) * slope + intercept
        present.add(k)
    missing = [k for k in range(n_slices) if k not in present]
    spacing = (col_spacing, row_spacing, abs(step))
    return volume, origin, step, spacing, missing


def load_rtstruct(inbox):
    for path in sorted(inbox.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "Modality", None) == "RTSTRUCT":
            return ds
    raise SystemExit(f"No RTSTRUCT in {inbox}")


def contours_for(rtstruct, roi_key):
    number = None
    roi_name = None
    for roi in rtstruct.StructureSetROISequence:
        key = "".join(ch for ch in str(roi.ROIName).lower() if ch.isalnum())
        if key == roi_key:
            number = int(roi.ROINumber)
            roi_name = str(roi.ROIName)
            break
    if number is None:
        raise SystemExit(f"ROI {roi_key} not found")
    for seq in rtstruct.ROIContourSequence:
        if int(seq.ReferencedROINumber) != number:
            continue
        polys = []
        for contour in seq.ContourSequence or []:
            data = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            polys.append(data)
        return roi_name, polys
    raise SystemExit(f"No contours for {roi_name}")


def fill_polygon(height, width, rows, cols):
    image = Image.new("1", (width, height), 0)
    points = [(float(c), float(r)) for r, c in zip(rows, cols)]
    ImageDraw.Draw(image).polygon(points, fill=1)
    return np.asarray(image, dtype=np.uint8)


def rasterize(polys, shape, origin, step, spacing):
    col_spacing, row_spacing, _ = spacing
    mask = np.zeros(shape, dtype=np.uint8)
    off_plane = 0
    for poly in polys:
        delta = poly - origin
        cols = delta[:, 0] / col_spacing
        rows = delta[:, 1] / row_spacing
        k = (poly[:, 2] - origin[2]) / step
        kk = int(round(float(np.median(k))))
        if abs(float(np.median(k)) - kk) > 0.2 or not (0 <= kk < shape[0]):
            off_plane += 1
            continue
        mask[kk] = np.maximum(mask[kk], fill_polygon(shape[1], shape[2], rows, cols))
    return mask, off_plane


def sitk_image(array, origin, spacing, pixel_id):
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(array)
    image.SetOrigin(tuple(float(v) for v in origin))
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return sitk.Cast(image, pixel_id)


def extractor():
    engine = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25,
        label=1,
        correctMask=True,
        preCrop=True,
    )
    engine.disableAllFeatures()
    engine.enableFeatureClassByName("shape")
    engine.enableFeatureClassByName("firstorder")
    engine.enableFeatureClassByName("glcm")
    return engine


def main():
    import logging

    logging.getLogger("radiomics").setLevel(logging.ERROR)

    slices = load_ct(INBOX)
    volume, origin, step, spacing, missing = stack_volume(slices)
    rtstruct = load_rtstruct(INBOX)
    engine = extractor()
    rows = []

    print(
        f"{PATIENT_ID}: {volume.shape[0]} slices, spacing {spacing[0]:.3f} x {spacing[1]:.3f} x {spacing[2]:.1f} mm",
        flush=True,
    )
    if missing:
        print(f"unreadable slices left as air: {missing}", flush=True)

    for roi_key, label in ROIS.items():
        roi_name, polys = contours_for(rtstruct, roi_key)
        mask, off_plane = rasterize(polys, volume.shape, origin, step, spacing)
        voxels = int(mask.sum())
        if voxels == 0:
            raise SystemExit(f"{label}: empty mask")
        voxel_cm3 = float(np.prod(spacing)) / 1000.0
        print(
            f"{label} ({roi_name}): {voxels} voxels, {voxels * voxel_cm3:.1f} cm3, off-plane contours {off_plane}",
            flush=True,
        )

        import SimpleITK as sitk

        result = engine.execute(
            sitk_image(volume, origin, spacing, sitk.sitkFloat32),
            sitk_image(mask, origin, spacing, sitk.sitkUInt8),
        )
        record = {"patient_id": PATIENT_ID, "roi": label}
        for key, value in result.items():
            if str(key).startswith("diagnostics_"):
                continue
            record[str(key)] = float(value)
        rows.append(record)
        for key in SHOW:
            print(f"  {key.removeprefix('original_'):<28} {record[key]:.4g}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False)
    print(
        f"wrote {OUT.relative_to(ROOT)} ({frame.shape[1] - 2} features, {frame.shape[0]} rois)",
        flush=True,
    )


if __name__ == "__main__":
    main()
