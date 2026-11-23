"""Reproducible visual review of CT contours and held-out calibration."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .config import resolve
from .dicom_io import contour_polygons, find_roi, load_ct_slices, scan_inbox
from .preprocess import build_masks, build_volume


def contour_review(cfg, ids, out_dir):
    """Save three CT slices per patient with GTV and generated ring overlays."""
    import pydicom

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for pid in ids:
        inbox = resolve(cfg, cfg["paths"]["raw_root"]) / pid / cfg["dicom"]["inbox_subdir"]
        scan = scan_inbox(inbox)
        slices, _ = load_ct_slices(scan.ct_paths)
        volume = build_volume(slices, cfg["dicom"]["missing_slice"])
        rt = pydicom.dcmread(scan.rtstruct_path, stop_before_pixels=True)
        masks, info = build_masks(rt, volume, cfg["rois"], find_roi, contour_polygons)
        gtv, ring = masks["gtv"], masks["periphery"]
        occupied = np.flatnonzero(gtv.any(axis=(1, 2)))
        if not len(occupied):
            raise ValueError(f"{pid}: empty GTV")
        planes = np.unique(np.rint(np.quantile(occupied, [0.25, 0.5, 0.75])).astype(int))
        fig, axes = plt.subplots(1, len(planes), figsize=(5 * len(planes), 5), squeeze=False)
        for ax, k in zip(axes[0], planes):
            ax.imshow(volume.array[k], cmap="gray", vmin=-1000, vmax=400)
            ax.contour(gtv[k], levels=[0.5], colors=["#ff624c"], linewidths=1.5)
            ax.contour(ring[k], levels=[0.5], colors=["#21d7cc"], linewidths=1)
            ax.set_title(f"{pid} · CT slice {k}")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{pid}.png", dpi=130)
        plt.close(fig)
        records.append({"patient_id": pid, "ct_slices": len(slices),
                        "gtv_voxels": int(gtv.sum()), "ring_voxels": int(ring.sum()),
                        "gtv_source": info["gtv"]["source"],
                        "ring_source": info["periphery"]["source"],
                        "reviewed_visually": False})
    pd.DataFrame(records).to_csv(out_dir / "review.csv", index=False)
    return out_dir


def calibration_summary(run_dir, bins=8):
    """Evaluate pooled out-of-fold probabilities, one row per patient and model."""
    run_dir = Path(run_dir)
    data = pd.read_csv(run_dir / "oof_predictions.csv")
    rows, bin_rows = [], []
    from sklearn.linear_model import LogisticRegression

    for (set_name, model), part in data.groupby(["set", "model"], sort=False):
        if part.patient_id.duplicated().any():
            raise ValueError(f"Duplicate patient in {set_name}/{model}")
        y = part.y.to_numpy(dtype=int)
        p = part.p.to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Invalid probabilities in {set_name}/{model}")
        clipped = np.clip(p, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped))
        slope = intercept = np.nan
        if np.std(logit) > 1e-8:
            # Descriptive calibration on held-out predictions; no model is refitted for evaluation.
            fit = LogisticRegression(C=1e6, solver="lbfgs").fit(logit[:, None], y)
            slope, intercept = float(fit.coef_[0, 0]), float(fit.intercept_[0])
        rows.append({"set": set_name, "model": model, "patients": len(y),
                     "observed_rate": float(y.mean()), "mean_predicted": float(p.mean()),
                     "calibration_in_large": float(y.mean() - p.mean()),
                     "calibration_intercept": intercept, "calibration_slope": slope,
                     "brier": float(brier_score_loss(y, p)),
                     "brier_skill_vs_prevalence": float(1 - brier_score_loss(y, p) / brier_score_loss(y, np.full(len(y), y.mean())))})
        groups = pd.qcut(pd.Series(p).rank(method="first"), q=min(bins, len(p)), labels=False)
        for group in sorted(groups.unique()):
            mask = (groups == group).to_numpy()
            bin_rows.append({"set": set_name, "model": model, "bin": int(group) + 1,
                             "n": int(mask.sum()), "mean_predicted": float(p[mask].mean()),
                             "observed_rate": float(y[mask].mean())})
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "calibration_summary.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(run_dir / "calibration_bins.csv", index=False)
    return summary
