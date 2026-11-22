"""One experiment: model ladder x models, nested CV, tables on disk.

Writes to ``<outputs_dir>/<run_id>/``:
  metrics.csv, comparisons.csv, selected_features.csv, oof_predictions.csv, run.json
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .clinical import build_dataset, columns_for
from .config import resolve
from .validation import (
    bootstrap_ci,
    holm,
    paired_bootstrap_auc,
    permutation_test,
    run_nested_cv,
    summarize,
)

# (a, b): "does set a add anything over set b?" (report Table 9, last row)
PAIRS = [("M1", "M0"), ("M2", "M1"), ("M5", "M2"), ("M5", "M1"), ("M6", "M5"), ("M6", "M1")]
PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "pyradiomics", "SimpleITK", "pydicom"]


def load_inputs(cfg):
    """Feature matrix X, labels y, ICC table (or None) and notes."""
    X, y, notes = build_dataset(cfg)
    icc_path = resolve(cfg, cfg["paths"]["features_dir"]) / "stability_icc.csv"
    icc = pd.read_csv(icc_path, index_col=0)["icc"] if icc_path.exists() else None
    return X, y, icc, notes


def _git_hash(root):
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def run_record(cfg, extra=None):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    record = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": cfg["seed"],
        "git_commit": _git_hash(cfg["_root"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    record.update(extra or {})
    return record


def run_experiment(cfg, X, y, icc=None, models=None, sets=None, permutations=0, out_dir=None, log=print):
    y = np.asarray(y)
    models = models or cfg["models"]
    ladder = {k: v for k, v in cfg["ladder"].items() if not sets or k in sets}
    v = cfg["validation"]
    out_dir = Path(out_dir) if out_dir else resolve(cfg, cfg["paths"]["outputs_dir"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, oof_rows, sel_rows, pooled_by = [], [], [], {}
    for set_name, spec in ladder.items():
        if spec.get("prior"):
            Xs, run_models = X, ["prior"]
        else:
            empty = [g for g in spec["groups"] if not columns_for(X, [g])]
            if empty:
                # a set with a missing group would silently be a smaller set under the wrong name
                log(f"[{set_name}] skipped: no columns for group(s) {empty}")
                continue
            cols = columns_for(X, spec["groups"])
            Xs, run_models = X[cols], models
        for model in run_models:
            started = time.time()
            result = run_nested_cv(Xs, y, model, cfg, icc)
            summary, pooled = summarize(y, result.oof)
            lo, hi = bootstrap_ci(y, pooled, n=v["bootstrap"], seed=cfg["seed"])
            counts = Counter(f for fold in result.folds for f in fold["selected"])
            n_radiomic = np.mean([sum(not f.startswith("clin__") for f in fold["selected"]) for fold in result.folds])
            row = {
                "set": set_name, "model": model, "n_features_in": Xs.shape[1],
                "n_radiomic_selected": float(n_radiomic), "auc_ci_low": lo, "auc_ci_high": hi, **summary,
            }
            if permutations and model != "prior":
                p_value, _ = permutation_test(Xs, y, model, cfg, icc, summary["auc"], permutations, cfg["seed"])
                row["permutation_p"] = p_value
            rows.append(row)
            pooled_by[(set_name, model)] = pooled
            oof_rows.append(pd.DataFrame({"patient_id": Xs.index, "y": y, "set": set_name, "model": model, "p": pooled}))
            for feature, count in counts.items():
                sel_rows.append({"set": set_name, "model": model, "feature": feature,
                                 "n_folds": count, "frequency": count / len(result.folds)})
            log(f"[{set_name}/{model}] AUC {summary['auc']:.3f} ({lo:.3f}-{hi:.3f}) "
                f"bal.acc {summary['balanced_accuracy']:.3f}  {time.time() - started:.0f}s")

    comparisons = []
    for model in models:
        for a, b in PAIRS:
            key_a, key_b = (a, model if a != "M0" else "prior"), (b, model if b != "M0" else "prior")
            if key_a in pooled_by and key_b in pooled_by:
                diff, lo, hi, p = paired_bootstrap_auc(y, pooled_by[key_a], pooled_by[key_b], n=v["bootstrap"], seed=cfg["seed"])
                comparisons.append({"model": model, "a": a, "b": b, "auc_diff": diff,
                                    "ci_low": lo, "ci_high": hi, "p": p})
    if comparisons:
        for row, adjusted in zip(comparisons, holm([c["p"] for c in comparisons])):
            row["p_holm"] = adjusted

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out_dir / "comparisons.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(out_dir / "selected_features.csv", index=False)
    if oof_rows:
        pd.concat(oof_rows, ignore_index=True).to_csv(out_dir / "oof_predictions.csv", index=False)
    record = run_record(cfg, {
        "n_patients": int(len(y)),
        "class_counts": {int(k): int((y == k).sum()) for k in np.unique(y)},
        "models_tried": sorted({r["model"] for r in rows}),
        "sets_tried": sorted({r["set"] for r in rows}),
        "n_configurations": len(rows),
        "permutations": permutations,
    })
    (out_dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
    log(f"wrote {out_dir}")
    return out_dir
