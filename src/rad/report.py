"""HTML results report with tables and figures (report FR12)."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from .validation import net_benefit  # noqa: E402


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _forest(metrics):
    labels = [f"{r.set} / {r.model}" for r in metrics.itertuples()]
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(labels) + 1.2))
    y = np.arange(len(labels))[::-1]
    ax.errorbar(metrics["auc"], y, xerr=[metrics["auc"] - metrics["auc_ci_low"], metrics["auc_ci_high"] - metrics["auc"]],
                fmt="o", color="#2b5fa5", ecolor="#8fb0dc", capsize=3)
    ax.axvline(0.5, color="#999", ls="--", lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("AUC (95% bootstrap CI, repeated nested CV)")
    return _png(fig)


def _calibration_and_dca(oof, best):
    sub = oof[(oof["set"] == best["set"]) & (oof["model"] == best["model"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    frac, mean = calibration_curve(sub["y"], sub["p"], n_bins=8, strategy="quantile")
    a.plot([0, 1], [0, 1], "--", color="#999")
    a.plot(mean, frac, "o-", color="#2b5fa5")
    a.set_xlabel("predicted probability")
    a.set_ylabel("observed fraction")
    a.set_title("Calibration")
    thresholds = np.linspace(0.05, 0.95, 50)
    model, treat_all = net_benefit(sub["y"].to_numpy(), sub["p"].to_numpy(), thresholds)
    b.plot(thresholds, model, color="#2b5fa5", label="model")
    b.plot(thresholds, treat_all, color="#c0504d", label="treat all")
    b.axhline(0, color="#999", lw=1, label="treat none")
    b.set_xlabel("threshold probability")
    b.set_ylabel("net benefit")
    b.set_title("Decision curve")
    b.legend(frameon=False)
    return _png(fig)


def _table(frame, floatfmt="{:.3f}"):
    return frame.to_html(index=False, border=0, classes="t", float_format=lambda v: floatfmt.format(v), na_rep="")


def build_report(out_dir):
    out_dir = Path(out_dir)
    metrics = pd.read_csv(out_dir / "metrics.csv")
    comparisons = pd.read_csv(out_dir / "comparisons.csv") if (out_dir / "comparisons.csv").stat().st_size > 5 else pd.DataFrame()
    selected = pd.read_csv(out_dir / "selected_features.csv") if (out_dir / "selected_features.csv").stat().st_size > 5 else pd.DataFrame()
    oof = pd.read_csv(out_dir / "oof_predictions.csv")
    record = json.loads((out_dir / "run.json").read_text())

    candidates = metrics[metrics["model"] != "prior"]
    best = candidates.sort_values("auc", ascending=False).iloc[0] if len(candidates) else None

    cols = ["set", "model", "auc", "auc_ci_low", "auc_ci_high", "balanced_accuracy", "f1_macro", "brier", "ece"]
    cols += [c for c in ("n_radiomic_selected", "permutation_p") if c in metrics]
    parts = [
        f"<h1>Results</h1><p>{record['n_patients']} patients, classes {html.escape(str(record['class_counts']))}; "
        f"{record['n_configurations']} configurations tried (all are listed). Seed {record['seed']}, "
        f"commit {html.escape(str(record['git_commit']))}. Exploratory single-centre result; "
        "no separate test set, so every number comes from nested cross-validation.</p>",
        "<h2>Performance</h2>", f'<img src="{_forest(candidates)}">' if len(candidates) else "",
        _table(metrics[cols]),
    ]
    if best is not None:
        parts += [f"<h2>Calibration and decision curve: {best['set']} / {best['model']}</h2>",
                  f'<img src="{_calibration_and_dca(oof, best)}">']
    calibration_path = out_dir / "calibration_summary.csv"
    if calibration_path.exists():
        calibration = pd.read_csv(calibration_path)
        parts += ["<h2>Held-out calibration summary</h2>",
                  "<p>Positive Brier skill improves on a constant prevalence prediction. "
                  "Calibration slope near 1 is desirable; these descriptive estimates use the same held-out patients.</p>",
                  _table(calibration)]
    if len(comparisons):
        parts += ["<h2>Does a richer input set add anything? (paired bootstrap, Holm-adjusted)</h2>", _table(comparisons, "{:.4f}")]
    if len(selected) and best is not None:
        top = (selected[(selected["set"] == best["set"]) & (selected["model"] == best["model"])]
               .sort_values("frequency", ascending=False).head(15))
        parts += [f"<h2>Selection frequency across outer folds: {best['set']} / {best['model']}</h2>", _table(top)]
    parts.append("<h2>Reproducibility</h2><pre>" + html.escape(json.dumps(
        {k: record[k] for k in ("created", "seed", "git_commit", "python", "packages")}, indent=2)) + "</pre>")

    style = ("body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#222}"
             "table.t{border-collapse:collapse;font-size:13px}table.t th,table.t td{padding:3px 9px;text-align:right}"
             "table.t th{border-bottom:2px solid #444}table.t td{border-bottom:1px solid #ddd}img{max-width:100%}")
    path = out_dir / "report.html"
    path.write_text(f"<!doctype html><meta charset='utf-8'><title>Results</title><style>{style}</style>" + "\n".join(parts))
    return path
