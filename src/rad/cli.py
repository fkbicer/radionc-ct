"""Command line entry point: ``python -m rad <command> --config configs/default.yaml``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, resolve


def _latest_run(cfg):
    runs = sorted(p for p in resolve(cfg, cfg["paths"]["outputs_dir"]).glob("*") if (p / "run.json").exists())
    if not runs:
        sys.exit("No run found; run `python -m rad run` first")
    return runs[-1]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rad", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="DICOM -> radiomic feature matrix (resumable)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true", help="redo patients that already have results")
    p.add_argument("--patients", help="comma-separated ids; default: every data/raw/<id>/inbox")

    sub.add_parser("combine", help="rebuild radiomics_wide.csv and audit.csv from the per-patient files")

    sub.add_parser("dataset", help="join clinical table, label and radiomics; write data/features/dataset.csv")

    sub.add_parser("select", help="show which features the selection chain keeps on the whole dataset (inspection only)")

    p = sub.add_parser("run", help="model ladder x models with repeated nested CV")
    p.add_argument("--models", help="comma-separated, default from config")
    p.add_argument("--sets", help="comma-separated ladder sets, e.g. M1,M6")
    p.add_argument("--repeats", type=int, help="override validation.repeats")
    p.add_argument("--permutations", type=int, default=0, help="label permutations per configuration (report: 1000)")

    p.add_argument("--no-report", action="store_true")

    p = sub.add_parser("explain", help="coefficients / SHAP of one refitted model")
    p.add_argument("--set", default="M6")
    p.add_argument("--model", default="logreg")
    p.add_argument("--run")

    p = sub.add_parser("report", help="HTML report of a run")
    p.add_argument("--run")
    p = sub.add_parser("quality", help="CT contour review sheets and held-out calibration tables")
    p.add_argument("--run", required=True)
    p.add_argument("--patients", default="LUNG1-001,LUNG1-003,LUNG1-005,LUNG1-006,LUNG1-007,LUNG1-008,LUNG1-009,LUNG1-010,LUNG1-011,LUNG1-012,LUNG1-014,LUNG1-017,LUNG1-020,LUNG1-021,LUNG1-023,LUNG1-025,LUNG1-026,LUNG1-029,LUNG1-030,LUNG1-031")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "extract":
        from .extraction import run_extraction

        ids = args.patients.split(",") if args.patients else None
        run_extraction(cfg, ids, args.workers, args.force)
    elif args.command == "combine":
        from .extraction import combine

        combine(cfg)
    elif args.command == "dataset":
        from .clinical import build_dataset
        from .config import resolve

        X, y, notes = build_dataset(cfg)
        for note in notes:
            print(note)
        out = resolve(cfg, cfg["paths"]["features_dir"]) / "dataset.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        X.assign(label=y).to_csv(out)
        n_rad = sum(not c.startswith("clin__") for c in X.columns)
        print(f"{len(X)} patients, {n_rad} radiomic + {X.shape[1] - n_rad} clinical features, "
              f"classes {y.value_counts().sort_index().to_dict()} -> {out}")
    elif args.command == "select":
        from .clinical import build_dataset
        from .selection import build_preprocessor

        X, y, _ = build_dataset(cfg)
        prep = build_preprocessor(cfg["selection"], seed=cfg["selection"]["seed"]).fit(X, y.to_numpy())
        print(f"{X.shape[1]} features in -> {len(prep.selected_features_)} kept (whole dataset; for inspection, "
              "not a performance estimate):")
        for name in prep.selected_features_:
            print("  ", name)
    elif args.command == "run":
        from .experiment import load_inputs, run_experiment
        from .report import build_report

        if args.repeats:
            cfg["validation"]["repeats"] = args.repeats
        X, y, icc, notes = load_inputs(cfg)
        for note in notes:
            print(note)
        out = run_experiment(
            cfg, X, y, icc,
            models=args.models.split(",") if args.models else None,
            sets=args.sets.split(",") if args.sets else None,
            permutations=args.permutations,
        )
        if not args.no_report:
            print("report:", build_report(out))
    elif args.command == "explain":
        from .experiment import load_inputs
        from .explain import explain

        X, y, icc, _ = load_inputs(cfg)
        out = Path(args.run) if args.run else _latest_run(cfg)
        table, path = explain(cfg, X, y, icc, args.set, args.model, out)
        print(table.head(10).to_string(index=False))
        print("wrote", path)
    elif args.command == "report":
        from .report import build_report

        print(build_report(Path(args.run) if args.run else _latest_run(cfg)))
    elif args.command == "quality":
        from .quality import calibration_summary, contour_review

        out = Path(args.run)
        summary = calibration_summary(out)
        print(summary[["set", "model", "brier", "brier_skill_vs_prevalence", "calibration_slope"]].to_string(index=False))
        print("contour review:", contour_review(cfg, args.patients.split(","), out / "contour_review"))


if __name__ == "__main__":
    main()
