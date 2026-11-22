"""Command line entry point: ``python -m rad <command> --config configs/default.yaml``."""

from __future__ import annotations

import argparse

from .config import load_config


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


if __name__ == "__main__":
    main()
