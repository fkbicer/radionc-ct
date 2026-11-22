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

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "extract":
        from .extraction import run_extraction

        ids = args.patients.split(",") if args.patients else None
        run_extraction(cfg, ids, args.workers, args.force)
    elif args.command == "combine":
        from .extraction import combine

        combine(cfg)


if __name__ == "__main__":
    main()
