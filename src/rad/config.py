"""Configuration loading.

Every path in the YAML file is relative to the project root, which is the
parent of the ``configs/`` directory that holds the file.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_root"] = str(path.parent.parent)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(cfg, relative):
    """Absolute path for a config-relative path."""
    p = Path(relative)
    return p if p.is_absolute() else Path(cfg["_root"]) / p
