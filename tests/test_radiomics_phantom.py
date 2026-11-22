"""PyRadiomics checks from report Table 15 (phantom, constant region).

Skipped when pyradiomics / SimpleITK are not installed.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("radiomics")
pytest.importorskip("SimpleITK")

from rad.config import load_config  # noqa: E402
from rad.preprocess import Volume  # noqa: E402
from rad.radiomics import build_extractor, extract_features  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _params():
    params = copy.deepcopy(load_config(REPO / "configs" / "default.yaml")["radiomics"]["params"])
    params["setting"].pop("resampledPixelSpacing", None)     # keep the phantom on its own grid
    params["setting"].pop("interpolator", None)
    return params


def _sphere(radius=12, shape=(40, 50, 50)):
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    c = (shape[0] / 2, shape[1] / 2, shape[2] / 2)
    return (((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) <= radius ** 2).astype(np.uint8)


def test_sphere_volume_matches_geometry():
    mask = _sphere()
    volume = Volume(np.random.default_rng(0).normal(-300, 50, mask.shape).astype(np.float32),
                    np.zeros(3), (1.0, 1.0, 1.0))
    features = extract_features(build_extractor(_params()), volume, mask, "gtv")
    assert features["gtv__original_shape_VoxelVolume"] == pytest.approx(mask.sum(), rel=1e-6)
    assert features["gtv__original_shape_MeshVolume"] == pytest.approx(4 / 3 * np.pi * 12 ** 3, rel=0.06)
    assert features["gtv__original_shape_Sphericity"] > 0.9


def test_constant_region_has_zero_entropy_and_unit_uniformity():
    mask = _sphere()
    volume = Volume(np.full(mask.shape, -500.0, dtype=np.float32), np.zeros(3), (1.0, 1.0, 1.0))
    features = extract_features(build_extractor(_params()), volume, mask, "gtv")
    assert features["gtv__original_firstorder_Entropy"] == pytest.approx(0.0, abs=1e-9)
    assert features["gtv__original_firstorder_Uniformity"] == pytest.approx(1.0, abs=1e-9)


def test_feature_names_are_prefixed_by_roi_and_cover_all_classes():
    mask = _sphere()
    volume = Volume(np.random.default_rng(1).normal(-300, 50, mask.shape).astype(np.float32),
                    np.zeros(3), (1.0, 1.0, 1.0))
    features = extract_features(build_extractor(_params()), volume, mask, "periphery")
    assert all(k.startswith("periphery__original_") for k in features)
    for cls in ("shape", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"):
        assert any(f"_{cls}_" in k for k in features), cls
