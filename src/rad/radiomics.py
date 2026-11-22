"""PyRadiomics wrapper. Feature names are ``<roi>__<pyradiomics name>``."""

from __future__ import annotations

import copy
import logging

import numpy as np


def build_extractor(params):
    from radiomics import featureextractor

    logging.getLogger("radiomics").setLevel(logging.ERROR)
    return featureextractor.RadiomicsFeatureExtractor(copy.deepcopy(params))


def _to_sitk(array, origin, spacing, pixel_type):
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(array)
    image.SetOrigin(tuple(float(v) for v in origin))
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return sitk.Cast(image, pixel_type)


def extract_features(extractor, volume, mask, prefix):
    """All enabled features for one ROI; diagnostics are dropped."""
    import SimpleITK as sitk

    result = extractor.execute(
        _to_sitk(volume.array, volume.origin, volume.spacing, sitk.sitkFloat32),
        _to_sitk(mask.astype(np.uint8), volume.origin, volume.spacing, sitk.sitkUInt8),
    )
    return {
        f"{prefix}__{key}": float(value)
        for key, value in result.items()
        if not str(key).startswith("diagnostics_")
    }
