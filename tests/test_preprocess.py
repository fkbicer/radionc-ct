"""Report Table 15: HU conversion, affine transformation, phantom, constant region."""

import numpy as np
import pytest

from rad.preprocess import (
    expand_ring,
    mm_to_pixel,
    pixel_to_mm,
    rasterize,
    raw_to_hu,
)


def test_hu_conversion():
    hu = raw_to_hu([24, 37, 1984], slope=1, intercept=-1024)
    assert hu.tolist() == [-1000, -987, 960]


def test_affine_transformation_sample_slice():
    origin = (-250.0, -143.0, -1170.5)
    spacing = (0.9766, 0.9766, 3.0)
    assert pixel_to_mm(0, 0, origin, spacing) == (-250.0, -143.0)
    x, y = pixel_to_mm(0, 1, origin, spacing)
    assert x == pytest.approx(-249.0234, abs=1e-4)
    x, y = pixel_to_mm(1, 0, origin, spacing)
    assert y == pytest.approx(-142.0234, abs=1e-4)
    row, col = mm_to_pixel(-249.0234, -143.0, origin, spacing)
    assert (row, col) == pytest.approx((0.0, 1.0), abs=1e-3)


def _circle_polys(radius_mm, spacing, origin, z_planes, center):
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    polys = []
    for k in z_planes:
        z = origin[2] + k * spacing[2]
        polys.append(np.column_stack([center[0] + radius_mm * np.cos(t), center[1] + radius_mm * np.sin(t), np.full_like(t, z)]))
    return polys


def test_phantom_sphere_volume():
    spacing = (1.0, 1.0, 1.0)
    origin = np.array([0.0, 0.0, 0.0])
    shape = (60, 100, 100)
    radius = 15.0
    center = (50.0, 50.0, 30.0)
    polys = []
    for k in range(shape[0]):
        r2 = radius ** 2 - (k - center[2]) ** 2
        if r2 > 0:
            polys += _circle_polys(np.sqrt(r2), spacing, origin, [k], center)
    mask, off = rasterize(polys, shape, origin, spacing)
    assert off == 0
    expected = 4 / 3 * np.pi * radius ** 3
    assert mask.sum() == pytest.approx(expected, rel=0.02)


def test_nested_polygons_make_a_hole():
    spacing = (1.0, 1.0, 1.0)
    origin = np.array([0.0, 0.0, 0.0])
    outer = _circle_polys(20, spacing, origin, [0], (50, 50))
    inner = _circle_polys(10, spacing, origin, [0], (50, 50))
    mask, _ = rasterize(outer + inner, (1, 100, 100), origin, spacing)
    assert mask[0, 50, 50] == 0
    assert mask[0, 50, 65] == 1


def test_ring_excludes_the_gtv_and_has_the_right_thickness():
    mask = np.zeros((40, 60, 60), dtype=np.uint8)
    mask[15:25, 25:35, 25:35] = 1
    ring = expand_ring(mask, margin_mm=3.0, spacing=(1.0, 1.0, 1.0))
    assert (ring & mask).sum() == 0
    assert ring[20, 30, 35:38].all()       # 3 voxels out on the +x side
    assert not ring[20, 30, 39]
