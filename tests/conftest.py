import numpy as np
import pandas as pd
import pytest

SELECTION = {
    "max_missing": 0.2, "near_constant_variance": 1e-8, "min_icc": 0.75,
    "corr_threshold": 0.9, "k_max": 15, "fallback_k": 5, "enet": {"C": 0.1, "l1_ratio": 0.5},
}

CONFIG = {
    "seed": 7,
    "selection": SELECTION,
    "validation": {"outer_folds": 4, "repeats": 2, "inner_folds": 3, "bootstrap": 200,
                   "permutation_repeats": 1, "n_jobs": 1},
    "models": ["logreg", "svm", "rf"],
}


def make_cohort(n=160, n_noise=120, signal=1.2, seed=0):
    """Fake cohort: a few informative radiomic features, many noise features, a
    near-duplicate pair, a constant column, a mostly-missing column, and two clinical columns."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    cols = {}
    for i in range(3):
        cols[f"gtv__informative_{i}"] = rng.normal(size=n) + signal * y
    cols["gtv__informative_0_copy"] = cols["gtv__informative_0"] + rng.normal(scale=0.01, size=n)
    for i in range(n_noise):
        cols[f"gtv__noise_{i}"] = rng.normal(size=n)
    cols["gtv__constant"] = np.full(n, 3.0)
    sparse = rng.normal(size=n)
    sparse[rng.random(n) < 0.5] = np.nan
    cols["gtv__mostly_missing"] = sparse
    cols["periphery__noise_0"] = rng.normal(size=n)
    cols["clin__age"] = rng.normal(60, 10, size=n) + 3 * y
    cols["clin__male"] = rng.integers(0, 2, size=n).astype(float)
    index = [f"P{i:03d}" for i in range(n)]
    return pd.DataFrame(cols, index=index), y


@pytest.fixture
def cohort():
    return make_cohort()


@pytest.fixture
def config():
    import copy
    return copy.deepcopy(CONFIG)
