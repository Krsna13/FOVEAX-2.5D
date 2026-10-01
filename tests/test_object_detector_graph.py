"""The faster cluster-graph conversion must give exactly the same clusters as the
plain scipy COO -> connected_components path."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception.object_detector import MockObjectDetector, _component_labels  # noqa: E402


def _partition(labels):
    order = np.argsort(labels, kind="stable")
    sl = labels[order]
    return sorted(tuple(g.tolist()) for g in np.split(order, np.flatnonzero(np.diff(sl)) + 1))


def _reference_labels(n, pairs):
    if len(pairs) == 0:
        adj = coo_matrix((n, n), dtype=bool)
    else:
        adj = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(adj, directed=False)[1]


@pytest.mark.parametrize("seed,n,radius", [(0, 400, 1.5), (1, 2500, 0.6), (2, 3000, 1.5), (3, 50, 5.0), (4, 1, 1.5)])
def test_same_partition_as_plain_scipy_path(seed, n, radius):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-12, 12, size=(n, 3))
    pairs = cKDTree(xyz).query_pairs(radius, output_type="ndarray")
    assert _partition(_component_labels(n, pairs)) == _partition(_reference_labels(n, pairs))


def test_no_pairs_means_every_point_is_its_own_cluster():
    pairs = np.zeros((0, 2), dtype=np.intp)
    labels = _component_labels(7, pairs)
    assert len(set(labels.tolist())) == 7


def test_falls_back_to_plain_path_if_the_private_kernel_is_unavailable(monkeypatch):
    import scipy.sparse as sp

    monkeypatch.delattr(sp, "_sparsetools", raising=False)
    monkeypatch.setitem(sys.modules, "scipy.sparse._sparsetools", None)  # makes the import fail
    rng = np.random.default_rng(9)
    xyz = rng.uniform(-6, 6, size=(600, 3))
    pairs = cKDTree(xyz).query_pairs(1.2, output_type="ndarray")
    assert _partition(_component_labels(600, pairs)) == _partition(_reference_labels(600, pairs))


def test_detector_clusters_unchanged_on_a_dense_scene():
    """End to end: two well separated blobs above the ground are two detections."""
    rng = np.random.default_rng(5)
    blob_a = rng.normal([5, 3, 1.0], 0.25, size=(300, 3))
    blob_b = rng.normal([-6, -4, 1.0], 0.25, size=(300, 3))
    ground = np.column_stack([rng.uniform(-10, 10, 800), rng.uniform(-10, 10, 800), np.zeros(800)])
    pts = np.vstack([blob_a, blob_b, ground])
    pts = np.column_stack([pts, np.ones(len(pts))]).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dets = MockObjectDetector().detect(pts, timestamp_s=0.0)
    assert len(dets) == 2
