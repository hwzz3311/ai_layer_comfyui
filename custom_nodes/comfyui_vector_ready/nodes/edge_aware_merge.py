"""Region-graph merge of quantized color blobs that doesn't cross strong edges.

Input is the LAB-quantized image plus an edge map (MASK, 0-1, 1=edge).
Adjacent regions with ΔE below threshold are unioned unless their shared
boundary passes through edge pixels — preserves designer-intended seams."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ._utils import (
    np_to_torch_image,
    torch_image_to_np,
    torch_mask_to_np,
)


def _label_regions(image_lab: np.ndarray) -> np.ndarray:
    """Label connected components per unique color in the quantized image."""
    # encode 3-channel color to a single int key
    key = (
        (image_lab[..., 0] * 1000).astype(np.int64) * 1_000_000
        + (image_lab[..., 1] * 1000).astype(np.int64) * 1_000
        + (image_lab[..., 2] * 1000).astype(np.int64)
    )
    labels = np.zeros(key.shape, dtype=np.int32)
    next_label = 1
    for color_key in np.unique(key):
        mask = key == color_key
        comp, n = ndimage.label(mask)
        labels[mask] = comp[mask] + next_label - 1
        next_label += n
    return labels


def _region_means(image_lab: np.ndarray, labels: np.ndarray):
    n = labels.max() + 1
    sums = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    flat_labels = labels.reshape(-1)
    flat_img = image_lab.reshape(-1, 3)
    np.add.at(sums, flat_labels, flat_img)
    np.add.at(counts, flat_labels, 1)
    counts = np.maximum(counts, 1)
    return sums / counts[:, None]


def _adjacency_with_edges(labels: np.ndarray, edge: np.ndarray):
    """Yield (label_a, label_b, edge_density) for every adjacent label pair."""
    pairs: dict[tuple[int, int], list[float]] = {}

    def _accumulate(a, b, e):
        diff = a != b
        if not diff.any():
            return
        ai = a[diff]
        bi = b[diff]
        ei = e[diff]
        for la, lb, ev in zip(ai, bi, ei):
            key = (int(min(la, lb)), int(max(la, lb)))
            pairs.setdefault(key, []).append(float(ev))

    _accumulate(labels[:, :-1], labels[:, 1:], np.maximum(edge[:, :-1], edge[:, 1:]))
    _accumulate(labels[:-1, :], labels[1:, :], np.maximum(edge[:-1, :], edge[1:, :]))
    return [(a, b, float(np.mean(vals))) for (a, b), vals in pairs.items()]


def _delta_e(c1: np.ndarray, c2: np.ndarray) -> float:
    return float(np.linalg.norm((c1 - c2) * np.array([100.0, 128.0, 128.0])))


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


class VR_EdgeAwareMerge:
    CATEGORY = "VectorReady/color"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_lab_quantized": ("IMAGE",),
                "edge_map": ("MASK",),
                "delta_e_threshold": ("FLOAT", {"default": 6.0, "min": 0.5, "max": 50.0}),
                "edge_density_max": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0}),
            }
        }

    def merge(self, image_lab_quantized, edge_map, delta_e_threshold, edge_density_max):
        arr = torch_image_to_np(image_lab_quantized)
        edges = torch_mask_to_np(edge_map)
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            img = arr[i]
            edge = edges[i] if edges.shape[0] > i else edges[0]
            labels = _label_regions(img)
            means = _region_means(img, labels)
            uf = _UnionFind(len(means))
            for a, b, ed in _adjacency_with_edges(labels, edge):
                if ed > edge_density_max:
                    continue
                if _delta_e(means[a], means[b]) < delta_e_threshold:
                    uf.union(a, b)
            # remap labels to root means
            roots = np.array([uf.find(l) for l in range(len(means))], dtype=np.int32)
            # recompute means for merged groups
            new_sums = np.zeros_like(means)
            new_counts = np.zeros(len(means), dtype=np.int64)
            for orig in range(len(means)):
                np.add.at(new_sums, roots[orig], means[orig])
                new_counts[roots[orig]] += 1
            new_counts = np.maximum(new_counts, 1)
            new_means = new_sums / new_counts[:, None]
            flat_labels = labels.reshape(-1)
            out[i] = new_means[roots[flat_labels]].reshape(img.shape)
        return (np_to_torch_image(out),)
