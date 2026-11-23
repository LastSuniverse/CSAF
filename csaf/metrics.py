"""Metric and interpolation utilities (verbatim from the research
pipeline, vad_mamba/eval.py and vad_mamba/ucf_free.py)."""

import numpy as np

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels > 0.5
    n_pos = int(pos.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels > 0.5
    n_pos = int(pos.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = pos[order].astype(np.float64)
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


@torch.no_grad()


def to_native(scores: np.ndarray, n_native: int) -> np.ndarray:
    if len(scores) == n_native:
        return scores
    return np.interp(np.linspace(0.0, 1.0, n_native), np.linspace(0.0, 1.0, len(scores)), scores)


def l2(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-5)


def smooth(scores, w=5):
    if w <= 1:
        return scores
    return np.convolve(np.asarray(scores, dtype=np.float64), np.ones(w) / w, mode="same")


def frame_labels(native, gt):
    y = np.zeros(native, dtype=np.int8)
    for a, b in (gt or []):
        a = int(a); b = min(int(b), native - 1)
        if b >= a:
            y[a:b + 1] = 1
    return y
