"""Evaluation metrics for DNA clustering.

Two complementary scores are provided:

* **Pairwise TS** -- ``tp / (tp + fp + fn)`` over all read pairs, where a pair
  is "predicted same" when the two reads land in the same cluster and "truly
  same" when they share the ground-truth label.  Also known as the
  *pairwise F1 / Jaccard* style score used in the paper.
* **Cluster accuracy** -- fraction of *true* clusters that are matched by a
  single predicted cluster covering >= ``coverage`` of them (mirrors the
  original ``performance_evaluation.evaluation``).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _cluster_to_label(clusters: Sequence[Sequence[int]], n: int) -> np.ndarray:
    pred = np.full(n, -1, dtype=np.int64)
    for cid, members in enumerate(clusters):
        for i in members:
            pred[i] = cid
    return pred


def pairwise_ts(clusters: Sequence[Sequence[int]], labels: Sequence[int]) -> Dict[str, float]:
    """Return tp / fp / fn / tn / ts over all read pairs."""
    n = len(labels)
    lab = np.asarray(labels)
    pred = _cluster_to_label(clusters, n)
    if (pred < 0).any():
        raise ValueError("Some reads were not assigned to any cluster.")

    true_same = lab[:, None] == lab[None, :]
    pred_same = pred[:, None] == pred[None, :]
    iu = np.triu_indices(n, k=1)
    t = true_same[iu]
    p = pred_same[iu]

    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    tn = int(np.sum(~t & ~p))
    denom = tp + fp + fn
    ts = tp / denom if denom else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "ts": ts}


def cluster_accuracy(clusters: Sequence[Sequence[int]], labels: Sequence[int],
                     coverage: float = 0.95) -> Dict[str, float]:
    """Fraction of true clusters matched by a >= ``coverage`` predicted cluster.

    A true cluster counts as matched if there exists an unused predicted cluster
    that is a subset of it and covers at least ``coverage`` of its reads.
    """
    lab = np.asarray(labels)

    true_clusters: Dict[int, set] = {}
    for idx, lb in enumerate(lab):
        true_clusters.setdefault(int(lb), set()).add(idx)

    pred_sets: List[set] = [set(c) for c in clusters]
    used = set()
    matches = 0
    for tc in true_clusters.values():
        size = len(tc)
        best = None
        best_overlap = 0
        for pid, ps in enumerate(pred_sets):
            if pid in used:
                continue
            if not ps.issubset(tc):
                continue
            if len(ps) >= coverage * size and len(ps) > best_overlap:
                best_overlap = len(ps)
                best = pid
        if best is not None:
            matches += 1
            used.add(best)
    acc = matches / len(true_clusters) if true_clusters else 0.0
    return {"accuracy": acc, "matched": matches, "true_clusters": len(true_clusters)}


def purity(clusters: Sequence[Sequence[int]], labels: Sequence[int]) -> float:
    """Weighted cluster purity."""
    lab = np.asarray(labels)
    total = 0
    correct = 0
    for c in clusters:
        if not len(c):
            continue
        members = lab[list(c)]
        vals, counts = np.unique(members, return_counts=True)
        correct += int(counts.max())
        total += len(c)
    return correct / total if total else 0.0


def report(clusters: Sequence[Sequence[int]], labels: Sequence[int]) -> Dict[str, float]:
    """Bundle all metrics into a single dict."""
    pw = pairwise_ts(clusters, labels)
    ca = cluster_accuracy(clusters, labels)
    return {
        "ts": pw["ts"], "tp": pw["tp"], "fp": pw["fp"], "fn": pw["fn"],
        "accuracy": ca["accuracy"], "matched": ca["matched"],
        "true_clusters": ca["true_clusters"],
        "purity": purity(clusters, labels),
        "n_clusters_pred": len(clusters),
    }
