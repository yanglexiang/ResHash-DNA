"""Clover-style clustering over binary hash codes.

Given for each read a binary hash code (list of 0/1 of fixed length) the
algorithm assigns reads to clusters sequentially, using a three-zone rule on the
Hamming distance to the nearest existing cluster representative:

* ``d <= hamming_safe``           -> join that cluster;
* ``hamming_suspect_low <= d <= hamming_suspect_high``
                                  -> verify with the exact edit distance to the
                                     representative: join if ``<= edit_threshold``,
                                     otherwise create a new cluster;
* ``d > hamming_suspect_high``    -> create a new cluster.

The representative of a cluster is the read that created it (its original DNA
string is stored so edit distance can be computed on demand).

This is an exact re-implementation of the paper algorithm; the nearest-cluster
search is vectorised with NumPy (``matrix @ vector``) for speed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:  # optional fast path
    from Levenshtein import distance as _levenshtein
except Exception:  # pragma: no cover - fallback
    _levenshtein = None


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (C extension if available, else DP)."""
    if _levenshtein is not None:
        return int(_levenshtein(a, b))
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


class HashClusterer:
    """Incremental hash-based clusterer.

    Parameters
    ----------
    hamming_safe:
        Hamming distance <= this -> direct assignment.
    hamming_suspect_low / hamming_suspect_high:
        Suspicious band verified with edit distance.
    edit_threshold:
        Max edit distance to the representative accepted inside the band.
    """

    def __init__(
        self,
        hamming_safe: int = 2,
        hamming_suspect_low: int = 3,
        hamming_suspect_high: int = 6,
        edit_threshold: int = 10,
    ):
        self.hamming_safe = hamming_safe
        self.hamming_suspect_low = hamming_suspect_low
        self.hamming_suspect_high = hamming_suspect_high
        self.edit_threshold = edit_threshold

        self._rep_codes: List[np.ndarray] = []   # one +-? no: {0,1} int8 per cluster
        self._rep_bits: List[float] = []         # number of ones per rep (for fast hamming)
        self._rep_dna: List[str] = []            # representative DNA string
        self.ref_dict: Dict[int, List[int]] = {} # cluster_id -> [seq indices]
        self.next_cluster_id = 0

    # ------------------------------------------------------------------
    def _nearest(self, code: np.ndarray) -> (int, int):
        """Return (best_cluster_id, min_hamming). best = -1 if none."""
        if not self._rep_codes:
            return -1, 10 ** 9
        mat = np.stack(self._rep_codes)                       # [C, bits]
        # Hamming = (#ones code) + (#ones rep) - 2 * dot
        ones = float(code.sum())
        hamming = ones + np.asarray(self._rep_bits) - 2.0 * (mat @ code)
        best = int(np.argmin(hamming))
        return best, int(round(hamming[best]))

    def _new_cluster(self, seq_idx: int, code: np.ndarray, dna: str) -> None:
        cid = self.next_cluster_id
        self.next_cluster_id += 1
        self.ref_dict[cid] = [seq_idx]
        self._rep_codes.append(code.astype(np.int8))
        self._rep_bits.append(float(code.sum()))
        self._rep_dna.append(dna)

    # ------------------------------------------------------------------
    def add(self, seq_idx: int, code: np.ndarray, dna: str) -> int:
        """Cluster one read; return its cluster id."""
        best, min_h = self._nearest(code)
        if best < 0:  # first read
            self._new_cluster(seq_idx, code, dna)
            return self.next_cluster_id - 1

        if min_h <= self.hamming_safe:
            self.ref_dict[best].append(seq_idx)
            return best

        if self.hamming_suspect_low <= min_h <= self.hamming_suspect_high:
            if edit_distance(dna, self._rep_dna[best]) <= self.edit_threshold:
                self.ref_dict[best].append(seq_idx)
                return best
            self._new_cluster(seq_idx, code, dna)
            return self.next_cluster_id - 1

        self._new_cluster(seq_idx, code, dna)
        return self.next_cluster_id - 1


def cluster_codes(
    hash_codes: Sequence[Sequence[int]],
    reads: Sequence[str],
    hamming_safe: int = 2,
    hamming_suspect_low: int = 3,
    hamming_suspect_high: int = 6,
    edit_threshold: int = 10,
) -> Dict[int, List[int]]:
    """Cluster reads given their binary hash codes.

    Returns ``{cluster_id: [read indices]}``.
    """
    cl = HashClusterer(
        hamming_safe=hamming_safe,
        hamming_suspect_low=hamming_suspect_low,
        hamming_suspect_high=hamming_suspect_high,
        edit_threshold=edit_threshold,
    )
    for idx, (code, dna) in enumerate(zip(hash_codes, reads)):
        cl.add(idx, np.asarray(code, dtype=np.int8), dna)
    return cl.ref_dict
