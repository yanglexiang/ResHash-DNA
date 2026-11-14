"""Binary hash-trie clustering (Clover-style, ported to a binary trie).

This is the reference implementation of the paper's clustering stage.  It
follows the original research code (``hash_clover_clustering.py``), which is a
``Clover``-style algorithm adapted from a quaternary nucleotide trie to a
**binary hash trie**:

* every read is mapped to a fixed-length binary code ``{0,1}^q``;
* a binary trie stores the **cluster-representative codes** (one leaf per
  cluster), edges labelled by bit values ``0`` / ``1``;
* a new read is compared against the cluster representatives.  Instead of the
  paper's abstract "tolerant trie traversal" we keep the exact three-zone rule
  used in the reference implementation:

    - Hamming distance ``d <= hamming_safe``          -> join that cluster;
    - ``hamming_suspect_low <= d <= hamming_suspect_high``
      -> verify with the exact edit distance to the representative; join if
         ``<= edit_threshold``, else start a new cluster;
    - ``d > hamming_suspect_high``                    -> start a new cluster.

The trie provides the index: representatives are stored in it and can be
retrieved (``get_all_clusters``) to compute the drift (Hamming) distance.  The
nearest-representative search visits the stored leaves and computes the Hamming
distance to each -- exactly the ``ClusterProcessor.cluster`` logic of the
original code, but organised around a ``Trie`` object so that the structure can
later be pruned / partitioned by leading bits for parallelism.

The representative of a cluster is the read that created it; its original DNA
string is stored so the exact edit distance can be computed on demand inside the
"suspicious" band.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

try:  # optional fast path for exact edit distance
    from Levenshtein import distance as _levenshtein
except Exception:  # pragma: no cover - fallback to pure-python DP
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


class TrieNode:
    """A node of the binary hash trie (two children: bit 0 and bit 1)."""

    __slots__ = ("children", "is_end", "cluster_id", "hash_code", "rep_dna")

    def __init__(self) -> None:
        self.children: List[Optional["TrieNode"]] = [None, None]
        self.is_end = False
        self.cluster_id: Optional[int] = None
        self.hash_code: str = ""
        self.rep_dna: str = ""


class HashTrie:
    """Binary hash trie storing one representative code per cluster."""

    def __init__(self, code_len: Optional[int] = None) -> None:
        self.root = TrieNode()
        self.code_len = code_len

    # ------------------------------------------------------------------
    def insert(self, code: str, cluster_id: int, rep_dna: str) -> None:
        """Insert a cluster representative ``code`` (binary string) into the trie."""
        if self.code_len is None:
            self.code_len = len(code)
        elif len(code) != self.code_len:
            raise ValueError(
                f"hash code length {len(code)} != trie length {self.code_len}"
            )
        node = self.root
        for ch in code:
            idx = int(ch)
            if node.children[idx] is None:
                node.children[idx] = TrieNode()
            node = node.children[idx]
        node.is_end = True
        node.cluster_id = cluster_id
        node.hash_code = code
        node.rep_dna = rep_dna

    def search_exact(self, code: str) -> Optional[TrieNode]:
        """Exact-match lookup of a code; returns the leaf node or ``None``."""
        node = self.root
        for ch in code:
            idx = int(ch)
            nxt = node.children[idx]
            if nxt is None:
                return None
            node = nxt
        return node if node.is_end else None

    def leaves(self) -> List[TrieNode]:
        """Return all cluster-representative leaves (the stored codes)."""
        out: List[TrieNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_end:
                out.append(node)
            for child in node.children:
                if child is not None:
                    stack.append(child)
        return out


class HashClusterer:
    """Incremental binary-hash-trie clusterer (Clover-style three-zone rule).

    Parameters
    ----------
    hamming_safe:
        Hamming distance ``<=`` this -> direct assignment.
    hamming_suspect_low / hamming_suspect_high:
        Suspicious band verified with the exact edit distance.
    edit_threshold:
        Max edit distance to the representative accepted inside the band.
    """

    def __init__(
        self,
        hamming_safe: int = 2,
        hamming_suspect_low: int = 3,
        hamming_suspect_high: int = 6,
        edit_threshold: int = 10,
        code_len: Optional[int] = None,
    ):
        self.hamming_safe = hamming_safe
        self.hamming_suspect_low = hamming_suspect_low
        self.hamming_suspect_high = hamming_suspect_high
        self.edit_threshold = edit_threshold

        self.tree = HashTrie(code_len=code_len)
        self.ref_dict: Dict[int, List[int]] = {}  # cluster_id -> [seq indices]
        self.next_cluster_id = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _hamming(code: str, other: str) -> int:
        if len(code) != len(other):
            return 10 ** 9
        return sum(c1 != c2 for c1, c2 in zip(code, other))

    def _nearest(self, code: str):
        """Return (best_cluster_id, min_hamming, best_leaf) over trie leaves."""
        best_cid: Optional[int] = None
        min_h = 10 ** 9
        best_leaf: Optional[TrieNode] = None
        for leaf in self.tree.leaves():
            h = self._hamming(code, leaf.hash_code)
            if h < min_h:
                min_h = h
                best_cid = leaf.cluster_id
                best_leaf = leaf
        return best_cid, min_h, best_leaf

    def _new_cluster(self, seq_idx: int, code: str, dna: str) -> int:
        cid = self.next_cluster_id
        self.next_cluster_id += 1
        self.ref_dict[cid] = [seq_idx]
        self.tree.insert(code, cid, dna)
        return cid

    # ------------------------------------------------------------------
    def add(self, seq_idx: int, code: str, dna: str) -> int:
        """Cluster one read; returns its cluster id."""
        best_cid, min_h, best_leaf = self._nearest(code)

        if best_cid is None:  # first read -> first cluster
            return self._new_cluster(seq_idx, code, dna)

        # 1. safe zone: direct assignment
        if min_h <= self.hamming_safe:
            self.ref_dict[best_cid].append(seq_idx)
            return best_cid

        # 2. suspicious zone: verify with exact edit distance
        if self.hamming_suspect_low <= min_h <= self.hamming_suspect_high:
            if edit_distance(dna, best_leaf.rep_dna) <= self.edit_threshold:
                self.ref_dict[best_cid].append(seq_idx)
                return best_cid
            return self._new_cluster(seq_idx, code, dna)

        # 3. out of range: new cluster
        return self._new_cluster(seq_idx, code, dna)


def _code_to_str(code: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in code)


def cluster_codes(
    hash_codes: Sequence[Sequence[int]],
    reads: Sequence[str],
    hamming_safe: int = 2,
    hamming_suspect_low: int = 3,
    hamming_suspect_high: int = 6,
    edit_threshold: int = 10,
) -> Dict[int, List[int]]:
    """Cluster reads given their binary hash codes using a binary hash trie.

    Parameters
    ----------
    hash_codes:
        ``[N, q]`` array-like of ``0/1`` codes (one row per read).
    reads:
        The original DNA strings (used for exact edit distance in the
        suspicious band).

    Returns
    -------
    ``{cluster_id: [read indices]}``.
    """
    code_len = len(hash_codes[0]) if len(hash_codes) else None
    cl = HashClusterer(
        hamming_safe=hamming_safe,
        hamming_suspect_low=hamming_suspect_low,
        hamming_suspect_high=hamming_suspect_high,
        edit_threshold=edit_threshold,
        code_len=code_len,
    )
    for idx, (code, dna) in enumerate(zip(hash_codes, reads)):
        cl.add(idx, _code_to_str(code), dna)
    return cl.ref_dict
