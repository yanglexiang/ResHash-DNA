"""Synthetic DNA clustering-benchmark generator.

Generates ``num_clusters`` "true" sequences and, for each, a noisy amplicon
family whose members are produced by applying independent deletions,
insertions and substitutions at a controlled per-base rate.  Optionally the
mutation rate is drawn per cluster from a range, mimicking heterogeneous
coverage in real amplicon data.

Output JSON::

    {"reads": ["ATCG...", ...], "labels": [0, 0, 1, ...],
     "meta": {"clusters": 30, ...}}

``labels`` are ground-truth cluster ids (used for scoring only).

Example
-------
    python -m reshash.generate_dataset --out data/demo/demo.json \
        --clusters 30 --total 1500 --min-size 20 --max-size 100 \
        --read-length 120 --error-rate 0.01 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Tuple

_ALPHABET = ["A", "T", "C", "G"]


def random_sequence(length: int) -> str:
    return "".join(random.choices(_ALPHABET, k=length))


def introduce_errors(original: str, error_rate: float) -> str:
    """Apply del/ins/sub at ~``error_rate`` per base (equal probability)."""
    sequence = list(original)
    i = 0
    while i < len(sequence):
        if random.random() < error_rate:
            op = random.choice(("insert", "delete", "substitute"))
            if op == "insert":
                sequence.insert(i, random.choice(_ALPHABET))
                i += 1
            elif op == "delete":
                del sequence[i]
                continue  # don't advance
            else:
                cur = sequence[i]
                sequence[i] = random.choice([b for b in _ALPHABET if b != cur])
        i += 1
    return "".join(sequence)


def _cluster_sizes(num_clusters: int, total: int, min_size: int, max_size: int) -> List[int]:
    sizes: List[int] = []
    remaining = total
    for _ in range(num_clusters - 1):
        max_possible = min(max_size, remaining - (num_clusters - len(sizes) - 1) * min_size)
        size = random.randint(min_size, max_possible)
        sizes.append(size)
        remaining -= size
    sizes.append(remaining)
    return sizes


def generate(
    num_clusters: int = 30,
    total: int = 1500,
    min_size: int = 20,
    max_size: int = 100,
    read_length: int = 120,
    error_rate: float = 0.01,
    error_rate_range: Tuple[float, float] | None = None,
    seed: int | None = None,
) -> Tuple[List[str], List[int]]:
    """Return ``(reads, labels)`` for a synthetic clustering benchmark."""
    if seed is not None:
        random.seed(seed)

    sizes = _cluster_sizes(num_clusters, total, min_size, max_size)

    reads: List[str] = []
    labels: List[int] = []
    for cluster_id, size in enumerate(sizes):
        original = random_sequence(read_length)
        rate = error_rate
        if error_rate_range is not None:
            rate = random.uniform(*error_rate_range)
        for _ in range(size):
            reads.append(introduce_errors(original, rate))
            labels.append(cluster_id)

    order = list(range(len(reads)))
    random.shuffle(order)
    reads = [reads[i] for i in order]
    labels = [labels[i] for i in order]
    return reads, labels


def save(reads: List[str], labels: List[int], path: str, meta: dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {"reads": reads, "labels": labels, "meta": meta or {}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic DNA clustering dataset.")
    ap.add_argument("--out", default="data/demo/demo.json")
    ap.add_argument("--clusters", type=int, default=30)
    ap.add_argument("--total", type=int, default=1500)
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--max-size", type=int, default=100)
    ap.add_argument("--read-length", type=int, default=120)
    ap.add_argument("--error-rate", type=float, default=0.01)
    ap.add_argument("--err-low", type=float, default=None,
                    help="If set along with --err-high, draw a per-cluster rate uniformly.")
    ap.add_argument("--err-high", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    err_range = None
    if args.err_low is not None and args.err_high is not None:
        err_range = (args.err_low, args.err_high)

    reads, labels = generate(
        num_clusters=args.clusters, total=args.total,
        min_size=args.min_size, max_size=args.max_size,
        read_length=args.read_length, error_rate=args.error_rate,
        error_rate_range=err_range, seed=args.seed,
    )

    counts = defaultdict(int)
    for lb in labels:
        counts[lb] += 1
    print(f"clusters={len(counts)} reads={len(reads)} "
          f"sizes {min(counts.values())}-{max(counts.values())}")

    save(reads, labels, args.out, meta={
        "clusters": len(counts), "reads": len(reads),
        "read_length": args.read_length, "error_rate": args.error_rate,
        "seed": args.seed,
    })
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
