"""Train the ResHash-DNA network and save a checkpoint.

Usage
-----
    python -m reshash.train \
        --data data/masked_reads.json \
        --out  weights/model.pth \
        --epochs 300 --hash-dim 64

Training procedure
------------------
1. Loads ``{reads, labels}`` (labels are only used to form positive / negative
   pairs; the loss is *not* a classifier loss).
2. Optimises ``EnhancedDNAHashResNet`` with the composite hash loss, using
   online perturbation to create (original, mutated) pairs whose edit distance
   is fed to the loss.  A cosine learning-rate schedule anneals ``lr`` to 0.
3. Periodically evaluates pairwise TS on a held-out slice and keeps the best
   checkpoint (by TS, tie-broken on fewer predicted clusters).
4. Saves the best checkpoint (``--out``), including architecture hyper-params
   so ``pipeline.cluster_reads`` can rebuild the model with no extra arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .model import EnhancedDNAHashResNet, encode_dna_sequence
from .loss import hash_loss
from .perturb import add_perturbation
from .cluster import cluster_codes
from .evaluate import pairwise_ts

# Default clustering parameters.
_CLUSTER_KW = dict(hamming_safe=2, hamming_suspect_low=3,
                   hamming_suspect_high=6, edit_threshold=10)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load(path: str) -> Tuple[List[str], List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["reads"], data["labels"]


@torch.no_grad()
def hash_reads(model: EnhancedDNAHashResNet, reads: Sequence[str],
               seq_len: int, batch_size: int = 256) -> np.ndarray:
    """Return ``{0,1}`` hash codes ``[N, hash_dim]`` for ``reads``."""
    model.eval()
    out = []
    for start in range(0, len(reads), batch_size):
        batch = encode_dna_sequence(reads[start:start + batch_size], max_len=seq_len)
        out.append(model.binary_codes(batch).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.hash_dim), dtype=np.int64)


def _codes_to_clusters(codes: np.ndarray, reads: Sequence[str],
                       cluster_kw: Dict | None = None) -> List[List[int]]:
    kw = _CLUSTER_KW if cluster_kw is None else cluster_kw
    ref = cluster_codes(codes, reads, **kw)
    return [sorted(v) for v in ref.values()]


def evaluate_ts(model: EnhancedDNAHashResNet, reads: Sequence[str],
                labels: Sequence[int], seq_len: int) -> Tuple[float, int]:
    codes = hash_reads(model, reads, seq_len)
    clusters = _codes_to_clusters(codes, reads)
    ts = pairwise_ts(clusters, labels)["ts"]
    return ts, len(clusters)


def _save_checkpoint(model, path, hp: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hyperparams": hp}, path)


def train(
    data: str,
    out: str,
    epochs: int = 300,
    hash_dim: int = 64,
    seq_len: int | None = None,
    train_pool: int | None = None,
    batch_size: int = 128,
    lr: float = 1e-3,
    eval_every: int = 20,
    eval_size: int = 400,
    seed: int = 0,
    width: int = 256,
    mlp_hidden: int = 256,
    num_blocks: int = 6,
    perturbation: float = 0.01,
    schedule: str = "cosine",
) -> EnhancedDNAHashResNet:
    """Full training routine. Returns the best model (also written to ``out``)."""
    set_seed(seed)
    reads, labels = _load(data)
    n = len(reads)

    data_len = max(len(r) for r in reads)
    if seq_len is None or seq_len < data_len:
        seq_len = data_len

    if train_pool is None or train_pool > n:
        train_pool = n
    pool_reads = reads[:train_pool]
    pool_labels = labels[:train_pool]

    encoded = encode_dna_sequence(pool_reads, max_len=seq_len)
    lab_t = torch.tensor(pool_labels, dtype=torch.long)

    model = EnhancedDNAHashResNet(
        seq_length=seq_len, hash_dim=hash_dim, width=width,
        mlp_hidden=mlp_hidden, num_blocks=num_blocks,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hp = dict(seq_length=seq_len, hash_dim=hash_dim, width=width,
              mlp_hidden=mlp_hidden, num_blocks=num_blocks, dropout_rate=0.3)

    ev_reads = pool_reads[-eval_size:] if n >= eval_size else pool_reads
    ev_labels = pool_labels[-eval_size:] if n >= eval_size else pool_labels

    best_ts, best_clusters, best_state = -1.0, 10 ** 9, None

    model.train()
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # learning-rate schedule
        if schedule == "cosine":
            cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / epochs))
            for g in opt.param_groups:
                g["lr"] = cur_lr

        idx = torch.randint(0, train_pool, (batch_size,))
        x = encoded[idx]
        y = lab_t[idx]

        pert, edit_dis = add_perturbation(x, del_rate=perturbation,
                                          sub_rate=perturbation,
                                          ins_rate=perturbation, seq_len=seq_len)
        combined = torch.cat([x, pert], dim=0)
        combined_labels = torch.cat([y, y], dim=0)

        opt.zero_grad(set_to_none=True)
        codes = model(combined)
        loss = hash_loss(codes, combined_labels, edit_dis)
        loss.backward()
        opt.step()

        if epoch % eval_every == 0 or epoch == epochs:
            ts, nclu = evaluate_ts(model, ev_reads, ev_labels, seq_len)
            # Prefer higher TS; tie-break on fewer spurious clusters.
            better = (ts > best_ts + 1e-9) or (abs(ts - best_ts) <= 1e-9 and nclu < best_clusters)
            if better:
                best_ts, best_clusters = ts, nclu
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"[{epoch:4d}/{epochs}] lr={cur_lr:.2e} loss={loss.item():.4f} "
                  f"eval_TS={ts:.4f} clusters={nclu} best_TS={best_ts:.4f} "
                  f"({time.time() - t0:.1f}s)")
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    _save_checkpoint(model, out, hp)
    print(f"checkpoint (best TS={best_ts:.4f}) -> {out}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the ResHash-DNA network.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="weights/model.pth")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hash-dim", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--train-pool", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-size", type=int, default=400)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--num-blocks", type=int, default=6)
    ap.add_argument("--perturbation", type=float, default=0.01)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "none"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train(
        data=args.data, out=args.out, epochs=args.epochs,
        hash_dim=args.hash_dim, seq_len=args.seq_len,
        batch_size=args.batch_size, train_pool=args.train_pool, lr=args.lr,
        eval_every=args.eval_every, eval_size=args.eval_size,
        width=args.width, num_blocks=args.num_blocks,
        perturbation=args.perturbation, schedule=args.schedule, seed=args.seed,
    )


if __name__ == "__main__":
    main()
