"""End-to-end pipeline: DNA reads -> clusters.

Loads a trained checkpoint, hashes every read, runs the Hamming/edit-distance
clustering, prints metrics (when labels are available) and writes a JSON
result::

    {"hash_values": ["0101...", ...], "clusters": [[0, 3, 7], [1, 2], ...]}

CLI
---
    python -m reshash.pipeline --data data/masked_reads.json \
        --weights weights/model.pth --out results/clusters.json

Python
------
    from reshash.pipeline import cluster_reads
    res = cluster_reads(reads, labels=None, weights="weights/model.pth")
    print(res["clusters"])
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .model import EnhancedDNAHashResNet, encode_dna_sequence
from .cluster import cluster_codes


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_model(weights: str, device: str = "cpu") -> EnhancedDNAHashResNet:
    """Rebuild the network from a checkpoint (handles both formats).

    Accepted checkpoint formats:
      * ``{"state_dict": ..., "hyperparams": {...}}`` (written by train.py)
      * a bare ``state_dict`` (as saved by the original research scripts)
    """
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        hp = ckpt.get("hyperparams", {})
    else:
        state = ckpt
        hp = {}

    # Infer hash_dim from the final linear layer if not recorded.
    hash_dim = hp.get("hash_dim")
    if hash_dim is None:
        hash_dim = int(state["fc.4.weight"].shape[0])
    seq_len = hp.get("seq_length", 152)

    model = EnhancedDNAHashResNet(
        seq_length=seq_len,
        hash_dim=hash_dim,
        width=hp.get("width", 256),
        mlp_hidden=hp.get("mlp_hidden", 256),
        num_blocks=hp.get("num_blocks", 6),
        dropout_rate=hp.get("dropout_rate", 0.3),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def hash_reads(model: EnhancedDNAHashResNet, reads: Sequence[str],
               seq_len: int, batch_size: int = 256) -> np.ndarray:
    """Return ``{0,1}`` codes ``[N, hash_dim]``."""
    out = []
    for start in range(0, len(reads), batch_size):
        batch = encode_dna_sequence(reads[start:start + batch_size], max_len=seq_len)
        out.append(model.binary_codes(batch).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.hash_dim), dtype=np.int64)


def codes_to_strings(codes: np.ndarray) -> List[str]:
    return ["".join(map(str, row.tolist())) for row in codes]


# --------------------------------------------------------------------------
# Main API
# --------------------------------------------------------------------------
def cluster_reads(
    reads: Sequence[str],
    labels: Optional[Sequence[int]] = None,
    weights: str = "weights/model.pth",
    hamming_safe: int = 2,
    hamming_suspect_low: int = 3,
    hamming_suspect_high: int = 6,
    edit_threshold: int = 10,
    device: str = "cpu",
    model: Optional[EnhancedDNAHashResNet] = None,
) -> Dict:
    """Hash ``reads`` with the trained network and cluster them.

    Parameters
    ----------
    reads:
        DNA strings.
    labels:
        Optional ground-truth labels; if given, metrics are computed.
    weights:
        Path to the checkpoint (ignored if ``model`` is provided).

    Returns
    -------
    dict with ``hash_values`` (binary strings), ``clusters`` (list of index
    lists) and, if ``labels`` given, ``metrics``.
    """
    if model is None:
        model = load_model(weights, device=device)
    seq_len = model.seq_length

    codes = hash_reads(model, reads, seq_len)
    ref = cluster_codes(
        codes, reads,
        hamming_safe=hamming_safe,
        hamming_suspect_low=hamming_suspect_low,
        hamming_suspect_high=hamming_suspect_high,
        edit_threshold=edit_threshold,
    )
    clusters = [sorted(members) for members in ref.values()]

    result: Dict = {
        "hash_values": codes_to_strings(codes),
        "clusters": clusters,
        "n_reads": len(reads),
        "n_clusters": len(clusters),
        "params": {
            "hamming_safe": hamming_safe,
            "hamming_suspect": [hamming_suspect_low, hamming_suspect_high],
            "edit_threshold": edit_threshold,
        },
    }

    if labels is not None:
        from .evaluate import report
        result["metrics"] = report(clusters, labels)

    return result


def run_file(data_path: str, weights: str, out: Optional[str] = None, **kw) -> Dict:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    reads = data["reads"]
    labels = data.get("labels")

    res = cluster_reads(reads, labels=labels, weights=weights, **kw)

    if labels is not None:
        m = res["metrics"]
        print(f"reads={res['n_reads']} clusters={res['n_clusters']} "
              f"TS={m['ts']:.4f} acc={m['accuracy']:.4f} purity={m['purity']:.4f}")
    else:
        print(f"reads={res['n_reads']} clusters={res['n_clusters']}")

    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"result -> {out}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="DNA reads -> clusters (end-to-end).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="weights/model.pth")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hamming-safe", type=int, default=2)
    ap.add_argument("--suspect-low", type=int, default=3)
    ap.add_argument("--suspect-high", type=int, default=6)
    ap.add_argument("--edit-threshold", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_file(
        args.data, args.weights, args.out,
        hamming_safe=args.hamming_safe,
        hamming_suspect_low=args.suspect_low,
        hamming_suspect_high=args.suspect_high,
        edit_threshold=args.edit_threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
