"""Training objective for the DNA hash network.

The loss combines four terms (see paper / original ``hash_loss``):

1. **positive** term -- pull same-cluster codes within ``pos_margin`` bits;
2. **negative** term -- push different-cluster codes beyond ``neg_margin`` bits;
3. **edit-distance** term -- for the (original, perturbed) read pairs, make the
   Hamming distance follow a sigmoid ramp of the *true* edit distance, so the
   codes stay ordered w.r.t. sequence similarity;
4. regularisers -- quantisation (codes near 0/1), bit balance, decorrelation.

Distance convention
-------------------
All margins are expressed in **bits** (Hamming).  The network outputs a
``tanh`` value in ``[-1, 1]``; for loss purposes we map it to a *soft* binary
code ``b = (h + 1) / 2`` in ``[0, 1]`` and use the identity

    Hamming(b_i, b_j) = sum_k b_ik (1 - b_jk) + (1 - b_ik) b_jk
                      = sum_k [b_ik + b_jk - 2 b_ik b_jk].

This keeps margins interpretable (0 .. hash_dim).  The previous
squared-Euclidean formulation saturated the ``tanh`` output and collapsed every
read onto a single code; working in the binary/Hamming scale avoids that.
"""

from __future__ import annotations

import torch

# Scale used by the edit-distance target ramp (same constants as research code).
_EDIT_SEQ_LEN = 152.0
_EDIT_SIG_CENTER = 20.0
_EDIT_SIG_SLOPE = 20.0
_EDIT_OFFSET = 10.0
_EDIT_MAX = 54.0


def _edit_target(edit_dis: torch.Tensor, hash_dim: int) -> torch.Tensor:
    """Sigmoid ramp mapping true edit distance -> target Hamming distance (bits).

    The raw ramp tops out around 64 for a 152-long read; it is rescaled into the
    ``[0, hash_dim]`` range so it is meaningful for any hash size.
    """
    raw = _EDIT_MAX / (1.0 + torch.exp(-(_EDIT_SIG_SLOPE * (edit_dis - _EDIT_SIG_CENTER)) / _EDIT_SEQ_LEN)) + _EDIT_OFFSET
    return raw / 64.0 * hash_dim


def _soft_hamming(b: torch.Tensor) -> torch.Tensor:
    """Pairwise soft Hamming distance (in bits) for codes in ``[0, 1]``."""
    s = b.sum(dim=1, keepdim=True)
    gram = b @ b.t()
    return torch.relu(s + s.t() - 2.0 * gram)


def hash_loss(
    hash_codes: torch.Tensor,
    labels: torch.Tensor,
    edit_dis: torch.Tensor,
    pos_margin: float = 0.15,
    neg_margin: float = 0.40,
    alpha: float = 1e-3,
    beta: float = 1e-3,
    gamma: float = 1e-3,
) -> torch.Tensor:
    """Compute the composite hash loss.

    Parameters
    ----------
    hash_codes:
        ``[2B, hash_dim]`` ``tanh`` outputs in ``[-1, 1]`` -- first ``B`` rows
        are originals, next ``B`` their perturbations.
    labels:
        ``[2B]`` cluster labels (originals and perturbations share label).
    edit_dis:
        ``[B]`` true edit distance between each (original, perturbation) pair.
    pos_margin, neg_margin:
        Hamming thresholds as a *fraction of hash_dim* (default: positives within
        15 % of bits, negatives at least 40 % of bits apart).
    """
    num_bits = int(hash_codes.shape[1])
    batch_size = int(hash_codes.shape[0])
    ori_b = batch_size // 2
    device = hash_codes.device

    b = (hash_codes + 1.0) * 0.5                       # tanh -> soft binary
    dist = _soft_hamming(b)                            # bits, 0..num_bits

    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    is_not_self = ~torch.eye(batch_size, dtype=torch.bool, device=device)

    pos_thr = pos_margin * num_bits
    neg_thr = neg_margin * num_bits

    # -- positives ---------------------------------------------------------
    pos_mask = label_matrix & is_not_self
    pos_vals = torch.relu(dist[pos_mask] - pos_thr)
    pos_loss = pos_vals.mean() if pos_vals.numel() else torch.zeros((), device=device)

    # -- negatives ---------------------------------------------------------
    neg_mask = ~label_matrix
    neg_vals = torch.relu(neg_thr - dist[neg_mask])
    neg_loss = neg_vals.mean() if neg_vals.numel() else torch.zeros((), device=device)

    # -- edit-distance term ------------------------------------------------
    idx = torch.arange(ori_b, device=device)
    edit_mask = torch.zeros_like(dist, dtype=torch.bool)
    edit_mask[idx, idx + ori_b] = True
    edit_mask[idx + ori_b, idx] = True

    target = torch.zeros_like(dist)
    target[idx, idx + ori_b] = edit_dis
    target[idx + ori_b, idx] = edit_dis
    target_ham = _edit_target(target, num_bits)
    edit_vals = torch.relu(dist[edit_mask] - target_ham[edit_mask])
    edit_loss = edit_vals.mean() if edit_vals.numel() else torch.zeros((), device=device)

    # -- regularisers ------------------------------------------------------
    quantization_loss = torch.mean((hash_codes.abs() - 1.0).pow(2))
    bit_mean = torch.mean(b, dim=0)                    # want 0.5 -> balanced
    balance_loss = torch.mean((bit_mean - 0.5).pow(2))
    centered = b - bit_mean
    cov = torch.matmul(centered.t(), centered) / batch_size
    ident = torch.eye(num_bits, device=device)
    decorrelation_loss = torch.mean(torch.abs(cov * (1 - ident)))

    return (pos_loss + neg_loss + 1.0 * edit_loss
            + alpha * quantization_loss
            + beta * balance_loss
            + gamma * decorrelation_loss)
