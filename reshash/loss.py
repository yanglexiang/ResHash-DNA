"""Training objective for the DNA hash network.

This is a faithful port of the original research ``hash_loss``
(see ``train_36_groups_to_excel.py``).  The loss combines four parts:

1. **positive / negative pairwise term** (``L_pair``): same-cluster codes are
   pulled within ``pos_margin``, different-cluster codes are pushed beyond
   ``neg_margin``.  Distances here are **squared Euclidean** distances between
   the *continuous* ``tanh`` outputs, which is what makes the objective
   differentiable -- binarising the codes with ``sign`` would kill the gradient.
2. **edit-distance term** (``L_edit``): for each (original, perturbed) read
   pair produced by the perturbation mechanism, the squared-Euclidean distance
   between the two hash vectors is required to stay below a sigmoid ramp of the
   pair's *true* edit distance, so the codes remain ordered w.r.t. sequence
   similarity.
3. **quantisation / balance / decorrelation regularisers** on the continuous
   output.

Scale convention
----------------
The two margins act on **different quantities**, which is the point stressed in
the paper:

* ``pos_margin = 2`` and ``neg_margin = 40`` are thresholds on the
  **squared-Euclidean** distance between ``tanh`` hash vectors.  A ``tanh``
  output lives in ``[-1, 1]^q``; the *original DNA reads* are of length
  ``L`` (~110-170), so a squared-Euclidean separation of 40 between the hash
  embeddings is readily reachable (it is not bounded by the 64-bit Hamming
  range; that 64 bound only applies to the *Hamming* distance of the final
  binary codes, which is a separate, downstream quantity).
* the Hamming distance of the binarised codes is at most ``q`` bits and is only
  used at clustering time, not in the training loss.

This matches the behaviour of the reference ``hash_loss`` and keeps the
gradient flowing through the continuous representation.
"""

from __future__ import annotations

import torch

# Constants of the edit-distance target ramp (same as the research code):
#   target(d) = 54 / (1 + exp(-(20 * (d - 20)) / 152)) + 10
_EDIT_SEQ_LEN = 152.0
_EDIT_SIG_CENTER = 20.0
_EDIT_SIG_SLOPE = 20.0
_EDIT_MAX = 54.0
_EDIT_OFFSET = 10.0


def _edit_target(edit_dis: torch.Tensor) -> torch.Tensor:
    """Sigmoid ramp mapping true edit distance -> target squared-Euclidean distance."""
    return _EDIT_MAX / (1.0 + torch.exp(-(_EDIT_SIG_SLOPE * (edit_dis - _EDIT_SIG_CENTER)) / _EDIT_SEQ_LEN)) + _EDIT_OFFSET


def hash_loss(
    hash_codes: torch.Tensor,
    labels: torch.Tensor,
    edit_dis: torch.Tensor,
    pos_margin: float = 2.0,
    neg_margin: float = 40.0,
    alpha: float = 1e-3,
    beta: float = 1e-3,
    gamma: float = 1e-3,
) -> torch.Tensor:
    """Composite hash loss (port of the original research implementation).

    Parameters
    ----------
    hash_codes:
        ``[2B, q]`` ``tanh`` outputs in ``[-1, 1]``.  The first ``B`` rows are
        the original reads, the next ``B`` their perturbations.
    labels:
        ``[2B]`` cluster labels; an original and its perturbation share a label.
    edit_dis:
        ``[B]`` true edit distance between each (original, perturbation) pair.
    pos_margin, neg_margin:
        Squared-Euclidean thresholds for positive / negative pairs
        (defaults 2 and 40, as in the reference code).
    """
    q = int(hash_codes.shape[1])
    batch_size = int(hash_codes.shape[0])
    ori_b = batch_size // 2
    device = hash_codes.device

    # Pairwise squared-Euclidean distances on the continuous codes.
    squared_norms = torch.sum(hash_codes ** 2, dim=1, keepdim=True)
    inner_products = torch.matmul(hash_codes, hash_codes.t())
    dist_matrix = torch.relu(squared_norms + squared_norms.t() - 2 * inner_products)

    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    identity_part = torch.eye(ori_b, dtype=torch.bool, device=device)

    # Mask selecting the (original, perturbation) pairs.
    edit_dist_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
    edit_dist_mask[:ori_b, ori_b:] = identity_part
    edit_dist_mask[ori_b:, :ori_b] = identity_part
    is_not_self = ~torch.eye(batch_size, dtype=torch.bool, device=device)

    # -- positive / negative pairwise term --------------------------------
    pos_mask = label_matrix & is_not_self
    pos_vals = torch.relu(dist_matrix[pos_mask] - pos_margin)
    pos_loss = pos_vals.mean() if pos_vals.numel() > 0 else torch.zeros((), device=device)

    neg_mask = ~label_matrix
    neg_vals = torch.relu(neg_margin - dist_matrix[neg_mask])
    neg_loss = neg_vals.mean() if neg_vals.numel() > 0 else torch.zeros((), device=device)

    # -- edit-distance term ------------------------------------------------
    target_edit_matrix = torch.zeros_like(dist_matrix)
    idx = torch.arange(ori_b, device=device)
    target_edit_matrix[idx, idx + ori_b] = edit_dis
    target_edit_matrix[idx + ori_b, idx] = edit_dis

    target = _edit_target(target_edit_matrix)
    edit_vals = torch.relu(dist_matrix[edit_dist_mask] - target[edit_dist_mask])
    edit_loss = edit_vals.mean() if edit_vals.numel() > 0 else torch.zeros((), device=device)

    # -- regularisers (on continuous codes) --------------------------------
    quantization_loss = torch.mean((hash_codes.pow(2) - 1.0).pow(2))
    bit_mean = torch.mean(hash_codes, dim=0)
    balance_loss = torch.mean(bit_mean.pow(2))
    codes_centered = hash_codes - bit_mean
    covariance_matrix = torch.matmul(codes_centered.t(), codes_centered) / batch_size
    identity_matrix = torch.eye(q, device=device)
    decorrelation_loss = torch.mean(torch.abs(covariance_matrix * (1 - identity_matrix)))

    total_loss = (pos_loss + neg_loss + 0.1 * edit_loss
                  + alpha * quantization_loss
                  + beta * balance_loss
                  + gamma * decorrelation_loss)
    return total_loss
