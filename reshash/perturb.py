"""Online read perturbation used during training.

Given a batch of one-hot reads ``[B, L, 4]`` we create a *mutated* copy by
randomly deleting / substituting / inserting bases at the given per-base rates.
The number of actual edits is returned so the loss can relate Hamming distance
to true edit distance.
"""

from __future__ import annotations

import random
from typing import Tuple

import torch

_BASES = [0, 1, 2, 3]  # A, T, C, G
_EMPTY = [0, 0, 0, 0]


def _index_of_one(row) -> int:
    """Return the position of the single 1 in a 4-vector, or -1 for padding."""
    for k in range(4):
        if row[k] == 1:
            return k
    return -1


def add_perturbation(
    sequences: torch.Tensor,
    del_rate: float = 0.01,
    sub_rate: float = 0.01,
    ins_rate: float = 0.01,
    seq_len: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(perturbed_sequences, edit_distances)``.

    Parameters
    ----------
    sequences:
        ``[B, L, 4]`` one-hot tensor.
    seq_len:
        Output length (defaults to the input length).  Truncation / padding
        edits are counted in the returned edit distance.
    """
    if seq_len is None:
        seq_len = sequences.shape[1]

    batch_size = sequences.shape[0]
    perturbed = torch.empty_like(sequences)
    edit_dis = torch.zeros(batch_size, dtype=torch.float32)

    for i in range(batch_size):
        seq = sequences[i].tolist()
        new_seq = []
        j = 0
        edit = 0
        while j < len(seq):
            # deletion
            if random.random() < del_rate:
                j += 1
                edit += 1
                continue
            # substitution
            if random.random() < sub_rate:
                cur = _index_of_one(seq[j])
                candidates = [k for k in _BASES if k != cur]
                new_base = [0, 0, 0, 0]
                new_base[random.choice(candidates)] = 1
                new_seq.append(new_base)
                j += 1
                edit += 1
                continue
            # insertion (before current base)
            if random.random() < ins_rate:
                new_base = [0, 0, 0, 0]
                new_base[random.choice(_BASES)] = 1
                new_seq.append(new_base)
                edit += 1
            new_seq.append(seq[j])
            j += 1

        if len(new_seq) > seq_len:
            edit += len(new_seq) - seq_len
            new_seq = new_seq[:seq_len]
        elif len(new_seq) < seq_len:
            pad = seq_len - len(new_seq)
            new_seq = new_seq + [_EMPTY] * pad
            edit += pad

        perturbed[i] = torch.tensor(new_seq, dtype=sequences.dtype)
        edit_dis[i] = float(edit)

    return perturbed, edit_dis
