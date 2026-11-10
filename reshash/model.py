"""Encoder network for DNA deep-hash clustering.

``EnhancedDNAHashResNet`` maps an one-hot encoded DNA read of fixed length onto
a ``hash_dim``-dimensional real vector in ``(-1, 1)``.  Signing the vector gives
a binary hash code whose **Hamming** distance approximates a *conservative*
version of the edit distance between the reads, which is what the downstream
clustering stage exploits.

The architecture is the union of the two variants found in the original
research scripts:

* two parallel convolution stems (kernel 3 and 5) -> richer local context,
* a 1x1 "fuse" projection + a stack of residual blocks,
* global max + average pooling (concatenated),
* a small MLP head with BatchNorm / Dropout, ``tanh`` output.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Nucleotide -> one-hot row.  ``N`` / anything unknown -> all zero (mask).
NUCLEOTIDES = {
    "A": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "G": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}
_PAD = [0, 0, 0, 0]


def encode_dna_sequence(reads: Sequence[str], max_len: int = 152) -> torch.Tensor:
    """One-hot encode a batch of DNA strings into ``[B, max_len, 4]``.

    Sequences longer than ``max_len`` are truncated; shorter ones are padded
    with all-zero rows (equivalent to ``N``).
    """
    encoded: List[List[List[int]]] = []
    for read in reads:
        seq = [NUCLEOTIDES.get(b, _PAD) for b in read[:max_len]]
        if len(seq) < max_len:
            seq.extend([_PAD] * (max_len - len(seq)))
        encoded.append(seq)
    return torch.tensor(encoded, dtype=torch.float32)


class ResidualBlock(nn.Module):
    """Standard 1-D residual block with projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(residual)
        return F.relu(out)


class EnhancedDNAHashResNet(nn.Module):
    """DNA read -> ``hash_dim``-dim embedding in ``(-1, 1)``.

    Parameters
    ----------
    seq_length:
        Fixed read length the network expects (only stored, the network is
        length-agnostic thanks to adaptive pooling).
    hash_dim:
        Number of hash bits.
    width, mlp_hidden, num_blocks, dropout_rate:
        Capacity / regularisation knobs.
    """

    def __init__(self, seq_length: int = 152, hash_dim: int = 64,
                 dropout_rate: float = 0.3, width: int = 256,
                 mlp_hidden: int = 256, num_blocks: int = 6):
        super().__init__()
        self.seq_length = seq_length
        self.hash_dim = hash_dim

        # Two parallel stems (kernel 3 / kernel 5) fused channel-wise.
        self.conv1 = nn.Conv1d(4, width // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(4, width // 2, kernel_size=5, padding=2)

        self.fuse = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(width, width) for _ in range(num_blocks)]
        )

        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(width * 2, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden, hash_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, L, 4] (one-hot) or already [B, 4, L].
        if x.dim() == 3 and x.shape[2] == 4:
            x = x.permute(0, 2, 1)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x = torch.cat([x1, x2], dim=1)

        x = self.fuse(x)
        x = self.res_blocks(x)

        x_max = self.gmp(x).squeeze(-1)
        x_avg = self.gap(x).squeeze(-1)
        x = torch.cat([x_max, x_avg], dim=1)

        x = self.fc(x)
        return torch.tanh(x)

    @torch.no_grad()
    def binary_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``{0,1}`` hash codes (sign of the tanh output)."""
        return (self.forward(x) > 0).int()
