# ResHash-DNA

Reference implementation of **ResHash-DNA**: residual deep hashing for
clustering error-prone DNA reads.

Each read is one-hot encoded and mapped by a residual convolutional encoder to a
compact real-valued vector in `(-1, 1)`; taking its sign yields a binary hash
code whose Hamming distance is a conservative proxy for sequence dissimilarity.
Reads are then clustered by Hamming distance, with an exact edit-distance check
only for codes that fall in an intermediate "suspicious" band.

```
one-hot read
     │
     ▼
EnhancedDNAHashResNet      (conv stem + residual trunk + MLP head)
     │  tanh → sign
     ▼
binary hash code (fixed length)
     │  Hamming + Levenshtein verification
     ▼
cluster assignment
```

## Contents

| Path | Description |
|------|-------------|
| `reshash/model.py` | `EnhancedDNAHashResNet` encoder and one-hot encoding |
| `reshash/loss.py` | Composite hash loss (pairwise + edit-distance-aware + quantisation / balance / decorrelation) |
| `reshash/perturb.py` | Online read perturbation (deletion / substitution / insertion) used during training |
| `reshash/train.py` | Training loop; produces a `.pth` checkpoint |
| `reshash/cluster.py` | Hash-based clustering (Hamming rule + Levenshtein verification) |
| `reshash/evaluate.py` | Evaluation metrics (pairwise TS, cluster accuracy, purity) |
| `reshash/pipeline.py` | End-to-end entry point: reads → clusters |
| `reshash/generate_dataset.py` | Synthetic read generator for benchmarking |
| `weights/model.pth` | Pretrained checkpoint (64-bit hash) |
| `data/masked_reads.json` | Example read set with ground-truth labels |

## Installation

```bash
pip install -r reshash/requirements.txt
```

## Usage

### Cluster a read set

```bash
python -m reshash.pipeline \
    --data data/masked_reads.json \
    --weights weights/model.pth \
    --out results/clusters.json
```

The output JSON contains the binary hash value of every read and the list of
cluster assignments:

```json
{
  "hash_values": ["0101...", "..."],
  "clusters": [[0, 3, 7], [1, 2], ...]
}
```

### Python API

```python
from reshash.pipeline import cluster_reads

result = cluster_reads(reads, labels=None, weights="weights/model.pth")
print(result["clusters"])
```

### Train from scratch

```bash
python -m reshash.train \
    --data data/masked_reads.json \
    --out  weights/model.pth \
    --epochs 300 --hash-dim 64
```

### Generate a synthetic benchmark

```bash
python -m reshash.generate_dataset \
    --out data/synthetic.json \
    --clusters 30 --total 1500 --min-size 20 --max-size 100 \
    --read-length 120 --error-rate 0.01 --seed 0
```

## Data format

Datasets are JSON files with two parallel lists:

```json
{
  "reads": ["ACGT...", "..."],
  "labels": [0, 0, 1, ...]
}
```

`labels` hold the ground-truth cluster of each read and are used only for
evaluation scoring.

## Citation

```bibtex
@article{reshash-dna,
  title   = {ResHash-DNA: Residual Deep Hashing for DNA Read Clustering},
  author  = {Yang, Lexiang and others},
  year    = {2025}
}
```
