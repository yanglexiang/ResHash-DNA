"""ResHash-DNA: residual deep hashing for DNA read clustering.

Public API
----------
* :class:`reshash.model.EnhancedDNAHashResNet` -- the encoder network
* :func:`reshash.pipeline.cluster_reads` -- reads -> clusters

``cluster_reads`` is imported lazily (PEP 562) so that running
``python -m reshash.pipeline`` does not trigger a double-import warning.
"""
from .model import EnhancedDNAHashResNet, encode_dna_sequence, NUCLEOTIDES

__all__ = [
    "EnhancedDNAHashResNet",
    "encode_dna_sequence",
    "NUCLEOTIDES",
    "cluster_reads",
]


def __getattr__(name):
    if name == "cluster_reads":
        from .pipeline import cluster_reads
        return cluster_reads
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
