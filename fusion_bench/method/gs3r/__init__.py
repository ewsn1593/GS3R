"""
GS3R: Global Singular Subspace Separation and Restoration (NeurIPS 2026).

Built on FusionBench (MIT, Anke Tang). The SVD merging utilities in ``utils/``
started from FusionBench's Task Singular Vectors (TSV) implementation
(fusion_bench/method/task_singular_vector), itself adapted from the official code of

- Gargiulo et al., "Task Singular Vectors: Reducing Task Interference in Model Merging",
  CVPR 2025. https://github.com/AntoAndGar/task_singular_vectors (Apache-2.0)

The optional auxiliary optimization (``wudi_merging*`` in ``utils/gs3r_utils.py``)
minimizes the objective of

- Cheng et al., "Whoever Started the Interference Should End It: Guiding Data-Free
  Model Merging via Task Vectors", ICML 2025. https://github.com/nathanielyvo/WUDI-Merging (MIT)

The restoration state, task-wise separation, global rank allocation, PRF, and the
truncated variants are original to this work.
"""
from .gs3r import GS3RForCLIP, GS3RForFlanT5
