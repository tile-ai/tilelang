# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Simple reduction schedule rule (subsumed by GeneralReduction for complex cases)."""

from .general_reduction import GeneralReduction


class Reduction(GeneralReduction):
    """Simple reduction schedule rule.

    This is a thin subclass of GeneralReduction that handles basic reduction
    patterns (sum, max, min) where the reduction dimension is straightforward.
    For complex multi-step reductions (softmax, layernorm, etc.), use
    GeneralReduction or LayerNormLike directly.
    """

    pass
