"""Feature fusion modules for multimodal WSI + Genomics."""

from .co_attention import GenomicGuidedCoAttention

__all__ = ["GenomicGuidedCoAttention"]
