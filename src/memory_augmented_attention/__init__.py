"""
Memory-Augmented Attention
==========================

A unified memory-augmented attention mechanism where current sequence tokens and
historical KV cache are both treated as timestamped memory entries in one shared
Memory Bank. Attention is computed over the entire bank with a temporal bias term
that naturally down-weights older memories.

Public API
----------
MemoryBank                  – stores and retrieves timestamped (K, V, Q) entries
TimeBias                    – learnable temporal decay / bias for attention scores
UnifiedMemoryAttention      – multi-head attention over the unified memory bank
MemoryAugmentedTransformerLayer – full transformer layer (attention + FFN)
"""

from .memory_bank import MemoryBank
from .attention import TimeBias, UnifiedMemoryAttention
from .model import MemoryAugmentedTransformerLayer

__all__ = [
    "MemoryBank",
    "TimeBias",
    "UnifiedMemoryAttention",
    "MemoryAugmentedTransformerLayer",
]
