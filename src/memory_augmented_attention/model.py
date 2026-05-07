"""
Memory-Augmented Transformer Layer
====================================

A full transformer encoder layer that uses :class:`UnifiedMemoryAttention`
instead of standard self-attention.  A shared :class:`MemoryBank` can be
passed to the layer so that multiple consecutive forward passes accumulate a
growing pool of historical context.

Classes
-------
MemoryAugmentedTransformerLayer  – attention + FFN + layer-norm residuals
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .attention import UnifiedMemoryAttention
from .memory_bank import MemoryBank


class MemoryAugmentedTransformerLayer(nn.Module):
    """
    One transformer encoder layer with unified memory-augmented attention.

    Architecture::

        x → [UnifiedMemoryAttention] → Add & Norm
          → [Feed-Forward (ReLU)]    → Add & Norm
          → output

    Parameters
    ----------
    d_model : int
        Model (embedding) dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Hidden dimension of the point-wise feed-forward network.
        Defaults to ``4 * d_model``.
    dropout : float
        Dropout probability applied inside attention and after each sub-layer.
    top_k : int, optional
        Passed to :class:`UnifiedMemoryAttention`; limits how many memory
        entries are included per forward pass.
    init_alpha : float
        Initial temporal-decay coefficient passed to :class:`TimeBias`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        top_k: Optional[int] = None,
        init_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model

        self.attn = UnifiedMemoryAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            top_k=top_k,
            init_alpha=init_alpha,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory_bank: Optional[MemoryBank] = None,
        seq_timestamps: Optional[Tensor] = None,
        current_step: int = 0,
        write_to_memory: bool = True,
        attn_mask: Optional[Tensor] = None,
        causal: bool = False,
        # 新增参数
        verdicts=None,
        source_authority: float = 0.5,
        memory_type: str = "episodic",
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, d_model)`` – current input sequence.
        memory_bank : MemoryBank, optional
            Shared memory bank updated across forward passes.
        seq_timestamps : Tensor, optional
            Shape ``(T,)`` – per-token global-step timestamps.
        current_step : int
            Global step of the first token in ``x``.
        write_to_memory : bool
            Whether to write the current sequence into ``memory_bank``.
        attn_mask : Tensor, optional
            Additive attention mask of shape ``(T, T_all)`` or
            ``(B, H, T, T_all)``.

        Returns
        -------
        Tensor
            Shape ``(B, T, d_model)``.
        """
        # --- Attention sub-layer (pre-norm style) ----------------------
        attn_out = self.attn(
            x,
            memory_bank=memory_bank,
            seq_timestamps=seq_timestamps,
            current_step=current_step,
            write_to_memory=write_to_memory,
            attn_mask=attn_mask,
            causal=causal,
            verdicts=verdicts,
            source_authority=source_authority,
            memory_type=memory_type,
        )
        x = self.norm1(x + self.drop1(attn_out))

        # --- Feed-forward sub-layer ------------------------------------
        x = self.norm2(x + self.drop2(self.ff(x)))
        return x
