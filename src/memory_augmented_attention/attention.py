"""
Unified Memory-Augmented Attention
====================================

This module realises the unified-memory attention formulation from the design
document:

    M_all = M_memory ∪ { (K_seq, V_seq, Q_seq, t_seq) }

    y_t = Attention(Q_t, K_all, V_all, TimeBias(t_t, t_all))

where the attention weight for the pair ``(t_query, t_key)`` is:

    softmax( Q K^T / sqrt(d_k) + TimeBias(t_query, t_key) )

Classes
-------
TimeBias                   – learnable temporal decay bias
UnifiedMemoryAttention     – full unified memory-augmented multi-head attention
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .memory_bank import MemoryBank


# ---------------------------------------------------------------------------
# TimeBias
# ---------------------------------------------------------------------------


class TimeBias(nn.Module):
    """
    Learnable temporal bias added to attention logits.

    For a query at global-step ``t_q`` and a key at global-step ``t_k`` the
    bias is:

        TimeBias(t_q, t_k) = -alpha * log(1 + |t_q - t_k|)

    where ``alpha`` is a learnable scalar initialised to ``init_alpha``.
    The logarithmic form gives a slow, monotonically-decreasing decay so that
    very-old memories are only mildly penalised relative to moderately-old
    ones, mimicking the power-law forgetting curve observed in human memory.

    Parameters
    ----------
    init_alpha : float
        Initial value of the decay coefficient.  ``alpha = 0`` means no decay.
    """

    def __init__(self, init_alpha: float = 1.0) -> None:
        super().__init__()
        if init_alpha == 0.0:
            # alpha = 0 means no temporal decay; store as log(epsilon) and
            # clamp at zero so the bias is identically zero.
            self.log_alpha = nn.Parameter(torch.tensor(float("-inf")))
        else:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def forward(self, t_query: Tensor, t_key: Tensor) -> Tensor:
        """
        Compute the time bias matrix.

        Parameters
        ----------
        t_query : Tensor
            Shape ``(tq,)`` – timestamps of query positions (int or float).
        t_key : Tensor
            Shape ``(tk,)`` – timestamps of key positions.

        Returns
        -------
        Tensor
            Shape ``(tq, tk)`` – bias to *add* to raw attention logits before
            the softmax.  Negative values penalise temporally distant pairs.
        """
        t_query = t_query.float().unsqueeze(1)  # (tq, 1)
        t_key = t_key.float().unsqueeze(0)       # (1, tk)
        delta = (t_query - t_key).abs()          # (tq, tk)
        return -self.alpha * torch.log1p(delta)


# ---------------------------------------------------------------------------
# UnifiedMemoryAttention
# ---------------------------------------------------------------------------


class UnifiedMemoryAttention(nn.Module):
    """
    Unified Memory-Augmented Multi-Head Attention.

    The attention pool consists of **all** keys and values from:

    1. The current input sequence (treated as short-term memory).
    2. Historical entries retrieved from a :class:`~memory_augmented_attention.memory_bank.MemoryBank`
       (long-term memory).

    After computing Q/K/V for the current sequence, the K/V/Q projections of
    the current sequence are written into the bank so that future calls can
    draw on them.

    Attention score formula::

        logit(q, k) = (Q · K^T) / sqrt(d_k) + TimeBias(t_q, t_k)

    Parameters
    ----------
    d_model : int
        Model (embedding) dimension.
    n_heads : int
        Number of attention heads.  Must divide ``d_model`` evenly.
    dropout : float
        Dropout probability applied to attention weights.
    top_k : int, optional
        If given, only the ``top_k`` most relevant memory entries (by
        dot-product similarity to the current query) are included in the
        attention pool.  ``None`` means use all stored entries.
    init_alpha : float
        Initial temporal-decay coefficient (see :class:`TimeBias`).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        top_k: Optional[int] = None,
        init_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.top_k = top_k

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.time_bias = TimeBias(init_alpha=init_alpha)
        self.dropout = nn.Dropout(dropout)

        self._scale = math.sqrt(self.d_k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: Tensor) -> Tensor:
        """``(B, T, d_model)`` → ``(B, H, T, d_k)``."""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """``(B, H, T, d_k)`` → ``(B, T, d_model)``."""
        B, H, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        memory_bank: Optional[MemoryBank] = None,
        seq_timestamps: Optional[Tensor] = None,
        current_step: int = 0,
        write_to_memory: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute unified memory-augmented attention.

        Parameters
        ----------
        x : Tensor
            Current input sequence, shape ``(B, T, d_model)``.
        memory_bank : MemoryBank, optional
            Bank of historical (K, V, Q, timestamp) entries.  When ``None``
            the module falls back to standard self-attention over ``x`` only.
        seq_timestamps : Tensor, optional
            Shape ``(T,)`` – global-step timestamps for each token in ``x``.
            When ``None``, timestamps ``[current_step, …, current_step + T - 1]``
            are used automatically.
        current_step : int
            Global step index corresponding to the *first* token in ``x``.
            Used when ``seq_timestamps`` is ``None`` and when writing to the
            memory bank.
        write_to_memory : bool
            If ``True`` (default), the K/V/Q projections of the current
            sequence are written into ``memory_bank`` after computing attention
            so that they are available in future calls.
        attn_mask : Tensor, optional
            Additive mask of shape ``(T_q, T_all)`` added to the raw logits
            *before* the softmax.  ``-inf`` values block specific positions.

        Returns
        -------
        Tensor
            Output of shape ``(B, T, d_model)``.
        """
        B, T, _ = x.shape
        device = x.device

        # ---- Timestamps for the current sequence ----------------------
        if seq_timestamps is None:
            seq_timestamps = torch.arange(
                current_step, current_step + T, device=device, dtype=torch.long
            )

        # ---- Project current sequence ---------------------------------
        Q = self._split_heads(self.q_proj(x))  # (B, H, T, d_k)
        K_seq = self._split_heads(self.k_proj(x))
        V_seq = self._split_heads(self.v_proj(x))

        # ---- Assemble memory pool ------------------------------------
        # Start with the current-sequence keys/values and their timestamps.
        K_all = K_seq  # (B, H, T_seq, d_k)
        V_all = V_seq
        t_all = seq_timestamps  # (T_seq,)

        if memory_bank is not None and len(memory_bank) > 0:
            # Retrieve from bank (optionally top-k per query).
            if self.top_k is not None:
                # Use the mean query vector as the probe; average over B and T
                # but keep the head dimension so the shape matches stored keys.
                probe = Q.detach().mean(dim=(0, 2))  # (H, d_k)
                mem_K, mem_V, _, mem_t = memory_bank.top_k_retrieve(
                    query=probe, k=self.top_k, device=device
                )
            else:
                mem_K, mem_V, _, mem_t = memory_bank.retrieve_all(device=device)

            # mem_K: (N_mem, H, d_k), mem_V: (N_mem, H, d_v)
            # We need shapes (B, H, N_mem, d_k) / (B, H, N_mem, d_v).
            N_mem = mem_K.shape[0]
            mem_K = mem_K.view(N_mem, self.n_heads, self.d_k)
            mem_V = mem_V.view(N_mem, self.n_heads, self.d_k)
            mem_K = mem_K.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)
            mem_V = mem_V.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)

            # Prepend memory to seq; chronologically memory is older.
            K_all = torch.cat([mem_K, K_seq], dim=2)   # (B, H, N_mem+T, d_k)
            V_all = torch.cat([mem_V, V_seq], dim=2)
            t_all = torch.cat([mem_t, seq_timestamps])  # (N_mem+T,)

        T_all = K_all.shape[2]

        # ---- Attention logits ----------------------------------------
        # (B, H, T, d_k) x (B, H, d_k, T_all) → (B, H, T, T_all)
        logits = torch.matmul(Q, K_all.transpose(-2, -1)) / self._scale

        # Add temporal bias: shape (T, T_all) → broadcast over (B, H).
        tb = self.time_bias(seq_timestamps.float(), t_all.float())  # (T, T_all)
        logits = logits + tb.unsqueeze(0).unsqueeze(0)

        if attn_mask is not None:
            # attn_mask: (T, T_all) or (B, H, T, T_all)
            logits = logits + attn_mask

        attn_weights = F.softmax(logits, dim=-1)   # (B, H, T, T_all)
        attn_weights = self.dropout(attn_weights)

        # ---- Weighted sum of values ----------------------------------
        out = torch.matmul(attn_weights, V_all)  # (B, H, T, d_k)
        out = self._merge_heads(out)             # (B, T, d_model)
        out = self.out_proj(out)

        # ---- Write current sequence into the bank --------------------
        if write_to_memory and memory_bank is not None:
            # Store per-token (H, d_k) tensors.
            # K_seq/V_seq shape: (B, H, T, d_k); use mean over batch dim.
            K_store = K_seq.mean(0).permute(1, 0, 2)  # (T, H, d_k)
            V_store = V_seq.mean(0).permute(1, 0, 2)
            Q_store = Q.mean(0).permute(1, 0, 2)
            memory_bank.write_sequence(
                keys=K_store,
                values=V_store,
                queries=Q_store,
                start_timestamp=current_step,
            )

        return out
