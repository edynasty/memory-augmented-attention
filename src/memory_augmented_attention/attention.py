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
# DynamicMemoryBias
# ---------------------------------------------------------------------------


class DynamicMemoryBias(nn.Module):
    """多维度动态记忆偏置

    从单一时间衰减扩展为 [时间, 可信度, 频率, 来源] 四维动态权重。
    权重由当前 query 上下文通过轻量 MLP 动态生成。

    公式: bias = sum(w_i * f_i) where w = MLP(query_context)
    - f_time = -alpha * log(1 + |t_q - t_k|)
    - f_verdict = verdict_value (标量 [-1, +1])
    - f_freq = log(1 + access_count) (归一化)
    - f_source = source_authority (标量 [0, 1])
    """

    def __init__(self, d_model: int, n_dims: int = 4, init_alpha: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_dims = n_dims

        # 保留 TimeBias 作为时间维度的子模块
        self.time_bias = TimeBias(init_alpha=init_alpha)

        # MLP 权重生成器：根据 query 上下文动态生成各维度权重
        self.weight_generator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_dims),
        )

        # 初始化偏置：让 w_time 初始为 1.0，其余为 0
        # 这样初始行为等同于纯 TimeBias（向后兼容）
        with torch.no_grad():
            self.weight_generator[-1].bias.copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0][:n_dims])
            )
            self.weight_generator[-1].weight.zero_()

    def forward(
        self,
        query_context: Tensor,    # (d_model,) 或 (B, d_model) - query 的平均嵌入
        t_query: Tensor,          # (T_q,) 查询时间戳
        t_key: Tensor,            # (T_k,) 键时间戳
        verdicts: Optional[Tensor] = None,     # (T_k,) 可信度 [-1, +1]
        frequencies: Optional[Tensor] = None,  # (T_k,) 访问次数
        sources: Optional[Tensor] = None,      # (T_k,) 来源权威性 [0, 1]
    ) -> Tensor:
        """
        Returns:
            bias: (T_q, T_k) 注意力偏置矩阵
        """
        T_q = t_query.shape[0]
        T_k = t_key.shape[0]

        # 1. 生成动态权重
        if query_context.dim() == 1:
            query_context = query_context.unsqueeze(0)
        weights = self.weight_generator(
            query_context)  # (1, n_dims) 或 (B, n_dims)
        weights = weights.squeeze(0)  # (n_dims,) - 用同一组权重广播到所有 query 位置

        # 2. 计算各维度特征
        # f_time: (T_q, T_k)
        f_time = self.time_bias(t_query, t_key)

        # 组合 bias 矩阵
        bias = weights[0] * f_time  # 时间维度始终有值

        # f_verdict: (T_k,) -> broadcast to (T_q, T_k)
        if verdicts is not None and self.n_dims > 1:
            f_verdict = verdicts.float().unsqueeze(0).expand(T_q, T_k)  # (T_q, T_k)
            bias = bias + weights[1] * f_verdict

        # f_freq: (T_k,) -> (T_q, T_k)
        if frequencies is not None and self.n_dims > 2:
            f_freq = torch.log1p(frequencies.float()).unsqueeze(
                0).expand(T_q, T_k)
            bias = bias + weights[2] * f_freq

        # f_source: (T_k,) -> (T_q, T_k)
        if sources is not None and self.n_dims > 3:
            f_source = sources.float().unsqueeze(0).expand(T_q, T_k)
            bias = bias + weights[3] * f_source

        return bias


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

        self.memory_bias = DynamicMemoryBias(
            d_model=d_model, n_dims=4, init_alpha=init_alpha)
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
        causal: bool = False,
        # 新增参数（可选，向后兼容）
        verdicts: Optional[Tensor] = None,
        source_authority: float = 0.5,
        memory_type: str = "episodic",
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
        causal : bool
            If ``True``, apply a causal (lower-triangular) mask to the current
            sequence tokens so position ``i`` cannot attend to positions
            ``> i`` within ``x``.  Memory-bank entries are always attended to
            (they represent strictly older history).

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

        metadata = None
        if memory_bank is not None and len(memory_bank) > 0:
            # Retrieve from bank (optionally top-k per query).
            if self.top_k is not None:
                # Use the mean query vector as the probe; average over B and T
                # but keep the head dimension so the shape matches stored keys.
                probe = Q.detach().mean(dim=(0, 2))  # (H, d_k)
                mem_K, mem_V, _, mem_t, metadata = memory_bank.top_k_retrieve_with_metadata(
                    query=probe, k=self.top_k, device=device
                )
            else:
                mem_K, mem_V, _, mem_t, metadata = memory_bank.retrieve_all_with_metadata(
                    device=device)

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

        # 计算 query 上下文向量（用于动态权重生成）
        query_context = x.mean(dim=1).mean(dim=0)  # (d_model,)

        # 组装元数据（来自 memory bank 检索）
        verdicts_all = None
        frequencies_all = None
        sources_all = None
        if metadata is not None:
            mem_verdicts = metadata['verdicts']   # (M,)
            mem_freqs = metadata['frequencies']   # (M,)
            mem_sources = metadata['sources']     # (M,)

            # 为当前序列 token 补充默认元数据
            seq_verdicts = torch.zeros(T, device=device)
            seq_freqs = torch.zeros(T, device=device)
            seq_sources = torch.ones(T, device=device) * 0.5

            verdicts_all = torch.cat([mem_verdicts, seq_verdicts])
            frequencies_all = torch.cat([mem_freqs, seq_freqs])
            sources_all = torch.cat([mem_sources, seq_sources])

        # 计算多维度偏置
        bias = self.memory_bias(
            query_context, seq_timestamps, t_all,
            verdicts=verdicts_all, frequencies=frequencies_all, sources=sources_all
        )
        logits = logits + bias.unsqueeze(0).unsqueeze(0)

        if attn_mask is not None:
            # attn_mask: (T, T_all) or (B, H, T, T_all)
            logits = logits + attn_mask

        if causal:
            # Causal mask applies only to the current-sequence tail of T_all;
            # memory-bank columns (if any) are strictly older and always
            # visible.
            cm = torch.zeros(T, T_all, device=device)
            seq_triu = torch.triu(
                torch.full((T, T), float("-inf"), device=device), diagonal=1,
            )
            cm[:, T_all - T:] = seq_triu
            logits = logits + cm

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
                verdicts=[0.0] * T if verdicts is None else verdicts.tolist(
                ) if isinstance(verdicts, Tensor) else verdicts,
                source_authorities=[source_authority] * T,
                memory_type=memory_type,
            )

        return out
