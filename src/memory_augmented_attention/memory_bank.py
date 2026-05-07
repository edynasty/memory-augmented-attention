"""
MemoryBank
==========

Unified Memory Bank that stores timestamped (K, V, Q) entries for both the
current-sequence tokens (short-term memory) and historical KV pairs
(long-term memory).

Design
------
* Each slot stores a key tensor ``K``, value tensor ``V``, query tensor ``Q``,
  and an integer *global step* timestamp ``t``.
* The bank has a fixed maximum capacity ``max_size``.  When full, the oldest
  entries (smallest timestamp) are evicted first.
* Top-K retrieval returns the ``k`` most relevant entries for a given query
  using dot-product similarity, so the Attention module only needs to attend
  over a manageable subset of the bank.
* An optional importance-based write filter lets callers skip tokens whose
  L2-norm falls below a threshold, avoiding cluttering the bank with
  uninformative entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Memory Type System
# ---------------------------------------------------------------------------


class MemoryType(str, Enum):
    """记忆类型枚举"""
    EPISODIC = "episodic"      # 情景型
    SEMANTIC = "semantic"      # 知识型
    PROCEDURAL = "procedural"  # 程序型

    @staticmethod
    def get_type_priors(memory_type: str) -> list:
        """返回 [w_time, w_verdict, w_freq, w_source] 维度先验权重"""
        priors = {
            "episodic": [0.8, 0.2, 0.5, 0.2],    # 时间高、可信度低
            "semantic": [0.2, 0.8, 0.2, 0.8],    # 可信度高、来源高
            "procedural": [0.4, 0.2, 0.8, 0.2],  # 频率高
        }
        return priors.get(memory_type, priors["episodic"])


# ---------------------------------------------------------------------------
# Memory Entry
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """One slot in the Memory Bank."""

    key: Tensor       # (head, d_k)
    value: Tensor     # (head, d_v)
    query: Tensor     # (head, d_k)
    timestamp: int    # global step at which this entry was written
    # 新增字段（向后兼容，均有默认值）
    verdict: float = 0.0           # 可信度 [-1.0, +1.0]，0=未验证
    access_count: int = 0          # 被检索次数
    source_authority: float = 0.5  # 来源权威性 [0, 1]
    memory_type: str = "episodic"  # "episodic" | "semantic" | "procedural"


# ---------------------------------------------------------------------------
# Consolidation Network
# ---------------------------------------------------------------------------


class ConsolidationNetwork(nn.Module):
    """可学习的记忆归纳网络：将 N 条旧记忆压缩为 N/ratio 条摘要"""

    def __init__(self, d_model: int, n_heads: int = 4, compression_ratio: int = 4):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.d_model = d_model
        self.n_heads = n_heads

        # Learnable summary queries（用于交叉注意力）
        self.summary_proj = nn.Linear(d_model, d_model)

        # Cross-attention: summary queries attend to old entries
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)

        # Output projection for keys, values, queries
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)

    def forward(self, old_keys: Tensor, old_values: Tensor,
                old_queries: Tensor, old_timestamps: Tensor):
        """
        Args:
            old_keys: (N, n_heads, d_k)
            old_values: (N, n_heads, d_v)
            old_queries: (N, n_heads, d_k)
            old_timestamps: (N,)
        Returns:
            summary_keys: (M, n_heads, d_k) where M = N // compression_ratio
            summary_values: (M, n_heads, d_v)
            summary_queries: (M, n_heads, d_k)
            summary_timestamps: (M,) 每个摘要取对应组的平均时间戳
        """
        N = old_keys.shape[0]
        n_heads = old_keys.shape[1]
        d_k = old_keys.shape[2]
        M = max(1, N // self.compression_ratio)

        # 将多头展平为 d_model
        old_flat = old_keys.reshape(N, -1)  # (N, d_model)

        # 生成 M 个 summary queries（使用均匀采样的 key 作为初始化）
        indices = torch.linspace(0, N - 1, M).long().to(old_keys.device)
        init_queries = old_flat[indices]  # (M, d_model)
        summary_q = self.summary_proj(init_queries)  # (M, d_model)

        # Cross-attention: (M, d_model) attends to (N, d_model)
        summary_q = summary_q.unsqueeze(0)  # (1, M, d_model)
        old_flat = old_flat.unsqueeze(0)     # (1, N, d_model)

        attn_out, _ = self.cross_attn(
            summary_q, old_flat, old_flat)  # (1, M, d_model)
        attn_out = attn_out.squeeze(0)  # (M, d_model)

        # 生成 summary KVQ
        s_keys = self.key_proj(attn_out).reshape(M, n_heads, d_k)
        s_values = self.value_proj(attn_out).reshape(M, n_heads, d_k)
        s_queries = self.query_proj(attn_out).reshape(M, n_heads, d_k)

        # 时间戳：按组平均
        chunk_size = N // M
        s_timestamps = torch.zeros(M, device=old_timestamps.device)
        for i in range(M):
            start = i * chunk_size
            end = min(start + chunk_size, N)
            s_timestamps[i] = old_timestamps[start:end].float().mean()

        return s_keys, s_values, s_queries, s_timestamps.long()


# ---------------------------------------------------------------------------
# Memory Bank
# ---------------------------------------------------------------------------


class MemoryBank:
    """
    Unified Memory Bank with timestamped (K, V, Q) entries.

    Parameters
    ----------
    max_size : int
        Maximum number of entries the bank can hold.  When the bank is full
        the oldest entries are evicted to make room.
    importance_threshold : float, optional
        Minimum L2-norm of the *value* vector required for an entry to be
        written.  Entries whose value norm is below this threshold are
        silently discarded.  Default ``0.0`` (accept everything).

    Attributes
    ----------
    entries : list[MemoryEntry]
        Chronologically ordered list of stored entries (oldest first).
    """

    def __init__(
        self,
        max_size: int = 1024,
        importance_threshold: float = 0.0,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if importance_threshold < 0.0:
            raise ValueError(
                f"importance_threshold must be >= 0, got {importance_threshold}"
            )
        self.max_size = max_size
        self.importance_threshold = importance_threshold
        self.entries: list[MemoryEntry] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        timestamp: int,
        verdict: float = 0.0,
        access_count: int = 0,
        source_authority: float = 0.5,
        memory_type: str = "episodic",
    ) -> bool:
        """
        Write a single entry to the bank.

        Parameters
        ----------
        key : Tensor
            Shape ``(..., d_k)``.
        value : Tensor
            Shape ``(..., d_v)``.
        query : Tensor
            Shape ``(..., d_k)``.
        timestamp : int
            Global step index for this entry.
        verdict : float
            Credibility score in [-1.0, +1.0]. Default 0.0 (unverified).
        access_count : int
            Initial access count. Default 0.
        source_authority : float
            Source authority in [0, 1]. Default 0.5.
        memory_type : str
            One of "episodic", "semantic", "procedural". Default "episodic".

        Returns
        -------
        bool
            ``True`` if the entry was accepted and stored, ``False`` if it was
            rejected by the importance filter.
        """
        if self.importance_threshold > 0.0:
            norm = value.detach().norm().item()
            if norm < self.importance_threshold:
                return False

        if len(self.entries) >= self.max_size:
            # Evict the oldest entry (front of the list).
            self.entries.pop(0)

        self.entries.append(
            MemoryEntry(
                key=key.detach(),
                value=value.detach(),
                query=query.detach(),
                timestamp=timestamp,
                verdict=verdict,
                access_count=access_count,
                source_authority=source_authority,
                memory_type=memory_type,
            )
        )
        return True

    def write_sequence(
        self,
        keys: Tensor,
        values: Tensor,
        queries: Tensor,
        start_timestamp: int,
        verdicts: Optional[List[float]] = None,
        source_authorities: Optional[List[float]] = None,
        memory_type: str = "episodic",
    ) -> int:
        """
        Write a batch of sequence-length entries at once.

        Parameters
        ----------
        keys : Tensor
            Shape ``(seq_len, ..., d_k)`` – one key per token.
        values : Tensor
            Shape ``(seq_len, ..., d_v)``.
        queries : Tensor
            Shape ``(seq_len, ..., d_k)``.
        start_timestamp : int
            Timestamp assigned to the first token; subsequent tokens receive
            ``start_timestamp + 1``, ``start_timestamp + 2``, …
        verdicts : Optional[List[float]]
            Per-token verdict scores. None uses default (0.0).
        source_authorities : Optional[List[float]]
            Per-token source authority. None uses default (0.5).
        memory_type : str
            Memory type for all entries in this batch. Default "episodic".

        Returns
        -------
        int
            The *next* available timestamp (``start_timestamp + seq_len``).
        """
        seq_len = keys.shape[0]
        for i in range(seq_len):
            v = verdicts[i] if verdicts is not None else 0.0
            sa = source_authorities[i] if source_authorities is not None else 0.5
            self.write(
                key=keys[i],
                value=values[i],
                query=queries[i],
                timestamp=start_timestamp + i,
                verdict=v,
                source_authority=sa,
                memory_type=memory_type,
            )
        return start_timestamp + seq_len

    # ------------------------------------------------------------------
    # Read / retrieval
    # ------------------------------------------------------------------

    def _build_metadata(self, entries: list, device=None) -> dict:
        """Build metadata dict from a list of MemoryEntry objects."""
        dev = device or (entries[0].key.device if entries else None)
        return {
            'verdicts': torch.tensor(
                [e.verdict for e in entries], dtype=torch.float, device=dev
            ),
            'frequencies': torch.tensor(
                [e.access_count for e in entries], dtype=torch.float, device=dev
            ),
            'sources': torch.tensor(
                [e.source_authority for e in entries], dtype=torch.float, device=dev
            ),
            'types': [e.memory_type for e in entries],
        }

    def retrieve_all(
        self, device: Optional[torch.device] = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return all stored entries as stacked tensors.

        Returns
        -------
        keys : Tensor
            Shape ``(N, ..., d_k)``.
        values : Tensor
            Shape ``(N, ..., d_v)``.
        queries : Tensor
            Shape ``(N, ..., d_k)``.
        timestamps : Tensor[int64]
            Shape ``(N,)`` – one scalar timestamp per entry.
        """
        if not self.entries:
            raise RuntimeError("MemoryBank is empty; nothing to retrieve.")

        keys = torch.stack([e.key for e in self.entries])
        values = torch.stack([e.value for e in self.entries])
        queries = torch.stack([e.query for e in self.entries])
        timestamps = torch.tensor(
            [e.timestamp for e in self.entries],
            dtype=torch.long,
            device=device or keys.device,
        )
        if device is not None:
            keys = keys.to(device)
            values = values.to(device)
            queries = queries.to(device)
        return keys, values, queries, timestamps

    def retrieve_all_with_metadata(
        self, device: Optional[torch.device] = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        """
        Return all stored entries as stacked tensors with metadata.

        Returns
        -------
        keys : Tensor
            Shape ``(N, ..., d_k)``.
        values : Tensor
            Shape ``(N, ..., d_v)``.
        queries : Tensor
            Shape ``(N, ..., d_k)``.
        timestamps : Tensor[int64]
            Shape ``(N,)`` – one scalar timestamp per entry.
        metadata : dict
            - 'verdicts': Tensor of shape (N,)
            - 'frequencies': Tensor of shape (N,)
            - 'sources': Tensor of shape (N,)
            - 'types': List[str] of length N
        """
        keys, values, queries, timestamps = self.retrieve_all(device=device)
        metadata = self._build_metadata(
            self.entries, device=device or keys.device)
        return keys, values, queries, timestamps, metadata

    def top_k_retrieve(
        self,
        query: Tensor,
        k: int,
        device: Optional[torch.device] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return the ``k`` most relevant entries for the given *query* vector.

        Relevance is measured by the dot product between ``query`` and each
        stored key (after L2-normalising both vectors).

        Parameters
        ----------
        query : Tensor
            Shape ``(d_k,)`` or ``(heads, d_k)`` – the probe vector.
        k : int
            Number of entries to return.  Clamped to ``len(self.entries)``.

        Returns
        -------
        keys, values, queries, timestamps
            Same shapes as :meth:`retrieve_all` but with the first dimension
            equal to ``min(k, len(self.entries))``.
        """
        if not self.entries:
            raise RuntimeError("MemoryBank is empty; nothing to retrieve.")

        k = min(k, len(self.entries))
        if k == len(self.entries):
            return self.retrieve_all(device=device)

        keys, values, queries, timestamps = self.retrieve_all(device=device)

        # Flatten head/spatial dims for similarity computation.
        q_flat = query.detach().reshape(-1).float()
        k_flat = keys.reshape(len(self.entries), -1).float()

        q_norm = F.normalize(q_flat.unsqueeze(0), dim=-1)  # (1, D)
        k_norm = F.normalize(k_flat, dim=-1)               # (N, D)
        scores = (k_norm @ q_norm.T).squeeze(-1)            # (N,)

        _, indices = scores.topk(k, dim=0)
        indices, _ = indices.sort()  # preserve chronological order

        return (
            keys[indices],
            values[indices],
            queries[indices],
            timestamps[indices],
        )

    def top_k_retrieve_with_metadata(
        self,
        query: Tensor,
        k: int,
        device: Optional[torch.device] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        """
        Return the ``k`` most relevant entries with metadata.

        Returns
        -------
        keys, values, queries, timestamps, metadata
            Same as :meth:`top_k_retrieve` plus a metadata dict with keys:
            - 'verdicts': Tensor of shape (K,)
            - 'frequencies': Tensor of shape (K,)
            - 'sources': Tensor of shape (K,)
            - 'types': List[str] of length K
        """
        if not self.entries:
            raise RuntimeError("MemoryBank is empty; nothing to retrieve.")

        k = min(k, len(self.entries))
        if k == len(self.entries):
            return self.retrieve_all_with_metadata(device=device)

        keys, values, queries, timestamps = self.retrieve_all(device=device)

        # Flatten head/spatial dims for similarity computation.
        q_flat = query.detach().reshape(-1).float()
        k_flat = keys.reshape(len(self.entries), -1).float()

        q_norm = F.normalize(q_flat.unsqueeze(0), dim=-1)  # (1, D)
        k_norm = F.normalize(k_flat, dim=-1)               # (N, D)
        scores = (k_norm @ q_norm.T).squeeze(-1)            # (N,)

        _, indices = scores.topk(k, dim=0)
        indices, _ = indices.sort()  # preserve chronological order

        selected_entries = [self.entries[i] for i in indices.tolist()]
        metadata = self._build_metadata(
            selected_entries, device=device or keys.device)

        return (
            keys[indices],
            values[indices],
            queries[indices],
            timestamps[indices],
            metadata,
        )

    # ------------------------------------------------------------------
    # Access tracking
    # ------------------------------------------------------------------

    def increment_access(self, indices: list) -> None:
        """增加指定索引条目的 access_count"""
        for idx in indices:
            if 0 <= idx < len(self.entries):
                self.entries[idx].access_count += 1

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self, network: 'ConsolidationNetwork', keep_newest: int) -> None:
        """使用归纳网络压缩旧条目为摘要

        Args:
            network: ConsolidationNetwork 实例
            keep_newest: 保留最新的 keep_newest 条不压缩
        """
        if len(self.entries) <= keep_newest:
            return  # 不需要压缩

        # 分离旧条目和新条目
        old_entries = self.entries[:-
                                   keep_newest] if keep_newest > 0 else self.entries[:]
        new_entries = self.entries[-keep_newest:] if keep_newest > 0 else []

        if len(old_entries) < network.compression_ratio:
            return  # 旧条目太少，不值得压缩

        # 提取旧条目张量
        device = old_entries[0].key.device
        old_keys = torch.stack([e.key for e in old_entries]).to(device)
        old_values = torch.stack([e.value for e in old_entries]).to(device)
        old_queries = torch.stack([e.query for e in old_entries]).to(device)
        old_timestamps = torch.tensor(
            [e.timestamp for e in old_entries], device=device
        )

        # 通过网络压缩
        with torch.no_grad():  # 归纳时不计算梯度（推理时使用）
            s_keys, s_values, s_queries, s_timestamps = network(
                old_keys, old_values, old_queries, old_timestamps
            )

        # 重建条目列表：摘要 + 保留的新条目
        summary_entries = []
        for i in range(s_keys.shape[0]):
            summary_entries.append(MemoryEntry(
                key=s_keys[i],
                value=s_values[i],
                query=s_queries[i],
                timestamp=int(s_timestamps[i].item()),
                verdict=0.0,
                access_count=0,
                source_authority=0.5,
                memory_type=MemoryType.SEMANTIC,  # 归纳后的知识
            ))

        self.entries = summary_entries + list(new_entries)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all entries from the bank."""
        self.entries.clear()

    def compress_by_time(self, keep_newest: int) -> None:
        """
        Retain only the ``keep_newest`` most recent entries.

        Parameters
        ----------
        keep_newest : int
            Number of entries to keep.
        """
        if keep_newest < len(self.entries):
            self.entries = self.entries[-keep_newest:]

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (
            f"MemoryBank(max_size={self.max_size}, "
            f"stored={len(self.entries)}, "
            f"importance_threshold={self.importance_threshold})"
        )
