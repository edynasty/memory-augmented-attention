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

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MemoryEntry:
    """One slot in the Memory Bank."""

    key: Tensor       # (head, d_k)
    value: Tensor     # (head, d_v)
    query: Tensor     # (head, d_k)
    timestamp: int    # global step at which this entry was written


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
            )
        )
        return True

    def write_sequence(
        self,
        keys: Tensor,
        values: Tensor,
        queries: Tensor,
        start_timestamp: int,
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

        Returns
        -------
        int
            The *next* available timestamp (``start_timestamp + seq_len``).
        """
        seq_len = keys.shape[0]
        for i in range(seq_len):
            self.write(
                key=keys[i],
                value=values[i],
                query=queries[i],
                timestamp=start_timestamp + i,
            )
        return start_timestamp + seq_len

    # ------------------------------------------------------------------
    # Read / retrieval
    # ------------------------------------------------------------------

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
