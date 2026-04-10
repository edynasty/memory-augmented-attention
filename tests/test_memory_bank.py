"""Tests for MemoryBank."""

import pytest
import torch

from memory_augmented_attention.memory_bank import MemoryBank, MemoryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_kv(d: int = 8, h: int = 2) -> tuple:
    """Return random (key, value, query) of shape (h, d)."""
    k = torch.randn(h, d)
    v = torch.randn(h, d)
    q = torch.randn(h, d)
    return k, v, q


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction():
    bank = MemoryBank(max_size=16)
    assert len(bank) == 0
    assert bank.max_size == 16
    assert bank.importance_threshold == 0.0


def test_invalid_max_size():
    with pytest.raises(ValueError, match="max_size"):
        MemoryBank(max_size=0)


def test_invalid_importance_threshold():
    with pytest.raises(ValueError, match="importance_threshold"):
        MemoryBank(max_size=4, importance_threshold=-0.1)


# ---------------------------------------------------------------------------
# Write / eviction
# ---------------------------------------------------------------------------


def test_write_single_entry():
    bank = MemoryBank(max_size=4)
    k, v, q = make_kv()
    accepted = bank.write(k, v, q, timestamp=0)
    assert accepted is True
    assert len(bank) == 1


def test_write_evicts_oldest_when_full():
    bank = MemoryBank(max_size=3)
    for i in range(3):
        k, v, q = make_kv()
        bank.write(k, v, q, timestamp=i)
    assert len(bank) == 3

    # Writing one more should evict timestamp=0.
    k, v, q = make_kv()
    bank.write(k, v, q, timestamp=3)
    assert len(bank) == 3
    timestamps = [e.timestamp for e in bank.entries]
    assert timestamps == [1, 2, 3]


def test_write_sequence():
    bank = MemoryBank(max_size=10)
    T = 5
    d, h = 8, 2
    keys = torch.randn(T, h, d)
    values = torch.randn(T, h, d)
    queries = torch.randn(T, h, d)
    next_step = bank.write_sequence(keys, values, queries, start_timestamp=10)
    assert next_step == 15
    assert len(bank) == T
    for i, entry in enumerate(bank.entries):
        assert entry.timestamp == 10 + i


def test_importance_threshold_rejects_small_values():
    bank = MemoryBank(max_size=4, importance_threshold=100.0)
    k = torch.randn(2, 8)
    v = torch.zeros(2, 8)  # near-zero norm
    q = torch.randn(2, 8)
    accepted = bank.write(k, v, q, timestamp=0)
    assert accepted is False
    assert len(bank) == 0


def test_importance_threshold_accepts_large_values():
    bank = MemoryBank(max_size=4, importance_threshold=0.1)
    k, v, q = make_kv(d=64)  # large random tensors have non-trivial norm
    accepted = bank.write(k, v, q, timestamp=0)
    assert accepted is True


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_retrieve_all_empty_raises():
    bank = MemoryBank(max_size=4)
    with pytest.raises(RuntimeError):
        bank.retrieve_all()


def test_retrieve_all_shapes():
    bank = MemoryBank(max_size=8)
    h, d = 2, 8
    for i in range(4):
        k, v, q = make_kv(d=d, h=h)
        bank.write(k, v, q, timestamp=i)

    keys, values, queries, timestamps = bank.retrieve_all()
    assert keys.shape == (4, h, d)
    assert values.shape == (4, h, d)
    assert queries.shape == (4, h, d)
    assert timestamps.shape == (4,)
    assert timestamps.tolist() == [0, 1, 2, 3]


def test_top_k_retrieve_fewer_than_k():
    bank = MemoryBank(max_size=8)
    for i in range(3):
        k, v, q = make_kv()
        bank.write(k, v, q, timestamp=i)

    probe = torch.randn(2, 8)
    keys, values, queries, timestamps = bank.top_k_retrieve(probe, k=10)
    # k > stored, should return all 3.
    assert keys.shape[0] == 3


def test_top_k_retrieve_exactly_k():
    bank = MemoryBank(max_size=16)
    for i in range(10):
        k, v, q = make_kv()
        bank.write(k, v, q, timestamp=i)

    probe = torch.randn(2, 8)
    keys, values, queries, timestamps = bank.top_k_retrieve(probe, k=4)
    assert keys.shape[0] == 4
    # Timestamps should be in ascending order (chronological).
    assert (timestamps[1:] >= timestamps[:-1]).all()


def test_top_k_retrieve_empty_raises():
    bank = MemoryBank(max_size=4)
    probe = torch.randn(2, 8)
    with pytest.raises(RuntimeError):
        bank.top_k_retrieve(probe, k=3)


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------


def test_clear():
    bank = MemoryBank(max_size=4)
    for i in range(3):
        k, v, q = make_kv()
        bank.write(k, v, q, timestamp=i)
    bank.clear()
    assert len(bank) == 0


def test_compress_by_time():
    bank = MemoryBank(max_size=10)
    for i in range(6):
        k, v, q = make_kv()
        bank.write(k, v, q, timestamp=i)
    bank.compress_by_time(keep_newest=3)
    assert len(bank) == 3
    assert bank.entries[0].timestamp == 3
    assert bank.entries[-1].timestamp == 5


def test_repr():
    bank = MemoryBank(max_size=32)
    assert "MemoryBank" in repr(bank)
    assert "max_size=32" in repr(bank)
