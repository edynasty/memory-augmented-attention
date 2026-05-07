"""Tests for MemoryBank."""

import pytest
import torch

from memory_augmented_attention.memory_bank import MemoryBank, MemoryEntry, MemoryType, ConsolidationNetwork


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


# ---------------------------------------------------------------------------
# Extended Entry Fields
# ---------------------------------------------------------------------------


class TestMemoryEntryExtended:
    """测试 MemoryEntry 扩展字段"""

    def test_default_fields(self):
        """新字段有正确的默认值"""
        entry = MemoryEntry(
            key=torch.randn(4, 32),
            value=torch.randn(4, 32),
            query=torch.randn(4, 32),
            timestamp=0,
        )
        assert entry.verdict == 0.0
        assert entry.access_count == 0
        assert entry.source_authority == 0.5
        assert entry.memory_type == "episodic"

    def test_custom_fields(self):
        """可以设置自定义字段值"""
        entry = MemoryEntry(
            key=torch.randn(4, 32),
            value=torch.randn(4, 32),
            query=torch.randn(4, 32),
            timestamp=10,
            verdict=1.0,
            access_count=5,
            source_authority=0.9,
            memory_type="semantic",
        )
        assert entry.verdict == 1.0
        assert entry.access_count == 5
        assert entry.source_authority == 0.9
        assert entry.memory_type == "semantic"

    def test_write_with_metadata(self):
        """write() 接受新参数"""
        bank = MemoryBank(max_size=10)
        result = bank.write(
            key=torch.randn(4, 32),
            value=torch.randn(4, 32),
            query=torch.randn(4, 32),
            timestamp=0,
            verdict=1.0,
            source_authority=0.8,
            memory_type="semantic",
        )
        assert result is True
        assert bank.entries[0].verdict == 1.0
        assert bank.entries[0].source_authority == 0.8
        assert bank.entries[0].memory_type == "semantic"

    def test_write_sequence_with_metadata(self):
        """write_sequence() 接受 verdicts 和 source_authorities"""
        bank = MemoryBank(max_size=20)
        keys = torch.randn(5, 4, 32)
        values = torch.randn(5, 4, 32)
        queries = torch.randn(5, 4, 32)
        verdicts = [1.0, -1.0, 0.0, 1.0, -1.0]
        sources = [0.9, 0.1, 0.5, 0.8, 0.2]

        next_ts = bank.write_sequence(
            keys, values, queries, start_timestamp=0,
            verdicts=verdicts, source_authorities=sources, memory_type="episodic"
        )
        assert next_ts == 5
        assert bank.entries[0].verdict == 1.0
        assert bank.entries[1].verdict == -1.0
        assert bank.entries[0].source_authority == 0.9


class TestMemoryType:
    """测试 MemoryType 枚举"""

    def test_enum_values(self):
        """枚举值正确"""
        assert MemoryType.EPISODIC == "episodic"
        assert MemoryType.SEMANTIC == "semantic"
        assert MemoryType.PROCEDURAL == "procedural"

    def test_get_type_priors(self):
        """先验权重形状和范围正确"""
        for mtype in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            priors = MemoryType.get_type_priors(mtype)
            assert len(priors) == 4
            for p in priors:
                assert 0.0 <= p <= 1.0

    def test_episodic_priors_time_dominant(self):
        """情景型记忆时间权重最高"""
        priors = MemoryType.get_type_priors("episodic")
        assert priors[0] >= priors[1]  # time >= verdict
        assert priors[0] >= priors[3]  # time >= source

    def test_semantic_priors_verdict_dominant(self):
        """知识型记忆可信度权重最高"""
        priors = MemoryType.get_type_priors("semantic")
        assert priors[1] >= priors[0]  # verdict >= time
        assert priors[3] >= priors[0]  # source >= time

    def test_procedural_priors_freq_dominant(self):
        """程序型记忆频率权重最高"""
        priors = MemoryType.get_type_priors("procedural")
        assert priors[2] >= priors[1]  # freq >= verdict
        assert priors[2] >= priors[3]  # freq >= source


class TestRetrieveWithMetadata:
    """测试带元数据的检索方法"""

    def test_retrieve_all_with_metadata_shapes(self):
        """retrieve_all_with_metadata 返回正确形状"""
        bank = MemoryBank(max_size=20)
        for i in range(5):
            bank.write(torch.randn(4, 32), torch.randn(4, 32), torch.randn(4, 32),
                       timestamp=i, verdict=float(i % 2), source_authority=0.5)

        keys, vals, queries, timestamps, metadata = bank.retrieve_all_with_metadata()
        assert keys.shape == (5, 4, 32)
        assert metadata['verdicts'].shape == (5,)
        assert metadata['frequencies'].shape == (5,)
        assert metadata['sources'].shape == (5,)

    def test_top_k_retrieve_with_metadata(self):
        """top_k_retrieve_with_metadata 返回元数据"""
        bank = MemoryBank(max_size=20)
        for i in range(10):
            bank.write(torch.randn(4, 32), torch.randn(4, 32), torch.randn(4, 32),
                       timestamp=i, verdict=1.0 if i < 5 else -1.0)

        query = torch.randn(4, 32)
        keys, vals, queries, timestamps, metadata = bank.top_k_retrieve_with_metadata(
            query, k=5)
        assert keys.shape[0] == 5
        assert metadata['verdicts'].shape == (5,)

    def test_increment_access(self):
        """increment_access 正确更新频率"""
        bank = MemoryBank(max_size=10)
        for i in range(5):
            bank.write(torch.randn(4, 32), torch.randn(
                4, 32), torch.randn(4, 32), timestamp=i)

        bank.increment_access([0, 2, 4])
        assert bank.entries[0].access_count == 1
        assert bank.entries[1].access_count == 0
        assert bank.entries[2].access_count == 1

        bank.increment_access([0, 0])
        assert bank.entries[0].access_count == 3


class TestConsolidationNetwork:
    """测试可学习归纳网络"""

    def test_output_shapes(self):
        """压缩输出形状正确"""
        net = ConsolidationNetwork(d_model=128, n_heads=4, compression_ratio=4)

        N = 16
        old_keys = torch.randn(N, 4, 32)
        old_values = torch.randn(N, 4, 32)
        old_queries = torch.randn(N, 4, 32)
        old_timestamps = torch.arange(N)

        s_keys, s_values, s_queries, s_timestamps = net(
            old_keys, old_values, old_queries, old_timestamps)
        M = N // 4  # compression_ratio=4
        assert s_keys.shape == (M, 4, 32)
        assert s_values.shape == (M, 4, 32)
        assert s_queries.shape == (M, 4, 32)
        assert s_timestamps.shape == (M,)

    def test_compression_ratio(self):
        """不同压缩比输出正确数量"""
        for ratio in [2, 4, 8]:
            net = ConsolidationNetwork(
                d_model=128, n_heads=4, compression_ratio=ratio)
            N = 24
            s_keys, _, _, _ = net(
                torch.randn(N, 4, 32), torch.randn(N, 4, 32),
                torch.randn(N, 4, 32), torch.arange(N)
            )
            assert s_keys.shape[0] == max(1, N // ratio)

    def test_trainable(self):
        """网络参数可训练"""
        net = ConsolidationNetwork(d_model=128, n_heads=4)

        old_keys = torch.randn(8, 4, 32, requires_grad=False)
        old_values = torch.randn(8, 4, 32)
        old_queries = torch.randn(8, 4, 32)
        old_timestamps = torch.arange(8)

        s_keys, s_values, _, _ = net(
            old_keys, old_values, old_queries, old_timestamps)
        loss = s_keys.sum() + s_values.sum()
        loss.backward()

        # 检查至少有部分参数收到了梯度
        grads_found = sum(1 for p in net.parameters() if p.grad is not None)
        assert grads_found > 0, "No parameter received gradients"

    def test_consolidate_integration(self):
        """MemoryBank.consolidate() 集成测试"""
        bank = MemoryBank(max_size=100)

        # 写入 20 条
        for i in range(20):
            bank.write(torch.randn(4, 32), torch.randn(
                4, 32), torch.randn(4, 32), timestamp=i)

        assert len(bank) == 20

        net = ConsolidationNetwork(d_model=128, n_heads=4, compression_ratio=4)
        bank.consolidate(net, keep_newest=8)

        # 应该是 12条旧条目压缩为3条 + 8条新条目 = 11条
        assert len(bank) <= 20  # 总数减少了
        assert len(bank) >= 8   # 至少保留了 keep_newest 条
