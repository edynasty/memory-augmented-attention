"""Tests for TimeBias, UnifiedMemoryAttention, and MemoryAugmentedTransformerLayer."""

import pytest
import torch
import torch.nn as nn

from memory_augmented_attention.attention import TimeBias, DynamicMemoryBias, UnifiedMemoryAttention
from memory_augmented_attention.memory_bank import MemoryBank
from memory_augmented_attention.model import MemoryAugmentedTransformerLayer


# ---------------------------------------------------------------------------
# TimeBias
# ---------------------------------------------------------------------------


class TestTimeBias:
    def test_output_shape(self):
        tb = TimeBias(init_alpha=1.0)
        t_q = torch.arange(3).float()
        t_k = torch.arange(5).float()
        out = tb(t_q, t_k)
        assert out.shape == (3, 5)

    def test_zero_lag_is_zero(self):
        """Bias for a query attending to itself (lag=0) must be 0."""
        tb = TimeBias(init_alpha=1.0)
        t = torch.tensor([0.0, 1.0, 2.0])
        bias = tb(t, t)
        # Diagonal should be -alpha * log(1) = 0.
        diag = bias.diagonal()
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5)

    def test_bias_is_non_positive(self):
        """All off-diagonal biases should be <= 0 (temporal penalty)."""
        tb = TimeBias(init_alpha=1.0)
        t_q = torch.tensor([5.0])
        t_k = torch.arange(10).float()
        bias = tb(t_q, t_k)
        assert (bias <= 1e-6).all()

    def test_older_keys_penalised_more(self):
        """Keys further in the past should receive a more-negative bias."""
        tb = TimeBias(init_alpha=1.0)
        t_q = torch.tensor([100.0])
        t_k = torch.tensor([99.0, 90.0, 50.0])
        bias = tb(t_q, t_k)  # (1, 3)
        # bias[:,0] > bias[:,1] > bias[:,2]  (less negative = better)
        assert bias[0, 0] > bias[0, 1] > bias[0, 2]

    def test_alpha_learned(self):
        """alpha parameter should change under gradient descent."""
        tb = TimeBias(init_alpha=1.0)
        t_q = torch.tensor([5.0])
        t_k = torch.tensor([0.0])
        loss = tb(t_q, t_k).sum()
        loss.backward()
        assert tb.log_alpha.grad is not None


# ---------------------------------------------------------------------------
# UnifiedMemoryAttention – without memory bank
# ---------------------------------------------------------------------------


class TestUnifiedMemoryAttentionNoBank:
    def setup_method(self):
        self.d_model = 16
        self.n_heads = 2
        self.attn = UnifiedMemoryAttention(
            d_model=self.d_model, n_heads=self.n_heads, dropout=0.0
        )

    def test_output_shape(self):
        B, T = 2, 5
        x = torch.randn(B, T, self.d_model)
        out = self.attn(x)
        assert out.shape == (B, T, self.d_model)

    def test_output_is_finite(self):
        x = torch.randn(2, 4, self.d_model)
        out = self.attn(x)
        assert torch.isfinite(out).all()

    def test_invalid_heads_raise(self):
        with pytest.raises(ValueError, match="divisible"):
            UnifiedMemoryAttention(d_model=15, n_heads=4)

    def test_gradients_flow(self):
        x = torch.randn(1, 3, self.d_model, requires_grad=True)
        out = self.attn(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# UnifiedMemoryAttention – with memory bank
# ---------------------------------------------------------------------------


class TestUnifiedMemoryAttentionWithBank:
    def setup_method(self):
        self.d_model = 16
        self.n_heads = 2
        self.attn = UnifiedMemoryAttention(
            d_model=self.d_model, n_heads=self.n_heads, dropout=0.0
        )
        self.bank = MemoryBank(max_size=32)

    def test_output_shape_with_bank(self):
        B, T = 2, 4
        x = torch.randn(B, T, self.d_model)
        # Pre-populate bank with a few entries.
        for i in range(5):
            self.attn(
                x[:, :1, :],
                memory_bank=self.bank,
                current_step=i,
                write_to_memory=True,
            )
        out = self.attn(x, memory_bank=self.bank, current_step=10)
        assert out.shape == (B, T, self.d_model)

    def test_bank_grows_after_write(self):
        bank = MemoryBank(max_size=64)
        B, T = 1, 3
        x = torch.randn(B, T, self.d_model)
        self.attn(x, memory_bank=bank, current_step=0, write_to_memory=True)
        assert len(bank) == T

    def test_bank_not_written_when_flag_false(self):
        bank = MemoryBank(max_size=64)
        x = torch.randn(1, 3, self.d_model)
        self.attn(x, memory_bank=bank, write_to_memory=False)
        assert len(bank) == 0

    def test_top_k_limits_memory_pool(self):
        attn = UnifiedMemoryAttention(
            d_model=self.d_model, n_heads=self.n_heads, dropout=0.0, top_k=3
        )
        bank = MemoryBank(max_size=32)
        # Populate 8 entries.
        for i in range(8):
            k = torch.randn(self.n_heads, self.d_model // self.n_heads)
            v = torch.randn(self.n_heads, self.d_model // self.n_heads)
            q = torch.randn(self.n_heads, self.d_model // self.n_heads)
            bank.write(k, v, q, timestamp=i)

        x = torch.randn(1, 2, self.d_model)
        out = attn(x, memory_bank=bank, current_step=10, write_to_memory=False)
        assert out.shape == (1, 2, self.d_model)

    def test_memory_across_two_forward_passes(self):
        """Output with memory should differ from output without it."""
        torch.manual_seed(42)
        bank = MemoryBank(max_size=64)
        attn = UnifiedMemoryAttention(
            d_model=self.d_model, n_heads=self.n_heads, dropout=0.0
        )
        attn.eval()

        x1 = torch.randn(1, 3, self.d_model)
        x2 = torch.randn(1, 3, self.d_model)

        # First pass – write x1 into bank.
        attn(x1, memory_bank=bank, current_step=0, write_to_memory=True)

        # Second pass without bank.
        out_no_mem = attn(x2, memory_bank=None, write_to_memory=False)
        # Second pass with bank (should see x1's KVs).
        out_with_mem = attn(x2, memory_bank=bank,
                            current_step=3, write_to_memory=False)

        diff = (out_no_mem - out_with_mem).abs().max().item()
        assert diff > 1e-5, (
            f"Expected outputs with and without memory to differ by > 1e-5, got {diff}"
        )


# ---------------------------------------------------------------------------
# MemoryAugmentedTransformerLayer
# ---------------------------------------------------------------------------


class TestTransformerLayer:
    def setup_method(self):
        self.layer = MemoryAugmentedTransformerLayer(
            d_model=16, n_heads=2, d_ff=32, dropout=0.0
        )

    def test_output_shape(self):
        x = torch.randn(2, 5, 16)
        out = self.layer(x)
        assert out.shape == (2, 5, 16)

    def test_residual_connection_changes_output(self):
        """Output must differ from raw input (residual does not reduce to identity)."""
        x = torch.randn(1, 4, 16)
        out = self.layer(x)
        assert not torch.allclose(x, out)

    def test_layer_with_memory_bank(self):
        bank = MemoryBank(max_size=64)
        x = torch.randn(1, 4, 16)
        out = self.layer(x, memory_bank=bank, current_step=0,
                         write_to_memory=True)
        assert out.shape == (1, 4, 16)
        assert len(bank) == 4  # 4 tokens written

    def test_multiple_forward_passes_accumulate_memory(self):
        bank = MemoryBank(max_size=64)
        layer = MemoryAugmentedTransformerLayer(
            d_model=16, n_heads=2, d_ff=32, dropout=0.0
        )
        layer.eval()

        for step in range(5):
            x = torch.randn(1, 3, 16)
            layer(x, memory_bank=bank, current_step=step * 3, write_to_memory=True)

        assert len(bank) == 15  # 5 passes × 3 tokens each

    def test_gradients_flow_through_layer(self):
        x = torch.randn(1, 3, 16, requires_grad=True)
        out = self.layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_no_memory_bank_is_valid(self):
        x = torch.randn(2, 6, 16)
        out = self.layer(x, memory_bank=None)
        assert out.shape == (2, 6, 16)


# ---------------------------------------------------------------------------
# DynamicMemoryBias
# ---------------------------------------------------------------------------


class TestDynamicMemoryBias:
    """测试多维度动态偏置"""

    def test_output_shape(self):
        """输出形状 (T_q, T_k)"""
        bias = DynamicMemoryBias(d_model=128, n_dims=4, init_alpha=1.0)

        query_ctx = torch.randn(128)
        t_q = torch.arange(8).float()
        t_k = torch.arange(16).float()

        out = bias(query_ctx, t_q, t_k)
        assert out.shape == (8, 16)

    def test_with_all_metadata(self):
        """传入全部元数据时输出正确"""
        bias = DynamicMemoryBias(d_model=128, n_dims=4, init_alpha=1.0)

        query_ctx = torch.randn(128)
        t_q = torch.arange(5).float()
        t_k = torch.arange(10).float()
        verdicts = torch.randn(10).clamp(-1, 1)
        freqs = torch.randint(0, 10, (10,)).float()
        sources = torch.rand(10)

        out = bias(query_ctx, t_q, t_k, verdicts, freqs, sources)
        assert out.shape == (5, 10)
        assert torch.isfinite(out).all()

    def test_no_metadata_degradation(self):
        """无元数据时退化为纯时间偏置"""
        bias = DynamicMemoryBias(d_model=128, n_dims=4, init_alpha=1.0)

        query_ctx = torch.randn(128)
        t_q = torch.arange(5).float()
        t_k = torch.arange(10).float()

        # 无元数据
        out_no_meta = bias(query_ctx, t_q, t_k)
        assert out_no_meta.shape == (5, 10)
        # 输出应为有限值
        assert torch.isfinite(out_no_meta).all()

    def test_gradient_flow(self):
        """梯度可以流经所有路径"""
        bias = DynamicMemoryBias(d_model=64, n_dims=4, init_alpha=1.0)

        query_ctx = torch.randn(64, requires_grad=True)
        t_q = torch.arange(4).float()
        t_k = torch.arange(8).float()
        verdicts = torch.randn(8).clamp(-1, 1)
        freqs = torch.randint(0, 5, (8,)).float()
        sources = torch.rand(8)

        out = bias(query_ctx, t_q, t_k, verdicts, freqs, sources)
        loss = out.sum()
        loss.backward()

        assert query_ctx.grad is not None
        # MLP 参数也有梯度
        for param in bias.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_initial_weights_time_dominant(self):
        """初始化时时间权重为 1，其余为 0"""
        bias = DynamicMemoryBias(d_model=64, n_dims=4, init_alpha=1.0)

        # 用零向量作为 context（不激活 MLP 的 weight 部分）
        query_ctx = torch.zeros(64)
        t_q = torch.arange(4).float()
        t_k = torch.arange(8).float()
        verdicts = torch.ones(8)

        out_with_verdict = bias(query_ctx, t_q, t_k, verdicts)
        out_no_verdict = bias(query_ctx, t_q, t_k)

        # 初始时 w_verdict ≈ 0，所以有无 verdict 差异应该很小
        diff = (out_with_verdict - out_no_verdict).abs().max()
        assert diff < 0.1  # 初始时近似相等


# ---------------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """测试向后兼容性"""

    def test_layer_old_api(self):
        """旧 API 调用方式不报错"""
        from memory_augmented_attention import MemoryBank, MemoryAugmentedTransformerLayer
        layer = MemoryAugmentedTransformerLayer(
            d_model=128, n_heads=4, top_k=16, init_alpha=1.0)
        bank = MemoryBank(max_size=64)

        x = torch.randn(2, 8, 128)
        out = layer(x, memory_bank=bank, current_step=0,
                    write_to_memory=True, causal=True)
        assert out.shape == (2, 8, 128)

    def test_layer_no_bank(self):
        """无 bank 时正常工作"""
        from memory_augmented_attention import MemoryAugmentedTransformerLayer
        layer = MemoryAugmentedTransformerLayer(d_model=128, n_heads=4)

        x = torch.randn(2, 8, 128)
        out = layer(x)
        assert out.shape == (2, 8, 128)

    def test_memory_bank_old_write(self):
        """MemoryBank 旧写入方式不报错"""
        bank = MemoryBank(max_size=10)
        result = bank.write(torch.randn(4, 32), torch.randn(
            4, 32), torch.randn(4, 32), timestamp=0)
        assert result is True

    def test_retrieve_all_still_4_elements(self):
        """retrieve_all() 仍返回 4 元素"""
        bank = MemoryBank(max_size=10)
        bank.write(torch.randn(4, 32), torch.randn(
            4, 32), torch.randn(4, 32), timestamp=0)
        result = bank.retrieve_all()
        assert len(result) == 4  # K, V, Q, T
