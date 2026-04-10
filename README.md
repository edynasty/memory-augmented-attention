# Memory-Augmented Attention

A PyTorch implementation of **Unified Memory-Augmented Attention**, where
current-sequence tokens (short-term memory) and historical KV pairs
(long-term memory) are stored in one shared, timestamped **Memory Bank** and
attended over jointly with a **temporal decay bias**.

---

## Motivation

Traditional Memory-Augmented Transformers keep the current sequence and the
external memory separate:

```
Attention(Q, [K_seq ; K_memory], [V_seq ; V_memory])
```

This library unifies both under one abstraction:

```
M_all = M_memory ∪ { (K_seq, V_seq, Q_seq, t_seq) }
y_t   = Attention(Q_t, K_all, V_all, TimeBias(t_t, t_all))
```

where the attention weight for the pair `(t_query, t_key)` is:

```
softmax( Q K^T / sqrt(d_k)  +  TimeBias(t_query, t_key) )
TimeBias(t_q, t_k) = -alpha * log(1 + |t_q - t_k|)
```

**Benefits**

| Property | Detail |
|---|---|
| Unified representation | No distinction between "current sequence" and "history" |
| Automatic temporal decay | Older entries receive a smaller (more-negative) bias |
| Unbounded context | The bank grows with every new token; old entries are evicted by LRU or compressed on demand |
| Scalable retrieval | Top-K dot-product retrieval keeps the attention pool manageable |

---

## Installation

```bash
pip install -e ".[dev]"   # editable install with test extras
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

---

## Quick Start

```python
import torch
from memory_augmented_attention import MemoryBank, MemoryAugmentedTransformerLayer

# Create a single transformer layer and a persistent memory bank.
layer = MemoryAugmentedTransformerLayer(
    d_model=256,
    n_heads=8,
    d_ff=1024,
    dropout=0.1,
    top_k=64,        # attend to at most 64 historical entries per forward pass
    init_alpha=1.0,  # initial temporal-decay strength
)
bank = MemoryBank(max_size=4096, importance_threshold=0.0)

global_step = 0
for batch in data_loader:
    x = batch["embeddings"]   # (B, T, 256)

    out = layer(
        x,
        memory_bank=bank,
        current_step=global_step,
        write_to_memory=True,  # write current seq into the bank after attention
    )

    global_step += x.shape[1]   # advance timestamp by sequence length
```

---

## API Reference

### `MemoryBank`

```python
MemoryBank(max_size=1024, importance_threshold=0.0)
```

| Method | Description |
|---|---|
| `write(key, value, query, timestamp)` | Write one entry; returns `False` if rejected by the importance filter |
| `write_sequence(keys, values, queries, start_timestamp)` | Write a full sequence; returns next available timestamp |
| `retrieve_all(device=None)` | Return all entries as stacked tensors `(keys, values, queries, timestamps)` |
| `top_k_retrieve(query, k, device=None)` | Return the `k` most relevant entries by cosine similarity |
| `compress_by_time(keep_newest)` | Discard old entries, keeping only the `keep_newest` most recent |
| `clear()` | Remove all entries |

### `TimeBias`

```python
TimeBias(init_alpha=1.0)
```

Learnable temporal-decay module.  The single learnable scalar `alpha`
(initialised to `init_alpha`) controls how strongly the model discounts old
memories.  `alpha = 0` recovers standard (time-agnostic) attention.

### `UnifiedMemoryAttention`

```python
UnifiedMemoryAttention(
    d_model, n_heads,
    dropout=0.0,
    top_k=None,       # None → use all stored entries
    init_alpha=1.0,
)
```

Multi-head attention over the unified memory pool.  When `write_to_memory=True`
(the default), the K/V/Q projections of the current sequence are written into
the supplied `MemoryBank` after each forward pass.

### `MemoryAugmentedTransformerLayer`

```python
MemoryAugmentedTransformerLayer(
    d_model, n_heads,
    d_ff=None,         # defaults to 4 * d_model
    dropout=0.1,
    top_k=None,
    init_alpha=1.0,
)
```

Complete encoder layer: `UnifiedMemoryAttention → Add & Norm → FFN → Add & Norm`.

---

## Design Details

### Unified Memory Formula

```
M_all = M_memory ∪ { (K_seq, V_seq, Q_seq, t_seq) }

y_t = Attention(Q_t, K_all, V_all, TimeBias(t_t, t_all))
    = softmax( (Q_t K_all^T) / sqrt(d_k) + TimeBias(t_t, t_all) ) · V_all
```

### Temporal Bias

```
TimeBias(t_q, t_k) = -alpha * log(1 + |t_q - t_k|)
```

The logarithmic form matches the power-law forgetting curve observed in human
memory: relatively recent entries are penalised gently while very distant
entries are penalised more, but the gap shrinks for very large lags.

### Memory Management

* **Eviction**: when the bank is full, the oldest entry is evicted (LRU).
* **Importance filter**: entries whose value L2-norm is below
  `importance_threshold` are discarded before writing.
* **Compression**: call `bank.compress_by_time(keep_newest)` to aggressively
  trim the bank between inference steps.
* **Top-K retrieval**: pass `top_k=N` to `UnifiedMemoryAttention` /
  `MemoryAugmentedTransformerLayer` to limit the attention pool size.

---

## Running Tests

```bash
pytest tests/ -v
```