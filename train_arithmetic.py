"""Train an arithmetic model with Memory-Augmented Attention.

This training setup is closer to how humans actually learn:

* The training distribution mixes 1- and 2-digit addition.
* Each training sample is a small **in-context lesson**: a variable-length
  prefix of ``a+b=cY`` / ``a+b=cN`` demonstrations, a separator ``|``, and
  a single query ``a+b=``.  The loss is computed only on the query's answer.
* Some of the demonstrations are *wrong* (marked with ``N``).  The model
  therefore must learn to trust the operation itself rather than blindly
  copy whatever number appears near ``=``.

At test time the demonstrations can be supplied two ways:
    1. **Prefix mode**: paste them at the start of the prompt (what the
       model saw during training).
    2. **Bank mode**: run a "warmup" forward pass on the demonstrations
       with ``write_to_memory=True`` so that their K/V projections enter
       the MemoryBank; then query with a short prompt.  Because MAA's
       attention is unified over bank + sequence, the model reaches the
       demonstrations either way.

No framework modifications are required: the verdict tokens ``Y``/``N``
are just regular characters whose K/V get stored in the bank alongside
everything else.
"""

from __future__ import annotations

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_augmented_attention import (
    MemoryBank, MemoryAugmentedTransformerLayer,
    ConsolidationNetwork, MemoryType,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D_MODEL = 192
N_HEADS = 6
D_FF = 512
DROPOUT = 0.1
TOP_K = 64
INIT_ALPHA = 1.0
N_LAYERS = 4

# Max sequence length seen during training (= few-shot prefix + query).
# Positional embedding capacity matches this so the same model can also
# be queried with even longer prompts at test time if needed.
MAX_SEQ_LEN = 96

# Number range for the two operands during training.
MAX_OPERAND = 99          # mix of 1- and 2-digit
MAX_DEMOS = 6             # up to this many in-context demonstrations
DEMO_WRONG_PROB = 0.3     # probability each demo is labelled wrong

BATCH_SIZE = 64
NUM_EPOCHS = 40
LR = 3e-4
NUM_TRAIN = 40000
NUM_EVAL = 2000

SEED = 42
CHECKPOINT_PATH = "arithmetic_maa_best.pt"


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick_device()

# ---------------------------------------------------------------------------
# Vocabulary — character-level.  'Y'/'N' encode the verdict of a demo;
# ';' separates demos; '|' separates the demonstration block from the query.
# ---------------------------------------------------------------------------
CHARS = "0123456789+-= ;|YN"
VOCAB = {c: i for i, c in enumerate(CHARS)}
INV_VOCAB = {i: c for c, i in VOCAB.items()}
VOCAB_SIZE = len(CHARS)
PAD_CHAR = " "
PAD_ID = VOCAB[PAD_CHAR]
SEP_DEMO = ";"
SEP_QUERY = "|"


def encode(text: str) -> list[int]:
    return [VOCAB[c] for c in text]


def decode(ids: list[int]) -> str:
    return "".join(INV_VOCAB[i] for i in ids)


def pad_to(text: str, length: int) -> str:
    return text.ljust(length)[:length]


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------
def _random_wrong_answer(true_ans: int) -> int:
    """Produce a plausible *wrong* answer close to the true one."""
    while True:
        delta = random.choice([-10, -5, -2, -1, 1, 2, 5, 10])
        cand = true_ans + delta
        if cand != true_ans and cand >= 0:
            return cand


def _random_demo() -> tuple[str, float]:
    """Return a single demonstration string 'a+b=cY' or 'a+b=cN' and its verdict score."""
    a = random.randint(0, MAX_OPERAND)
    b = random.randint(0, MAX_OPERAND)
    true_ans = a + b
    if random.random() < DEMO_WRONG_PROB:
        ans = _random_wrong_answer(true_ans)
        verdict = "N"
        verdict_score = -1.0
    else:
        ans = true_ans
        verdict = "Y"
        verdict_score = 1.0
    return f"{a}+{b}={ans}{verdict}", verdict_score


def generate_sample() -> tuple[str, int, int, list[float]]:
    """Generate one training sample.

    Returns
    -------
    text : str
        Full sequence: ``<demos>|<query>=<answer>`` (padded later).
    ans_start, ans_end : int
        Character offsets (inclusive-exclusive) of the query's answer digits
        in *text*, so the loss mask can be set correctly.
    demo_verdicts : list[float]
        Verdict score for each demo (+1.0 = correct, -1.0 = wrong).
        Used for MemoryBank metadata during evaluation (not during training).
    """
    n_demos = random.randint(0, MAX_DEMOS)
    demo_results = [_random_demo() for _ in range(n_demos)]
    demo_strings = [d[0] for d in demo_results]
    demo_verdicts = [d[1] for d in demo_results]
    demo_block = SEP_DEMO.join(demo_strings)

    # Query uses the same distribution as demos (so the model sees both 1-
    # and 2-digit addition as the thing-to-predict).
    a = random.randint(0, MAX_OPERAND)
    b = random.randint(0, MAX_OPERAND)
    true_ans = a + b
    query_prompt = f"{a}+{b}="
    answer_str = str(true_ans)

    if demo_block:
        prefix = demo_block + SEP_QUERY + query_prompt
    else:
        prefix = query_prompt

    text = prefix + answer_str
    ans_start = len(prefix)
    ans_end = len(text)
    return text, ans_start, ans_end, demo_verdicts


def build_dataset(n: int, seq_len: int = MAX_SEQ_LEN):
    """Build (inputs, targets, answer_mask) tensors on DEVICE.

    Input is ``tokens[:-1]`` and target is ``tokens[1:]`` (next-token).
    The mask is 1.0 at positions whose *target* is a query-answer digit,
    otherwise 0.0.  Pad positions are also 0.0.
    """
    inputs, targets, masks = [], [], []
    kept = 0
    attempts = 0
    while kept < n:
        attempts += 1
        text, ans_start, ans_end, _demo_verdicts = generate_sample()
        # +1 PAD so the last answer digit has a valid target.
        if len(text) + 1 > seq_len:
            continue
        padded = pad_to(text, seq_len)
        ids = encode(padded)

        target = ids[1:]
        mask = [0.0] * len(target)
        # Answer positions in *target* are shifted by -1 vs. *text*.
        # Extend supervision one step past the last answer digit so the
        # model learns to emit a PAD token when the answer is complete.
        # Without this, greedy decoding never knows when to stop.
        t_ans_start = ans_start - 1
        t_ans_end = ans_end - 1
        for i in range(t_ans_start, t_ans_end + 1):
            if 0 <= i < len(mask):
                mask[i] = 1.0

        inputs.append(ids[:-1])
        targets.append(target)
        masks.append(mask)
        kept += 1

    return (
        torch.tensor(inputs, dtype=torch.long, device=DEVICE),
        torch.tensor(targets, dtype=torch.long, device=DEVICE),
        torch.tensor(masks, dtype=torch.float, device=DEVICE),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ArithmeticMAA(nn.Module):
    """Character-level causal transformer with memory-augmented attention."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
        top_k: int = TOP_K,
        init_alpha: float = INIT_ALPHA,
        n_layers: int = N_LAYERS,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            MemoryAugmentedTransformerLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, top_k=top_k, init_alpha=init_alpha,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, x, memory_bank=None, current_step=0, write_to_memory=False, verdicts=None):
        B, T = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Input length {T} exceeds MAX_SEQ_LEN={self.max_seq_len}."
            )
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for layer in self.layers:
            h = layer(
                h, memory_bank=memory_bank,
                current_step=current_step, write_to_memory=write_to_memory,
                causal=True,
                verdicts=verdicts,
            )
        return self.head(self.norm(h))


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, optimizer, train_x, train_y, train_m,
                    all_params=None):
    model.train()
    total_loss, n_batches = 0.0, 0
    clip_params = all_params if all_params is not None else model.parameters()

    indices = torch.randperm(train_x.size(0), device=DEVICE)
    for start in range(0, train_x.size(0), BATCH_SIZE):
        idx = indices[start:start + BATCH_SIZE]
        bx, by, bm = train_x[idx], train_y[idx], train_m[idx]

        optimizer.zero_grad()
        logits = model(bx, memory_bank=None, write_to_memory=False)

        per_token_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), by.reshape(-1), reduction="none",
        )
        mask = bm.reshape(-1)
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, eval_x, eval_y, eval_m):
    model.eval()
    total_loss_sum, correct, total = 0.0, 0, 0

    for start in range(0, eval_x.size(0), BATCH_SIZE):
        bx = eval_x[start:start + BATCH_SIZE]
        by = eval_y[start:start + BATCH_SIZE]
        bm = eval_m[start:start + BATCH_SIZE]

        logits = model(bx, memory_bank=None, write_to_memory=False)
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), by.reshape(-1), reduction="none",
        )
        mask = bm.reshape(-1)
        total_loss_sum += (per_token_loss * mask).sum().item()
        total += mask.sum().item()

        preds = logits.argmax(dim=-1)
        correct += ((preds == by).float() * bm).sum().item()

    return total_loss_sum / max(1, total), correct / max(1, total)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print(
        f"Task: mixed-width addition (operands 0..{MAX_OPERAND}) with "
        f"in-context demos (0..{MAX_DEMOS} per sample, ~{int(DEMO_WRONG_PROB*100)}% wrong)."
    )
    print(f"Building {NUM_TRAIN} train / {NUM_EVAL} eval samples…")

    train_x, train_y, train_m = build_dataset(NUM_TRAIN)
    eval_x, eval_y, eval_m = build_dataset(NUM_EVAL)

    model = ArithmeticMAA().to(DEVICE)

    # 可学习归纳网络（用于推理时压缩记忆库）
    consolidation_net = ConsolidationNetwork(
        d_model=D_MODEL, n_heads=N_HEADS, compression_ratio=4,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_consol = sum(p.numel()
                          for p in consolidation_net.parameters() if p.requires_grad)
    print(
        f"Model parameters: {n_params:,} (+{n_params_consol:,} consolidation)")
    print(f"Vocab: {CHARS!r}  (size {VOCAB_SIZE})")
    print(f"MAX_SEQ_LEN: {MAX_SEQ_LEN}")

    all_params = list(model.parameters()) + \
        list(consolidation_net.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS)

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, optimizer, train_x, train_y, train_m,
            all_params=all_params)
        eval_loss, eval_acc = evaluate(model, eval_x, eval_y, eval_m)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}  "
            f"eval_acc={eval_acc:.4f}  lr={lr_now:.2e}"
        )
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'consolidation_net_state_dict': consolidation_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
            }, CHECKPOINT_PATH)

    print(f"\nBest eval accuracy: {best_acc:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    # === 训练后：演示可学习归纳 ===
    print("\n--- Consolidation Demo ---")
    demo_bank = MemoryBank(max_size=256)
    # 用模型生成一些 KV 写入 bank
    dummy_input = torch.randint(0, VOCAB_SIZE, (1, 16), device=DEVICE)
    with torch.no_grad():
        for step in range(8):
            _ = model(dummy_input, memory_bank=demo_bank,
                      current_step=step, write_to_memory=True)
    print(f"Before consolidation: {len(demo_bank)} entries")
    consolidation_net.eval()
    with torch.no_grad():
        demo_bank.consolidate(consolidation_net, keep_newest=32)
    print(f"After consolidation: {len(demo_bank)} entries")


def load_checkpoint(model: ArithmeticMAA, path: str = CHECKPOINT_PATH,
                    consolidation_net: ConsolidationNetwork | None = None,
                    optimizer: torch.optim.Optimizer | None = None,
                    device: torch.device = DEVICE) -> dict:
    """Load a checkpoint with backward compatibility.

    Supports both old format (plain state_dict) and new dict format.
    """
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        model.load_state_dict(raw['model_state_dict'])
        if consolidation_net is not None and 'consolidation_net_state_dict' in raw:
            consolidation_net.load_state_dict(
                raw['consolidation_net_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in raw:
            optimizer.load_state_dict(raw['optimizer_state_dict'])
        return raw
    else:
        # Old format: raw is model.state_dict() directly
        model.load_state_dict(raw)
        return {'model_state_dict': raw, 'epoch': 0, 'best_acc': 0.0}


if __name__ == "__main__":
    main()
