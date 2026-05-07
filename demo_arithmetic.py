"""Evaluate the checkpoint trained with `train_arithmetic.py`.

The training task bakes in **in-context learning with noisy lessons**:
each training sample is a small "classroom" — a few ``a+b=cY``/``a+b=cN``
demonstrations followed by a real query.  The trained model therefore
knows to (a) read demos, (b) ignore demos it can tell are wrong, and
(c) still compute the answer even when no demos are provided.

This script probes how well those three skills transfer to harder
out-of-distribution problems.

Conditions
----------
    Cold         : no demos at all
    Prefix       : 5 correct demos pasted in front of the query
    Prefix-noisy : 5 demos in front of the query, 2 of them wrong
    Bank         : 5 correct demos are pre-written into the MemoryBank
                   (separate warmup forward pass), query fed alone
    BankDynamic  : mixed demos (correct+wrong) written into MemoryBank
                   with verdict metadata, leveraging DynamicMemoryBias

Levels
------
    L1  single-digit       a+b   (a,b ∈ 0..9)
    L2  two-digit          ab+cd (a,b ∈ 10..99)        — in-training distribution
    L3  three-digit        abc+def                     — out of distribution
    L4  four-digit         abcd+efgh                   — far out of distribution
    L5  three-operand      a+b+c (0..9)                — syntax OOD
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Callable

import torch

from train_arithmetic import (
    ArithmeticMAA,
    CHECKPOINT_PATH,
    DEVICE,
    INV_VOCAB,
    MAX_SEQ_LEN,
    MemoryBank,
    PAD_CHAR,
    PAD_ID,
    SEP_DEMO,
    SEP_QUERY,
    VOCAB,
    encode,
    pad_to,
    _random_wrong_answer,
)


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------
@dataclass
class Level:
    name: str
    generate: Callable[[], tuple[str, str]]   # returns (query_prompt, answer)
    # for seeding examples (same distribution)
    demo_generate: Callable[[], tuple[str, str]]
    max_ans_len: int


def _rand(lo: int, hi: int) -> int:
    return random.randint(lo, hi)


def _q(a: int, b: int) -> tuple[str, str]:
    return f"{a}+{b}=", str(a + b)


def _q3(a: int, b: int, c: int) -> tuple[str, str]:
    return f"{a}+{b}+{c}=", str(a + b + c)


def gen_L1() -> tuple[str, str]:
    return _q(_rand(0, 9), _rand(0, 9))


def gen_L2() -> tuple[str, str]:
    return _q(_rand(10, 99), _rand(10, 99))


def gen_L3() -> tuple[str, str]:
    return _q(_rand(100, 999), _rand(100, 999))


def gen_L4() -> tuple[str, str]:
    return _q(_rand(1000, 9999), _rand(1000, 9999))


def gen_L5() -> tuple[str, str]:
    return _q3(_rand(0, 9), _rand(0, 9), _rand(0, 9))


LEVELS: list[Level] = [
    Level("L1 1-digit       a+b",      gen_L1, gen_L1, max_ans_len=2),
    Level("L2 2-digit     ab+cd",      gen_L2, gen_L2, max_ans_len=3),
    Level("L3 3-digit   abc+def",      gen_L3, gen_L3, max_ans_len=4),
    Level("L4 4-digit abcd+efgh",      gen_L4, gen_L4, max_ans_len=5),
    Level("L5 3-operand  a+b+c",       gen_L5, gen_L5, max_ans_len=2),
]


# ---------------------------------------------------------------------------
# Demo construction
# ---------------------------------------------------------------------------
def _build_demo_block(
    demo_gen: Callable[[], tuple[str, str]],
    n_correct: int,
    n_wrong: int,
) -> tuple[str, list[str], list[float]]:
    """Return a ``;``-joined string of ``a+b=cY``/``a+b=cN`` demonstrations.

    Returns
    -------
    demo_block : str
        Joined demo string.
    demo_items : list[str]
        Individual demo strings (before joining).
    verdicts : list[float]
        Per-demo verdict values (+1.0 for correct, -1.0 for wrong).
    """
    items = []
    verdicts = []
    for _ in range(n_correct):
        prompt, ans = demo_gen()
        items.append(f"{prompt}{ans}Y")
        verdicts.append(1.0)
    for _ in range(n_wrong):
        prompt, ans = demo_gen()
        wrong = _random_wrong_answer(int(ans))
        items.append(f"{prompt}{wrong}N")
        verdicts.append(-1.0)
    # Shuffle items and verdicts together
    combined = list(zip(items, verdicts))
    random.shuffle(combined)
    items, verdicts = zip(*combined) if combined else ([], [])
    items, verdicts = list(items), list(verdicts)
    return SEP_DEMO.join(items), items, verdicts


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_answer(
    model,
    full_prompt: str,
    *,
    max_new_tokens: int,
    memory_bank: MemoryBank | None = None,
    current_step: int = 0,
) -> str:
    """Greedy char-level decoding starting from ``full_prompt``.

    Stops at a space or when ``max_new_tokens`` chars have been emitted.

    NOTE: The input is right-padded to MAX_SEQ_LEN to match training conditions.
    During training, all sequences are padded to MAX_SEQ_LEN.  The
    DynamicMemoryBias computes ``query_context = x.mean(dim=1).mean(dim=0)``
    over the full input, so padding length affects the bias weights.  Padding
    at inference ensures the same distribution of query_context values that
    the model learned with.
    """
    generated = ""
    current = full_prompt
    for _ in range(max_new_tokens):
        length = min(len(current), MAX_SEQ_LEN)
        ids = encode(current[-length:])
        real_len = len(ids)
        # Pad to MAX_SEQ_LEN to match training (query_context depends on seq length)
        while len(ids) < MAX_SEQ_LEN:
            ids.append(PAD_ID)
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        logits = model(
            x, memory_bank=memory_bank,
            current_step=current_step, write_to_memory=False,
        )
        # Read prediction at the last real (non-pad) position
        next_id = logits[0, real_len - 1].argmax().item()
        next_char = INV_VOCAB[next_id]
        if next_char in (PAD_CHAR, SEP_DEMO, SEP_QUERY, "Y", "N"):
            break
        if not next_char.isdigit() and generated:
            break
        generated += next_char
        current += next_char
    return generated.strip()


@torch.no_grad()
def seed_bank_with_demos(
    model,
    bank: MemoryBank,
    demo_block: str,
    demos: list[str] | None = None,
    verdicts_list: list[float] | None = None,
) -> int:
    """Run one forward pass over the demo block to write its K/V into bank.

    Parameters
    ----------
    demos : list[str], optional
        Individual demo strings used to build token-level verdicts.
    verdicts_list : list[float], optional
        Per-demo verdict values (+1.0 or -1.0). When provided together
        with *demos*, a token-level verdict tensor is constructed and
        passed to the model so that DynamicMemoryBias can leverage
        credibility information.
    """
    if not demo_block:
        return 0
    # Append a trailing separator so the demo block is "self-contained".
    text = demo_block + SEP_QUERY
    length = min(len(text), MAX_SEQ_LEN)
    ids = encode(text[:length])
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    # Build token-level verdict tensor
    verdicts_tensor = None
    if verdicts_list is not None and demos is not None:
        token_verdicts: list[float] = []
        sep_demo_ids = encode(SEP_DEMO)
        for i, (demo, verdict) in enumerate(zip(demos, verdicts_list)):
            demo_ids = encode(demo)
            token_verdicts.extend([verdict] * len(demo_ids))
            # Add separator tokens (verdict=0) between demos
            if i < len(demos) - 1:
                token_verdicts.extend([0.0] * len(sep_demo_ids))
        # SEP_QUERY token(s) have verdict=0
        sep_query_ids = encode(SEP_QUERY)
        token_verdicts.extend([0.0] * len(sep_query_ids))
        # Truncate/pad to match sequence length
        token_verdicts = token_verdicts[:len(ids)]
        while len(token_verdicts) < len(ids):
            token_verdicts.append(0.0)
        verdicts_tensor = torch.tensor(token_verdicts, device=DEVICE)

    model(x, memory_bank=bank, current_step=0, write_to_memory=True,
          verdicts=verdicts_tensor)
    return length


# ---------------------------------------------------------------------------
# One evaluation run
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_condition(
    model,
    level: Level,
    *,
    condition: str,
    n_trials: int,
    n_correct: int,
    n_wrong: int,
    verbose: int = 0,
) -> float:
    correct = 0
    shown = 0
    for _ in range(n_trials):
        query_prompt, true_ans = level.generate()

        if condition == "cold":
            full_prompt = query_prompt
            bank = None
            start_step = 0
        else:
            demo_block, demo_items, verdicts = _build_demo_block(
                level.demo_generate, n_correct, n_wrong)
            if condition == "bank":
                bank = MemoryBank(max_size=2048)
                start_step = seed_bank_with_demos(model, bank, demo_block)
                full_prompt = query_prompt
            elif condition == "bank_dynamic":
                bank = MemoryBank(max_size=2048)
                start_step = seed_bank_with_demos(
                    model, bank, demo_block,
                    demos=demo_items, verdicts_list=verdicts,
                )
                full_prompt = query_prompt
            else:  # prefix / prefix_noisy
                bank = None
                start_step = 0
                full_prompt = demo_block + SEP_QUERY + \
                    query_prompt if demo_block else query_prompt

        pred = greedy_answer(
            model, full_prompt,
            max_new_tokens=level.max_ans_len,
            memory_bank=bank,
            current_step=start_step,
        )
        ok = pred == true_ans
        correct += ok
        if shown < verbose:
            tag = "OK " if ok else "FAIL"
            print(
                f"    [{tag}] {full_prompt[:60]!r:<62} -> {pred!r:<6} (expected {true_ans!r})")
            shown += 1

    return correct / max(1, n_trials)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CHECKPOINT_PATH)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--show", type=int, default=2,
                        help="per-cell verbose sample count")
    parser.add_argument("--demos", type=int, default=5,
                        help="number of demos for prefix/bank modes")
    parser.add_argument("--wrong", type=int, default=2,
                        help="number of wrong demos for prefix-noisy mode")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = ArithmeticMAA().to(DEVICE)
    checkpoint = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Device: {DEVICE}")
    print(
        f"Trials/cell: {args.trials}  Demos: {args.demos}  Noisy wrong: {args.wrong}\n")

    header = (
        f"{'Level':22s} | {'Cold':>8s} | {'Prefix':>8s} | "
        f"{'PrefixNoisy':>12s} | {'Bank':>8s} | {'BankDynamic':>12s}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for level in LEVELS:
        print(f"\n>>> {level.name}")
        print("  cold:")
        cold = evaluate_condition(
            model, level, condition="cold", n_trials=args.trials,
            n_correct=0, n_wrong=0, verbose=args.show,
        )
        print(f"  prefix (all correct, n={args.demos}):")
        pref = evaluate_condition(
            model, level, condition="prefix", n_trials=args.trials,
            n_correct=args.demos, n_wrong=0, verbose=args.show,
        )
        print(
            f"  prefix-noisy ({args.demos - args.wrong} correct, {args.wrong} wrong):")
        pref_n = evaluate_condition(
            model, level, condition="prefix", n_trials=args.trials,
            n_correct=max(0, args.demos - args.wrong), n_wrong=args.wrong,
            verbose=args.show,
        )
        print(f"  bank (demos pre-written to MemoryBank, n={args.demos}):")
        bank_score = evaluate_condition(
            model, level, condition="bank", n_trials=args.trials,
            n_correct=args.demos, n_wrong=0, verbose=args.show,
        )
        print(
            f"  bank-dynamic (mixed demos with verdict metadata, "
            f"{args.demos - args.wrong} correct + {args.wrong} wrong):")
        bank_dyn = evaluate_condition(
            model, level, condition="bank_dynamic", n_trials=args.trials,
            n_correct=max(0, args.demos - args.wrong), n_wrong=args.wrong,
            verbose=args.show,
        )
        rows.append((level.name, cold, pref, pref_n, bank_score, bank_dyn))

    print("\n\nSummary")
    print("=======")
    print(header)
    print("-" * len(header))
    for name, cold, pref, pref_n, bank_score, bank_dyn in rows:
        print(
            f"{name:22s} | {cold*100:7.1f}% | {pref*100:7.1f}% | "
            f"{pref_n*100:11.1f}% | {bank_score*100:7.1f}% | {bank_dyn*100:11.1f}%"
        )

    print(
        "\nLegend:\n"
        "  Cold        — no demos.  Weight-only knowledge.\n"
        "  Prefix      — 5 correct demos in the prompt (in-context learning).\n"
        "  PrefixNoisy — 3 correct + 2 wrong demos.  Tests whether the model\n"
        "                reads the Y/N verdicts instead of blindly copying.\n"
        "  Bank        — the demos are first digested into the MemoryBank;\n"
        "                the query is then asked alone.  Stresses MAA's\n"
        "                unified-memory story.\n"
        "  BankDynamic — mixed demos (correct+wrong) written into MemoryBank\n"
        "                with verdict metadata.  Tests DynamicMemoryBias's\n"
        "                ability to up-weight correct memories.\n"
    )


if __name__ == "__main__":
    main()
