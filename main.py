import argparse
import random
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL_ID = "google/gemma-3-4b-it"
GRID_SIZE = 4
WALK_LEN = 200
DEFAULT_NUM_TRIALS = 20
PAPER_NUM_TRIALS = 1000
DEFAULT_SHOW_EXAMPLES = 5

WORDS = [
    "toy", "ink", "city", "air",
    "wing", "cat", "jam", "zoo",
    "baby", "rock", "leaf", "ship",
    "lamp", "fork", "star", "bell",
]
WORD_SET = set(WORDS)

ANSWER_RE = re.compile(r"^\s*(?:\[ANSWER\]\s*)?(?P<word>[A-Za-z]+)\b")


@dataclass
class TrialResult:
    seed: int
    last_word: str
    valid_next_tokens: List[str]
    instruction_raw: str
    instruction_guess: str | None
    prefilled_raw: str
    prefilled_guess: str | None

    @property
    def instruction_ok(self) -> bool:
        return self.instruction_guess in self.valid_next_tokens

    @property
    def prefilled_ok(self) -> bool:
        return self.prefilled_guess in self.valid_next_tokens

    @property
    def instruction_parsed(self) -> bool:
        return self.instruction_guess is not None

    @property
    def prefilled_parsed(self) -> bool:
        return self.prefilled_guess is not None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the delayed next-token prediction effect from "
        "'Language Models Struggle to Use Representations Learned In-Context'."
    )
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--grid-size", type=int, default=GRID_SIZE,
                   help="Grid size (expects len(vocab) == grid_size^2).")
    p.add_argument("--walk-len", type=int, default=WALK_LEN)
    p.add_argument("--num-trials", type=int, default=DEFAULT_NUM_TRIALS,
                   help=f"Word assignments to evaluate (paper uses {PAPER_NUM_TRIALS}).")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--show-examples", type=int, default=DEFAULT_SHOW_EXAMPLES)
    p.add_argument("--quiet", action="store_true",
                   help="Print only summary statistics.")
    return p.parse_args()


def neighbors(i: int, j: int, n: int) -> List[Tuple[int, int]]:
    out = []
    if i > 0: out.append((i - 1, j))
    if i < n - 1: out.append((i + 1, j))
    if j > 0: out.append((i, j - 1))
    if j < n - 1: out.append((i, j + 1))
    return out


def parse_answer(text: str) -> str | None:
    m = ANSWER_RE.match(text)
    if m is None:
        return None
    word = m.group("word").lower()
    return word if word in WORD_SET else None


def run_generation(
    model, processor, user_text: str, *, prefill: str | None = None, max_new_tokens: int
) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prefill is not None:
        prompt += prefill
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(gen_ids, skip_special_tokens=True)


def evaluate_trial(
    model, processor, *, seed: int, words: Sequence[str], grid_size: int, walk_len: int
) -> TrialResult:
    rng = random.Random(seed)
    shuffled = list(words)
    rng.shuffle(shuffled)
    pos2word = {(i, j): shuffled[i * grid_size + j] for i in range(grid_size) for j in range(grid_size)}
    word2pos = {w: p for p, w in pos2word.items()}

    pos = rng.choice(list(pos2word))
    walk = [pos2word[pos]]
    for _ in range(walk_len - 1):
        pos = rng.choice(neighbors(pos[0], pos[1], grid_size))
        walk.append(pos2word[pos])

    last_word = walk[-1]
    gold_valid = [pos2word[p] for p in neighbors(*word2pos[last_word], grid_size)]
    seq = " ".join(walk)

    instruction_raw = run_generation(
        model, processor,
        "Your job is to predict the next word in a sequence of words. "
        f"Generate the token [ANSWER], then generate the next word in the sequence.\n[SEQUENCE] {seq}",
        max_new_tokens=12,
    )
    prefilled_raw = run_generation(
        model, processor, "Continue the sequence of words.",
        prefill=f"[SEQUENCE] {seq}", max_new_tokens=4,
    )

    return TrialResult(
        seed=seed,
        last_word=last_word,
        valid_next_tokens=gold_valid,
        instruction_raw=instruction_raw,
        instruction_guess=parse_answer(instruction_raw),
        prefilled_raw=prefilled_raw,
        prefilled_guess=parse_answer(prefilled_raw),
    )


def main() -> None:
    args = parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto"
    ).eval()
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)

    if len(WORDS) != args.grid_size ** 2:
        raise ValueError(f"Expected {args.grid_size ** 2} words for {args.grid_size}x{args.grid_size} grid.")
    tokenizer = processor.tokenizer
    bad = [(w, ids) for w in WORDS if len(ids := tokenizer.encode(w, add_special_tokens=False)) != 1]
    if bad:
        raise ValueError(f"Multi-token words: {', '.join(f'{w}={ids}' for w, ids in bad)}")

    results = [
        evaluate_trial(model, processor, seed=s, words=WORDS, grid_size=args.grid_size, walk_len=args.walk_len)
        for s in range(args.base_seed, args.base_seed + args.num_trials)
    ]

    n, vocab_size = len(results), len(WORDS)
    summary = {
        "num_trials": n,
        "instruction_acc": sum(r.instruction_ok for r in results) / n,
        "prefilled_acc": sum(r.prefilled_ok for r in results) / n,
        "instruction_parse_rate": sum(r.instruction_parsed for r in results) / n,
        "prefilled_parse_rate": sum(r.prefilled_parsed for r in results) / n,
        "uniform_vocab_chance_acc": sum(len(r.valid_next_tokens) / vocab_size for r in results) / n,
        "avg_num_valid_next_tokens": sum(len(r.valid_next_tokens) for r in results) / n,
    }

    print("=== Configuration ===")
    print({"model_id": args.model_id, "grid_size": args.grid_size, "walk_len": args.walk_len,
           "num_trials": args.num_trials, "paper_num_trials": PAPER_NUM_TRIALS, "base_seed": args.base_seed})
    print()
    print("=== Summary ===")
    print(summary)
    print()

    if not args.quiet:
        print("=== Example trials ===")
        for r in results[:args.show_examples]:
            print({"seed": r.seed, "last_word": r.last_word, "valid_next_tokens": r.valid_next_tokens,
                   "instruction_guess": r.instruction_guess, "instruction_ok": r.instruction_ok,
                   "instruction_raw": r.instruction_raw, "prefilled_guess": r.prefilled_guess,
                   "prefilled_ok": r.prefilled_ok, "prefilled_raw": r.prefilled_raw})
        print()

        print("=== Instruction failures ===")
        for r in [r for r in results if not r.instruction_ok][:args.show_examples]:
            print({"seed": r.seed, "last_word": r.last_word, "valid_next_tokens": r.valid_next_tokens,
                   "instruction_guess": r.instruction_guess, "instruction_raw": r.instruction_raw})
        print()

        print("=== Prefilled failures ===")
        for r in [r for r in results if not r.prefilled_ok][:args.show_examples]:
            print({"seed": r.seed, "last_word": r.last_word, "valid_next_tokens": r.valid_next_tokens,
                   "prefilled_guess": r.prefilled_guess, "prefilled_raw": r.prefilled_raw})
        print()


if __name__ == "__main__":
    main()