import argparse
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL_ID = "google/gemma-3-4b-it"

# Paper-aligned default slice:
# - model: gemma-3-4b-it
# - topology: 4x4 grid
# - context length: 200
# The paper evaluates 1000 word assignments; this script defaults to fewer
# assignments for practical runtime on CPU-only machines.
GRID_SIZE = 4
WALK_LEN = 200
DEFAULT_NUM_TRIALS = 20
PAPER_NUM_TRIALS = 1000
DEFAULT_SHOW_EXAMPLES = 5

WORDS = [
    "toy",
    "ink",
    "city",
    "air",
    "wing",
    "cat",
    "jam",
    "zoo",
    "baby",
    "rock",
    "leaf",
    "ship",
    "lamp",
    "fork",
    "star",
    "bell",
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
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the delayed next-token prediction effect from "
            "'Language Models Struggle to Use Representations Learned In-Context'."
        )
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=GRID_SIZE,
        help="Grid size. This script currently expects len(vocab) == grid_size^2.",
    )
    parser.add_argument(
        "--walk-len",
        type=int,
        default=WALK_LEN,
        help="Length of the random walk.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help=(
            "Number of word assignments to evaluate. "
            f"The paper uses {PAPER_NUM_TRIALS}, but the default here is lower for runtime."
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="First seed to evaluate.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=DEFAULT_SHOW_EXAMPLES,
        help="How many trial rows to print in full.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-trial example rows and print only summary statistics.",
    )
    return parser.parse_args()


def make_grid(
    words: Sequence[str], n: int, rng: random.Random
) -> Tuple[Dict[Tuple[int, int], str], Dict[str, Tuple[int, int]]]:
    shuffled = list(words)
    rng.shuffle(shuffled)

    pos2word: Dict[Tuple[int, int], str] = {}
    word2pos: Dict[str, Tuple[int, int]] = {}
    k = 0
    for i in range(n):
        for j in range(n):
            word = shuffled[k]
            pos2word[(i, j)] = word
            word2pos[word] = (i, j)
            k += 1
    return pos2word, word2pos


def neighbors(i: int, j: int, n: int) -> List[Tuple[int, int]]:
    out = []
    if i > 0:
        out.append((i - 1, j))
    if i < n - 1:
        out.append((i + 1, j))
    if j > 0:
        out.append((i, j - 1))
    if j < n - 1:
        out.append((i, j + 1))
    return out


def random_walk(
    pos2word: Dict[Tuple[int, int], str], n: int, steps: int, rng: random.Random
) -> List[str]:
    pos = rng.choice(list(pos2word.keys()))
    walk = [pos2word[pos]]
    for _ in range(steps - 1):
        pos = rng.choice(neighbors(pos[0], pos[1], n))
        walk.append(pos2word[pos])
    return walk


def valid_next_tokens(
    last_word: str,
    word2pos: Dict[str, Tuple[int, int]],
    pos2word: Dict[Tuple[int, int], str],
    n: int,
) -> List[str]:
    i, j = word2pos[last_word]
    return [pos2word[p] for p in neighbors(i, j, n)]


def build_instruction_prompt(walk: Sequence[str]) -> str:
    return (
        "Your job is to predict the next word in a sequence of words. "
        "Generate the token [ANSWER], then generate the next word in the sequence.\n"
        f"[SEQUENCE] {' '.join(walk)}"
    )


def build_prefilled_prefix(walk: Sequence[str]) -> str:
    return f"[SEQUENCE] {' '.join(walk)}"


def parse_prefixed_answer(text: str, vocab: Sequence[str]) -> str | None:
    match = ANSWER_RE.match(text)
    if match is None:
        return None

    word = match.group("word").lower()
    if word not in WORD_SET:
        return None
    return word


def build_generation_prompt(processor, user_text: str, prefill: str | None) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if prefill is not None:
        prompt += prefill
    return prompt


def run_generation(
    model,
    processor,
    user_text: str,
    *,
    prefill: str | None = None,
    max_new_tokens: int,
) -> str:
    prompt = build_generation_prompt(processor, user_text, prefill)
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(gen_ids, skip_special_tokens=True)


def validate_setup(processor, words: Sequence[str], grid_size: int) -> None:
    if len(words) != grid_size * grid_size:
        raise ValueError(
            f"Expected {grid_size * grid_size} words for a {grid_size}x{grid_size} grid, "
            f"but found {len(words)}."
        )

    bad_words = []
    tokenizer = processor.tokenizer
    for word in words:
        piece_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(piece_ids) != 1:
            bad_words.append((word, piece_ids))

    if bad_words:
        detail = ", ".join(f"{word}={piece_ids}" for word, piece_ids in bad_words)
        raise ValueError(f"Expected one-token words, but found multi-token entries: {detail}")


def evaluate_trial(
    model,
    processor,
    *,
    seed: int,
    words: Sequence[str],
    grid_size: int,
    walk_len: int,
) -> TrialResult:
    rng = random.Random(seed)
    pos2word, word2pos = make_grid(words, grid_size, rng)
    walk = random_walk(pos2word, grid_size, walk_len, rng)

    last_word = walk[-1]
    gold_valid = valid_next_tokens(last_word, word2pos, pos2word, grid_size)

    instruction_raw = run_generation(
        model,
        processor,
        build_instruction_prompt(walk),
        max_new_tokens=12,
    )
    instruction_guess = parse_prefixed_answer(instruction_raw, words)

    prefilled_raw = run_generation(
        model,
        processor,
        "Continue the sequence of words.",
        prefill=build_prefilled_prefix(walk),
        max_new_tokens=4,
    )
    prefilled_guess = parse_prefixed_answer(prefilled_raw, words)

    return TrialResult(
        seed=seed,
        last_word=last_word,
        valid_next_tokens=gold_valid,
        instruction_raw=instruction_raw,
        instruction_guess=instruction_guess,
        prefilled_raw=prefilled_raw,
        prefilled_guess=prefilled_guess,
    )


def summarize_results(results: Sequence[TrialResult], vocab_size: int) -> Dict[str, float]:
    n = len(results)
    instruction_acc = sum(r.instruction_ok for r in results) / n
    prefilled_acc = sum(r.prefilled_ok for r in results) / n
    instruction_parse_rate = sum(r.instruction_parsed for r in results) / n
    prefilled_parse_rate = sum(r.prefilled_parsed for r in results) / n
    chance_acc = sum(len(r.valid_next_tokens) / vocab_size for r in results) / n
    avg_neighbors = sum(len(r.valid_next_tokens) for r in results) / n

    return {
        "num_trials": float(n),
        "instruction_acc": instruction_acc,
        "prefilled_acc": prefilled_acc,
        "instruction_parse_rate": instruction_parse_rate,
        "prefilled_parse_rate": prefilled_parse_rate,
        "uniform_vocab_chance_acc": chance_acc,
        "avg_num_valid_next_tokens": avg_neighbors,
    }


def print_examples(results: Sequence[TrialResult], limit: int) -> None:
    print("=== Example trials ===")
    for result in results[:limit]:
        print(
            {
                "seed": result.seed,
                "last_word": result.last_word,
                "valid_next_tokens": result.valid_next_tokens,
                "instruction_guess": result.instruction_guess,
                "instruction_ok": result.instruction_ok,
                "instruction_raw": result.instruction_raw,
                "prefilled_guess": result.prefilled_guess,
                "prefilled_ok": result.prefilled_ok,
                "prefilled_raw": result.prefilled_raw,
            }
        )
    print()


def print_failures(results: Sequence[TrialResult], limit: int) -> None:
    instruction_failures = [r for r in results if not r.instruction_ok]
    prefilled_failures = [r for r in results if not r.prefilled_ok]

    print("=== Instruction failures ===")
    for result in instruction_failures[:limit]:
        print(
            {
                "seed": result.seed,
                "last_word": result.last_word,
                "valid_next_tokens": result.valid_next_tokens,
                "instruction_guess": result.instruction_guess,
                "instruction_raw": result.instruction_raw,
            }
        )
    print()

    print("=== Prefilled failures ===")
    for result in prefilled_failures[:limit]:
        print(
            {
                "seed": result.seed,
                "last_word": result.last_word,
                "valid_next_tokens": result.valid_next_tokens,
                "prefilled_guess": result.prefilled_guess,
                "prefilled_raw": result.prefilled_raw,
            }
        )
    print()


def print_summary(args: argparse.Namespace, summary: Dict[str, float]) -> None:
    print("=== Configuration ===")
    print(
        {
            "model_id": args.model_id,
            "grid_size": args.grid_size,
            "walk_len": args.walk_len,
            "num_trials": args.num_trials,
            "paper_num_trials": PAPER_NUM_TRIALS,
            "base_seed": args.base_seed,
        }
    )
    print()

    print("=== Summary ===")
    print(summary)
    print()


def load_model_and_processor(model_id: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    ).eval()
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    return model, processor


def main() -> None:
    args = parse_args()
    model, processor = load_model_and_processor(args.model_id)
    validate_setup(processor, WORDS, args.grid_size)

    results = []
    for seed in range(args.base_seed, args.base_seed + args.num_trials):
        results.append(
            evaluate_trial(
                model,
                processor,
                seed=seed,
                words=WORDS,
                grid_size=args.grid_size,
                walk_len=args.walk_len,
            )
        )

    summary = summarize_results(results, vocab_size=len(WORDS))
    print_summary(args, summary)

    if not args.quiet:
        print_examples(results, args.show_examples)
        print_failures(results, args.show_examples)


if __name__ == "__main__":
    main()
