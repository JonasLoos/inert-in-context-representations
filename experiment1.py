import re
from typing import Callable, Any
from dataclasses import dataclass
import random
from collections import defaultdict

from utils import (
    CmdLineArgs,
    call_model,
    experiment,
    Grid,
    print_results,
    save_results,
)

# Task-specific patterns (group name "word" required for parse_answer).
RE_ANSWER = r"\[ANSWER\]\s*(?P<word>[A-Za-z]+)"
# First alphabetic token of the model reply (single-word answers, no tag).
RE_FIRST_TOKEN = r"^\s*(?P<word>[A-Za-z]+)"
# Continuation after prefill / "continue": optional leading [ANSWER], then the word.
RE_CONTINUATION = r"^\s*(?:\[ANSWER\]\s*)?(?P<word>[A-Za-z]+)"


@dataclass(frozen=True)
class Args(CmdLineArgs):
    model_id: str = "google/gemma-3-4b-it"
    grid_size: str = "4x4"
    walk_len: int = 200
    num_trials: int = 100


REGISTERED_EXPERIMENTS = set()
def experiment1(fn: Callable[[Grid], Any]):
    REGISTERED_EXPERIMENTS.add(fn)
    return fn


def parse_answer(result: str, pattern: str, *, last: bool = False) -> str | None:
    text = result.strip().rstrip("<end_of_turn>").strip()
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    m = matches[-1] if last else matches[0]
    return m.group("word")


@experiment
def instruction(grid: Grid):
    result = call_model(
        messages=[{"role": "user", "content": (
            "Your job is to predict the next word in a sequence of words. "
            "Generate the token [ANSWER], then generate the next word in the sequence.\n"
            f"[SEQUENCE] {' '.join(grid.walk)}"
        )}],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in grid.valid_next


@experiment
def prefilled(grid: Grid):
    result = call_model(
        messages=[{"role": "user", "content": "Continue the sequence of words."}],
        prefill=f"[SEQUENCE] {' '.join(grid.walk)}",
        max_new_tokens=4,
    )
    return parse_answer(result, RE_CONTINUATION) in grid.valid_next


@experiment
def multi_turn(grid: Grid):
    result = call_model(
        messages=[
            {"role": "user", "content": "Generate a sequence of words that follow a pattern. Start with [SEQUENCE]."},
            {"role": "assistant", "content": f"[SEQUENCE] {' '.join(grid.walk)}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
        ],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in grid.valid_next


@experiment
def multi_turn_1_example(grid: Grid):
    result = call_model(
        messages=[
            {"role": "user", "content": "Generate a sequence of words that follow a pattern."},
            {"role": "assistant", "content": f"[SEQUENCE] {' '.join(grid.walk[:-1])}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
            {"role": "assistant", "content": f"[ANSWER] {grid.walk[-1]}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
        ],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in grid.valid_next


@experiment
def multi_turn_2_examples(grid: Grid):
    result = call_model(
        messages=[
            {"role": "user", "content": "Generate a sequence of words that follow a pattern."},
            {"role": "assistant", "content": f"[SEQUENCE] {' '.join(grid.walk[:-2])}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
            {"role": "assistant", "content": f"[ANSWER] {grid.walk[-2]}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
            {"role": "assistant", "content": f"[ANSWER] {grid.walk[-1]}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
        ],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in grid.valid_next


@experiment
def turn_by_turn(grid: Grid):
    result = call_model(
        messages=[
            *[
                x for w in grid.walk for x in [
                    {"role": "user", "content": "Give me a word."},
                    {"role": "assistant", "content": w},
                ]
            ],
            {"role": "user", "content": "Give me a word."},
        ],
        prefill=None,
        max_new_tokens=4,
    )
    return parse_answer(result, RE_FIRST_TOKEN) in grid.valid_next


@experiment
def chain_of_thought(grid: Grid):
    result = call_model(
        messages=[{"role": "user", "content": (
            "Your job is to predict the next word in a sequence of words. "
            "First, think step by step about what pattern connects consecutive words in the sequence. "
            "Then, on a new line, write [ANSWER] followed by the next word.\n"
            f"[SEQUENCE] {' '.join(grid.walk)}"
        )}],
        prefill=None,
        max_new_tokens=420,
    )
    return parse_answer(result, RE_ANSWER, last=True) in grid.valid_next


@experiment
def prefill_with_separator(grid: Grid):
    result = call_model(
        messages=[{"role": "user", "content": "Output [SEQUENCE] followed by a sequence of words, then [ANSWER] and one more word that follows the pattern."}],
        prefill=f"[SEQUENCE] {' '.join(grid.walk)}\n[ANSWER]",
        max_new_tokens=4,
    )
    return parse_answer(result, RE_CONTINUATION) in grid.valid_next


@experiment
def reflection(grid: Grid):
    result = call_model(
        messages=[
            {"role": "user", "content": f"You are given a sequence of words. Answer with ONLY 'Sequence acknowledged'. No explanation, no punctuation, no other text.\n[SEQUENCE] {' '.join(grid.walk)}"},
            {"role": "assistant", "content": "Sequence acknowledged"},
            {"role": "user", "content": "Repeat the last word in the sequence. Answer with ONLY a single word. No explanation, no punctuation, no other text."},
            {"role": "assistant", "content": grid.walk[-1]},
            {"role": "user", "content": "Predict the next word in the sequence. Answer with ONLY a single word. No explanation, no punctuation, no other text."},
        ],
        prefill=None,
        max_new_tokens=4,
    )
    return parse_answer(result, RE_FIRST_TOKEN) in grid.valid_next


@experiment
def pair_format(grid: Grid):
    result = call_model(
        messages=[{"role": "user", "content": (
            "Below are word-to-word transitions from a sequence. "
            "Your job is to predict the next word in a sequence of words. "
            "Generate the token [ANSWER], then generate the next word in the sequence.\n"
            "[TRANSITIONS] " + ", ".join(f"{a} -> {b}" for a, b in zip(grid.walk, grid.walk[1:])) +
            f"\n[LAST WORD] {grid.walk[-1]}"
        )}],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in grid.valid_next


@experiment
def system_message(grid: Grid):
    result = call_model(
        messages=[
            {"role": "system", "content": f"[SEQUENCE] {' '.join(grid.walk)}"},
            {"role": "user", "content": (
                "Your job is to predict the next word in a sequence of words given in the system message. "
                "Generate the token [ANSWER], then generate the next word in the sequence.\n"
            )},
        ],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in grid.valid_next


@experiment
def partial_prefill(grid: Grid):
    result = call_model(
        messages=[{"role": "user", "content": (
            "Continue this sequence of words.\n"
            f"[SEQUENCE] {' '.join(grid.walk[:len(grid.walk)//2])}"
        )}],
        prefill=" ".join(grid.walk[len(grid.walk)//2:]),
        max_new_tokens=4,
    )
    return parse_answer(result, RE_CONTINUATION) in grid.valid_next


if __name__ == "__main__":
    args = Args.parse()
    results = defaultdict(list)
    for _ in range(args.num_trials):
        random.seed(args.base_seed + _)
        grid = Grid(args.grid_size).walk(args.walk_len)
        for experiment in REGISTERED_EXPERIMENTS:
            results[experiment].append(experiment(grid))
    print_results(results)
    save_results(results)

