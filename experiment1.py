import re
from dataclasses import dataclass

from utils import experiment, generate_walk, call_model, CmdLineArgs, generate_grid, run_experiments

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


def generate_walk(args: Args):
    grid = generate_grid(args.grid_size)
    walk = ...
    return ..., ...


def parse_answer(result: str, pattern: str, *, last: bool = False) -> str | None:
    text = result.strip().rstrip("<end_of_turn>").strip()
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    m = matches[-1] if last else matches[0]
    return m.group("word")


@experiment
def instruction(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[{"role": "user", "content": (
            "Your job is to predict the next word in a sequence of words. "
            "Generate the token [ANSWER], then generate the next word in the sequence.\n"
            f"[SEQUENCE] {' '.join(walk)}"
        )}],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in next


@experiment
def prefilled(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[{"role": "user", "content": "Continue the sequence of words."}],
        prefill=f"[SEQUENCE] {' '.join(walk)}",
        max_new_tokens=4,
    )
    return parse_answer(result, RE_CONTINUATION) in next


@experiment
def multi_turn(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[
            {"role": "user", "content": "Generate a sequence of words that follow a pattern. Start with [SEQUENCE]."},
            {"role": "assistant", "content": f"[SEQUENCE] {' '.join(walk)}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
        ],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in next


@experiment
def multi_turn_1_example(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[
            {"role": "user", "content": "Generate a sequence of words that follow a pattern."},
            {"role": "assistant", "content": f"[SEQUENCE] {' '.join(walk[:-1])}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
            {"role": "assistant", "content": f"[ANSWER] {walk[-1]}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
        ],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in next


@experiment
def multi_turn_2_examples(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[
            {"role": "user", "content": "Generate a sequence of words that follow a pattern."},
            {"role": "assistant", "content": f"[SEQUENCE] {' '.join(walk[:-2])}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
            {"role": "assistant", "content": f"[ANSWER] {walk[-2]}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
            {"role": "assistant", "content": f"[ANSWER] {walk[-1]}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
        ],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in next


@experiment
def turn_by_turn(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[
            *[
                x for w in walk for x in [
                    {"role": "user", "content": "Give me a word."},
                    {"role": "assistant", "content": w},
                ]
            ],
            {"role": "user", "content": "Give me a word."},
        ],
        prefill=None,
        max_new_tokens=4,
    )
    return parse_answer(result, RE_FIRST_TOKEN) in next


@experiment
def chain_of_thought(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[{"role": "user", "content": (
            "Your job is to predict the next word in a sequence of words. "
            "First, think step by step about what pattern connects consecutive words in the sequence. "
            "Then, on a new line, write [ANSWER] followed by the next word.\n"
            f"[SEQUENCE] {' '.join(walk)}"
        )}],
        prefill=None,
        max_new_tokens=420,
    )
    return parse_answer(result, RE_ANSWER, last=True) in next


@experiment
def prefill_with_separator(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[{"role": "user", "content": "Output [SEQUENCE] followed by a sequence of words, then [ANSWER] and one more word that follows the pattern."}],
        prefill=f"[SEQUENCE] {' '.join(walk)}\n[ANSWER]",
        max_new_tokens=4,
    )
    return parse_answer(result, RE_CONTINUATION) in next


@experiment
def reflection(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[
            {"role": "user", "content": f"You are given a sequence of words. Answer with ONLY 'Sequence acknowledged'. No explanation, no punctuation, no other text.\n[SEQUENCE] {' '.join(walk)}"},
            {"role": "assistant", "content": "Sequence acknowledged"},
            {"role": "user", "content": "Repeat the last word in the sequence. Answer with ONLY a single word. No explanation, no punctuation, no other text."},
            {"role": "assistant", "content": walk[-1]},
            {"role": "user", "content": "Predict the next word in the sequence. Answer with ONLY a single word. No explanation, no punctuation, no other text."},
        ],
        prefill=None,
        max_new_tokens=4,
    )
    return parse_answer(result, RE_FIRST_TOKEN) in next


@experiment
def pair_format(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[{"role": "user", "content": (
            "Below are word-to-word transitions from a sequence. "
            "Your job is to predict the next word in a sequence of words. "
            "Generate the token [ANSWER], then generate the next word in the sequence.\n"
            "[TRANSITIONS] " + ", ".join(f"{a} -> {b}" for a, b in zip(walk, walk[1:])) +
            f"\n[LAST WORD] {walk[-1]}"
        )}],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in next


@experiment
def system_message(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[
            {"role": "system", "content": f"[SEQUENCE] {' '.join(walk)}"},
            {"role": "user", "content": (
                "Your job is to predict the next word in a sequence of words given in the system message. "
                "Generate the token [ANSWER], then generate the next word in the sequence.\n"
            )},
        ],
        prefill=None,
        max_new_tokens=12,
    )
    return parse_answer(result, RE_ANSWER) in next


@experiment
def partial_prefill(args: Args):
    walk, next = generate_walk(args)
    result = call_model(
        messages=[{"role": "user", "content": (
            "Continue this sequence of words.\n"
            f"[SEQUENCE] {' '.join(walk[:len(walk)//2])}"
        )}],
        prefill=" ".join(walk[len(walk)//2:]),
        max_new_tokens=4,
    )
    return parse_answer(result, RE_CONTINUATION) in next


if __name__ == "__main__":
    args = Args.parse_args()
    run_experiments(args)
