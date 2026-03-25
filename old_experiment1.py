"""Experiment 1: In-Context Representation Learning Does Not Imply Robust Next-Token Prediction.

Replicates Experiment 1 from "Language Models Struggle to Use Representations Learned In-Context"
(Lepori et al., 2025). Evaluates next-token prediction accuracy under an Instruction Condition
(walk in user message) vs. a Prefilled Condition (walk in assistant prefill).
"""

import argparse
from datetime import datetime
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple
from pathlib import Path
import json

from tqdm import trange

from old_utils import (
    WORDS, Message, neighbors, parse_answer, run_generation,
    print_table, load_model, validate_words,
)


@dataclass
class Condition:
    """A prompting setup defined by an arbitrary message history plus an optional prefill."""
    name: str
    messages: Callable[[List[str]], List[Message]]
    prefill: Callable[[str], str | None] = lambda walk: None
    max_new_tokens: int = 12


CONDITIONS: List[Condition] = [
    # Replicates the Instruction Condition from the paper (Experiment 1).
    Condition(
        name="instruction",
        messages=lambda walk: [{"role": "user", "content": (
            "Your job is to predict the next word in a sequence of words. "
            "Generate the token [ANSWER], then generate the next word in the sequence.\n"
            f"[SEQUENCE] {' '.join(walk)}"
        )}],
        prefill=lambda walk: None,
        max_new_tokens=12,
    ),
    # Replicates the Prefilled Condition from the paper (Experiment 1).
    Condition(
        name="prefilled",
        messages=lambda walk: [{"role": "user", "content": "Continue the sequence of words."}],
        prefill=lambda walk: f"[SEQUENCE] {' '.join(walk)}",
        max_new_tokens=4,
    ),
    # Custom: Walk is embedded in previous assistant message.
    Condition(
        name="multi-turn",
        messages=lambda walk: [
            {"role": "user", "content": "Generate a sequence of words that follow a pattern. Start with [SEQUENCE]."},
            {"role": "assistant", "content": f"[SEQUENCE] {' '.join(walk)}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
        ],
        prefill=lambda walk: None,
        max_new_tokens=12,
    ),
    # Custom: Walk is embedded in previous assistant message, except the last word is given as example (1-shot learning).
    Condition(
        name="multi-turn-1-example",
        messages=lambda walk: [
            {"role": "user", "content": "Generate a sequence of words that follow a pattern."},
            {"role": "assistant", "content": f"[SEQUENCE] {' '.join(walk[:-1])}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
            {"role": "assistant", "content": f"[ANSWER] {walk[-1]}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
        ],
        prefill=lambda walk: None,
        max_new_tokens=12,
    ),
    # Custom: Walk is embedded in previous assistant message, except the last two words are given as example (2-shot learning).
    Condition(
        name="multi-turn-2-examples",
        messages=lambda walk: [
            {"role": "user", "content": "Generate a sequence of words that follow a pattern."},
            {"role": "assistant", "content": f"[SEQUENCE] {' '.join(walk[:-2])}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
            {"role": "assistant", "content": f"[ANSWER] {walk[-2]}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
            {"role": "assistant", "content": f"[ANSWER] {walk[-1]}"},
            {"role": "user", "content": "Generate the next word in the sequence. Start with [ANSWER]."},
        ],
        prefill=lambda walk: None,
        max_new_tokens=12,
    ),
    # Custom: Walk is embedded fully in the message history, where each assistant message contains one word from the walk.
    Condition(
        name="turn-by-turn",
        messages=lambda walk: [
            *[
                x for w in walk for x in [
                    {"role": "user", "content": "Give me a word."},
                    {"role": "assistant", "content": w},
                ]
            ],
            {"role": "user", "content": "Give me a word."},
        ],
        prefill=lambda walk: None,
        max_new_tokens=4,
    ),
    # Hypothesis: explicitly prompting CoT reasoning can help a non-reasoning model
    # deploy otherwise-inert representations (paper shows reasoning models do better).
    Condition(
        name="chain-of-thought",
        messages=lambda walk: [{"role": "user", "content": (
            "Your job is to predict the next word in a sequence of words. "
            "First, think step by step about what pattern connects consecutive words in the sequence. "
            "Then, on a new line, write [ANSWER] followed by the next word.\n"
            f"[SEQUENCE] {' '.join(walk)}"
        )}],
        prefill=lambda walk: None,
        max_new_tokens=420,
    ),
    # Hypothesis: the prefilled condition works because there is literally zero gap.
    # Inserting a brief natural-language separator in an otherwise-prefilled response
    # tests how fragile the prefill advantage is.
    Condition(
        name="prefill-with-separator",
        messages=lambda walk: [{"role": "user", "content": "Output [SEQUENCE] followed by a sequence of words, then [ANSWER] and one more word that follows the pattern."}],
        prefill=lambda walk: f"[SEQUENCE] {' '.join(walk)}\n[ANSWER]",
        max_new_tokens=4,
    ),
    # Hypothesis: having the model "engage" with the walk before predicting — even in a
    # trivial way — warms up the representation for deployment. The assistant confirms
    # the last word (demonstrating awareness of the walk), then is asked to predict.
    Condition(
        name="reflection",
        messages=lambda walk: [
            {"role": "user", "content": f"You are given a sequence of words. Answer with ONLY 'Sequence acknowledged'. No explanation, no punctuation, no other text.\n[SEQUENCE] {' '.join(walk)}"},
            {"role": "assistant", "content": f"Sequence acknowledged"},
            {"role": "user", "content": f"Repeat the last word in the sequence. Answer with ONLY a single word. No explanation, no punctuation, no other text."},
            {"role": "assistant", "content": walk[-1]},
            {"role": "user", "content": f"Predict the next word in the sequence. Answer with ONLY a single word. No explanation, no punctuation, no other text."},
        ],
        prefill=lambda walk: None,
        max_new_tokens=4,
    ),
    # Hypothesis: presenting transitions as explicit pairs ("A -> B") rather than a flat
    # sequence makes the relational structure more salient and easier to deploy.
    Condition(
        name="pair-format",
        messages=lambda walk: [{"role": "user", "content": (
            "Below are word-to-word transitions from a sequence. "
            "Your job is to predict the next word in a sequence of words. "
            "Generate the token [ANSWER], then generate the next word in the sequence.\n"
            "[TRANSITIONS] " + ", ".join(f"{a} -> {b}" for a, b in zip(walk, walk[1:])) +
            f"\n[LAST WORD] {walk[-1]}"
        )}],
        prefill=lambda walk: None,
        max_new_tokens=12,
    ),
    # Hypothesis: placing the walk in the system message (a privileged context slot that
    # models are trained to attend to) may improve deployment over the user message.
    Condition(
        name="system-message",
        messages=lambda walk: [
            {"role": "system", "content": f"[SEQUENCE] {' '.join(walk)}"},
            {"role": "user", "content": (
                "Your job is to predict the next word in a sequence of words given in the system message. "
                "Generate the token [ANSWER], then generate the next word in the sequence.\n"
            )},
        ],
        prefill=lambda walk: None,
        max_new_tokens=12,
    ),
    # Hypothesis: if the prefill condition works because tokens are in the model's own
    # output space, then splitting the walk — first half in the user message, second half
    # prefilled — should still enable deployment, since the representation is built
    # incrementally and the critical final tokens are in assistant space.
    Condition(
        name="partial-prefill",
        messages=lambda walk: [{"role": "user", "content": (
            "Continue this sequence of words.\n"
            f"[SEQUENCE] {' '.join(walk[:len(walk)//2])}"
        )}],
        prefill=lambda walk: " ".join(walk[len(walk)//2:]),
        max_new_tokens=4,
    ),
]


@dataclass
class ConditionResult:
    raw: str
    guess: str | None
    ok: bool


@dataclass
class TrialResult:
    seed: int
    last_word: str
    valid_next_tokens: List[str]
    conditions: Dict[str, ConditionResult]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate prompting conditions on the delayed next-token prediction task from "
        "'Language Models Struggle to Use Representations Learned In-Context'."
    )
    p.add_argument("--model-id", default="google/gemma-3-4b-it")
    p.add_argument("--grid-size", type=str, default="4x4", help="Grid dimensions as RxC (e.g. 4x4, 5x5, 16x1).")
    p.add_argument("--walk-len", type=int, default=200)
    p.add_argument("--num-trials", type=int, default=100, help="Number of word assignments to evaluate.")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--show-examples", type=int, default=5)
    return p.parse_args()


def evaluate_trial(
    model, processor,
    *, seed: int, words: Sequence[str], grid_rows: int, grid_cols: int, walk_len: int,
    conditions: List[Condition],
) -> TrialResult:
    rng = random.Random(seed)
    sampled = rng.sample(list(words), grid_rows * grid_cols)
    pos2word = {(i, j): sampled[i * grid_cols + j] for i in range(grid_rows) for j in range(grid_cols)}
    word2pos = {w: p for p, w in pos2word.items()}

    pos = rng.choice(list(pos2word))
    walk = [pos2word[pos]]
    for _ in range(walk_len - 1):
        pos = rng.choice(neighbors(pos[0], pos[1], grid_rows, grid_cols))
        walk.append(pos2word[pos])

    last_word = walk[-1]
    valid_next = [pos2word[p] for p in neighbors(*word2pos[last_word], grid_rows, grid_cols)]

    cond_results: Dict[str, ConditionResult] = {}
    for cond in conditions:
        raw = run_generation(model, processor, cond.messages(walk), prefill=cond.prefill(walk), max_new_tokens=cond.max_new_tokens)
        guess = parse_answer(raw, sampled)
        cond_results[cond.name] = ConditionResult(raw=raw, guess=guess, ok=guess in valid_next)

    return TrialResult(seed=seed, last_word=last_word, valid_next_tokens=valid_next, conditions=cond_results)


def main() -> None:
    args = parse_args()

    model, processor = load_model(args.model_id)

    parts = args.grid_size.split("x")
    grid_rows, grid_cols = int(parts[0]), int(parts[1])
    grid_cells = grid_rows * grid_cols

    if len(WORDS) < grid_cells:
        raise ValueError(f"Need at least {grid_cells} words for {grid_rows}x{grid_cols} grid, got {len(WORDS)}.")
    validate_words(WORDS, processor)

    results = [
        evaluate_trial(
            model, processor,
            seed=s, words=WORDS, grid_rows=grid_rows, grid_cols=grid_cols, walk_len=args.walk_len,
            conditions=CONDITIONS,
        )
        for s in trange(args.base_seed, args.base_seed + args.num_trials, desc="Evaluating trials")
    ]

    n = len(results)
    cond_names = [c.name for c in CONDITIONS]

    print("=== Configuration ===")
    print(f"Model: {args.model_id}")
    print(f"Grid size: {grid_rows}x{grid_cols}")
    print(f"Walk length: {args.walk_len}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Base seed: {args.base_seed}")
    print(f"Conditions: {cond_names}")
    print()
    print("=== Summary ===")
    print(f"Number of trials: {n}")
    print(f"Uniform vocab chance accuracy: {sum(len(r.valid_next_tokens) / grid_cells for r in results) / n:7.2%}")
    print(f"Average number of valid next tokens: {sum(len(r.valid_next_tokens) for r in results) / n}")
    print()
    print_table([{
        "condition": name,
        "accuracy": f"{sum(r.conditions[name].ok for r in results) / n:7.2%}",
        "parse rate": f"{sum(r.conditions[name].guess is not None for r in results) / n:7.2%}",
    } for name in cond_names])
    print()

    print("=== Example trials ===")
    ex_rows = []
    for r in results[:args.show_examples]:
        row = {"seed": r.seed, "last_word": r.last_word, "valid_next_tokens": r.valid_next_tokens}
        for name in cond_names:
            cr = r.conditions[name]
            row[f"{name}"] = ("✓" if cr.ok else "✗") + " " + (cr.guess or "???")
        ex_rows.append(row)
    if ex_rows:
        print_table(ex_rows)
    print()

    for name in cond_names:
        failures = [r for r in results if not r.conditions[name].ok]
        print(f"=== {name} failures ({len(failures)}/{n}) ===")
        fail_rows = []
        for r in failures[:args.show_examples]:
            cr = r.conditions[name]
            fail_rows.append({"seed": r.seed, "last_word": r.last_word,
                               "valid_next_tokens": r.valid_next_tokens,
                               "guess": cr.guess, "raw answer": cr.raw.replace("\n", "\\n")})
        if fail_rows:
            print_table(fail_rows)
        else:
            print("(no failures)")
        print()

    Path("results").mkdir(exist_ok=True)
    with open(Path("results") / f"exp1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        data = {
            "experiment": "experiment1",
            "args": vars(args),
            "results": [
                {**vars(r), "conditions": {k: vars(v) for k, v in r.conditions.items()}}
                for r in results
            ],
        }
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
