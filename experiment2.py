"""Experiment 2: In-Context Representation Learning Does Not Imply Adaptive World Modeling.

Replicates Experiment 2 from "Language Models Struggle to Use Representations Learned In-Context"
(Lepori et al., 2025). The Adaptive World Modeling (AWM) task has two components:
  1. A random walk over a latent grid/line state space (same as Experiment 1).
  2. Few-shot examples of a mapping rule (e.g. state (i,j) -> state (i+1,j)).
The model must apply the rule to a held-out query state, which requires deploying the
in-context topology it inferred from the walk.

We also implement the explicit-topology baseline from Section 4, where the grid structure
is described verbatim (as "Coordinates: i j Item: word") instead of via the random walk.
"""

import argparse
from datetime import datetime
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import json

from tqdm import trange

from utils import (
    WORDS, Message, neighbors, parse_answer, run_generation,
    print_table, load_model, validate_words,
)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """A mapping rule from one grid position to another."""
    name: str
    # Returns the target position given source (i, j) and grid dimensions,
    # or None if the source is invalid for this rule.
    apply: Callable[[int, int, int, int], Optional[Tuple[int, int]]]


RULES: List[Rule] = [
    Rule(
        name="1-step-down",
        apply=lambda i, j, rows, cols: (i + 1, j) if i + 1 < rows else None,
    ),
    Rule(
        name="2-step-down",
        apply=lambda i, j, rows, cols: (i + 2, j) if i + 2 < rows else None,
    ),
    # 3-step rule: (i,j) -> (i+2, j+1); only meaningful for grids with >=3 rows and >=2 cols.
    Rule(
        name="3-step",
        apply=lambda i, j, rows, cols: (i + 2, j + 1) if i + 2 < rows and j + 1 < cols else None,
    ),
]

# How many few-shot examples to include per rule/topology combination.
# The paper uses 10 for most cases, but only 6 for 4x4 grid + 2-step-down
# (since only 8 valid input states exist and some are held out as queries).
DEFAULT_NUM_EXAMPLES = 10


def get_num_examples(rule: Rule, grid_rows: int, grid_cols: int) -> int:
    """Return the number of few-shot examples for a given rule/topology."""
    valid_inputs = [
        (i, j)
        for i in range(grid_rows) for j in range(grid_cols)
        if rule.apply(i, j, grid_rows, grid_cols) is not None
    ]
    # Reserve at least 1 for the query; leave a small buffer.
    return min(DEFAULT_NUM_EXAMPLES, len(valid_inputs) - 2)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_examples(examples: List[Tuple[str, str]]) -> str:
    lines = "\n".join(f"Input: {inp} Output: {out}" for inp, out in examples)
    return f"[EXAMPLES]\n{lines}"


def build_awm_prefill(
    walk: List[str],
    examples: List[Tuple[str, str]],
    query_input: str,
) -> str:
    """Prefill for the AWM condition: walk + few-shot examples + query."""
    seq = f"[SEQUENCE] {' '.join(walk)}"
    ex = _format_examples(examples)
    return f"{seq}\n\n{ex}\n\n[QUERY]\nInput: {query_input} Output:"


def build_explicit_prefill(
    pos2word: Dict[Tuple[int, int], str],
    grid_rows: int,
    grid_cols: int,
    examples: List[Tuple[str, str]],
    query_input: str,
) -> str:
    """Prefill for the explicit-topology baseline: coordinate list + few-shot examples + query."""
    coord_lines = "\n".join(
        f"Coordinates: {i} {j} Item: {pos2word[(i, j)]}"
        for i in range(grid_rows) for j in range(grid_cols)
    )
    topo = f"[TOPOLOGY]\n{coord_lines}"
    ex = _format_examples(examples)
    return f"{topo}\n\n{ex}\n\n[QUERY]\nInput: {query_input} Output:"


AWM_USER_MESSAGE = (
    "You are given a sequence of words and few-shot examples of a mapping rule. "
    "Each example shows an input word and the output word it maps to under the rule. "
    "Predict the output word for the query input. "
    "Respond with a single word only."
)

EXPLICIT_USER_MESSAGE = (
    "You are given a description of a grid where each cell has coordinates and a word, "
    "followed by few-shot examples of a mapping rule. "
    "Each example shows an input word and the output word it maps to under the rule. "
    "Predict the output word for the query input. "
    "Respond with a single word only."
)


# ---------------------------------------------------------------------------
# Trial result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AWMConditionResult:
    raw: str
    guess: Optional[str]
    expected: str
    ok: bool


@dataclass
class AWMTrialResult:
    seed: int
    rule: str
    condition: str  # "awm" or "explicit"
    query_input: str
    expected_output: str
    result: AWMConditionResult


# ---------------------------------------------------------------------------
# Trial evaluation
# ---------------------------------------------------------------------------

def evaluate_awm_trial(
    model, processor,
    *,
    seed: int,
    words: Sequence[str],
    grid_rows: int,
    grid_cols: int,
    walk_len: int,
    rule: Rule,
    num_examples: int,
    run_explicit: bool,
) -> List[AWMTrialResult]:
    """
    Run one trial (one word assignment) for both AWM and (optionally) explicit-topology conditions.
    Returns a list of AWMTrialResult, one per condition run.
    """
    rng = random.Random(seed)
    sampled = rng.sample(list(words), grid_rows * grid_cols)
    pos2word = {(i, j): sampled[i * grid_cols + j] for i in range(grid_rows) for j in range(grid_cols)}
    word2pos = {w: p for p, w in pos2word.items()}

    # Build random walk.
    pos = rng.choice(list(pos2word))
    walk = [pos2word[pos]]
    for _ in range(walk_len - 1):
        pos = rng.choice(neighbors(pos[0], pos[1], grid_rows, grid_cols))
        walk.append(pos2word[pos])

    # Collect all valid (input_word, output_word) pairs for this rule.
    valid_pairs: List[Tuple[str, str]] = []
    for (i, j), inp_word in pos2word.items():
        target = rule.apply(i, j, grid_rows, grid_cols)
        if target is not None:
            valid_pairs.append((inp_word, pos2word[target]))

    if len(valid_pairs) < num_examples + 1:
        raise ValueError(
            f"Not enough valid pairs ({len(valid_pairs)}) for rule '{rule.name}' "
            f"on {grid_rows}x{grid_cols} grid with {num_examples} examples + 1 query."
        )

    # Sample examples and a held-out query.
    rng.shuffle(valid_pairs)
    example_pairs = valid_pairs[:num_examples]
    query_input, expected_output = valid_pairs[num_examples]

    results = []

    # --- AWM condition (walk as prefill) ---
    awm_prefill = build_awm_prefill(walk, example_pairs, query_input)
    awm_messages: List[Message] = [{"role": "user", "content": AWM_USER_MESSAGE}]
    raw = run_generation(model, processor, awm_messages, prefill=awm_prefill, max_new_tokens=4)
    guess = parse_answer(raw, sampled)
    results.append(AWMTrialResult(
        seed=seed,
        rule=rule.name,
        condition="awm",
        query_input=query_input,
        expected_output=expected_output,
        result=AWMConditionResult(raw=raw, guess=guess, expected=expected_output, ok=guess == expected_output),
    ))

    # --- Explicit topology condition ---
    if run_explicit:
        exp_prefill = build_explicit_prefill(pos2word, grid_rows, grid_cols, example_pairs, query_input)
        exp_messages: List[Message] = [{"role": "user", "content": EXPLICIT_USER_MESSAGE}]
        raw = run_generation(model, processor, exp_messages, prefill=exp_prefill, max_new_tokens=4)
        guess = parse_answer(raw, sampled)
        results.append(AWMTrialResult(
            seed=seed,
            rule=rule.name,
            condition="explicit",
            query_input=query_input,
            expected_output=expected_output,
            result=AWMConditionResult(raw=raw, guess=guess, expected=expected_output, ok=guess == expected_output),
        ))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Adaptive World Modeling (Experiment 2) from "
        "'Language Models Struggle to Use Representations Learned In-Context'."
    )
    p.add_argument("--model-id", default="google/gemma-3-4b-it")
    p.add_argument("--grid-size", type=str, default="4x4", help="Grid dimensions as RxC (e.g. 4x4, 5x5, 16x1).")
    p.add_argument("--walk-len", type=int, default=200)
    p.add_argument("--rules", nargs="+", default=["1-step-down", "2-step-down"],
                   choices=[r.name for r in RULES],
                   help="Which rules to evaluate.")
    p.add_argument("--num-trials", type=int, default=100, help="Number of word assignments to evaluate.")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--show-examples", type=int, default=5)
    p.add_argument("--no-explicit", action="store_true",
                   help="Skip the explicit-topology baseline condition.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model, processor = load_model(args.model_id)

    parts = args.grid_size.split("x")
    grid_rows, grid_cols = int(parts[0]), int(parts[1])
    grid_cells = grid_rows * grid_cols

    if len(WORDS) < grid_cells:
        raise ValueError(f"Need at least {grid_cells} words for {grid_rows}x{grid_cols} grid, got {len(WORDS)}.")
    validate_words(WORDS, processor)

    active_rules = [r for r in RULES if r.name in args.rules]
    run_explicit = not args.no_explicit

    all_results: List[AWMTrialResult] = []

    for rule in active_rules:
        num_examples = get_num_examples(rule, grid_rows, grid_cols)
        print(f"\n--- Rule: {rule.name} | Examples: {num_examples} ---")
        for s in trange(args.base_seed, args.base_seed + args.num_trials, desc=f"{rule.name}"):
            trial_results = evaluate_awm_trial(
                model, processor,
                seed=s,
                words=WORDS,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                walk_len=args.walk_len,
                rule=rule,
                num_examples=num_examples,
                run_explicit=run_explicit,
            )
            all_results.extend(trial_results)

    n = args.num_trials
    conditions = ["awm"] + (["explicit"] if run_explicit else [])

    print("\n=== Configuration ===")
    print(f"Model: {args.model_id}")
    print(f"Grid size: {grid_rows}x{grid_cols}")
    print(f"Walk length: {args.walk_len}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Base seed: {args.base_seed}")
    print(f"Rules: {[r.name for r in active_rules]}")
    print(f"Conditions: {conditions}")

    print("\n=== Summary ===")
    summary_rows = []
    for rule in active_rules:
        for cond in conditions:
            rule_cond = [r for r in all_results if r.rule == rule.name and r.condition == cond]
            if not rule_cond:
                continue
            acc = sum(r.result.ok for r in rule_cond) / len(rule_cond)
            parse_rate = sum(r.result.guess is not None for r in rule_cond) / len(rule_cond)
            summary_rows.append({
                "rule": rule.name,
                "condition": cond,
                "n": len(rule_cond),
                "accuracy": f"{acc:7.2%}",
                "parse rate": f"{parse_rate:7.2%}",
            })
    if summary_rows:
        print_table(summary_rows)

    print("\n=== Example trials ===")
    for rule in active_rules:
        for cond in conditions:
            rule_cond = [r for r in all_results if r.rule == rule.name and r.condition == cond]
            print(f"\n-- {rule.name} / {cond} --")
            ex_rows = [
                {
                    "seed": r.seed,
                    "query": r.query_input,
                    "expected": r.expected_output,
                    "guess": r.result.guess or "???",
                    "ok": "✓" if r.result.ok else "✗",
                    "raw": r.result.raw.replace("\n", "\\n")[:60],
                }
                for r in rule_cond[:args.show_examples]
            ]
            if ex_rows:
                print_table(ex_rows)

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"exp2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        data = {
            "experiment": "experiment2",
            "args": vars(args),
            "results": [
                {
                    "seed": r.seed,
                    "rule": r.rule,
                    "condition": r.condition,
                    "query_input": r.query_input,
                    "expected_output": r.expected_output,
                    **vars(r.result),
                }
                for r in all_results
            ],
        }
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
