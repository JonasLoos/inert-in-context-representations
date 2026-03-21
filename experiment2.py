"""Experiment 2: In-Context Representation Learning Does Not Imply Adaptive World Modeling.

Replicates Experiment 2 from "Language Models Struggle to Use Representations Learned In-Context"
(Lepori et al., 2025). The Adaptive World Modeling (AWM) task has two components:
  1. A random walk over a latent grid/line state space (same as Experiment 1).
  2. Few-shot examples of a mapping rule (e.g. state (i,j) -> state (i+1,j)).
The model must apply the rule to a held-out query state, which requires deploying the
in-context topology it inferred from the walk.

Conditions vary where/how the walk and examples are presented, mirroring experiment1.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import random
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
    # 3-step rule: (i,j) -> (i+2, j+1); meaningful for grids with >=3 rows and >=2 cols.
    Rule(
        name="3-step",
        apply=lambda i, j, rows, cols: (i + 2, j + 1) if i + 2 < rows and j + 1 < cols else None,
    ),
]

DEFAULT_NUM_EXAMPLES = 10


def get_num_examples(rule: Rule, grid_rows: int, grid_cols: int) -> int:
    """Return the number of few-shot examples for a given rule/topology.

    Mirrors the paper: 10 for most settings, fewer when the rule has limited valid inputs
    (e.g. 4x4 grid + 2-step-down has only 8 valid pairs, so the paper uses 6).
    """
    valid_count = sum(
        1
        for i in range(grid_rows) for j in range(grid_cols)
        if rule.apply(i, j, grid_rows, grid_cols) is not None
    )
    return min(DEFAULT_NUM_EXAMPLES, valid_count - 2)


# ---------------------------------------------------------------------------
# Shared prompt helpers
# ---------------------------------------------------------------------------

def _examples_text(examples: List[Tuple[str, str]]) -> str:
    return "\n".join(f"Input: {inp} Output: {out}" for inp, out in examples)


def _coord_text(pos2word: Dict[Tuple[int, int], str], grid_rows: int, grid_cols: int) -> str:
    return "\n".join(
        f"Coordinates: {i} {j} Item: {pos2word[(i, j)]}"
        for i in range(grid_rows) for j in range(grid_cols)
    )


def _transitions_text(walk: List[str]) -> str:
    return ", ".join(f"{a}->{b}" for a, b in zip(walk, walk[1:]))


# ---------------------------------------------------------------------------
# AWM condition definitions
# ---------------------------------------------------------------------------

# Build signature: (walk, examples, query_input, pos2word, grid_rows, grid_cols)
#                  -> (messages, prefill | None)
BuildFn = Callable[
    [List[str], List[Tuple[str, str]], str, Dict[Tuple[int, int], str], int, int],
    Tuple[List[Message], Optional[str]],
]


@dataclass
class AWMCondition:
    name: str
    build: BuildFn
    max_new_tokens: int = 12


CONDITIONS: List[AWMCondition] = [
    # Replicates the paper's AWM setup (instruction format).
    # Walk, examples, and query all in a single user message.
    AWMCondition(
        name="awm",
        build=lambda walk, ex, q, p2w, rows, cols: (
            [{"role": "user", "content": (
                "You are given a sequence of words that encodes a hidden spatial structure, "
                "followed by examples of a mapping rule. "
                "Predict the output word for the query by inferring the structure from the sequence. "
                "Generate the token [ANSWER], then generate the output word.\n\n"
                f"[SEQUENCE] {' '.join(walk)}\n\n"
                f"[EXAMPLES]\n{_examples_text(ex)}\n\n"
                f"[QUERY]\nInput: {q}"
            )}],
            None,
        ),
    ),
    # Explicit-topology baseline from the paper.
    # Grid coordinates given verbatim instead of the walk — tests rule-learning ability
    # independently of in-context topology induction.
    AWMCondition(
        name="explicit",
        build=lambda walk, ex, q, p2w, rows, cols: (
            [{"role": "user", "content": (
                "You are given a description of a grid where each cell has coordinates and a word, "
                "followed by examples of a mapping rule. "
                "Predict the output word for the query. "
                "Generate the token [ANSWER], then generate the output word.\n\n"
                f"[TOPOLOGY]\n{_coord_text(p2w, rows, cols)}\n\n"
                f"[EXAMPLES]\n{_examples_text(ex)}\n\n"
                f"[QUERY]\nInput: {q}"
            )}],
            None,
        ),
    ),
    # Walk in a completed assistant turn (analogous to experiment1's multi-turn condition).
    # Requires the model to use the walk it previously "generated" to answer the AWM question.
    AWMCondition(
        name="awm-multi-turn",
        build=lambda walk, ex, q, p2w, rows, cols: (
            [
                {"role": "user", "content": "Generate a sequence of words that follow a pattern. Start with [SEQUENCE]."},
                {"role": "assistant", "content": f"[SEQUENCE] {' '.join(walk)}"},
                {"role": "user", "content": (
                    "Given examples of a mapping rule applied to words in the sequence above, "
                    "predict the output word for the query. "
                    "Generate [ANSWER] then the output word.\n\n"
                    f"[EXAMPLES]\n{_examples_text(ex)}\n\n"
                    f"[QUERY]\nInput: {q}"
                )},
            ],
            None,
        ),
    ),
    # Walk as assistant prefill, examples+query in the user message
    # (analogous to experiment1's partial-prefill condition).
    # The model sees the task first, then generates the walk in its own response space,
    # then must predict the answer — testing whether prefill-space representations help.
    AWMCondition(
        name="awm-seq-prefill",
        build=lambda walk, ex, q, p2w, rows, cols: (
            [{"role": "user", "content": (
                "You will be shown a sequence of words encoding a hidden structure. "
                "After the sequence, generate [ANSWER] followed by the output word for the query.\n\n"
                f"[EXAMPLES]\n{_examples_text(ex)}\n\n"
                f"[QUERY]\nInput: {q}"
            )}],
            f"[SEQUENCE] {' '.join(walk)}\n[ANSWER]",
        ),
        max_new_tokens=4,
    ),
    # Walk in the system message (analogous to experiment1's system-message condition).
    # Tests whether the privileged system context slot improves topology deployment.
    AWMCondition(
        name="awm-system",
        build=lambda walk, ex, q, p2w, rows, cols: (
            [
                {"role": "system", "content": f"[SEQUENCE] {' '.join(walk)}"},
                {"role": "user", "content": (
                    "You are given a sequence of words in the system message that encodes a hidden "
                    "spatial structure, followed by examples of a mapping rule. "
                    "Predict the output word for the query. "
                    "Generate [ANSWER] then the output word.\n\n"
                    f"[EXAMPLES]\n{_examples_text(ex)}\n\n"
                    f"[QUERY]\nInput: {q}"
                )},
            ],
            None,
        ),
    ),
    # Chain-of-thought: ask the model to reason about topology before answering
    # (analogous to experiment1's chain-of-thought condition).
    AWMCondition(
        name="awm-cot",
        build=lambda walk, ex, q, p2w, rows, cols: (
            [{"role": "user", "content": (
                "You are given a sequence of words that encodes a hidden spatial structure, "
                "followed by examples of a mapping rule. "
                "First, think step by step about the spatial structure and what rule the examples demonstrate. "
                "Then, on a new line, write [ANSWER] followed by the output word for the query.\n\n"
                f"[SEQUENCE] {' '.join(walk)}\n\n"
                f"[EXAMPLES]\n{_examples_text(ex)}\n\n"
                f"[QUERY]\nInput: {q}"
            )}],
            None,
        ),
        max_new_tokens=512,
    ),
    # Explicit topology + chain-of-thought: tests whether explicit topology + reasoning
    # brings the model to ceiling performance.
    AWMCondition(
        name="explicit-cot",
        build=lambda walk, ex, q, p2w, rows, cols: (
            [{"role": "user", "content": (
                "You are given a description of a grid, followed by examples of a mapping rule. "
                "First, reason step by step about what rule the examples demonstrate. "
                "Then, on a new line, write [ANSWER] followed by the output word for the query.\n\n"
                f"[TOPOLOGY]\n{_coord_text(p2w, rows, cols)}\n\n"
                f"[EXAMPLES]\n{_examples_text(ex)}\n\n"
                f"[QUERY]\nInput: {q}"
            )}],
            None,
        ),
        max_new_tokens=512,
    ),
    # Walk presented as explicit A->B transitions rather than a flat sequence
    # (analogous to experiment1's pair-format condition).
    # The relational structure is made explicit rather than requiring the model to infer it.
    AWMCondition(
        name="awm-pair-format",
        build=lambda walk, ex, q, p2w, rows, cols: (
            [{"role": "user", "content": (
                "You are given transitions between words that encode a hidden spatial structure, "
                "followed by examples of a mapping rule. "
                "Predict the output word for the query. Generate [ANSWER] then the output word.\n\n"
                f"[TRANSITIONS] {_transitions_text(walk)}\n\n"
                f"[EXAMPLES]\n{_examples_text(ex)}\n\n"
                f"[QUERY]\nInput: {q}"
            )}],
            None,
        ),
    ),
    # Model first acknowledges the walk (demonstrating awareness), then answers
    # (analogous to experiment1's reflection condition).
    AWMCondition(
        name="awm-reflection",
        build=lambda walk, ex, q, p2w, rows, cols: (
            [
                {"role": "user", "content": (
                    f"You are given a sequence of words. "
                    f"Answer with ONLY 'Sequence acknowledged'. No other text.\n"
                    f"[SEQUENCE] {' '.join(walk)}"
                )},
                {"role": "assistant", "content": "Sequence acknowledged"},
                {"role": "user", "content": (
                    "Given examples of a mapping rule applied to the sequence above, "
                    "predict the output word for the query. "
                    "Generate [ANSWER] then the output word.\n\n"
                    f"[EXAMPLES]\n{_examples_text(ex)}\n\n"
                    f"[QUERY]\nInput: {q}"
                )},
            ],
            None,
        ),
    ),
    # Ablation: examples and query only, no walk at all.
    # Tests how much of the model's performance can be attributed to pattern-matching
    # the examples without any topology information.
    AWMCondition(
        name="awm-no-walk",
        build=lambda walk, ex, q, p2w, rows, cols: (
            [{"role": "user", "content": (
                "You are given examples of a mapping rule between words. "
                "Predict the output word for the query by inferring the rule. "
                "Generate [ANSWER] then the output word.\n\n"
                f"[EXAMPLES]\n{_examples_text(ex)}\n\n"
                f"[QUERY]\nInput: {q}"
            )}],
            None,
        ),
    ),
    # Examples presented as rule-labeled pairs rather than raw input/output lines.
    # Makes the relational structure of the few-shot examples more explicit.
    AWMCondition(
        name="awm-arrow-format",
        build=lambda walk, ex, q, p2w, rows, cols: (
            [{"role": "user", "content": (
                "You are given a sequence of words that encodes a hidden spatial structure, "
                "followed by examples of a mapping rule shown as arrows. "
                "Predict the output word for the query. Generate [ANSWER] then the output word.\n\n"
                f"[SEQUENCE] {' '.join(walk)}\n\n"
                "[EXAMPLES] " + ", ".join(f"{inp}->{out}" for inp, out in ex) + "\n\n"
                f"[QUERY]\nInput: {q}"
            )}],
            None,
        ),
    ),
]

CONDITION_NAMES = [c.name for c in CONDITIONS]


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
    condition: str
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
    conditions: List[AWMCondition],
) -> List[AWMTrialResult]:
    """Run all conditions for one trial (one word assignment)."""
    rng = random.Random(seed)
    sampled = rng.sample(list(words), grid_rows * grid_cols)
    pos2word = {(i, j): sampled[i * grid_cols + j] for i in range(grid_rows) for j in range(grid_cols)}

    # Build random walk.
    pos = rng.choice(list(pos2word))
    walk = [pos2word[pos]]
    for _ in range(walk_len - 1):
        pos = rng.choice(neighbors(pos[0], pos[1], grid_rows, grid_cols))
        walk.append(pos2word[pos])

    # Collect all valid (input_word, output_word) pairs for this rule.
    valid_pairs: List[Tuple[str, str]] = [
        (pos2word[(i, j)], pos2word[rule.apply(i, j, grid_rows, grid_cols)])
        for i in range(grid_rows) for j in range(grid_cols)
        if rule.apply(i, j, grid_rows, grid_cols) is not None
    ]

    if len(valid_pairs) < num_examples + 1:
        raise ValueError(
            f"Not enough valid pairs ({len(valid_pairs)}) for rule '{rule.name}' "
            f"on {grid_rows}x{grid_cols} grid with {num_examples} examples + 1 query."
        )

    rng.shuffle(valid_pairs)
    example_pairs = valid_pairs[:num_examples]
    query_input, expected_output = valid_pairs[num_examples]

    output: List[AWMTrialResult] = []
    for cond in conditions:
        messages, prefill = cond.build(walk, example_pairs, query_input, pos2word, grid_rows, grid_cols)
        raw = run_generation(model, processor, messages, prefill=prefill, max_new_tokens=cond.max_new_tokens)
        guess = parse_answer(raw, sampled)
        output.append(AWMTrialResult(
            seed=seed, rule=rule.name, condition=cond.name,
            query_input=query_input, expected_output=expected_output,
            result=AWMConditionResult(raw=raw, guess=guess, expected=expected_output, ok=guess == expected_output),
        ))

    return output


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
                   choices=[r.name for r in RULES], help="Which rules to evaluate.")
    p.add_argument("--conditions", nargs="+", default=CONDITION_NAMES,
                   choices=CONDITION_NAMES, help="Which prompt conditions to evaluate.")
    p.add_argument("--num-trials", type=int, default=100)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--show-examples", type=int, default=3)
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
    active_conditions = [c for c in CONDITIONS if c.name in args.conditions]

    all_results: List[AWMTrialResult] = []
    for rule in active_rules:
        num_examples = get_num_examples(rule, grid_rows, grid_cols)
        print(f"\n--- Rule: {rule.name} | Examples: {num_examples} ---")
        for s in trange(args.base_seed, args.base_seed + args.num_trials, desc=rule.name):
            all_results.extend(evaluate_awm_trial(
                model, processor,
                seed=s,
                words=WORDS,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                walk_len=args.walk_len,
                rule=rule,
                num_examples=num_examples,
                conditions=active_conditions,
            ))

    n = args.num_trials

    print("\n=== Configuration ===")
    print(f"Model: {args.model_id}")
    print(f"Grid size: {grid_rows}x{grid_cols}")
    print(f"Walk length: {args.walk_len}")
    print(f"Number of trials: {n}")
    print(f"Base seed: {args.base_seed}")
    print(f"Rules: {[r.name for r in active_rules]}")
    print(f"Conditions: {[c.name for c in active_conditions]}")

    print("\n=== Summary ===")
    summary_rows = []
    for rule in active_rules:
        for cond in active_conditions:
            subset = [r for r in all_results if r.rule == rule.name and r.condition == cond.name]
            if not subset:
                continue
            acc = sum(r.result.ok for r in subset) / len(subset)
            parse_rate = sum(r.result.guess is not None for r in subset) / len(subset)
            summary_rows.append({
                "rule": rule.name,
                "condition": cond.name,
                "n": len(subset),
                "accuracy": f"{acc:7.2%}",
                "parse rate": f"{parse_rate:7.2%}",
            })
    if summary_rows:
        print_table(summary_rows)

    print("\n=== Example trials ===")
    for rule in active_rules:
        for cond in active_conditions:
            subset = [r for r in all_results if r.rule == rule.name and r.condition == cond.name]
            print(f"\n-- {rule.name} / {cond.name} --")
            ex_rows = [
                {
                    "seed": r.seed,
                    "query": r.query_input,
                    "expected": r.expected_output,
                    "guess": r.result.guess or "???",
                    "ok": "✓" if r.result.ok else "✗",
                    "raw": r.result.raw.replace("\n", "\\n")[:60],
                }
                for r in subset[:args.show_examples]
            ]
            if ex_rows:
                print_table(ex_rows)

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"exp2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({
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
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
