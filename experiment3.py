"""Experiment 3: Grid reconstruction from a walk.

This task is intentionally independent from the paper replicated by experiment1.py
and experiment2.py. The model is given a random walk over a hidden word-labeled
grid and asked to reconstruct a full grid that is consistent with that walk.

Because a walk only identifies the latent grid up to symmetries (and can still be
ambiguous beyond that), the primary metric is whether the predicted grid is a valid
permutation of the words whose adjacency structure is consistent with the walk.
We also report whether the prediction matches the sampled target grid up to grid
symmetry as a secondary diagnostic.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import trange

from utils import WORDS, Message, neighbors, run_generation, print_table, load_model, validate_words


Grid = List[List[str]]

GRID_TAG_RE = re.compile(r"\[GRID\](?P<body>.*)", re.IGNORECASE | re.DOTALL)
WORD_RE = re.compile(r"[A-Za-z]+")
EXAMPLE_SEED_OFFSET = 1_000_000


@dataclass(frozen=True)
class AnswerFormat:
    name: str
    instruction: Callable[[int, int, bool], str]
    render: Callable[[Grid], str]
    max_new_tokens: int = 256


@dataclass(frozen=True)
class ReconstructionCondition:
    name: str
    style: str
    include_shape: bool
    answer_format: AnswerFormat
    max_new_tokens: int

    def build_messages(self, walk: List[str], example_walk: List[str], example_grid: Grid, rows: int, cols: int) -> List[Message]:
        instruction = build_instruction(rows, cols, self.include_shape, self.answer_format)
        if self.style == "direct":
            return [{"role": "user", "content": f"{instruction}\n[WALK] {' '.join(walk)}"}]
        if self.style == "one-shot":
            return [
                {"role": "user", "content": f"Here is one solved example.\n{instruction}\n[WALK] {' '.join(example_walk)}"},
                {"role": "assistant", "content": render_answer(example_grid, self.answer_format)},
                {"role": "user", "content": f"Now solve a new instance.\n{instruction}\n[WALK] {' '.join(walk)}"},
            ]
        raise ValueError(f"Unknown style: {self.style}")


@dataclass
class GridConditionResult:
    raw: str
    parsed_grid: Optional[Grid]
    parse_ok: bool
    permutation_ok: bool
    walk_consistent: bool
    symmetry_match: bool
    exact_match: bool
    best_symmetry_cell_accuracy: float


@dataclass
class GridTrialResult:
    seed: int
    walk: List[str]
    target_grid: Grid
    conditions: Dict[str, GridConditionResult]


def _rows_instruction(rows: int, cols: int, include_shape: bool) -> str:
    if include_shape:
        return f"After [GRID], print {rows} lines with {cols} words per line, separated by spaces."
    return "After [GRID], print the grid with one row per line and spaces between words."


def _markdown_instruction(rows: int, cols: int, include_shape: bool) -> str:
    if include_shape:
        return f"After [GRID], print a markdown table with {rows} rows and {cols} columns, containing only the grid words."
    return "After [GRID], print a markdown table with one table row per grid row, containing only the grid words."


def _json_instruction(rows: int, cols: int, include_shape: bool) -> str:
    if include_shape:
        return f"After [GRID], print a JSON-style nested list with {rows} row lists of length {cols}."
    return "After [GRID], print a JSON-style nested list with one list per grid row."


def _render_rows(grid: Grid) -> str:
    return "\n".join(" ".join(row) for row in grid)


def _render_markdown(grid: Grid) -> str:
    return "\n".join("| " + " | ".join(row) + " |" for row in grid)


def _render_json(grid: Grid) -> str:
    return json.dumps(grid)


ANSWER_FORMATS: List[AnswerFormat] = [
    AnswerFormat(name="rows", instruction=_rows_instruction, render=_render_rows, max_new_tokens=192),
    AnswerFormat(name="markdown-table", instruction=_markdown_instruction, render=_render_markdown, max_new_tokens=256),
    AnswerFormat(name="json", instruction=_json_instruction, render=_render_json, max_new_tokens=256),
]


def build_instruction(rows: int, cols: int, include_shape: bool, answer_format: AnswerFormat) -> str:
    parts = [
        "You are given a random walk over a hidden grid of words.",
        "Consecutive words in the walk always come from horizontally or vertically adjacent cells.",
        "Reconstruct one full grid arrangement that is consistent with the walk.",
        "Use each distinct word from the walk exactly once.",
    ]
    if include_shape:
        parts.append(f"The grid has shape {rows}x{cols}.")
    parts.append("Any rotation or reflection is acceptable if it is consistent with the walk.")
    parts.append(answer_format.instruction(rows, cols, include_shape))
    parts.append("Do not add any explanation before or after the grid.")
    return "\n".join(parts)


CONDITIONS: List[ReconstructionCondition] = [
    ReconstructionCondition(
        name=f"{style}-{'shape' if include_shape else 'no-shape'}-{answer_format.name}",
        style=style,
        include_shape=include_shape,
        answer_format=answer_format,
        max_new_tokens=answer_format.max_new_tokens,
    )
    for style in ["direct", "one-shot"]
    for include_shape in [True, False]
    for answer_format in ANSWER_FORMATS
]

CONDITION_NAMES = [c.name for c in CONDITIONS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate grid reconstruction from a random walk.")
    p.add_argument("--model-id", default="google/gemma-3-4b-it")
    p.add_argument("--grid-size", type=str, default="4x4", help="Grid dimensions as RxC (e.g. 4x4, 5x5, 16x1).")
    p.add_argument("--walk-len", type=int, default=200)
    p.add_argument("--num-trials", type=int, default=100)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--conditions", nargs="+", default=CONDITION_NAMES, choices=CONDITION_NAMES)
    p.add_argument("--show-examples", type=int, default=3)
    p.add_argument("--max-walk-attempts", type=int, default=256,
                   help="Retries used to generate a walk that visits every cell.")
    return p.parse_args()


def grid_from_mapping(pos2word: Dict[Tuple[int, int], str], rows: int, cols: int) -> Grid:
    return [[pos2word[(i, j)] for j in range(cols)] for i in range(rows)]


def generate_covering_walk(
    rng: random.Random, pos2word: Dict[Tuple[int, int], str], rows: int, cols: int, walk_len: int, max_attempts: int
) -> List[str]:
    for _ in range(max_attempts):
        pos = rng.choice(list(pos2word))
        walk = [pos2word[pos]]
        for _ in range(walk_len - 1):
            pos = rng.choice(neighbors(pos[0], pos[1], rows, cols))
            walk.append(pos2word[pos])
        if len(set(walk)) == rows * cols:
            return walk
    raise ValueError(
        f"Failed to generate a walk that visits every cell for a {rows}x{cols} grid with walk_len={walk_len}. "
        "Increase --walk-len or --max-walk-attempts."
    )


def sample_trial(seed: int, words: Sequence[str], rows: int, cols: int, walk_len: int, max_walk_attempts: int) -> Tuple[Grid, List[str], List[str]]:
    rng = random.Random(seed)
    sampled = rng.sample(list(words), rows * cols)
    pos2word = {(i, j): sampled[i * cols + j] for i in range(rows) for j in range(cols)}
    grid = grid_from_mapping(pos2word, rows, cols)
    walk = generate_covering_walk(rng, pos2word, rows, cols, walk_len, max_walk_attempts)
    return grid, walk, sampled


def render_answer(grid: Grid, answer_format: AnswerFormat) -> str:
    return f"[GRID]\n{answer_format.render(grid)}"


def parse_grid(text: str, allowed_words: Sequence[str], rows: int, cols: int) -> Optional[Grid]:
    allowed = set(allowed_words)

    def extract(segment: str) -> List[str]:
        return [tok.lower() for tok in WORD_RE.findall(segment) if tok.lower() in allowed]

    tagged = GRID_TAG_RE.search(text)
    tokens = extract(tagged.group("body")) if tagged else []
    if len(tokens) < rows * cols:
        tokens = extract(text)
    if len(tokens) < rows * cols:
        return None
    flat = tokens[: rows * cols]
    return [flat[i * cols:(i + 1) * cols] for i in range(rows)]


def is_permutation(grid: Grid, expected_words: Sequence[str]) -> bool:
    flat = [word for row in grid for word in row]
    return sorted(flat) == sorted(expected_words)


def walk_consistency(grid: Grid, walk: Sequence[str]) -> bool:
    rows, cols = len(grid), len(grid[0])
    flat = [word for row in grid for word in row]
    if len(flat) != len(set(flat)):
        return False
    word2pos = {word: (i, j) for i, row in enumerate(grid) for j, word in enumerate(row)}
    return all(word2pos[b] in neighbors(*word2pos[a], rows, cols) for a, b in zip(walk, walk[1:]))


def transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def rotate90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])]


def rotate180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def rotate270(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)][::-1]


def flip_horizontal(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def flip_vertical(grid: Grid) -> Grid:
    return grid[::-1]


def symmetry_variants(grid: Grid) -> List[Grid]:
    candidates = [
        grid,
        flip_horizontal(grid),
        flip_vertical(grid),
        rotate180(grid),
    ]
    if len(grid) == len(grid[0]):
        candidates.extend([
            rotate90(grid),
            rotate270(grid),
            transpose(grid),
            flip_horizontal(transpose(grid)),
        ])

    unique: List[Grid] = []
    seen = set()
    for candidate in candidates:
        key = tuple(tuple(row) for row in candidate)
        if key not in seen:
            seen.add(key)
            unique.append([list(row) for row in key])
    return unique


def cell_accuracy(pred: Grid, target: Grid) -> float:
    rows, cols = len(target), len(target[0])
    correct = sum(pred[i][j] == target[i][j] for i in range(rows) for j in range(cols))
    return correct / (rows * cols)


def evaluate_trial(
    model, processor,
    *,
    seed: int,
    words: Sequence[str],
    rows: int,
    cols: int,
    walk_len: int,
    conditions: List[ReconstructionCondition],
    max_walk_attempts: int,
) -> GridTrialResult:
    target_grid, walk, sampled = sample_trial(seed, words, rows, cols, walk_len, max_walk_attempts)
    example_grid, example_walk, _ = sample_trial(seed + EXAMPLE_SEED_OFFSET, words, rows, cols, walk_len, max_walk_attempts)
    target_symmetries = symmetry_variants(target_grid)

    condition_results: Dict[str, GridConditionResult] = {}
    for cond in conditions:
        raw = run_generation(
            model,
            processor,
            cond.build_messages(walk, example_walk, example_grid, rows, cols),
            max_new_tokens=cond.max_new_tokens,
        )
        parsed_grid = parse_grid(raw, sampled, rows, cols)
        permutation_ok = parsed_grid is not None and is_permutation(parsed_grid, sampled)
        consistent = parsed_grid is not None and permutation_ok and walk_consistency(parsed_grid, walk)
        exact_match = parsed_grid == target_grid if parsed_grid is not None else False
        symmetry_match = parsed_grid in target_symmetries if parsed_grid is not None else False
        best_cell_acc = (
            max(cell_accuracy(parsed_grid, variant) for variant in target_symmetries)
            if parsed_grid is not None else 0.0
        )
        condition_results[cond.name] = GridConditionResult(
            raw=raw,
            parsed_grid=parsed_grid,
            parse_ok=parsed_grid is not None,
            permutation_ok=permutation_ok,
            walk_consistent=consistent,
            symmetry_match=symmetry_match,
            exact_match=exact_match,
            best_symmetry_cell_accuracy=best_cell_acc,
        )

    return GridTrialResult(seed=seed, walk=walk, target_grid=target_grid, conditions=condition_results)


def main() -> None:
    args = parse_args()

    model, processor = load_model(args.model_id)

    parts = args.grid_size.split("x")
    rows, cols = int(parts[0]), int(parts[1])
    grid_cells = rows * cols

    if len(WORDS) < grid_cells:
        raise ValueError(f"Need at least {grid_cells} words for {rows}x{cols} grid, got {len(WORDS)}.")
    validate_words(WORDS, processor)

    active_conditions = [c for c in CONDITIONS if c.name in args.conditions]
    results = [
        evaluate_trial(
            model,
            processor,
            seed=seed,
            words=WORDS,
            rows=rows,
            cols=cols,
            walk_len=args.walk_len,
            conditions=active_conditions,
            max_walk_attempts=args.max_walk_attempts,
        )
        for seed in trange(args.base_seed, args.base_seed + args.num_trials, desc="Evaluating trials")
    ]

    print("=== Configuration ===")
    print(f"Model: {args.model_id}")
    print(f"Grid size: {rows}x{cols}")
    print(f"Walk length: {args.walk_len}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Base seed: {args.base_seed}")
    print(f"Conditions: {[c.name for c in active_conditions]}")
    print()

    print("=== Summary ===")
    summary_rows = []
    n = len(results)
    for cond in active_conditions:
        subset = [trial.conditions[cond.name] for trial in results]
        summary_rows.append({
            "condition": cond.name,
            "parse rate": f"{sum(r.parse_ok for r in subset) / n:7.2%}",
            "permutation": f"{sum(r.permutation_ok for r in subset) / n:7.2%}",
            "walk consistent": f"{sum(r.walk_consistent for r in subset) / n:7.2%}",
            "symmetry match": f"{sum(r.symmetry_match for r in subset) / n:7.2%}",
            "avg best cell acc": f"{sum(r.best_symmetry_cell_accuracy for r in subset) / n:7.2%}",
        })
    print_table(summary_rows)
    print()

    print("=== Example trials ===")
    for cond in active_conditions:
        print(f"--- {cond.name} ---")
        rows_out = []
        for trial in results[:args.show_examples]:
            result = trial.conditions[cond.name]
            rows_out.append({
                "seed": trial.seed,
                "parse": "✓" if result.parse_ok else "✗",
                "perm": "✓" if result.permutation_ok else "✗",
                "walk": "✓" if result.walk_consistent else "✗",
                "sym": "✓" if result.symmetry_match else "✗",
                "cell acc": f"{result.best_symmetry_cell_accuracy:6.2%}",
                "raw": result.raw.replace("\n", "\\n")[:80],
            })
        if rows_out:
            print_table(rows_out)
        print()

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"exp3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "experiment3",
            "args": vars(args),
            "results": [
                {
                    "seed": result.seed,
                    "walk": result.walk,
                    "target_grid": result.target_grid,
                    "conditions": {name: vars(cond_result) for name, cond_result in result.conditions.items()},
                }
                for result in results
            ],
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
