import argparse
from datetime import datetime
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple
from pathlib import Path
import json

import torch
from tqdm import trange
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

WORDS = [
    "toy", "ink", "city", "air",
    "wing", "cat", "jam", "zoo",
    "baby", "rock", "leaf", "ship",
    "lamp", "fork", "star", "bell",
    "dog", "hat", "cup", "sun",
    "box", "pen", "key", "bag",
    "arm", "leg", "eye", "ear",
    "bed", "car", "map", "egg",
    "ice", "mud", "fox", "net",
    "rod", "ant", "bee", "dew",
    "oak", "nut",
]

ANSWER_RE = re.compile(r"^\s*(?:.*\[ANSWER\]\s*)?(?P<word>[A-Za-z]+)\b")

# A message is {"role": "user"|"assistant"|"system", "content": str}.
Message = Dict[str, str]


@dataclass
class Condition:
    """A prompting setup defined by an arbitrary message history plus an optional prefill."""
    name: str
    messages: Callable[[List[str]], List[Message]]
    prefill: Callable[[str], str|None] = lambda walk: None
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
    # Hypothesis: the instruction condition fails because of special tokens between
    # the walk and the prediction, not because the representations are inert. Prefilling
    # "[ANSWER]" in the assistant response eliminates most of the gap while keeping the
    # walk in the user message.
    Condition(
        name="instruction-answer-prefill",
        messages=lambda walk: [{"role": "user", "content": (
            "Your job is to predict the next word in a sequence of words. "
            "Generate the token [ANSWER], then generate the next word in the sequence.\n"
            f"[SEQUENCE] {' '.join(walk)}"
        )}],
        prefill=lambda walk: "[ANSWER] ",
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
        max_new_tokens=200,
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
            {"role": "user", "content": (
                f"Here is a sequence of words:\n[SEQUENCE] {' '.join(walk)}\n"
                "What is the last word in the sequence? Answer in format `[ANSWER] <word>`."
            )},
            {"role": "assistant", "content": f"[ANSWER] {walk[-1]}"},
            {"role": "user", "content": "Predict what word comes next in the sequence. Answer in format `[ANSWER] <word>`."},
        ],
        prefill=lambda walk: None,
        max_new_tokens=12,
    ),
    # Hypothesis: presenting transitions as explicit pairs ("A -> B") rather than a flat
    # sequence makes the relational structure more salient and easier to deploy.
    Condition(
        name="pair-format",
        messages=lambda walk: [{"role": "user", "content": (
            "Below are word-to-word transitions from a sequence. Predict what comes after the last word. "
            "Write [ANSWER] then the word.\n"
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
                "The system message contains a sequence of words that follow a pattern. "
                "Predict the next word. Write [ANSWER] then the word."
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
    p.add_argument("--num-trials", type=int, default=50, help="Number of word assignments to evaluate.")  # paper uses 1000
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--show-examples", type=int, default=5)
    p.add_argument("--quiet", action="store_true", help="Print only summary statistics.")
    return p.parse_args()


def neighbors(i: int, j: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    out = []
    if i > 0: out.append((i - 1, j))
    if i < rows - 1: out.append((i + 1, j))
    if j > 0: out.append((i, j - 1))
    if j < cols - 1: out.append((i, j + 1))
    return out


def parse_answer(text: str, words: Sequence[str]) -> str | None:
    m = ANSWER_RE.match(text)
    if m is None:
        return None
    word = m.group("word").lower()
    return word if word in words else None


def run_generation(
    model, processor, messages: List[Message], *, prefill: str | None = None, max_new_tokens: int
) -> str:
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prefill is not None:
        prompt += prefill
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(gen_ids, skip_special_tokens=True)


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


def print_table(rows: List[Dict[str, str]]) -> None:
    cols = list(rows[0].keys())
    widths = [max(len(str(c)), max(len(str(row[c])) for row in rows)) for c in cols]
    header = "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[c]).ljust(w) for c, w in zip(cols, widths)))


def main() -> None:
    args = parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto"
    ).eval()
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)

    parts = args.grid_size.split("x")
    grid_rows, grid_cols = int(parts[0]), int(parts[1])
    grid_cells = grid_rows * grid_cols

    if len(WORDS) < grid_cells:
        raise ValueError(f"Need at least {grid_cells} words for {grid_rows}x{grid_cols} grid, got {len(WORDS)}.")
    tokenizer = processor.tokenizer
    bad = [(w, ids) for w in WORDS if len(ids := tokenizer.encode(w, add_special_tokens=False)) != 1]
    if bad:
        raise ValueError(f"Multi-token words: {', '.join(f'{w}={ids}' for w, ids in bad)}")

    results = [
        evaluate_trial(
            model, processor,
            seed=s, words=WORDS, grid_rows=grid_rows, grid_cols=grid_cols, walk_len=args.walk_len,
            conditions=CONDITIONS,
        )
        for s in trange(args.base_seed, args.base_seed + args.num_trials, desc="Evaluating trials")
    ]

    n, vocab_size = len(results), len(WORDS)
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
    print(f"Uniform vocab chance accuracy: {sum(len(r.valid_next_tokens) / vocab_size for r in results) / n:7.2%}")
    print(f"Average number of valid next tokens: {sum(len(r.valid_next_tokens) for r in results) / n}")
    print()
    print_table([{
        "condition": name,
        "accuracy": f"{sum(r.conditions[name].ok for r in results) / n:7.2%}",
        "parse rate": f"{sum(r.conditions[name].guess is not None for r in results) / n:7.2%}",
    } for name in cond_names])
    print()

    if not args.quiet:
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

    # save results to json file
    Path("results").mkdir(exist_ok=True)
    with open(Path("results") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        data = {
            "args": vars(args),
            "results": [
                {**vars(r), "conditions": {k: vars(v) for k, v in r.conditions.items()}}
                for r in results
            ],
        }
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
