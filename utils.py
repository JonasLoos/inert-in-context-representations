import argparse
from dataclasses import dataclass, fields
import random


# Single-token words used to label graph states.
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


@dataclass(frozen=True)
class CmdLineArgs:
    model_id: str
    grid_size: str
    walk_len: int
    num_trials: int
    base_seed: int = 42  # With trial_index, walk RNG uses base_seed + trial_index.
    trial_index: int = 0
    show_examples: int = 5

    @classmethod
    def parse(cls):
        proto = cls()
        argparser = argparse.ArgumentParser()
        for field in fields(cls):
            argparser.add_argument(
                f"--{field.name}",
                type=field.type,
                default=getattr(proto, field.name),
            )
        ns = argparser.parse_args()
        kwargs = {f.name: getattr(ns, f.name) for f in fields(cls)}
        return cls(**kwargs)


def neighbors(i: int, j: int, rows: int, cols: int) -> list[tuple[int, int]]:
    """Return valid grid neighbors of cell (i, j)."""
    out: list[tuple[int, int]] = []
    if i > 0:
        out.append((i - 1, j))
    if i < rows - 1:
        out.append((i + 1, j))
    if j > 0:
        out.append((i, j - 1))
    if j < cols - 1:
        out.append((i, j + 1))
    return out


class Grid:
    def __init__(self, size: str):
        parts = size.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid grid size {size!r}, expected RxC (e.g. 4x4).")
        self.rows, self.cols = int(parts[0]), int(parts[1])
        if self.rows < 1 or self.cols < 1:
            raise ValueError(f"Invalid grid dimensions {self.rows}x{self.cols}.")
        cells = self.rows * self.cols
        if len(WORDS) < self.cells:
            raise ValueError(
                f"Need at least {self.cells} words for {self.rows}x{self.cols} grid, got {len(WORDS)}."
            )
        self.words = random.sample(WORDS, self.cells)
        self.pos2word = {(i, j): self.words[i * self.cols + j] for i in range(self.rows) for j in range(self.cols)}
        self.word2pos = {w: p for p, w in self.pos2word.items()}
        self.walk = []

    def walk(self, walk_len: int):
        for _ in range(walk_len):
            pos = random.choice(list(self.pos2word))
            self.walk.append(self.pos2word[pos])
            self.last = self.pos2word[pos]
            self.valid_next = [self.pos2word[p] for p in neighbors(*self.word2pos[self.last], self.rows, self.cols)]
        self.last = self.walk[-1]
        self.valid_next = [self.pos2word[p] for p in neighbors(*self.word2pos[self.last], self.rows, self.cols)]
        return self


def call_model(messages: list[dict[str, str]], *, prefill: str | None = None, max_new_tokens: int):
    ...
