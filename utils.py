import argparse
from dataclasses import dataclass, fields



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
    base_seed: int = 42
    show_examples: int = 5

    @classmethod
    def parse_args(cls):
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


def generate_grid(size: str):
    ...


def call_model(messages: list[dict[str, str]], *, prefill: str | None = None, max_new_tokens: int):
    ...


REGISTERED_EXPERIMENTS = set()


def experiment(func):
    REGISTERED_EXPERIMENTS.add(func)
    return func


def run_experiments(args: CmdLineArgs):
    ...
