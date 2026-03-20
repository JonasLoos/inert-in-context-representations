"""Shared utilities for in-context representation learning experiments."""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

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

ANSWER_TAG_RE = re.compile(r"\[ANSWER\]\s*(?P<word>[A-Za-z]+)")
FIRST_WORD_RE = re.compile(r"(?P<word>[A-Za-z]+)")

# A message is {"role": "user"|"assistant"|"system", "content": str}.
Message = Dict[str, str]


def neighbors(i: int, j: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    """Return valid grid neighbors of cell (i, j)."""
    out = []
    if i > 0: out.append((i - 1, j))
    if i < rows - 1: out.append((i + 1, j))
    if j > 0: out.append((i, j - 1))
    if j < cols - 1: out.append((i, j + 1))
    return out


def parse_answer(text: str, words: Sequence[str]) -> Optional[str]:
    """Extract a valid word from model output, checking for [ANSWER] tag first."""
    m = ANSWER_TAG_RE.search(text) or FIRST_WORD_RE.search(text)
    if m is None:
        return None
    word = m.group("word").lower()
    return word if word in words else None


def run_generation(
    model, processor, messages: List[Message], *, prefill: Optional[str] = None, max_new_tokens: int
) -> str:
    """Run greedy decoding, optionally with an assistant prefill."""
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prefill is not None:
        prompt += prefill
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(gen_ids, skip_special_tokens=True)


def print_table(rows: List[Dict[str, str]]) -> None:
    """Print a simple fixed-width text table."""
    cols = list(rows[0].keys())
    widths = [max(len(str(c)), max(len(str(row[c])) for row in rows)) for c in cols]
    header = "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[c]).ljust(w) for c, w in zip(cols, widths)))


def load_model(model_id: str):
    """Load model and processor onto available device."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto"
    ).eval()
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    return model, processor


def validate_words(words: Sequence[str], processor) -> None:
    """Raise if any word in the list is not a single token."""
    tokenizer = processor.tokenizer
    bad = [(w, ids) for w in words if len(ids := tokenizer.encode(w, add_special_tokens=False)) != 1]
    if bad:
        raise ValueError(f"Multi-token words: {', '.join(f'{w}={ids}' for w, ids in bad)}")
