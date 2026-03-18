# pip install -U "torch>=2.4" "transformers>=4.51.3" accelerate sentencepiece

import os
import random
import re
from typing import Dict, List, Tuple

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL_ID = "google/gemma-3-4b-it"

# Paper-inspired minimal setup:
# - model: gemma-3-4b-it
# - topology: 4x4 grid
# - context length: 200 for Gemma-4b / Instruction / 16-Grid
GRID_SIZE = 4
WALK_LEN = 200
SEED = 7

WORDS = [
    "toy", "ink", "city", "air",
    "wing", "cat", "jam", "zoo",
    "baby", "rock", "leaf", "ship",
    "lamp", "fork", "star", "bell",
]

random.seed(SEED)
torch.manual_seed(SEED)


def make_grid(words: List[str], n: int) -> Tuple[Dict[Tuple[int, int], str], Dict[str, Tuple[int, int]]]:
    shuffled = words[:]
    random.shuffle(shuffled)
    pos2word = {}
    word2pos = {}
    k = 0
    for i in range(n):
        for j in range(n):
            pos2word[(i, j)] = shuffled[k]
            word2pos[shuffled[k]] = (i, j)
            k += 1
    return pos2word, word2pos


def neighbors(i: int, j: int, n: int) -> List[Tuple[int, int]]:
    out = []
    if i > 0:
        out.append((i - 1, j))
    if i < n - 1:
        out.append((i + 1, j))
    if j > 0:
        out.append((i, j - 1))
    if j < n - 1:
        out.append((i, j + 1))
    return out


def random_walk(pos2word: Dict[Tuple[int, int], str], n: int, steps: int) -> List[str]:
    pos = random.choice(list(pos2word.keys()))
    walk = [pos2word[pos]]
    for _ in range(steps - 1):
        pos = random.choice(neighbors(pos[0], pos[1], n))
        walk.append(pos2word[pos])
    return walk


def valid_next_tokens(last_word: str, word2pos: Dict[str, Tuple[int, int]], pos2word: Dict[Tuple[int, int], str], n: int) -> List[str]:
    i, j = word2pos[last_word]
    return [pos2word[p] for p in neighbors(i, j, n)]


def build_instruction_prompt(walk: List[str]) -> str:
    # Matches the Figure 1 "Instruction" style closely.
    return (
        "Your job is to predict the next word in a sequence of words. "
        "Generate the token [ANSWER], then generate the next word in the sequence.\n"
        f"[SEQUENCE] {' '.join(walk)}"
    )


def build_prefilled_prompt(walk: List[str]) -> str:
    # Matches the Figure 1 "Prefilled" style closely.
    return f"[SEQUENCE] {' '.join(walk)}"


def extract_first_vocab_match(text: str, vocab: List[str]) -> str | None:
    toks = re.findall(r"[A-Za-z]+", text.lower())
    vocab_set = set(vocab)
    for tok in toks:
        if tok in vocab_set:
            return tok
    return None


def run_chat(model, processor, user_text: str, prefill: str | None = None, max_new_tokens: int = 8) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(prefill is None),
    )
    if prefill is not None:
        prompt += prefill

    inputs = processor(
        text=prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(gen_ids, skip_special_tokens=True)


def main():
    token = os.environ.get("HF_TOKEN")
    if token is None:
        print("Set HF_TOKEN in your environment after accepting the Gemma license on Hugging Face.")
        return

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        token=token,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)

    pos2word, word2pos = make_grid(WORDS, GRID_SIZE)
    walk = random_walk(pos2word, GRID_SIZE, WALK_LEN)

    last_word = walk[-1]
    gold_valid = valid_next_tokens(last_word, word2pos, pos2word, GRID_SIZE)

    print("=== Hidden topology (for us, not the model) ===")
    for i in range(GRID_SIZE):
        print([pos2word[(i, j)] for j in range(GRID_SIZE)])
    print()

    print("Last walk token:", last_word)
    print("Valid next tokens:", gold_valid)
    print()

    # Instruction condition: model must emit extra stuff before the next word.
    instr_prompt = build_instruction_prompt(walk)
    instr_raw = run_chat(model, processor, instr_prompt, prefill=None, max_new_tokens=12)
    instr_guess = extract_first_vocab_match(instr_raw, WORDS)

    print("=== Instruction condition ===")
    print("Raw generation:", repr(instr_raw))
    print("Parsed guess:", instr_guess)
    print("Is valid next token?:", instr_guess in gold_valid if instr_guess else False)
    print()

    # Prefilled condition: the walk is placed in the assistant prefix, so the model continues immediately.
    prefilled_user = "Continue the sequence of words."
    prefilled_prefix = build_prefilled_prompt(walk)
    prefilled_raw = run_chat(model, processor, prefilled_user, prefill=prefilled_prefix, max_new_tokens=4)
    prefilled_guess = extract_first_vocab_match(prefilled_raw, WORDS)

    print("=== Prefilled condition ===")
    print("Raw generation:", repr(prefilled_raw))
    print("Parsed guess:", prefilled_guess)
    print("Is valid next token?:", prefilled_guess in gold_valid if prefilled_guess else False)


if __name__ == "__main__":
    main()