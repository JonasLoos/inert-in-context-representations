# Do Language Models Struggle to Use Representations Learned In-Context?

A partial replication and slight extension of the paper [Language Models Struggle to Use Representations Learned In-Context](https://arxiv.org/abs/2602.04212) (Experiment 1) by Lepori et al. (Google DeepMind).


## Setup

The model is given a random **4×4 (or 16×1 linear) grid** of words drawn from a 42-word single-token vocabulary. A random walk of 200 steps is performed on the grid and the model sees the full walk sequence. It is then asked to predict the **next word** given the last word. We test 12 different prompting strategies with 100 trials each and measure whether the model correctly names a valid neighbor.

**Chance baseline** (guessing uniformly from the full vocabulary):
- 4×4 grid: ~19.75%
- 16×1 linear: ~12.12%



## Conditions

| Condition | Description |
|---|---|
| `instruction` *(paper)* | Walk shown in system prompt, model asked for next word |
| `prefilled` *(paper)* | Response prefilled with the walk sequence |
| `multi-turn` *(new)* | Walk presented as a conversation history |
| `multi-turn-1-example` *(new)* | 1-shot in-context example then multi-turn |
| `multi-turn-2-examples` *(new)* | 2-shot in-context examples then multi-turn |
| `turn-by-turn` *(new)* | Each walk step presented in a separate turn |
| `chain-of-thought` *(new)* | Model asked to reason step-by-step |
| `prefill-with-separator` *(new)* | Prefilled with separator tokens |
| `reflection` *(new)* | Model reflects on the sequence structure first |
| `pair-format` *(new)* | Walk presented as explicit transitions (A → B) |
| `system-message` *(new)* | Walk placed in system message |
| `partial-prefill` *(new)* | Half user message, half prefilled |

## Results

### Gemma-3-4B-IT

| Condition | 4×4 accuracy | 4×4 parse rate | 16×1 accuracy | 16×1 parse rate |
|---|---|---|---|---|
| instruction *(paper)* | 36% | 93% | 43% | 94% |
| prefilled *(paper)* | 87% | 100% | 96% | 100% |
| multi-turn | 26% | 100% | 37% | 100% |
| multi-turn-1-example | 83% | 100% | 93% | 99% |
| multi-turn-2-examples | 67% | 100% | 67% | 97% |
| turn-by-turn | 75% | 100% | 92% | 100% |
| chain-of-thought | 33% | 100% | 47% | 99% |
| prefill-with-separator | 30% | 100% | 44% | 99% |
| reflection | 54% | 99% | 63% | 95% |
| pair-format | 95% | 100% | 98% | 100% |
| system-message | 28% | 100% | 34% | 100% |
| partial-prefill | 89% | 100% | 93% | 98% |
| *Chance baseline* | *19.75%* | — | *12.12%* | — |

### Gemma-3-27B-IT

| Condition | 4×4 accuracy | 4×4 parse rate | 16×1 accuracy | 16×1 parse rate |
|---|---|---|---|---|
| instruction *(baseline)* | 38% | 100% | 57% | 100% |
| prefilled | 98% | 100% | 99% | 100% |
| multi-turn | 25% | 100% | 42% | 99% |
| multi-turn-1-example | 95% | 100% | 93% | 100% |
| multi-turn-2-examples | 94% | 100% | 95% | 100% |
| turn-by-turn | 82% | 100% | 95% | 100% |
| chain-of-thought | 72% | 100% | 77% | 99% |
| prefill-with-separator | 17% | 100% | 30% | 100% |
| reflection | 72% | 100% | 83% | 100% |
| pair-format | 100% | 100% | 100% | 100% |
| system-message | 32% | 100% | 44% | 100% |
| partial-prefill | 97% | 100% | 98% | 99% |
| *Chance baseline* | *19.75%* | — | *12.12%* | — |


## Key Findings

... and ideas what these results may mean:

- **The instruction baseline is consistently weak** (26–57%), well below many other conditions, replicating the paper's finding that in-context learned representations remain largely "inert" when the model simply follows an instruction.
- **Prefilling and partial-prefill are very strong** (87–99%), replicating the paper's finding. When the next word prediction task resembles the pre-training next-token prediction setting, performance is high.
- **Multi-turn (bare) performs poorly** (25–42%), often worse than the instruction baseline. Distributing the walk across turns without explicit examples provides no benefit.
- **Few-shot examples help dramatically** (multi-turn +1 example: 83–95%, vs bare multi-turn: 25–42%). A single in-context example showing how to answer is enough to unlock most of the available performance. Adding a second example yields inconsistent gains and can even hurt.
- **Turn-by-turn format works well** (75–95%), especially on the linear grid. It frames the task as the model actively predicting each step, which makes the final prediction a natural continuation.
- **Model scale helps, but doesn't solve the core problem.** Gemma-3-27B shows clear gains on chain-of-thought (+39pp on 4×4) and reflection (+18pp), and reaches 100% on pair-format. But the instruction baseline only improves by ~2–14pp, confirming the inertness phenomenon persists at scale.
- **Pair-format is trivial** (95–100%), suggesting that making relational transitions *explicit* (A → B) may allow the model to directly read of possible continuations instead of inferring graph structure.

**Overall, while this confirms that models sometimes struggle to use representations learned in-context, this is not always the case and heavily depends on prompting/formatting.**

**Limitations**:
- We don't test in-context representation quality, but only task performance.
- We only evaluate two models from the same family on 100 trials per condition and a walk length of 200, which may not transfer to other settings.
- We don't further investigate why the models do better or worse on some tasks, the interpretations are just ideas.
