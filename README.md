# llama.cpp soft reasoning budget

A small patch for [llama.cpp](https://github.com/ggml-org/llama.cpp) that changes how
`--reasoning-budget` ends a model's thinking: instead of hard-forcing the end tag when the
budget runs out, it makes the end tag more likely as the budget is consumed, so the model
winds down and closes its reasoning on its own.

Status: proposed upstream in [ggml-org/llama.cpp#28932](https://github.com/ggml-org/llama.cpp/issues/28932). The same code lives in
[`NightPoetry/llama.cpp@reasoning-budget-soft-ramp`](https://github.com/NightPoetry/llama.cpp/tree/reasoning-budget-soft-ramp)
(on top of master, 8172e65 era) — that branch is the PR source; this repo is the
patch + docs + data home. License: MIT, same as llama.cpp.

## The problem

`--reasoning-budget N` caps a thinking model's reasoning. When the cap is hit today, the
sampler enters a FORCING state: it sets every logit except the forced end sequence to -inf
and injects the end tag mid-thought. Two failure modes follow on small thinking models:

1. The model is interrupted mid-sentence. Some models then produce a worse answer than if
   they had been allowed to wrap up.
2. The budget is spent entirely inside the reasoning block, the block is force-closed, and
   the answer comes out empty.

## What the patch changes

Three things, all in `common/reasoning-budget.cpp` (~40 lines):

1. **Soft wind-down ramp.** In the last quarter of the budget, each sampling step adds a
   quadratic logit bias to the end sequence's first token. The bias grows as the remaining
   count drops (and keeps growing past zero), so the model closes the reasoning block by
   itself instead of being cut off. FORCING remains in the code as a backstop path but is no
   longer entered at zero.
2. **Anti-empty window.** After the reasoning block closes, EOS is suppressed for the next
   64 tokens, so the model writes the answer body instead of stopping with an empty answer.
3. **Multi-block re-arm** behavior is preserved: a new `<think>` start tag resets the
   budget window.

## Measured behavior

Repetition-inducing tasks (30-item list + 800-word story), character 12-gram repeat rate,
Spark-X2.5-4B-Q8 on a 16GB V100 (sm_70):

| config | merged repeat rate | worst single-gram dup |
|---|---|---|
| baseline (temp 0.6 / top_k 20 / rep_penalty 1.0) | 0.271 | 41 (hard loop) |
| + DRY (`--dry-penalty-last-n 1024 --dry-multiplier 0.8 --dry-base 1.75`) | 0.085 | 2 |
| + DRY + `--reasoning-budget 4096` (this patch) | **0.039** | 3 |

Budget wall behavior with this patch (budget 4096, thinking on, same over-thinking-prone
prompt, three runs): run 1 hit the client max_tokens cap before the budget; run 2 closed on
its own at 1470 tokens; run 3 stopped right at the budget line and produced a complete,
correct answer. The hard-injected budget message was never observed.

Same prompt, budget 4000, without the patch: still digit-by-digit converting at the wall,
no answer.

## How to use

```bash
git am 0001-*.patch        # or: git apply the .diff
# build as usual; flags unchanged:
llama-server -m model.gguf --reasoning-budget 4096 --jinja ...
```

No new flags. Behavior differs from stock only near and past the budget wall: the end tag
becomes progressively more likely, and the injected budget message is no longer part of the
normal path.

If you prefer an opt-in switch, the ramp is one `if` — gating it on a
`--reasoning-budget-soft` flag is a two-line change; feedback welcome in the upstream thread.

## Scope notes

- The patch does not touch repetition handling. The DRY numbers above are shown because the
  two problems (loops and over-thinking) show up together on small thinking models; DRY is
  in mainline and needs `--dry-penalty-last-n` set explicitly (default 0 = off).
- Tested with spark2_5-style hybrid-SWA models and Qwen3-style reasoning templates on
  sm_70. Reports from other architectures are welcome.
