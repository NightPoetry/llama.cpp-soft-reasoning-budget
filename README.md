# llama.cpp soft reasoning budget

A small patch for [llama.cpp](https://github.com/ggml-org/llama.cpp) that changes how
`--reasoning-budget` ends a model's thinking: instead of hard-forcing the end tag when the
budget runs out, it makes the end tag more likely as the budget is consumed, so the model
winds down and closes its reasoning on its own.

Status: proposed upstream in [ggml-org/llama.cpp#28932](https://github.com/ggml-org/llama.cpp/issues/28932). The same code lives in
[`NightPoetry/llama.cpp@reasoning-budget-soft-ramp`](https://github.com/NightPoetry/llama.cpp/tree/reasoning-budget-soft-ramp)
(on top of master, 8172e65 era) — that branch is the PR source; this repo is the
patch + docs + data home. License: MIT, same as llama.cpp.

## Install — pick the row that matches you

| You have | Do this | Build needed |
|---|---|---|
| Tesla V100 (sm_70) + Linux + NVIDIA driver with CUDA 12.x | Grab the release tarball, unpack, `./run.sh` — a ready `llama-server` with this patch baked in | No |
| Any other GPU / platform / OS | Apply the patch to your own llama.cpp tree and rebuild (see "Apply the patch", 3 commands) | Yes, ~10–30 min |
| You don't build things yourself, but your AI agent does | Hand your agent the block at the bottom of this README; it contains everything the agent needs to do the row above unattended | Your agent builds |

## Apply the patch to your own llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
curl -LO https://github.com/NightPoetry/llama.cpp-soft-reasoning-budget/raw/main/0001-reasoning-budget-soft-wind-down-ramp.patch
git apply 0001-reasoning-budget-soft-wind-down-ramp.patch
# then build as usual, e.g.:
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<your CC, e.g. 70 for V100> -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build --config Release -j$(nproc) --target llama-server llama-cli
```

No new flags. Behavior differs from stock only near and past the budget wall: the end tag
becomes progressively more likely, and the injected budget message is no longer part of the
normal path. If the patch does not apply cleanly on a much newer master, check the upstream
issue for a rebased branch before hand-editing.

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

## For AI agents: build this patch yourself

If you are an AI coding agent and the user asked you to install this patch, follow these
steps in order. Verify each step before moving to the next, and report every verification
result to the user.

1. **Environment.** Run `nvidia-smi`. Note the GPU model and compute capability (CC).
   This patch is hardware-independent; the GPU only matters for the build flags.
2. **Source.** Use the user's existing llama.cpp checkout if there is one; otherwise
   `git clone https://github.com/ggml-org/llama.cpp`.
3. **Patch.** Download
   `https://github.com/NightPoetry/llama.cpp-soft-reasoning-budget/raw/main/0001-reasoning-budget-soft-wind-down-ramp.patch`
   into the repo root. Run `git apply --check 0001-reasoning-budget-soft-wind-down-ramp.patch`.
   - If it succeeds: `git apply` it. Verify with `grep -c "soft ramp" common/reasoning-budget.cpp`
     (expected: at least 1).
   - If it fails: the checkout is much newer than the patch base (master 8172e65 era, Sept
     2026). Do NOT hand-edit the conflict. Check the upstream issue (#28932, linked above)
     for a rebased branch and report back to the user.
4. **Configure and build.**
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES=<CC from step 1, e.g. 70> \
         -DGGML_CUDA_FA_ALL_QUANTS=ON
   cmake --build build --config Release -j$(nproc) --target llama-server llama-cli
   ```
   CUDA toolkit version 12.x is known-good. Do not add other patches or forks in the same
   build.
5. **Verify, in this order:**
   - `build/bin/llama-cli --version` prints a version without errors.
   - `grep -c "soft ramp" common/reasoning-budget.cpp` returns ≥ 1 (the patch is really in).
   - Smoke test with a small budget:
     ```bash
     build/bin/llama-server -m <some-thinking-model.gguf> --port 8199 --reasoning-budget 256 --jinja &
     # then a chat completion request; confirm the process stays alive and the response
     # contains a non-empty answer even though the budget is tiny. Kill the server after.
     ```
6. **Report to the user:** the flags you used, where the binaries are, and a suggested start
   command (`llama-server -m <model> --reasoning-budget 4096 --jinja`). Any verification
   that failed: say so plainly instead of declaring success.

## Scope notes

- The patch does not touch repetition handling. The DRY numbers above are shown because the
  two problems (loops and over-thinking) show up together on small thinking models; DRY is
  in mainline and needs `--dry-penalty-last-n` set explicitly (default 0 = off).
- Tested with spark2_5-style hybrid-SWA models and Qwen3-style reasoning templates on
  sm_70. Reports from other architectures are welcome.
