# Weight-space programming with small editors

Can **Qwen3.5-4B** and **Qwen3.5-9B**, working as agents in a Python sandbox, edit the weights of **Qwen3.5-0.8B** directly,
with no gradients and no training, so that it scores higher on held-out math? This repo answers that question and
compares the editors against scripted baselines (non-LLM edits) of the same size.

This replicates the setup of MIT CSAIL's
["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming) (Experiment 8) with
small open editors. In MIT's run, GPT-6 Astra raised Qwen3.5-0.8B on MATH-500 from 7.9% to 48.1%, mostly by removing a
"repetition" direction that drove reasoning loops. MIT's numbers are cited here as a reference only and are not rerun.

Results: `results/REPORT.md` (written at the end). Decisions, deviations and failures: [NOTES.md](NOTES.md).

## Research questions
1. Do the 4B and 9B editors beat the unedited model on held-out math?
2. Do they beat a scripted difference-of-means ablation with the same edit size?
3. Does any gain come from fixing loops or from better reasoning? Measured by completion rate and accuracy on completed answers.
4. How do 4B and 9B compare? (The larger editor is always labelled "9B".)

## Layout
```
config.yaml     all hyperparameters: models, sampling (frozen after Milestone 1), budgets, seeds, splits, paths
evaluate.py     data prep and decontamination, grading (math-verify, IFEval), vLLM target engine, held-out evals, KL
harness.py      sandbox: tools, AST and runtime no-gradient guards, budget counter, exact edit diff log and rebuild
editors.py      agent loop for the 4B/9B editors (L0/L1 prompts), editor server, dry run, campaign driver
baselines.py    scripted diff-of-means ablation, same-size random edit, MathNeuro-style scaling, LoRA SFT
analyze.py      paired tests (bootstrap, McNemar), seed aggregation, tables -> results/analysis.json, tables.md
tests/          harness guards and grading tests (pytest)
third_party/    official IFEval checkers (vendored from lm_eval 0.4.9, Apache-2.0)
results/        per-run JSON, edit logs and diffs, held-out evals, REPORT.md
NOTES.md        decisions, deviations, inaccessible sources, failures
assets/         local models and datasets (not committed; see download.sh)
work/           derived files: text-only target export, prepared splits, nltk data (not committed)
```

## Setup
Package management uses [uv](https://docs.astral.sh/uv/), on a uv-managed CPython 3.12.
```bash
uv sync                               # vllm 0.33, torch 2.14 (cu130), transformers 5.20, math-verify, peft, ...
uv run python -c "import nltk; nltk.download('punkt_tab', download_dir='work/nltk_data')"   # IFEval checkers
uv run python evaluate.py prepare     # text-only export of the target + practice / held-out splits + hygiene check
uv run pytest -q tests/
```
`config.yaml: paths.cuda_home` points FlashInfer's JIT at the CUDA 13 toolkit. The system `nvcc` is 11.5, which is too old for the RTX PRO 6000 (SM 120).

## Protocol (matched to MIT's `think35`)
- **Target:** text-only export of Qwen3.5-0.8B, thinking on.
- **Prompt:** user turn = problem + `"\nPlease reason step by step, and put your final answer within \boxed{}."`
- **Sampling:** temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, at most **32,768** new tokens. Each request is seeded by (problem, sample), so every checkpoint sees the same random draws.
- **Samples per problem:** MATH-500 ×4, GSM8K test ×2, IFEval ×2. Score = mean over all samples.
- **Grading:** the last `\boxed{}` answer, checked with math-verify. IFEval uses the strict prompt-level official checkers.
- **"Cut"** = the answer hit max tokens. **"Completed"** = it finished.

## Sandbox (what an editor gets)
- **Tools:**
  - `eval(practice_ids, original=False)`: budgeted, at most 512 practice ids per call. Held-out ids are rejected.
  - `forward(texts, layers)`: free. Returns token log-probs, the residual stream, and MLP inputs/outputs.
  - `get_tensor`, `apply_edit`, `revert`, `list_tensors`.
  - L1 only: `diff_of_means`, `project_out`.
- **Budget per run:** 50 eval calls and 80 Python executions, with a 16 h wall-clock safety stop.
- **No gradients:**
  - An AST check runs before execution and refuses `backward`, `torch.optim`, `torch.autograd`, `torch.func`, `requires_grad`, file I/O and dunders.
  - Code then executes under `torch.no_grad()` with backward, autograd, optimizers and `enable_grad` patched to raise.
  - Both layers are tested.
- **Edit log:** every edit is stored as an exact diff (changed indices + new values). `harness.EditState.rebuild()`
  replays the log onto the base weights and checks a sha256 hash at every step, so any final checkpoint can be rebuilt from
  its log alone.

## Running the milestones
```bash
uv run python evaluate.py run base results/base/heldout --sets math500,ifeval,gsm8k_test   # M1 base model
uv run python baselines.py scripted        # M2 scripted diff-of-means + random same-size edit
uv run pytest -q tests/                    # M3 harness checks
uv run python editors.py dry               # M4 hand-written scripted editor, end to end
uv run python editors.py campaign 4B       # M5 4B x {L0, L1} x 5 seeds (starts the editor server itself)
uv run python editors.py campaign 9B       # M5 9B x {L0, L1} x 5 seeds
uv run python baselines.py mathneuro && uv run python baselines.py lora    # M6 extra baselines
uv run python baselines.py heldout <name> math500,ifeval,gsm8k_test         # held-out eval of a baseline checkpoint
uv run python evaluate.py kl <ckpt_dir> <out.json>                          # KL(base || edit) on wikitext
uv run python analyze.py                   # M7 tables and tests -> results/analysis.json, results/tables.md
```

## Monitoring long evals
Generations hit the 32k cap often, so one MATH-500 pass of the base model is about 58M tokens and takes 2-3 h; decode
speed is limited by memory bandwidth on reading the KV cache.
- **While running:** each eval writes `<set>_partial.jsonl` (one graded sample per line, as it finishes) and prints a progress line
  every minute: `[math500] done/total | acc so far | cut so far | tok/s`.
- **Practice eval calls inside editor runs:** these write `results/runs/<run>/evals/NNN_partial.jsonl`.
- **Caveat:** accuracy so far is biased early on, because short answers finish first; only the final number counts.
