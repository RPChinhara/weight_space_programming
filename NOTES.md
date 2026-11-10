# NOTES: decisions, deviations, inaccessible sources, failures

Dates are 2026-11-10 unless noted.

## Sources
- MIT post (https://rikiyat.github.io/blog/weight-space-programming): fetched.
- MIT logs (github.com/RikiyaT/lm_understand-logs, `exp8_math/`): public. I read README, `code/evaluate.py`,
  `code/profiles.py`, `code/mathqa.py`, `code/practice35.py`, `runs/p50c/final_answer.md`, `notices/*`, and the base model's
  chat template, config and generation config. Copies are in the session scratchpad and are not committed.
- Not available from MIT: the agent harness code (their README says it is not released) and the per-run work folders
  (activations). Their Experiment 8 prompts are in `runs/*/turns/t001/prompt.txt`; I did not copy them. My task prompt is
  my own (`editors.task_message`).

## Evaluation protocol: matched to MIT "think35" (profiles.py `qwen35think`, evaluate.py `SAMPLING_THINK35`)
- Target: text-only export of Qwen3.5-0.8B (architecture `Qwen3_5ForCausalLM`; tensors `model.language_model.*` copied
  bit-exact from `assets/models/Qwen3.5-0.8B`; the visual tower and MTP head are dropped). This is the same export MIT
  evaluated (their `base_model/Qwen3.5-0.8B-text/config.json`). The chat template is byte-identical to MIT's.
- Thinking on. User turn = problem + "\n" + "Please reason step by step, and put your final answer within \boxed{}."
- temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, max 32,768 new tokens. min_p is not set by MIT, so it is
  0 (the vLLM default).
- Samples per problem: MATH-500 4, GSM8K test 2, IFEval 2. Score = mean over all samples.
- Grading: the last `\boxed{}` of the text after the last `</think>`, or of the whole text if there is none (as MIT's
  `mathqa.extract`), checked with math-verify 0.9.0 `verify(parse('\boxed{gold}'), parse('\boxed{pred}'))`.
  IFEval: strict prompt-level, using the official checkers (vendored from lm_eval 0.4.9 into `third_party/ifeval`) on the
  answer after `</think>`. The answer is empty if thinking never closed, as in MIT.
- Seeds. MIT used one engine seed (0). I seed each request with hash(seed 0, problem id, sample index), so every checkpoint
  gets the same random draws on the same problem. That gives paired comparisons with less noise (common random numbers).
  This is a deliberate small deviation in the RNG scheme only; the sampling distribution is unchanged.
- Edit size: the text model has 752,393,024 weights. MIT's "0.016%" for 131,072 weights divides by the full 0.8B release
  (vision tower included). I report % of the 752M text weights; 131,072 entries = 0.0174% on this basis.

## Data (Milestone 1 hygiene check): `results/data_hygiene.json`
- Practice: GSM8K main/train 7,469 (7,473 minus 4 with a 13-word overlap with held-out; same count as MIT),
  MATH train 7,407 (7,500 minus 91 with 13-word overlap and 2 with no boxed answer; MIT kept 7,275 because they also
  decontaminated against AMC23/Minerva/OlympiadBench/AIME24/25), AIME 90.
- Exact-match overlap of MATH-500 with MATH train: 0. With the AIME practice set: 0. Normalised-text overlap with any held-out
  problem: 0. I also dropped 13-gram overlaps from the practice side (MIT's rule).
- Deviation: `AI-MO/aimo-validation-aime` (as downloaded) has 90 problems (AIME 2022-2024). MIT's AIME practice set had 714
  (1983-2023). The asset was used as given; AIME is a minor part of the practice pool.
- Held-out: MATH-500 (500), GSM8K main/test (1,319), IFEval (541). The sandbox `eval` tool only accepts practice ids
  (tested), and the held-out files live in `work/data/heldout`, which the sandbox cannot open (no file access).

## Environment
- uv project (`pyproject.toml`, `uv.lock`), uv-managed CPython 3.12.13, vllm 0.33.0, torch 2.14.0+cu130,
  transformers 5.20.2. Earlier in the session I briefly pip-installed packages into the conda base; all 45 were
  uninstalled at the user's request, and the conda base is back to how it was.
- `/usr/bin/nvcc` is CUDA 11.5, so FlashInfer's JIT fell back for SM 120. `CUDA_HOME` is set to `/usr/local/cuda` (13.0)
  from `config.yaml`.
- The editor server and target share the single RTX PRO 6000 (96 GB): editor `gpu_memory_utilization` 0.36 and target 0.52
  during editor runs. Held-out evals with no editor running use 0.88. This only affects batching and memory, not sampling.
- In-place weight reload into vLLM (`model.load_weights`) was verified. Editing layer 15 `down_proj` changes greedy output, and
  restoring the base tensor gives the identical greedy text and cumulative logprob.
- transformers prints a warning about an "incorrect regex pattern" in the tokenizer (aimed at Mistral tokenizers). HF and
  vLLM use the same `tokenizer.json`, so tokenization is consistent between the forward tool and generation.

## Harness rules (enforced in code, tested in `tests/test_harness.py`)
- AST check before execution: only whitelisted imports; no `torch.optim`/`torch.autograd`/`torch.func`; no attributes
  `backward`, `grad`, `enable_grad`, `set_grad_enabled`, `requires_grad_`, file I/O (`load`, `save`, `fromfile`, ...);
  no `requires_grad=` keyword; no dunders; no `open`/`exec`/`getattr`/`__import__`. Builtins are restricted.
- Runtime: code runs under `torch.no_grad()` with `Tensor.backward`, `autograd.grad/backward`, `Optimizer.__init__`,
  `enable_grad`, `set_grad_enabled(True)` patched to raise.
- MIT's edit rules are adopted: embeddings/output layer and final norm are frozen, and an edit that only multiplies a whole
  tensor by a constant is refused.
- AST-rejected code does not count as a Python execution, matching MIT's rule that a failed request does not use a
  forward pass. Every turn counts toward `max_turns` (160), which bounds the run.
- `forward()` is free (not budgeted), as in the spec. Only `eval` calls and Python executions are budgeted.
  MIT counted teacher-forced passes as passes; the spec here does not.
- `eval(ids, original=True)` runs the unedited model on the same ids with the same seeds (MIT's "both" mode in spirit). It
  costs one call.

## Compute (decided before any results)
- Decode at 32k contexts is limited by memory bandwidth on reading the KV cache: about 7-8k tok/s, measured 5.4k tok/s on
  128 real 32k-token practice generations. One base MATH-500 pass (2,000 samples, mostly running to the cap) takes about 2+ h.
- The user confirmed: keep MIT's protocol exactly (bf16 KV cache, 32,768 max tokens, 4 samples). No FP8 KV, no lower cap.
- Held-out sets for EVERY final checkpoint: MATH-500 (primary) and IFEval (collateral), plus KL on wikitext.
  GSM8K test (2 samples, about 3 h per looping checkpoint) is run for the base model and the four scripted/gradient
  baselines only, not for the 20 editor runs. This is a compute choice fixed before seeing any editor result.
- Per editor run there is a wall-clock safety stop of 16 h (`editor.max_run_hours`), end_reason `time_limit`. It is fixed before any run.
