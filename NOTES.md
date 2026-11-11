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

## Milestone 1: base reproduction (results/base/heldout/heldout_summary.json)
- MATH-500, 4 samples x 500: accuracy **8.05%** (MIT: 7.9%), cut at 32,768 tokens **92.35%** (MIT: 93.0%),
  completion 7.65%, accuracy on completed answers 99.3% (152/153), mean 30,700 tokens; 11,325 s.
  This is close to MIT, so no investigation was needed.
- config.yaml FROZEN at 2026-11-10 19:50 (sha256 fb9b269ec60682cc...). Changes made before the freeze:
  max_num_seqs 512 -> 256 (preemption), max_run_hours added, heldout_sets_all added, cuda_home added.
- Logging change at the freeze point, with no effect on sampling: Target.generate now drives the vLLM engine step by step
  (add_request/step, FINAL_ONLY outputs, same SamplingParams and seeds) so partial results can be logged. The base MATH-500
  run used llm.generate. Both paths submit the same requests.

## Kernels (2026-11-10, before any edited-model result)
- At the user's request, added `flash-linear-attention` 0.6.0 and `causal-conv1d` 1.7.0 (built from source with uv, no build
  isolation, CUDA 13 toolkit, TORCH_CUDA_ARCH_LIST=12.0). The HF model (forward tool, activations, KL, LoRA, MathNeuro)
  now uses the fused GatedDeltaNet / causal-conv kernels instead of the torch fallback: 8 x 4k-token forward 3.9 s -> 1.0 s.
- Check: log-prob of a fixed 71-token text: vLLM -153.05, HF fallback -153.31, HF fla+causal-conv1d -152.57, i.e. within
  about 0.01 nats/token (bf16 kernel differences). vLLM generation (all accuracy numbers) is unaffected.
- MathNeuro was computed with the fallback kernels; its importance statistics are activation norms, insensitive to this.
  LoRA training with BOTH fused packages was about 4x slower (160/250 steps in 25.7 min). With fla alone it was the
  fastest (250 steps in about 7.5 min, against about 9 min on the torch fallback). At the user's request `causal-conv1d`
  was removed (`uv remove`), and fla is kept for everything. The final LoRA baseline (`results/baselines/lora/`) was
  trained with fla only. The fallback-kernel run is kept for reference in `results/baselines/lora_fallback_kernels/` (not
  evaluated), and the aborted fla+causal-conv1d log is `results/baselines/lora_fallback_kernels/lora_train_fused_kernels_aborted.log`.
- Final kernel setup: vLLM (generation) as is; HF uses fla GatedDeltaNet kernels + torch causal conv.

## Milestone 4: dry run (results/runs/dryrun_scripted/, results/logs/m4_dryrun.log)
- The first attempt failed: in my hand-written script, all 8 practice answers looped, so the "productive" set fell back to the
  same texts and diff_of_means returned a zero vector, which became NaN after normalisation. apply_edit's NaN guard refused it,
  so there was no edit and no checkpoint. Fixes: the script uses reference solutions when no answer finished.
  diff_of_means/project_out now raise a clear error on a zero difference (tested). eval progress lines go to stderr, not the
  editor's stdout. TRITON_PRINT_AUTOTUNING=0 (vLLM's JIT monitor had turned on autotune prints that reached the editor's
  output). fla kernels are warmed up when a session starts.
- Second attempt passed: 7 turns, 2 eval calls, 5 executions, 1 rejected (`.backward`), edit of 2 x 64 down_proj columns
  (129,284 entries changed; the rest round to the same bf16 values). The diff-log rebuild is bit-identical to the saved
  checkpoint. The held-out pipeline check on 32 MATH-500 problems ran end to end; it is a pipeline check, not a result.

## Milestone 2: scripted baseline (results/baselines/{collect,scripted,random}/)
- Collect: the base model answered 512 practice problems (256 GSM8K + 256 MATH, 1 sample): 12.1% correct, 88.3% cut,
  completed-only 98.3%. That gave 128 looping texts (window = last 1,024 tokens) and 60 productive texts (window = 1,024
  tokens before </think>).
- Separation d' of looping vs productive along each layer's diff-of-means direction (MLP output): every layer separates
  strongly; top six: 9 (7.59), 13 (7.54), 11 (7.31), 17 (7.30), 14 (7.29), 16 (7.22). Pairs of the top 4 were screened on
  256 practice problems (disjoint from the collect set), 6 eval calls: [9,13] 15.6%, [9,17] 11.7%, [9,11] 10.5%, [11,13] 9.8%,
  [13,17] 9.4%, [11,17] 6.6%. Chosen: [9, 13]. MIT's edit used layers 15 and 19. My practice-only rule did not select them,
  and I did not override it.
- Held-out MATH-500: scripted 8.55% (cut 92.2%, 128,470 entries = 0.0171%); random same-size, norm-matched edit 7.05%
  (cut 93.4%, 125,783 entries). Base 8.05%. The scripted edit did not reproduce MIT's loop fix. Significance: see REPORT.

## Sandbox fix after the first live editor run (2026-11-11 08:40)
- The first live run (4B-L0-s0, started 07:55) showed that the AST check blocked harmless code: `type(x)` (the editor lost
  turns 2-7 to it and then tried `import builtins`), `torch.nn.functional` (the attribute ban on "functional") and
  `re.compile` (the attribute ban on "compile"). `hasattr`, `AttributeError`, `NameError` were also missing from the builtins.
  These were over-broad rules on my side. None of them gives access to gradients or files.
- Fix: those names are allowed. All gradient/file/dunder rules are unchanged (torch.autograd.functional stays blocked through
  "autograd"). Tests added for both the new allowances and the still-blocked calls (52 pass). The rejected blocks of
  the discarded run now pass, except `import builtins`, which stays rejected.
- The partial run (35 min, 9 turns, 4 eval calls, no edit yet) is kept, unused, in
  `results/runs_discarded/editor_4B_L0_s0_overrestrictive_sandbox/`. The 4B campaign restarts from seed 0. This affects
  no completed condition (no editor run had finished).
