# Weight-space programming with small editors: report

_Narrative not written yet._


## Generated tables

Protocol: Qwen3.5-0.8B (text), thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, max 32768 new tokens; MATH-500 x4, IFEval x2, GSM8K test x2. Paired seeds per (problem, sample). Bootstrap: 10000 resamples over problems; McNemar exact on paired samples.

### Headline: MATH-500 (4 samples x 500 problems; mean over seeds for editor conditions)

| Condition | Runs | Acc % (mean ± SD) | Min-max | Completion % | Completed-only acc % | Δ vs base (pooled, 95% CI) | p | Δ vs scripted | p | % weights edited (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 8.1 ± 0.0 | 8.1-8.1 | 7.6 | 99.3 | - | - | - | - | 0.0000 |
| scripted | 1 | 8.6 ± 0.0 | 8.6-8.6 | 7.8 | 99.4 | +0.5 [-0.7, +1.8] | 0.433 | - | - | 0.0171 |
| random | 1 | 7.0 ± 0.0 | 7.0-7.0 | 6.6 | 100.0 | -1.0 [-2.1, +0.1] | 0.0712 | -1.5 | 0.0074 | 0.0167 |
| mathneuro | 1 | not evaluated | | | | | | | | |
| lora | 1 | not evaluated | | | | | | | | |
| 4B-L0 | 1 | not evaluated | | | | | | | | |

### Per run (MATH-500)

| Condition | Run | Acc % | Completion % | Completed-only acc % | Mean tokens | Δ vs base | p (boot) | p (McNemar) | Δ vs scripted | p | Entries edited | % | End | Evals used | Execs used | Failure mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | base | 8.1 | 7.6 | 99.3 | 30700 | - | - | - | - | - | 0 | 0.0000 | - | - | - | - |
| scripted | scripted | 8.6 | 7.8 | 99.4 | 30663 | +0.5 | 0.433 | 0.474 | - | - | 128470 | 0.0171 | - | - | - | - |
| random | random | 7.0 | 6.6 | 100.0 | 30999 | -1.0 | 0.0712 | 0.108 | -1.5 | 0.0074 | 125783 | 0.0167 | - | - | - | - |
| mathneuro | mathneuro | not evaluated | | | | - | - | - | - | - | 3469255 | 0.4611 | - | - | - | - |
| lora | lora | not evaluated | | | | - | - | - | - | - | 264915484 | 35.2097 | - | - | - | - |
| 4B-L0 | s0 | not evaluated | | | | - | - | - | - | - | 0 | 0.0000 | - | - | - | - |

### Collateral (all final checkpoints)

| Condition | Run | GSM8K test acc % | GSM8K completion % | IFEval strict prompt % | KL(base‖edit) nats/token (wikitext) | PPL base → edit |
|---|---|---|---|---|---|---|
| base | base | - | - | 19.4 | - | - |
| scripted | scripted | - | - | - | - | - |
| random | random | - | - | - | - | - |
| mathneuro | mathneuro | - | - | - | - | - |
| lora | lora | - | - | - | - | - |
| 4B-L0 | s0 | - | - | - | - | - |


## Sources

- Base: `results/base/heldout/` (`math500_per_problem.json`, `ifeval_per_problem.json`, `gsm8k_test_per_problem.json`)
- Baselines: `results/baselines/<name>/heldout/*_per_problem.json`, `final/manifest.json`, `kl.json`; layer choice `results/baselines/scripted/selection.json`
- Editor runs: `results/runs/editor_<size>_<level>_s<seed>/` (`run.json`, `transcript.jsonl`, `edit_log.jsonl` + `diffs/`, `evals/`, `final/manifest.json`, `heldout/*_per_problem.json`, `kl.json`)
- All statistics: `results/analysis.json` (written by `analyze.py` from the files above)
