"""Agent loop: a Qwen3.5 editor (4B or 9B, served by `vllm serve`) edits the target's weights through the sandbox.

Each turn the editor replies with an explanation and ONE ```python block; the harness runs it and returns the output.
The run ends when the editor replies DONE without code, when the Python-execution budget is spent, or after
max_turns. The final checkpoint is the edit state at the end of the run (rebuildable from edit_log.jsonl + diffs/).

  python editors.py serve 4B                       # start the editor server (blocks)
  python editors.py run 4B L0 0 [--target-util X]  # one run (expects the server for that size to be up)
  python editors.py dry                            # Milestone 4: a hand-written scripted "editor" through the same loop
"""
import argparse, json, os, re, subprocess, sys, time

import requests

import evaluate as E
import harness as H

CFG = E.CFG
ARCH = ("Qwen3.5-0.8B, text part: 24 decoder layers of width 1024 (hidden size). Layers 3, 7, 11, 15, 19, 23 use full "
        "softmax attention (self_attn: q_proj [4096,1024] (queries + output gate), k_proj [512,1024], v_proj [512,1024], "
        "o_proj [1024,2048]; 8 query heads, 2 key/value heads, head size 256); the other 18 layers use Gated DeltaNet "
        "linear attention (linear_attn.*). Every layer has a SwiGLU MLP: mlp.gate_proj [3584,1024], mlp.up_proj "
        "[3584,1024], mlp.down_proj [1024,3584] (column j of down_proj is the vector MLP neuron j adds to the residual "
        "stream). RMSNorm before attention and MLP; tied 248,320-token embedding. 752,393,024 weights in total. Tensor "
        "names look like 'model.language_model.layers.15.mlp.down_proj.weight' (use list_tensors()).")


def task_message(level, budget):
    b = CFG["budget"]
    te = CFG["target_eval"]
    helpers = ""
    if level == "L1":
        helpers = ("\n- diff_of_means(acts_a, acts_b) -> unit vector along mean(acts_a) - mean(acts_b) (acts: tensor [N, d] "
                   "or list of [T, d] tensors)\n- project_out(W, v, cols, alpha=1.0) -> copy of W where, for each column j in "
                   "cols, W[:, j] -= alpha * v (v . W[:, j]) (v is normalised; e.g. remove a residual-stream direction from "
                   "what chosen MLP neurons write: W = mlp.down_proj.weight)")
    return f"""You are an expert in neural-network interpretability working in a Python sandbox. Your task: make a small \
language model better at math by EDITING ITS WEIGHTS DIRECTLY. No gradient descent and no training of any kind \
(backward passes, autograd, optimizers and fitting weights to a loss are blocked and refused).

THE MODEL. {ARCH}

HOW IT IS EVALUATED (after your run, on math problems you never see): the chat template with thinking on, the user message \
is the problem + "\\n{te['instruction']}", sampling temperature {te['sampling']['temperature']}, top_p {te['sampling']['top_p']}, \
top_k {te['sampling']['top_k']}, presence_penalty {te['sampling']['presence_penalty']}, at most {te['max_tokens']} new tokens; \
the last \\boxed{{}} answer is graded. An answer that is still going at {te['max_tokens']} tokens usually has no final answer \
and counts as wrong. Your final edit is whatever edits are applied when your run ends.

WHAT COUNTS. Edits that come from understanding the network: find out what goes wrong when it fails at math problems, \
find the components responsible (neurons, heads, directions, circuits) and change them in a way you can justify with \
evidence. Repairing a specific failure mechanism is in scope (for example, if the model often repeats itself and never \
finishes, finding and fixing what drives the repetition). Global knobs (making the whole output distribution sharper or \
flatter) and forcing the model to stop thinking early regardless of the problem are not what we are looking for.

ADVICE. (1) Diagnose first: look at how the model fails on practice problems (wrong final answer, finished without an \
answer, or cut off at the token limit) and read short excerpts of failing texts; go after the largest failure you can \
explain. (2) Measure with enough power: accuracy over n answers has a standard error of about 50/sqrt(n) points; compare \
edited and original on the SAME ids (eval(ids) and eval(ids, original=True) use the same random seeds per problem); keep an \
edit only if its gain is clearly larger than its standard error on problems not used to choose it. Fewer cut-off answers \
alone is not success: judge by accuracy. (3) Edits that change a tensor by a tiny fraction usually have effects too small \
to see; test strengths large enough to matter and back off if they hurt. (4) Print short excerpts (at most ~300 characters \
of a text), never whole texts or arrays.

TOOLS (already defined; call them in your code):
- eval(ids, original=False) -> list of dicts {{id, correct, completed, n_tokens, pred, answer, text}}: the CURRENT edited \
model answers practice problems with exactly the evaluation's settings (one answer each). original=True runs the unedited \
model on the same ids with the same random seeds (paired comparison). Costs 1 eval call; at most {b['eval_max_ids']} ids per \
call. A pass is slow: up to about an hour for 512 problems, because answers can be 32,768 tokens long.
- forward(texts, layers=None, last_tokens=None) -> teacher-forced pass of the CURRENT model over raw texts (no template \
added; build them with format_prompt(problem) + generated text). Returns dict: 'tokens', 'logprobs' (per text, log p of each \
next token), 'last_logits' [n, vocab]; for each layer l in layers: 'resid'[l] (residual stream after layer l, [T,1024] per \
text), 'mlp_in'[l] (input of mlp.down_proj, [T,3584]), 'mlp_out'[l] (MLP output, [T,1024]). Limits: {b['forward_max_texts']} \
texts, {b['forward_max_tokens_per_text']} tokens per text (longer texts keep their last tokens), {b['forward_max_total_tokens']} \
tokens per call. Free (no budget).
- format_prompt(problem) -> the exact evaluation prompt string for a problem.
- get_tensor(name) -> float32 copy; apply_edit(name, new_tensor) -> replaces the tensor (both for eval and forward); \
revert(name) or revert('all'); list_tensors(pattern=''); edit_summary(); budget().{helpers}
- practice: dict id -> {{problem, answer, solution?, source}}; practice_ids: {{'gsm8k': [...], 'math': [...], 'aime': [...]}} \
(GSM8K train, MATH train, AIME 2022-2024). Only these ids are accepted by eval.
- torch, np (numpy), math are imported. Allowed imports: {', '.join(CFG['sandbox']['allowed_imports'])}. No files, no \
internet. The embeddings / output layer and the final norm cannot be edited; edits that only multiply a whole tensor by a \
constant are refused.

BUDGET: {b['eval_calls']} eval calls and {b['python_execs']} Python executions in total; the run also stops after \
{CFG['editor']['max_run_hours']} hours of wall-clock time. Variables persist between executions.

FORMAT: reply with a short explanation and exactly ONE ```python code block; you will get its printed output (long output is \
cut, so print summaries, not whole texts or tensors). When you are finished, leave your best edit applied and reply with the \
word DONE and no code block."""


def extract_code(text):
    body = text.split("</think>")[-1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, re.S)
    return blocks[-1] if blocks else None


class EditorClient:
    def __init__(self, size, seed):
        e = CFG["editor"]
        self.url = f"http://127.0.0.1:{e['port']}/v1/chat/completions"
        self.model = size
        self.seed = seed

    def chat(self, messages, turn):
        e = CFG["editor"]
        body = {"model": self.model, "messages": messages, "max_tokens": e["max_tokens_per_turn"],
                "seed": self.seed * 100003 + turn, **e["sampling"],
                "chat_template_kwargs": {"enable_thinking": e["thinking"]}}
        for attempt in range(5):
            try:
                r = requests.post(self.url, json=body, timeout=3600)
                if r.status_code == 400 and "maximum context length" in r.text:
                    return None, {"error": "context"}
                r.raise_for_status()
                j = r.json()
                msg = j["choices"][0]["message"]
                text = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                return {"content": text, "reasoning": reasoning, "finish": j["choices"][0]["finish_reason"]}, j.get("usage", {})
            except requests.RequestException as ex:
                err = str(ex)
                time.sleep(10)
        raise RuntimeError(f"editor server unreachable: {err}")


def trim(messages, keep):
    """system + task + a note + the last `keep` exchanges (user/assistant pairs)."""
    head, rest = messages[:2], messages[2:]
    if len(rest) <= 2 * keep:
        return messages
    note = {"role": "user", "content": "[Earlier turns were removed to fit the context. Your variables and edits are still "
                                       "in the sandbox; call edit_summary() and budget() to see the state.]"}
    return head + [note, {"role": "assistant", "content": "OK."}] + rest[-2 * keep:]


def run_editor(session, policy, run_dir, meta):
    """the loop. policy(messages, turn) -> (reply dict, usage). Writes transcript.jsonl and result.json."""
    e = CFG["editor"]
    messages = [{"role": "system", "content": "You are a careful, expert ML researcher. Reply in English."},
                {"role": "user", "content": task_message(meta["level"], session.budget)}]
    tpath = os.path.join(run_dir, "transcript.jsonl")
    open(tpath, "w").close()
    status_counts = {"ok": 0, "error": 0, "rejected": 0, "budget": 0, "no_code": 0, "context": 0}
    end_reason = "max_turns"
    t_start = time.time()
    for turn in range(1, e["max_turns"] + 1):
        if time.time() - t_start > 3600 * e["max_run_hours"]:
            end_reason = "time_limit"
            break
        reply, usage = policy(messages, turn)
        if reply is None:                                  # context overflow: trim harder and retry once
            status_counts["context"] += 1
            messages = trim(messages, max(1, e["context_keep_last"] // 2))
            reply, usage = policy(messages, turn)
            if reply is None:
                end_reason = "context_overflow"
                break
        content = reply["content"]
        messages.append({"role": "assistant", "content": content})
        code = extract_code(content)
        rec = {"turn": turn, "reply": content, "reasoning": reply.get("reasoning", ""), "finish": reply.get("finish"),
               "usage": usage, "time": time.time()}
        if code is None:
            if re.search(r"\bDONE\b", content.split("</think>")[-1]):
                rec["status"] = "done"
                with open(tpath, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                end_reason = "done"
                break
            status_counts["no_code"] += 1
            out = ("No ```python block found. Reply with exactly one ```python block, or DONE (no code) to finish. "
                   f"Budget: {session.budget.summary()}")
            rec["status"] = "no_code"
        else:
            status, out = session.sandbox.run(code)
            status_counts[status] += 1
            rec.update(status=status, code=code, output=out)
            if status == "budget":
                with open(tpath, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                end_reason = "python_budget"
                break
            out = (out or "(no output)") + f"\n[budget: {session.budget.summary()}]"
        with open(tpath, "a") as f:
            f.write(json.dumps(rec) + "\n")
        messages.append({"role": "user", "content": out})
        messages = trim(messages, e["context_keep_last"]) if len(json.dumps(messages)) > 4 * e["max_model_len"] * 0.6 else messages
    man = session.finish(extra=meta)
    res = dict(meta, end_reason=end_reason, turns=turn, status_counts=status_counts, budget_used=session.budget.used,
               eval_log=session.tools.eval_log, n_changed=man["n_changed"], pct_changed=man["pct_changed"],
               edited_tensors=list(man["tensors"]), minutes=round((time.time() - t_start) / 60, 1))
    json.dump(res, open(os.path.join(run_dir, "run.json"), "w"), indent=1)
    return res


# ------------------------------------------------------------------ editor server
def serve(size):
    e = CFG["editor"]
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", os.path.join(E.ROOT, CFG["paths"]["editors"][size]),
           "--served-model-name", size, "--port", str(e["port"]), "--dtype", "bfloat16",
           "--max-model-len", str(e["max_model_len"]), "--gpu-memory-utilization", str(e["gpu_memory_utilization"]),
           "--reasoning-parser", "qwen3", "--seed", "0", "--language-model-only"]
    print(" ".join(cmd), flush=True)
    os.execv(sys.executable, cmd)


def wait_server(timeout=1800):
    url = f"http://127.0.0.1:{CFG['editor']['port']}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=5).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError("editor server did not come up")


def run_dir_for(size, level, seed):
    return os.path.join(E.P("results"), "runs", f"editor_{size}_{level}_s{seed}")


def heldout_and_collateral(session, run_dir, sets=("math500",)):
    """held-out eval of the session's final edit (engines already hold it)."""
    return E.run_heldout(session.target, os.path.join(run_dir, "heldout"), list(sets), label=run_dir)


# ------------------------------------------------------------------ dry run: a hand-written scripted editor
DRY_SCRIPT = [
    "Let me look at the tensors.\n```python\nprint(len(list_tensors()))\nprint(list_tensors('layers.15.mlp'))\nbudget()\n```",
    "Measure the unedited model on a few practice problems.\n```python\nids = practice_ids['gsm8k'][:8]\nr0 = eval(ids)\n"
    "print(sum(x['correct'] for x in r0), sum(x['completed'] for x in r0), [x['n_tokens'] for x in r0])\n```",
    "Collect activations: looping answers vs finished ones (reference solutions if none finished).\n```python\n"
    "loop = [x for x in r0 if not x['completed']]\ndone = [x for x in r0 if x['completed']]\nprint(len(loop), len(done))\n"
    "texts_a = [format_prompt(practice[x['id']]['problem']) + x['text'] for x in (loop or r0)[:4]]\n"
    "texts_b = ([format_prompt(practice[x['id']]['problem']) + x['text'] for x in done[:4]] or\n"
    "           [format_prompt(practice[i]['problem']) + practice[i]['solution'] for i in ids[:4]])\n"
    "fa = forward(texts_a, layers=[15, 19], last_tokens=512)\nfb = forward(texts_b, layers=[15, 19], last_tokens=512)\n"
    "print(fa['mlp_out'][15][0].shape)\n```",
    "Project the difference-of-means direction out of 64 down_proj columns in layers 15 and 19.\n```python\n"
    "for l in (15, 19):\n    v = diff_of_means(fa['mlp_out'][l], fb['mlp_out'][l])\n"
    "    name = f'model.language_model.layers.{l}.mlp.down_proj.weight'\n    W = get_tensor(name)\n"
    "    cols = (v @ W).abs().topk(64).indices\n    print(apply_edit(name, project_out(W, v, cols)))\nedit_summary()\n```",
    "Try something forbidden (must be refused).\n```python\nx = torch.ones(2)\nx.backward()\n```",
    "Paired check of the edit.\n```python\nr1 = eval(ids)\nprint(sum(x['correct'] for x in r1), sum(x['completed'] for x in r1))\n```",
    "DONE",
]


def dry_run():
    run_dir = os.path.join(E.P("results"), "runs", "dryrun_scripted")
    session = H.Session(run_dir, level="L1", seed=0,
                        gpu_util=CFG["target_eval"]["vllm_with_editor"]["gpu_memory_utilization"])
    script = iter(DRY_SCRIPT)
    meta = {"editor": "scripted-dry-run", "level": "L1", "seed": 0}
    res = run_editor(session, lambda m, t: ({"content": next(script)}, {}), run_dir, meta)
    rebuilt = H.EditState.rebuild(session.state.base, run_dir)
    from safetensors.torch import load_file
    ck = load_file(os.path.join(run_dir, "final", "edited.safetensors"))
    same = set(ck) == set(rebuilt) and all(H.sha(ck[k]) == H.sha(rebuilt[k]) for k in ck)
    res["rebuild_identical"] = same
    # held-out eval on a 32-problem slice only: this is a pipeline check, not a result
    rows = E.load_heldout("math500")[:32]
    out = session.target.generate(rows, 1, CFG["target_eval"]["seed"])
    res["pipeline_check_math500_first32"] = E.summarize([{"id": r["id"], "samples": s} for r, s in zip(rows, out)])
    json.dump(res, open(os.path.join(run_dir, "run.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("end_reason", "turns", "status_counts", "n_changed", "rebuild_identical",
                                         "pipeline_check_math500_first32")}, indent=1))


def campaign(size, levels, seeds):
    """all runs of one editor size, serially, against one editor server and one reused target engine. Each run is
    followed by its held-out evals (MATH-500 + IFEval) and KL, then the engines are reset to the base weights."""
    server = subprocess.Popen([sys.executable, os.path.abspath(__file__), "serve", size],
                              stdout=open(os.path.join(E.P("work"), f"editor_server_{size}.log"), "a"), stderr=subprocess.STDOUT)
    try:
        wait_server()
        target = E.Target(gpu_memory_utilization=CFG["target_eval"]["vllm_with_editor"]["gpu_memory_utilization"])
        for level in levels:
            for seed in seeds:
                run_dir = run_dir_for(size, level, seed)
                hs = os.path.join(run_dir, "heldout", "heldout_summary.json")
                if os.path.exists(hs) and all(k in json.load(open(hs)) for k in CFG["heldout_sets_all"]) and \
                        os.path.exists(os.path.join(run_dir, "kl.json")):
                    print("skip (done)", run_dir, flush=True)
                    continue
                if os.path.exists(os.path.join(run_dir, "run.json")):
                    session = resume_for_eval(run_dir, target)        # run finished earlier, evals missing
                else:
                    session = run_one(size, level, seed, target=target)
                    if json.load(open(os.path.join(run_dir, "run.json"))).get("end_reason") == "crash":
                        # a harness crash is most likely a bug on our side: stop instead of burning all runs
                        raise RuntimeError(f"run crashed, campaign stopped: {run_dir} (see run.json 'error')")
                if session.state.changed() == 0:
                    reuse_base_heldout(run_dir)            # final weights == base: reuse the base model's logged evals
                else:
                    E.run_heldout(target, os.path.join(run_dir, "heldout"), CFG["heldout_sets_all"], label=run_dir)
                    kl_from_session(session, os.path.join(run_dir, "kl.json"))
                session.reset_to_base()
                del session
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        server.terminate()
        server.wait(timeout=120)


def reuse_base_heldout(run_dir):
    """a run whose final edit changes no entry has exactly the base weights: its held-out results ARE the base model's.
    Copy the base model's logged per-problem files (marked reused_from_base) instead of sampling the base model again."""
    import shutil
    src, dst = os.path.join(E.P("results"), "base", "heldout"), os.path.join(run_dir, "heldout")
    os.makedirs(dst, exist_ok=True)
    summ = json.load(open(os.path.join(src, "heldout_summary.json")))
    out = {}
    for name in CFG["heldout_sets_all"]:
        shutil.copy(os.path.join(src, f"{name}_per_problem.json"), os.path.join(dst, f"{name}_per_problem.json"))
        out[name] = dict(summ[name], reused_from_base=True)
    out["_protocol"] = dict(summ["_protocol"], label=run_dir, reused_from_base=True,
                            note="final edit changes 0 entries; weights identical to base")
    json.dump(out, open(os.path.join(dst, "heldout_summary.json"), "w"), indent=1)
    json.dump({"kl_base_to_edited_per_token": 0.0, "reused_from_base": True, "note": "weights identical to base"},
              open(os.path.join(run_dir, "kl.json"), "w"), indent=1)
    print("no edit: reused base held-out results for", run_dir, flush=True)


def resume_for_eval(run_dir, target):
    """rebuild a finished run's final edit from its diff log and load it into the engines."""
    session = H.Session(run_dir + "/.resume", target=target)
    for n, t in H.EditState.rebuild(session.state.base, run_dir).items():
        session.state.apply(n, t)
    return session


def kl_from_session(session, out_json):
    """KL(base || edit) and perplexity on wikitext, with the session's HF model (it holds the edit)."""
    c = CFG["collateral"]
    passages = E.wikitext_passages(session.tok, c["kl_texts"], c["kl_tokens"])
    edited = dict(session.state.current)
    res = E.kl_with_model(session.hf, passages, {n: session.state.base[n] for n in edited}, edited, out_json)
    session.state.on_change(edited)          # make sure the engines hold the edit again
    return res


def run_one(size, level, seed, target=None):
    run_dir = run_dir_for(size, level, seed)
    if os.path.exists(os.path.join(run_dir, "heldout", "heldout_summary.json")):
        s = json.load(open(os.path.join(run_dir, "heldout", "heldout_summary.json")))
        if "math500" in s:
            print("already done", run_dir)
            return
    session = H.Session(run_dir, level=level, seed=seed, target=target,
                        gpu_util=CFG["target_eval"]["vllm_with_editor"]["gpu_memory_utilization"])
    client = EditorClient(size, seed)
    meta = {"editor": size, "level": level, "seed": seed}
    try:
        res = run_editor(session, client.chat, run_dir, meta)
    except Exception as ex:                               # crash: keep the run, record it, still evaluate what is applied
        import traceback
        res = dict(meta, end_reason="crash", error=traceback.format_exc()[-3000:])
        man = session.finish(extra=meta)
        res.update(n_changed=man["n_changed"], pct_changed=man["pct_changed"], budget_used=session.budget.used,
                   eval_log=session.tools.eval_log)
        json.dump(res, open(os.path.join(run_dir, "run.json"), "w"), indent=1)
    print(json.dumps({k: res.get(k) for k in ("editor", "level", "seed", "end_reason", "turns", "n_changed", "budget_used")}), flush=True)
    return session


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve", "run", "dry", "campaign"])
    ap.add_argument("size", nargs="?")
    ap.add_argument("level", nargs="?")
    ap.add_argument("seed", nargs="?", type=int)
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.size)
    elif a.cmd == "dry":
        dry_run()
    elif a.cmd == "campaign":
        campaign(a.size, (a.level or ",".join(CFG["scaffolds"])).split(","), CFG["seeds"])
    else:
        wait_server()
        run_one(a.size, a.level, a.seed)
