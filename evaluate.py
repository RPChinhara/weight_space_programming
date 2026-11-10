"""Evaluation of target checkpoints: data preparation, grading, the vLLM target engine, held-out benchmarks, KL.

Protocol (MIT "think35", lm_understand-logs/exp8_math/code/evaluate.py): chat template with thinking on, user turn =
problem + "\\n" + instruction, sampling from config.yaml, at most 32,768 new tokens; the last \\boxed{} answer (of the
text after the last </think>, or of the whole text if there is none) graded with math-verify. Accuracy = mean over all
samples. "cut" = finish_reason == "length" (hit max tokens); "completed" = finish_reason == "stop".

  python evaluate.py prepare                      # text-only export of the target + practice / held-out files
  python evaluate.py run <ckpt_dir|base> <out_dir> [--sets math500,gsm8k_test,ifeval]
  python evaluate.py kl <ckpt_dir> <out_json>     # KL(base || ckpt) and perplexity on wikitext-103 validation
"""
import argparse, glob, hashlib, json, os, re, shutil, sys, time

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
os.environ.setdefault("CUDA_HOME", CFG["paths"]["cuda_home"])
os.environ["PATH"] = os.path.join(CFG["paths"]["cuda_home"], "bin") + os.pathsep + os.environ.get("PATH", "")


def P(key):
    return os.path.join(ROOT, CFG["paths"][key])


def read_jsonl(path):
    return [json.loads(l) for l in open(path)]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ answer extraction and grading (as MIT mathqa.py)
def _box_at(text, idx):
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return r[1:k].strip() or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return last_boxed(text)


def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    from math_verify import parse, verify
    try:
        return bool(verify(parse("\\boxed{" + str(gold) + "}"), parse("\\boxed{" + pred + "}")))
    except Exception:
        return False


def grade_ifeval(row, text):
    """strict prompt-level IFEval on the answer after </think> (empty if thinking never closed), official checkers."""
    os.environ.setdefault("NLTK_DATA", os.path.join(P("work"), "nltk_data"))
    sys.path.insert(0, ROOT)
    from third_party.ifeval import instructions_registry as IR
    resp = text.split("</think>")[-1].strip() if "</think>" in text else ""
    ok = []
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
        ins = IR.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
        args = ins.get_instruction_args()
        if args and "prompt" in args:
            ins.build_description(prompt=row["prompt"])
        ok.append(bool(resp) and bool(ins.check_following(resp)))
    return all(ok)


# ------------------------------------------------------------------ data preparation
def _norm(s):
    return " ".join(s.lower().split())


def _grams(s, n):
    w = _norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def prepare():
    import pandas as pd
    D = P("data")
    # text-only export of the target (the VL wrapper's language model, tensor names unchanged), as MIT evaluated it
    src, dst = P("target_src"), P("target")
    if not os.path.exists(os.path.join(dst, "model.safetensors")):
        from safetensors.torch import load_file, save_file
        os.makedirs(dst, exist_ok=True)
        c = json.load(open(os.path.join(src, "config.json")))
        t = dict(c["text_config"], architectures=["Qwen3_5ForCausalLM"], tie_word_embeddings=True)
        json.dump(t, open(os.path.join(dst, "config.json"), "w"), indent=2)
        sd = {}
        for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
            sd.update({k: v for k, v in load_file(f).items() if k.startswith("model.language_model.")})
        save_file(sd, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
            shutil.copy(os.path.join(src, f), dst)
        json.dump({"eos_token_id": t["eos_token_id"], "use_cache": True}, open(os.path.join(dst, "generation_config.json"), "w"))

    # held-out
    m500 = [{"id": "math500/" + r["unique_id"], "problem": r["problem"], "answer": r["answer"],
             "subject": r["subject"], "level": r["level"]} for r in read_jsonl(f"{D}/math500/test.jsonl")]
    gt = pd.read_parquet(f"{D}/gsm8k/main/test-00000-of-00001.parquet")
    gsm_test = [{"id": f"gsm8k_test_{i}", "problem": q, "answer": a.split("####")[-1].strip().replace(",", "")}
                for i, (q, a) in enumerate(zip(gt["question"], gt["answer"]))]
    ife = [{"id": f"ifeval_{r['key']}", **r} for r in read_jsonl(f"{D}/ifeval/ifeval_input_data.jsonl")]
    H = P("heldout")
    write_jsonl(f"{H}/math500.jsonl", m500)
    write_jsonl(f"{H}/gsm8k_test.jsonl", gsm_test)
    write_jsonl(f"{H}/ifeval.jsonl", ife)

    # practice
    g = pd.read_parquet(f"{D}/gsm8k/main/train-00000-of-00001.parquet")
    gsm = [{"id": f"gsm8k_train_{i}", "source": "gsm8k", "problem": q, "answer": a.split("####")[-1].strip().replace(",", ""),
            "solution": a.split("####")[0].strip()} for i, (q, a) in enumerate(zip(g["question"], g["answer"]))]
    math = []
    for sub in sorted(os.listdir(f"{D}/math")):
        f = f"{D}/math/{sub}/train-00000-of-00001.parquet"
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        for i, r in df.iterrows():
            math.append({"id": f"math_train_{sub}_{i}", "source": "math", "subject": sub, "level": r["level"],
                         "problem": r["problem"], "answer": last_boxed(r["solution"]), "solution": r["solution"]})
    a = pd.read_parquet(glob.glob(f"{D}/aime_hist/data/*.parquet")[0])
    aime = [{"id": f"aime_{r['id']}", "source": "aime", "problem": r["problem"], "answer": str(r["answer"]),
             "solution": r.get("solution")} for _, r in a.iterrows()]

    n = CFG["data"]["decontam_ngram"]
    held = m500 + gsm_test
    held_norm = {_norm(r["problem"]) for r in held}
    held_grams = set().union(*(_grams(r["problem"], n) for r in held))
    m500_exact = {r["problem"] for r in m500}
    stats = {}
    out = {}
    for name, rows in (("gsm8k_train", gsm), ("math_train", math), ("aime", aime)):
        exact = sum(r["problem"] in m500_exact for r in rows)
        normm = sum(_norm(r["problem"]) in held_norm for r in rows)
        gram = sum(bool(_grams(r["problem"], n) & held_grams) for r in rows)
        noans = sum(r["answer"] in (None, "") for r in rows)
        kept = [r for r in rows if r["answer"] not in (None, "") and _norm(r["problem"]) not in held_norm
                and not (_grams(r["problem"], n) & held_grams)]
        stats[name] = {"total": len(rows), "exact_match_math500": exact, "normalized_match_heldout": normm,
                       f"{n}gram_overlap_heldout": gram, "no_answer": noans, "kept": len(kept)}
        out[name] = kept
        write_jsonl(f"{P('practice')}/{name}.jsonl", kept)
    # re-check after removal: must be zero
    kept_all = out["gsm8k_train"] + out["math_train"] + out["aime"]
    stats["after_removal_exact_match_math500"] = sum(r["problem"] in m500_exact for r in kept_all)
    stats["after_removal_normalized_match_heldout"] = sum(_norm(r["problem"]) in held_norm for r in kept_all)
    stats["heldout_sizes"] = {"math500": len(m500), "gsm8k_test": len(gsm_test), "ifeval": len(ife)}
    os.makedirs(P("results"), exist_ok=True)
    json.dump(stats, open(os.path.join(P("results"), "data_hygiene.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


def load_practice():
    rows = []
    for name in ("gsm8k_train", "math_train", "aime"):
        rows += read_jsonl(f"{P('practice')}/{name}.jsonl")
    return {r["id"]: r for r in rows}


def load_heldout(name):
    return read_jsonl(f"{P('heldout')}/{name}.jsonl")


# ------------------------------------------------------------------ target engine
def req_seed(base, pid, k):
    return int(hashlib.sha256(f"{base}|{pid}|{k}".encode()).hexdigest()[:8], 16)


class Target:
    """vLLM engine of the target, in-process (so edited tensors can be loaded in place) and the chat formatting."""

    def __init__(self, gpu_memory_utilization=None, model_path=None):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM
        v = CFG["target_eval"]["vllm"]
        self.path = model_path or P("target")
        self.llm = LLM(model=self.path, dtype="bfloat16", seed=0, enable_prefix_caching=False,
                       gpu_memory_utilization=gpu_memory_utilization or v["gpu_memory_utilization"],
                       max_model_len=v["max_model_len"], max_num_seqs=v["max_num_seqs"])
        self.tok = self.llm.get_tokenizer()

    def _model(self):
        e = self.llm.llm_engine
        for path in ("model_executor.driver_worker.worker.model_runner.model",
                     "engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model"):
            obj = e
            try:
                for a in path.split("."):
                    obj = getattr(obj, a)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("cannot reach the vLLM model object")

    def load_tensors(self, tensors):
        """replace tensors in place: {hf_name: bf16 tensor}. vLLM's own loader maps names / fused weights."""
        import torch
        m = self._model()
        with torch.no_grad():
            m.load_weights([(k, v.to(torch.bfloat16)) for k, v in tensors.items()])
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass

    def prompt(self, text):
        te = CFG["target_eval"]
        return self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=te["thinking"])

    def generate(self, rows, n, seed, kind="math"):
        """rows: dicts with id, problem (math) or prompt (ifeval). Returns per row a list of n sample dicts."""
        from vllm import SamplingParams
        te = CFG["target_eval"]
        prompts, params, owner = [], [], []
        for i, r in enumerate(rows):
            text = r["prompt"] if kind == "ifeval" else r["problem"] + "\n" + te["instruction"]
            p = self.prompt(text)
            for k in range(n):
                prompts.append(p)
                params.append(SamplingParams(max_tokens=te["max_tokens"], seed=req_seed(seed, r["id"], k), **te["sampling"]))
                owner.append(i)
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        res = [[] for _ in rows]
        for i, o in zip(owner, outs):
            s = o.outputs[0]
            if s.finish_reason not in ("stop", "length"):
                raise RuntimeError(f"unexpected finish_reason {s.finish_reason}")
            r = rows[i]
            rec = {"finish": s.finish_reason, "n_tokens": len(s.token_ids), "text": s.text}
            if kind == "ifeval":
                rec["correct"] = grade_ifeval(r, s.text)
            else:
                rec["pred"] = extract(s.text)
                rec["correct"] = is_correct(rec["pred"], r["answer"])
            res[i].append(rec)
        return res


def summarize(per):
    """per: list of {id, samples:[{correct, finish, n_tokens}]} -> summary numbers."""
    s = [x for r in per for x in r["samples"]]
    n = len(s)
    done = [x for x in s if x["finish"] == "stop"]
    return {"n_problems": len(per), "n_samples": n,
            "acc": sum(x["correct"] for x in s) / n,
            "cut_share": sum(x["finish"] == "length" for x in s) / n,
            "completion_rate": len(done) / n,
            "completed_only_acc": (sum(x["correct"] for x in done) / len(done)) if done else None,
            "n_completed": len(done),
            "mean_tokens": sum(x["n_tokens"] for x in s) / n}


def run_heldout(target, out_dir, sets, label=""):
    """evaluate the target's current weights on held-out sets; writes per-problem results and a summary."""
    os.makedirs(out_dir, exist_ok=True)
    te = CFG["target_eval"]
    summ_path = os.path.join(out_dir, "heldout_summary.json")
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    for name in sets:
        if name in summ:
            continue
        rows = load_heldout(name)
        t0 = time.time()
        res = target.generate(rows, te["samples"][name], te["seed"], kind="ifeval" if name == "ifeval" else "math")
        per = [{"id": r["id"], "samples": [{k: v for k, v in x.items() if k != "text"} for x in samp]}
               for r, samp in zip(rows, res)]
        json.dump(per, open(os.path.join(out_dir, f"{name}_per_problem.json"), "w"))
        with open(os.path.join(out_dir, f"{name}_texts_first_sample.jsonl"), "w") as f:
            for r, samp in zip(rows, res):
                f.write(json.dumps({"id": r["id"], "text": samp[0]["text"]}, ensure_ascii=False) + "\n")
        summ[name] = dict(summarize(per), seconds=round(time.time() - t0), samples_per_problem=te["samples"][name])
        summ["_protocol"] = {"label": label, "sampling": te["sampling"], "max_tokens": te["max_tokens"],
                             "thinking": te["thinking"], "seed": te["seed"], "config_frozen": CFG["frozen"]}
        json.dump(summ, open(summ_path, "w"), indent=1)
        print(name, json.dumps(summ[name]), flush=True)
    return summ


def load_ckpt_tensors(ckpt):
    """the tensors a checkpoint changes relative to the base export (a dir with model.safetensors or edited.safetensors)."""
    from safetensors.torch import load_file
    f = os.path.join(ckpt, "edited.safetensors")
    return load_file(f) if os.path.exists(f) else load_file(os.path.join(ckpt, "model.safetensors"))


# ------------------------------------------------------------------ collateral: KL / perplexity on wikitext
def wikitext_passages(tok, n, n_tokens):
    import pandas as pd
    df = pd.read_parquet(glob.glob(f"{P('data')}/wikitext/wikitext-103-raw-v1/validation-*.parquet")[0])
    text = "".join(df["text"].tolist())
    ids = tok(text)["input_ids"]
    out = []
    for i in range(n):
        out.append(ids[i * n_tokens:(i + 1) * n_tokens])
    return out


def kl_with_model(hf, passages, base_tensors, edited_tensors, out_json=None):
    """mean per-token KL(base || edited) and perplexities on the passages. hf: HF model; the named tensors are swapped
    in (checkpoint names); the model is left holding the EDITED tensors."""
    import math
    import torch
    params = dict(hf.named_parameters())

    def put(tensors):
        for k, v in tensors.items():
            params[k.replace("model.language_model.", "model.")].data.copy_(v.to(params[k.replace("model.language_model.", "model.")].dtype))

    def lp(p):
        return torch.log_softmax(hf(torch.tensor([p], device=hf.device)).logits[0].float(), -1)
    kls, nll_b, nll_e, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for p in passages:                    # base and edited log-probs of one passage at a time (GPU memory only)
            put(base_tensors)
            b = lp(p)
            put(edited_tensors)
            e = lp(p)
            kls += (b.exp() * (b - e)).sum(-1)[:-1].sum().item()
            tgt = torch.tensor(p[1:], device=hf.device)[:, None]
            nll_b += -b[:-1].gather(1, tgt).sum().item()
            nll_e += -e[:-1].gather(1, tgt).sum().item()
            n += len(p) - 1
    res = {"kl_base_to_edited_per_token": kls / n, "ppl_base": math.exp(nll_b / n), "ppl_edited": math.exp(nll_e / n),
           "n_tokens": n, "n_passages": len(passages)}
    if out_json:
        json.dump(res, open(out_json, "w"), indent=1)
    print("kl", json.dumps(res), flush=True)
    return res


def kl_eval(ckpt, out_json):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    c = CFG["collateral"]
    tok = AutoTokenizer.from_pretrained(P("target"))
    m = AutoModelForCausalLM.from_pretrained(P("target"), dtype=torch.bfloat16).cuda().eval()
    edited = load_ckpt_tensors(ckpt)
    base = load_file(os.path.join(P("target"), "model.safetensors"))
    return kl_with_model(m, wikitext_passages(tok, c["kl_texts"], c["kl_tokens"]), {k: base[k] for k in edited}, edited, out_json)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "kl"])
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--sets", default="math500,gsm8k_test,ifeval")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "run":
        t = Target()
        if a.ckpt != "base":
            t.load_tensors(load_ckpt_tensors(a.ckpt))
        run_heldout(t, a.out, a.sets.split(","), label=a.ckpt)
    else:
        kl_eval(a.ckpt, a.out)
