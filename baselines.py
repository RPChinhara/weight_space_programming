"""Non-LLM baselines on the same practice data, each written as an edit checkpoint and evaluated on held-out sets.

  python baselines.py collect      # base model answers collect_ids practice problems (1 sample): looping vs productive
  python baselines.py scripted     # difference-of-means loop direction projected out of 64 down_proj columns in 2 layers
                                   # (layers chosen on practice data only) + random edit of the same size; held-out evals
  python baselines.py mathneuro    # MathNeuro-style scaling of math-specific MLP weights
  python baselines.py lora         # LoRA SFT on practice reference solutions (gradient baseline), merged
  python baselines.py heldout <name> [sets]   # held-out eval of results/baselines/<name>/final

Scripted direction, explicitly:
  looping texts    = base answers cut at max_tokens; window = their last `window_tokens` tokens
  productive texts = base answers that finished; window = the `window_tokens` tokens before </think> (still reasoning)
  per layer l: u_l = unit(mean_loop(mlp_out_l) - mean_prod(mlp_out_l)) on window positions (mlp_out = what the MLP writes)
  separation d'_l = gap of per-text mean projections on u_l / pooled std
  columns: the n_cols down_proj columns j with the largest |E_loop[a_j] * (u_l . W[:, j])| (the neurons that write u_l most)
  edit: W[:, cols] -= alpha * u_l u_l^T W[:, cols]
  layer choice: top `candidate_layers` layers by d'; every pair of the top 4 is screened by one practice eval call on
  `screen_ids` problems (disjoint from the collect set); the pair with the best accuracy (then completion) is kept.
"""
import gzip, itertools, json, os, random, sys, time

import torch

import evaluate as E
import harness as H

CFG = E.CFG
OUT = os.path.join(E.P("results"), "baselines")


def practice_sample(n, seed, exclude=()):
    """n practice ids, half GSM8K train and half MATH train, fixed by seed."""
    prac = E.load_practice()
    rng = random.Random(seed)
    g = sorted(k for k in prac if k.startswith("gsm8k_train_") and k not in exclude)
    m = sorted(k for k in prac if k.startswith("math_train_") and k not in exclude)
    return rng.sample(g, n // 2) + rng.sample(m, n - n // 2)


def collect(target=None):
    path = os.path.join(OUT, "collect", "gens.json.gz")
    if os.path.exists(path):
        return json.load(gzip.open(path, "rt"))
    s = CFG["scripted"]
    ids = practice_sample(s["collect_ids"], s["collect_seed"])
    prac = E.load_practice()
    t = target or E.Target()
    t0 = time.time()
    res = t.generate([prac[i] for i in ids], 1, CFG["target_eval"]["seed"])
    gens = [{"id": i, "finish": r[0]["finish"], "n_tokens": r[0]["n_tokens"], "correct": r[0]["correct"],
             "text": r[0]["text"]} for i, r in zip(ids, res)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(gens, gzip.open(path, "wt"))
    summ = E.summarize([{"id": g["id"], "samples": [g]} for g in gens])
    summ["seconds"] = round(time.time() - t0)
    json.dump(summ, open(os.path.join(OUT, "collect", "summary.json"), "w"), indent=1)
    print("collect", json.dumps(summ), flush=True)
    return gens


def windows(gens, tok, prac, fmt):
    """(texts, class) with token windows: list of (token ids, start, end, is_loop)."""
    s = CFG["scripted"]
    W = s["window_tokens"]
    out = []
    think_end = tok.convert_tokens_to_ids("</think>")
    loop = [g for g in gens if g["finish"] == "length"][: s["max_texts_per_class"]]
    prod = [g for g in gens if g["finish"] == "stop"][: s["max_texts_per_class"]]
    for g, is_loop in [(g, True) for g in loop] + [(g, False) for g in prod]:
        p = tok(fmt(prac[g["id"]]["problem"]), add_special_tokens=False)["input_ids"]
        a = tok(g["text"], add_special_tokens=False)["input_ids"]
        if is_loop:
            end = len(a)
        else:
            end = a.index(think_end) if think_end in a else len(a)
        start = max(0, end - W)
        ids = (p + a[:end])[-(end - start) - 2048:]           # keep up to 2048 tokens of left context before the window
        out.append((ids, len(ids) - (end - start), len(ids), is_loop))
    return out


def layer_stats(hf, wins):
    """per text and layer: mean mlp_out [1024] and mean mlp_in [3584] over the window positions."""
    L = len(hf.model.layers)
    cache = {}
    hooks = [hf.model.layers[l].mlp.down_proj.register_forward_hook(
        lambda m, i, o, l=l: cache.__setitem__(l, (i[0][0], o[0]))) for l in range(L)]
    per = []
    try:
        with torch.no_grad():
            for ids, a, b, is_loop in wins:
                cache.clear()
                hf(torch.tensor([ids], device=hf.device))
                per.append({"loop": is_loop, "out": torch.stack([cache[l][1][a:b].float().mean(0).cpu() for l in range(L)]),
                            "in": torch.stack([cache[l][0][a:b].float().mean(0).cpu() for l in range(L)])})
    finally:
        for h in hooks:
            h.remove()
    return per


def directions(per, base):
    L = per[0]["out"].shape[0]
    lo = torch.stack([p["out"] for p in per if p["loop"]])
    pr = torch.stack([p["out"] for p in per if not p["loop"]])
    a_lo = torch.stack([p["in"] for p in per if p["loop"]]).mean(0)
    info = []
    for l in range(L):
        u = lo[:, l].mean(0) - pr[:, l].mean(0)
        u = u / u.norm()
        pl, pp = lo[:, l] @ u, pr[:, l] @ u
        dprime = float((pl.mean() - pp.mean()) / ((pl.var() + pp.var()) / 2).sqrt())
        W = base[f"model.language_model.layers.{l}.mlp.down_proj.weight"].float()
        score = (a_lo[l] * (u @ W)).abs()
        cols = score.topk(CFG["scripted"]["n_cols"]).indices
        info.append({"layer": l, "u": u, "dprime": dprime, "cols": cols})
    return info


def project_edit(base, layer, u, cols, alpha):
    n = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    return n, H.project_out(base[n].float(), u, cols, alpha)


def scripted():
    s = CFG["scripted"]
    out_dir = os.path.join(OUT, "scripted")
    os.makedirs(out_dir, exist_ok=True)
    target = E.Target()
    gens = collect(target)
    sess = H.Session(os.path.join(out_dir, "log"), level="L1", target=target)
    prac = sess.tools.practice
    wins = windows(gens, sess.tok, prac, sess.tools.format_prompt)
    n_loop = sum(w[3] for w in wins)
    print("windows: loop", n_loop, "productive", len(wins) - n_loop, flush=True)
    if n_loop < 8 or len(wins) - n_loop < 8:
        raise RuntimeError("too few looping or productive practice answers for a direction")
    per = layer_stats(sess.hf, wins)
    info = directions(per, sess.state.base)
    torch.save(info, os.path.join(out_dir, "directions.pt"))
    ranked = sorted(info, key=lambda x: -x["dprime"])
    cand = [x["layer"] for x in ranked[: s["candidate_layers"]]]
    pairs = list(itertools.combinations(sorted(cand[:4]), s["n_layers"]))
    screen_ids = practice_sample(s["screen_ids"], s["screen_seed"], exclude={g["id"] for g in gens})
    screens = []
    for pair in pairs:
        sess.state.revert("all")
        for l in pair:
            sess.state.apply(*project_edit(sess.state.base, l, info[l]["u"], info[l]["cols"], s["alpha"]))
        r = sess.tools.eval(screen_ids)
        rec = {"layers": list(pair), "acc": sum(x["correct"] for x in r) / len(r),
               "completion": sum(x["completed"] for x in r) / len(r)}
        screens.append(rec)
        print("screen", rec, flush=True)
    best = max(screens, key=lambda x: (x["acc"], x["completion"]))
    sess.state.revert("all")
    for l in best["layers"]:
        sess.state.apply(*project_edit(sess.state.base, l, info[l]["u"], info[l]["cols"], s["alpha"]))
    sel = {"dprime_by_layer": {x["layer"]: x["dprime"] for x in info}, "candidates": cand, "pairs_screened": screens,
           "chosen_layers": best["layers"], "n_loop_texts": n_loop, "n_prod_texts": len(wins) - n_loop,
           "screen_ids": screen_ids, "eval_calls_used": sess.budget.used["eval_calls"]}
    json.dump(sel, open(os.path.join(out_dir, "selection.json"), "w"), indent=1)
    man = sess.state.save_checkpoint(os.path.join(out_dir, "final"), extra={"method": "scripted_diff_of_means", **{
        k: sel[k] for k in ("chosen_layers",)}})
    print("scripted edit", best, "changed", man["n_changed"], f"{man['pct_changed']:.4f}%", flush=True)
    E.run_heldout(target, os.path.join(out_dir, "heldout"), ["math500"], label="scripted")
    delta_norm = sum(float((sess.state.current[n].float() - sess.state.base[n].float()).norm() ** 2)
                     for n in sess.state.current) ** 0.5

    # random edit: same layers, random unit direction, random columns, Frobenius norm of the change matched
    rd = os.path.join(OUT, "random")
    g = torch.Generator().manual_seed(CFG["random_edit"]["seed"])
    sess.state.revert("all")
    rsess_state = H.EditState(sess.state.base, log_dir=os.path.join(rd, "log"), on_change=sess.state.on_change)
    raw = {}
    for l in best["layers"]:
        n = f"model.language_model.layers.{l}.mlp.down_proj.weight"
        u = torch.randn(1024, generator=g)
        cols = torch.randperm(3584, generator=g)[: s["n_cols"]]
        raw[n] = (H.project_out(rsess_state.base[n].float(), u, cols) - rsess_state.base[n].float())
    rnorm = sum(float(d.norm() ** 2) for d in raw.values()) ** 0.5
    for n, d in raw.items():
        rsess_state.apply(n, rsess_state.base[n].float() + d * (delta_norm / rnorm))
    man_r = rsess_state.save_checkpoint(os.path.join(rd, "final"), extra={
        "method": "random_same_size", "layers": best["layers"], "target_frobenius": delta_norm})
    print("random edit changed", man_r["n_changed"], flush=True)
    E.run_heldout(target, os.path.join(rd, "heldout"), ["math500"], label="random")
    rsess_state.revert("all")


# ------------------------------------------------------------------ MathNeuro-style scaling
def mathneuro():
    c = CFG["mathneuro"]
    out_dir = os.path.join(OUT, "mathneuro")
    os.makedirs(out_dir, exist_ok=True)
    hf, tok = H.hf_model()
    prac = E.load_practice()
    rng = random.Random(0)
    mids = rng.sample(sorted(k for k in prac if k.startswith("math_train_") or k.startswith("gsm8k_train_")), c["math_texts"])
    math_texts = [prac[i]["problem"] + "\n" + prac[i]["solution"] for i in mids]
    gen_ids = E.wikitext_passages(tok, c["general_texts"], 512)       # wikitext validation passages
    import pandas as pd, glob
    # wikitext TRAIN for the importance pass (validation is used for KL)
    df = pd.read_parquet(sorted(glob.glob(f"{E.P('data')}/wikitext/wikitext-103-raw-v1/train-*.parquet"))[0])
    tr = tok("".join(df["text"].tolist()[:20000]))["input_ids"]
    gen_ids = [tr[i * 512:(i + 1) * 512] for i in range(c["general_texts"])]
    targets = [(l, t) for l in range(len(hf.model.layers)) for t in c["targets"]]

    def act_norms(seqs):
        """per (layer, module): sum over tokens of x_j^2 for each input feature j (Wanda / MathNeuro statistic)."""
        acc = {}
        hooks = []
        for l, t in targets:
            mod = hf.model.layers[l].get_submodule(t)
            hooks.append(mod.register_forward_hook(lambda m, i, o, key=(l, t): acc.__setitem__(
                key, acc.get(key, 0) + (i[0].float() ** 2).reshape(-1, i[0].shape[-1]).sum(0))))
        with torch.no_grad():
            for ids in seqs:
                hf(torch.tensor([ids[:1024]], device=hf.device))
        for h in hooks:
            h.remove()
        return {k: v.sqrt() for k, v in acc.items()}

    m_ids = [tok(t, add_special_tokens=False)["input_ids"] for t in math_texts]
    am, ag = act_norms(m_ids), act_norms(gen_ids)
    base = H.load_base()
    st = H.EditState(base, log_dir=os.path.join(out_dir, "log"))
    for l, t in targets:
        n = f"model.language_model.layers.{l}.{t}.weight"
        W = base[n].float().cuda()
        k = max(1, int(c["topk_frac"] * W.shape[1]))
        im, ig = W.abs() * am[(l, t)][None], W.abs() * ag[(l, t)][None]
        top_m = torch.zeros_like(W, dtype=torch.bool).scatter_(1, im.topk(k, dim=1).indices, True)
        top_g = torch.zeros_like(W, dtype=torch.bool).scatter_(1, ig.topk(k, dim=1).indices, True)
        mask = top_m & ~top_g
        st.apply(n, torch.where(mask, W * c["scale"], W).cpu())
    man = st.save_checkpoint(os.path.join(out_dir, "final"), extra={"method": "mathneuro", **c})
    print("mathneuro changed", man["n_changed"], f"{man['pct_changed']:.3f}%", flush=True)
    del hf
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ LoRA SFT (gradient baseline)
def lora():
    """LoRA SFT on practice reference solutions in the evaluation's chat format (thinking block = the reference
    solution, answer = the boxed final answer); merged and stored as an edit checkpoint."""
    from peft import LoraConfig, get_peft_model
    c = CFG["lora"]
    out_dir = os.path.join(OUT, "lora")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(c["seed"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(E.P("target"))
    model = AutoModelForCausalLM.from_pretrained(E.P("target"), dtype=torch.bfloat16).cuda()
    prac = E.load_practice()
    rng = random.Random(c["seed"])
    ids = rng.sample(sorted(k for k in prac if k.startswith(("gsm8k_train_", "math_train_"))), c["n_train"])
    te = CFG["target_eval"]
    data = []
    for i in ids:
        r = prac[i]
        prompt = tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\n" + te["instruction"]}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)
        resp = r["solution"].strip() + "\n</think>\n\nThe final answer is $\\boxed{" + r["answer"] + "}$.<|im_end|>"
        p, a = tok(prompt, add_special_tokens=False)["input_ids"], tok(resp, add_special_tokens=False)["input_ids"]
        x = (p + a)[: c["max_len"]]
        data.append((x, len(p)))
    pcfg = LoraConfig(r=c["r"], lora_alpha=c["alpha"], lora_dropout=c["dropout"], target_modules=c["targets"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, pcfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c["lr"])
    steps = (len(data) // c["batch_size"]) * c["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    model.train()
    log = []
    step = 0
    for ep in range(c["epochs"]):
        rng.shuffle(data)
        for b in range(0, len(data) - c["batch_size"] + 1, c["batch_size"]):
            opt.zero_grad()
            tot = 0.0
            for x, plen in data[b:b + c["batch_size"]]:          # micro-batches of one (no padding with linear attention)
                inp = torch.tensor([x], device="cuda")
                lab = inp.clone()
                lab[0, :plen] = -100
                loss = model(input_ids=inp, labels=lab).loss / c["batch_size"]
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                log.append({"step": step, "loss": tot})
                print("lora step", step, "/", steps, "loss", round(tot, 4), flush=True)
    model = model.merge_and_unload()
    sd = {n: p.detach().cpu() for n, p in model.named_parameters()}
    base = H.load_base()
    st = H.EditState(base, log_dir=os.path.join(out_dir, "log"), refuse_rescale=False)
    for n in base:
        hn = H.hf_name(n)
        if hn in sd and not torch.equal(H.bits(sd[hn].to(base[n].dtype)), H.bits(base[n])):
            st.apply(n, sd[hn])
    json.dump(log, open(os.path.join(out_dir, "train_log.json"), "w"))
    n_lora = sum(p.numel() for p in get_peft_model(AutoModelForCausalLM.from_pretrained(E.P("target"), dtype=torch.bfloat16),
                                                     pcfg).parameters() if p.requires_grad)
    man = st.save_checkpoint(os.path.join(out_dir, "final"), extra={"method": "lora_sft", "n_lora_params": n_lora, **c})
    print("lora changed", man["n_changed"], f"{man['pct_changed']:.2f}%", "lora params", n_lora, flush=True)


def heldout(name, sets):
    d = os.path.join(OUT, name)
    t = E.Target()
    t.load_tensors(E.load_ckpt_tensors(os.path.join(d, "final")))
    E.run_heldout(t, os.path.join(d, "heldout"), sets, label=name)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        collect()
    elif cmd == "scripted":
        scripted()
    elif cmd == "mathneuro":
        mathneuro()
    elif cmd == "lora":
        lora()
    elif cmd == "heldout":
        heldout(sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "math500").split(","))
