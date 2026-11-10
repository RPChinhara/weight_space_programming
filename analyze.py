"""Tables and paired tests from the logged runs in results/. Writes results/analysis.json and results/tables.md.

Paired design: every checkpoint was sampled with the same per-(problem, sample) seeds, so samples are paired.
  - paired bootstrap over problems (resample the 500 MATH-500 problems; statistic = difference in mean accuracy)
  - McNemar exact test on the paired per-sample outcomes (samples of one problem are not independent, so the bootstrap is
    the primary test; McNemar is reported as the spec asks)
Editor conditions: 5 seeds each; per-seed numbers, mean +/- SD (and min-max) over seeds, and a pooled test on the
per-problem accuracy averaged over the seeds.
"""
import glob, json, os

import numpy as np
from scipy.stats import binomtest

import evaluate as E

CFG = E.CFG
R = E.P("results")


def load_per(d, name="math500"):
    f = os.path.join(d, f"{name}_per_problem.json")
    if not os.path.exists(f):
        return None
    per = json.load(open(f))
    return {r["id"]: r["samples"] for r in per}


def arr(per, ids):
    """[problems, samples] arrays of correct and finished."""
    c = np.array([[x["correct"] for x in per[i]] for i in ids], dtype=float)
    f = np.array([[x["finish"] == "stop" for x in per[i]] for i in ids], dtype=float)
    return c, f


def metrics(per, ids):
    c, f = arr(per, ids)
    n_done = f.sum()
    return {"acc": c.mean(), "completion_rate": f.mean(), "cut_share": 1 - f.mean(),
            "completed_only_acc": (c * f).sum() / n_done if n_done else float("nan"),
            "acc_on_cut": (c * (1 - f)).sum() / (1 - f).sum() if (1 - f).sum() else float("nan"),
            "mean_tokens": float(np.mean([[x["n_tokens"] for x in per[i]] for i in ids]))}


def boot(a, b, n, seed):
    """paired bootstrap over problems: a, b = per-problem mean accuracy. Returns diff, 95% CI, two-sided p."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"diff": float(d.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "p_boot": float(min(1.0, max(p, 1.0 / n)))}


def mcnemar(ca, cb):
    """exact McNemar on paired samples (flattened)."""
    a, b = ca.flatten().astype(bool), cb.flatten().astype(bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only": n10, "b_only": n01, "p_mcnemar": float(p)}


def conditions():
    """name -> list of (label, dir with heldout files, manifest path, run.json or None)."""
    out = {"base": [("base", os.path.join(R, "base", "heldout"), None, None)]}
    for b in ("scripted", "random", "mathneuro", "lora"):
        d = os.path.join(R, "baselines", b)
        if os.path.exists(d):
            out[b] = [(b, os.path.join(d, "heldout"), os.path.join(d, "final", "manifest.json"), None)]
    for size in CFG["editor_sizes"]:
        for level in CFG["scaffolds"]:
            runs = []
            for s in CFG["seeds"]:
                d = os.path.join(R, "runs", f"editor_{size}_{level}_s{s}")
                if os.path.exists(d):
                    runs.append((f"s{s}", os.path.join(d, "heldout"), os.path.join(d, "final", "manifest.json"),
                                 os.path.join(d, "run.json")))
            if runs:
                out[f"{size}-{level}"] = runs
    return out


def failure_mode(run, man, m, base_m, test):
    """category of an editor run (kept, never dropped)."""
    if run is None:
        return "missing"
    if run.get("end_reason") == "crash":
        return "crash"
    if man is None or man.get("n_changed", 0) == 0:
        return "no_edit"
    if m is None:
        return "no_heldout_eval"
    if test["p_boot"] < 0.05 and test["diff"] < 0:
        return "broke_model" if m["acc"] < 0.5 * base_m["acc"] else "significant_regression"
    if test["p_boot"] < 0.05 and test["diff"] > 0:
        return "significant_gain"
    return "edit_no_significant_change"


def main():
    a = CFG["analysis"]
    conds = conditions()
    base = load_per(conds["base"][0][1])
    if base is None:
        raise SystemExit("no base held-out eval yet")
    ids = sorted(base)
    cb, _ = arr(base, ids)
    scripted = load_per(os.path.join(R, "baselines", "scripted", "heldout"))
    cs = arr(scripted, ids)[0] if scripted else None
    res = {"n_problems": len(ids), "conditions": {}}
    for name, runs in conds.items():
        rows = []
        accs = []
        for label, d, manp, runp in runs:
            per = load_per(d)
            man = json.load(open(manp)) if manp and os.path.exists(manp) else None
            run = json.load(open(runp)) if runp and os.path.exists(runp) else None
            row = {"label": label, "pct_changed": man["pct_changed"] if man else 0.0,
                   "n_changed": man["n_changed"] if man else 0}
            if run:
                row.update(end_reason=run.get("end_reason"), turns=run.get("turns"), budget_used=run.get("budget_used"),
                           status_counts=run.get("status_counts"))
            for other in ("gsm8k_test", "ifeval"):
                po = load_per(d, other)
                if po:
                    pb = load_per(conds["base"][0][1], other)
                    oids = sorted(po)
                    row[other] = metrics(po, oids)
                    if pb and name != "base":
                        row[other + "_vs_base"] = boot(arr(po, oids)[0].mean(1), arr(pb, oids)[0].mean(1), a["bootstrap"], a["seed"])
            klf = os.path.join(os.path.dirname(d), "kl.json")
            if os.path.exists(klf):
                row["kl"] = json.load(open(klf))
            if per is not None:
                c, _ = arr(per, ids)
                accs.append(c.mean(1))
                row["math500"] = metrics(per, ids)
                if name != "base":
                    row["vs_base"] = dict(boot(c.mean(1), cb.mean(1), a["bootstrap"], a["seed"]), **mcnemar(c, cb))
                    if cs is not None and name != "scripted":
                        row["vs_scripted"] = dict(boot(c.mean(1), cs.mean(1), a["bootstrap"], a["seed"]), **mcnemar(c, cs))
            if run is not None:
                row["failure_mode"] = failure_mode(run, man, row.get("math500"), metrics(base, ids), row.get("vs_base", {"p_boot": 1, "diff": 0}))
            rows.append(row)
        agg = {}
        done = [r for r in rows if "math500" in r]
        for k in ("acc", "completion_rate", "completed_only_acc", "mean_tokens"):
            v = np.array([r["math500"][k] for r in done], dtype=float)
            if len(v):
                agg[k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                          "min": float(np.nanmin(v)), "max": float(np.nanmax(v)), "n": int(len(v))}
        agg["pct_changed"] = {"mean": float(np.mean([r["pct_changed"] for r in rows])),
                              "max": float(np.max([r["pct_changed"] for r in rows]))}
        if accs and name != "base":
            pooled = np.mean(accs, 0)
            agg["pooled_vs_base"] = boot(pooled, cb.mean(1), a["bootstrap"], a["seed"])
            if cs is not None and name != "scripted":
                agg["pooled_vs_scripted"] = boot(pooled, cs.mean(1), a["bootstrap"], a["seed"])
        if any("failure_mode" in r for r in rows):
            fm = {}
            for r in rows:
                fm[r.get("failure_mode")] = fm.get(r.get("failure_mode"), 0) + 1
            agg["failure_modes"] = fm
        res["conditions"][name] = {"runs": rows, "agg": agg}
    json.dump(res, open(os.path.join(R, "analysis.json"), "w"), indent=1, default=float)
    open(os.path.join(R, "tables.md"), "w").write(tables(res))
    print(tables(res))


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def tables(res):
    L = []
    L.append("### Headline: MATH-500 (4 samples x 500 problems; mean over seeds for editor conditions)\n")
    L.append("| Condition | Runs | Acc % (mean ± SD) | Min-max | Completion % | Completed-only acc % | Δ vs base (pooled, 95% CI) | p | Δ vs scripted | p | % weights edited (mean) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, c in res["conditions"].items():
        g = c["agg"]
        if "acc" not in g:
            L.append(f"| {name} | {len(c['runs'])} | not evaluated | | | | | | | | |")
            continue
        vb = g.get("pooled_vs_base") or {}
        vs = g.get("pooled_vs_scripted") or {}
        L.append(f"| {name} | {g['acc']['n']} | {pct(g['acc']['mean'])} ± {pct(g['acc']['sd'])} | {pct(g['acc']['min'])}-{pct(g['acc']['max'])} | "
                 f"{pct(g['completion_rate']['mean'])} | {pct(g['completed_only_acc']['mean'])} | "
                 + (f"{100*vb['diff']:+.1f} [{100*vb['ci95'][0]:+.1f}, {100*vb['ci95'][1]:+.1f}] | {vb['p_boot']:.3g} | " if vb else "- | - | ")
                 + (f"{100*vs['diff']:+.1f} | {vs['p_boot']:.3g} | " if vs else "- | - | ")
                 + f"{g['pct_changed']['mean']:.4f} |")
    L.append("\n### Per run (MATH-500)\n")
    L.append("| Condition | Run | Acc % | Completion % | Completed-only acc % | Mean tokens | Δ vs base | p (boot) | p (McNemar) | Δ vs scripted | p | Entries edited | % | End | Evals used | Execs used | Failure mode |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, c in res["conditions"].items():
        for r in c["runs"]:
            m = r.get("math500")
            vb, vs = r.get("vs_base", {}), r.get("vs_scripted", {})
            bu = r.get("budget_used") or {}
            L.append(f"| {name} | {r['label']} | " + (f"{pct(m['acc'])} | {pct(m['completion_rate'])} | {pct(m['completed_only_acc'])} | {m['mean_tokens']:.0f} | " if m else "not evaluated | | | | ")
                     + (f"{100*vb['diff']:+.1f} | {vb['p_boot']:.3g} | {vb['p_mcnemar']:.3g} | " if vb else "- | - | - | ")
                     + (f"{100*vs['diff']:+.1f} | {vs['p_boot']:.3g} | " if vs else "- | - | ")
                     + f"{r['n_changed']} | {r['pct_changed']:.4f} | {r.get('end_reason', '-')} | {bu.get('eval_calls', '-')} | {bu.get('python_execs', '-')} | {r.get('failure_mode', '-')} |")
    L.append("\n### Collateral (all final checkpoints)\n")
    L.append("| Condition | Run | GSM8K test acc % | GSM8K completion % | IFEval strict prompt % | KL(base‖edit) nats/token (wikitext) | PPL base → edit |")
    L.append("|---|---|---|---|---|---|---|")
    for name, c in res["conditions"].items():
        for r in c["runs"]:
            g8, ife, kl = r.get("gsm8k_test"), r.get("ifeval"), r.get("kl")
            kl_s = f"{kl['kl_base_to_edited_per_token']:.4f} | {kl['ppl_base']:.2f} → {kl['ppl_edited']:.2f}" if kl else "- | -"
            L.append(f"| {name} | {r['label']} | {pct(g8['acc']) if g8 else '-'} | {pct(g8['completion_rate']) if g8 else '-'} | "
                     f"{pct(ife['acc']) if ife else '-'} | {kl_s} |")
    fm = {n: c["agg"].get("failure_modes") for n, c in res["conditions"].items() if c["agg"].get("failure_modes")}
    if fm:
        L.append("\n### Failure modes of editor runs\n")
        L.append("| Condition | " + " | ".join(sorted({k for v in fm.values() for k in v})) + " |")
        keys = sorted({k for v in fm.values() for k in v})
        L.append("|---|" + "---|" * len(keys))
        for n, v in fm.items():
            L.append(f"| {n} | " + " | ".join(str(v.get(k, 0)) for k in keys) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
