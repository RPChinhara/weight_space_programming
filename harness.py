"""The editor's sandbox: tools over the target's weights, no-gradient enforcement, budget counter, edit diff log.

Pieces (each usable on its own, which the tests rely on):
  EditState  the target's tensors (base + edits), every apply/revert logged as an exact bf16 diff; rebuild() replays a
             log onto the base and must give identical tensors
  Budget     eval calls (each <= eval_max_ids practice ids) and Python executions
  Tools      the editor-facing functions: forward, eval, get_tensor, apply_edit, revert (+ L1 helpers)
  Sandbox    executes the editor's code: AST check first (imports, gradients, file access, dunders), then exec with
             restricted builtins, under torch.no_grad(), with backward / autograd / optimizers patched to raise

Tensor names are the checkpoint's (model.language_model.layers.N....); the HF model uses model.layers.N....
"""
import ast, builtins, contextlib, gzip, hashlib, io, json, os, signal, time, traceback

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
import evaluate as E  # noqa: E402

CFG = E.CFG


class SandboxViolation(Exception):
    pass


class BudgetExceeded(Exception):
    pass


def bits(t):
    """integer view of a tensor's bit pattern (exact comparisons, also for bf16 and NaN-free floats)."""
    return t.contiguous().view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def sha(t):
    return hashlib.sha256(bits(t).numpy().tobytes()).hexdigest()


# ------------------------------------------------------------------ edit state and diff log
class EditState:
    def __init__(self, base, log_dir=None, frozen=(), refuse_rescale=True, on_change=None):
        """base: {name: tensor} (CPU; bf16, a few float32 norms). log_dir: where edit_log.jsonl and diffs/ are written (None = no log)."""
        self.base = base
        self.current = {}                      # name -> bf16 tensor, only for edited names
        self.frozen = set(frozen)
        self.refuse_rescale = refuse_rescale
        self.on_change = on_change             # callback({name: tensor}) pushing tensors to the engines
        self.log_dir = log_dir
        self.step = 0
        if log_dir:
            os.makedirs(os.path.join(log_dir, "diffs"), exist_ok=True)
            open(os.path.join(log_dir, "edit_log.jsonl"), "w").close()

    def names(self):
        return list(self.base)

    def get(self, name):
        if name not in self.base:
            raise KeyError(f"no tensor named {name!r}; see list_tensors()")
        return self.current.get(name, self.base[name]).clone()

    def _log(self, entry, diff=None):
        self.step += 1
        entry = dict(entry, step=self.step, time=time.time())
        if self.log_dir:
            if diff is not None:
                f = f"diffs/{self.step:04d}.pt"
                torch.save(diff, os.path.join(self.log_dir, f))
                entry["diff_file"] = f
            with open(os.path.join(self.log_dir, "edit_log.jsonl"), "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        return entry

    def apply(self, name, new):
        if name not in self.base:
            raise KeyError(f"no tensor named {name!r}")
        if name in self.frozen:
            raise SandboxViolation(f"{name} is frozen (embeddings / output layer / final norm cannot be edited)")
        new = torch.as_tensor(new).detach().cpu()
        b = self.base[name]
        if tuple(new.shape) != tuple(b.shape):
            raise ValueError(f"shape {tuple(new.shape)} != {tuple(b.shape)}")
        if not torch.isfinite(new.float()).all():
            raise ValueError("new tensor has NaN or inf")
        new = new.to(b.dtype)
        if self.refuse_rescale:
            bf = b.float()
            denom = (bf * bf).sum()
            if denom > 0:
                c = (new.float() * bf).sum() / denom
                if not torch.equal(new, b) and torch.equal((c * bf).to(b.dtype), new):
                    raise SandboxViolation("edits that only multiply a whole tensor by a constant are refused")
        cur = self.current.get(name, b)
        idx = (bits(new) != bits(cur)).flatten().nonzero().flatten()
        diff = {"name": name, "idx": idx, "val": new.flatten()[idx].clone()}
        self.current[name] = new
        if torch.equal(bits(new), bits(b)):
            del self.current[name]
        e = self._log({"op": "apply", "name": name, "n_changed_vs_prev": int(idx.numel()),
                       "n_changed_vs_base": self.changed(name), "sha256_after": sha(new)}, diff)
        if self.on_change:
            self.on_change({name: new})
        return e

    def revert(self, name="all"):
        names = list(self.current) if name == "all" else [name]
        for n in names:
            if n not in self.base:
                raise KeyError(n)
            cur = self.current.pop(n, None)
            if cur is None:
                continue
            idx = (bits(cur) != bits(self.base[n])).flatten().nonzero().flatten()
            diff = {"name": n, "idx": idx, "val": self.base[n].flatten()[idx].clone()}
            self._log({"op": "revert", "name": n, "n_changed_vs_prev": int(idx.numel()), "n_changed_vs_base": 0,
                       "sha256_after": sha(self.base[n])}, diff)
            if self.on_change:
                self.on_change({n: self.base[n]})
        return names

    def changed(self, name=None):
        """number of entries that differ from the base (one tensor or all)."""
        names = [name] if name else list(self.current)
        tot = 0
        for n in names:
            if n in self.current:
                tot += int((bits(self.current[n]) != bits(self.base[n])).sum())
        return tot

    def save_checkpoint(self, out_dir, extra=None):
        from safetensors.torch import save_file
        os.makedirs(out_dir, exist_ok=True)
        tensors = {k: v.contiguous() for k, v in self.current.items()}
        if tensors:
            save_file(tensors, os.path.join(out_dir, "edited.safetensors"))
        total = sum(v.numel() for v in self.base.values())
        man = {"tensors": {k: {"shape": list(v.shape), "n_changed": self.changed(k), "sha256": sha(v),
                               "sha256_base": sha(self.base[k])} for k, v in self.current.items()},
               "n_changed": self.changed(), "n_params_total": total,
               "pct_changed": 100.0 * self.changed() / total, **(extra or {})}
        json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
        return man

    @staticmethod
    def rebuild(base, log_dir):
        """replay edit_log.jsonl + diffs onto the base: {name: tensor} of every tensor that differs from the base."""
        cur = {}
        for line in open(os.path.join(log_dir, "edit_log.jsonl")):
            e = json.loads(line)
            if "diff_file" not in e:
                continue
            d = torch.load(os.path.join(log_dir, e["diff_file"]))
            n = d["name"]
            t = cur.get(n, base[n]).clone().flatten()
            t[d["idx"]] = d["val"]
            cur[n] = t.view(base[n].shape)
            if sha(cur[n]) != e["sha256_after"]:
                raise RuntimeError(f"rebuild mismatch at step {e['step']} ({n})")
        return {k: v for k, v in cur.items() if not torch.equal(bits(v), bits(base[k]))}


def load_base():
    from safetensors.torch import load_file
    return load_file(os.path.join(E.P("target"), "model.safetensors"))


# ------------------------------------------------------------------ budget
class Budget:
    def __init__(self, eval_calls, eval_max_ids, python_execs):
        self.limits = {"eval_calls": eval_calls, "python_execs": python_execs}
        self.eval_max_ids = eval_max_ids
        self.used = {"eval_calls": 0, "python_execs": 0}

    def take(self, kind):
        if self.used[kind] >= self.limits[kind]:
            raise BudgetExceeded(f"{kind} budget exhausted ({self.used[kind]}/{self.limits[kind]})")
        self.used[kind] += 1

    def left(self, kind):
        return self.limits[kind] - self.used[kind]

    def summary(self):
        return {k: f"{self.used[k]}/{self.limits[k]} used" for k in self.used}


# ------------------------------------------------------------------ L1 helpers (generic linear algebra)
def diff_of_means(acts_a, acts_b):
    """unit vector along mean(acts_a) - mean(acts_b). acts: a tensor [N, d] or a list of tensors [*, d]."""
    def stack(a):
        if isinstance(a, (list, tuple)):
            a = torch.cat([torch.as_tensor(x).reshape(-1, torch.as_tensor(x).shape[-1]).float() for x in a])
        return torch.as_tensor(a).float().reshape(-1, a.shape[-1])
    v = stack(acts_a).mean(0) - stack(acts_b).mean(0)
    if not torch.isfinite(v).all() or v.norm() < 1e-8:
        raise ValueError("diff_of_means: the two sets have (almost) the same mean; no direction")
    return v / v.norm()


def project_out(W, v, cols, alpha=1.0):
    """W' = W with alpha * v v^T removed from the given columns: W'[:, j] = W[:, j] - alpha * v (v . W[:, j]).
    W: [d_out, d_in] (e.g. mlp.down_proj.weight, whose columns are what each MLP neuron writes), v: [d_out]."""
    W = torch.as_tensor(W).float().clone()
    v = torch.as_tensor(v).float()
    if not torch.isfinite(v).all() or v.norm() < 1e-8:
        raise ValueError("project_out: v must be a finite, non-zero vector")
    v = v / v.norm()
    cols = torch.as_tensor(cols, dtype=torch.long)
    sub = W[:, cols]
    W[:, cols] = sub - alpha * torch.outer(v, v @ sub)
    return W


# ------------------------------------------------------------------ tools
class Tools:
    """editor-facing tools. target: evaluate.Target (vLLM) or None; hf: HF model or None (forward unavailable)."""

    def __init__(self, state, budget, target=None, hf=None, tok=None, run_dir=None, practice=None, seed=0):
        self.state, self.budget, self.target, self.hf, self.tok = state, budget, target, hf, tok
        self.run_dir = run_dir
        self.practice = practice if practice is not None else E.load_practice()
        self.seed = seed
        self.n_eval = 0
        self.eval_log = []
        b = CFG["budget"]
        self.fwd_limits = (b["forward_max_texts"], b["forward_max_tokens_per_text"], b["forward_max_total_tokens"])

    # -- weights
    def list_tensors(self, pattern=""):
        return [(n, tuple(t.shape)) for n, t in self.state.base.items() if pattern in n]

    def get_tensor(self, name):
        """a float32 copy of the tensor's current value."""
        return self.state.get(name).float()

    def apply_edit(self, name, new_tensor):
        e = self.state.apply(name, new_tensor)
        return {"name": name, "entries_changed_vs_base": e["n_changed_vs_base"],
                "total_entries_changed": self.state.changed()}

    def revert(self, name="all"):
        return self.state.revert(name)

    def edit_summary(self):
        return {n: self.state.changed(n) for n in self.state.current}

    # -- evaluation on practice problems
    def eval(self, practice_ids, original=False):
        """generate one answer per practice id with the evaluation's sampling; returns a list of dicts
        {id, correct, completed, n_tokens, pred, answer, text}. Costs one eval call (<= eval_max_ids ids).
        original=True: the unedited model on the same ids (same random seeds), also one call."""
        ids = list(practice_ids)
        bad = [i for i in ids if i not in self.practice]
        if bad:
            raise ValueError(f"eval accepts practice ids only; rejected: {bad[:5]}")
        if not ids:
            raise ValueError("no ids")
        if len(ids) > self.budget.eval_max_ids:
            raise ValueError(f"at most {self.budget.eval_max_ids} ids per eval call (got {len(ids)})")
        self.budget.take("eval_calls")
        rows = [self.practice[i] for i in ids]
        t0 = time.time()
        if original and self.state.current:
            self.target.load_tensors({n: self.state.base[n] for n in self.state.current})
            try:
                res = self.target.generate(rows, 1, CFG["target_eval"]["seed"], partial_path=self._partial(), tag=f"eval{self.n_eval + 1}-orig")
            finally:
                self.target.load_tensors(dict(self.state.current))
        else:
            res = self.target.generate(rows, 1, CFG["target_eval"]["seed"], partial_path=self._partial(),
                                       tag=f"eval{self.n_eval + 1}" + ("-orig" if original else ""))
        out = [{"id": r["id"], "correct": bool(s[0]["correct"]), "completed": s[0]["finish"] == "stop",
                "n_tokens": s[0]["n_tokens"], "pred": s[0]["pred"], "answer": r["answer"], "text": s[0]["text"]}
               for r, s in zip(rows, res)]
        self.n_eval += 1
        rec = {"call": self.n_eval, "original": original, "edits": self.edit_summary(), "n": len(out),
               "acc": sum(o["correct"] for o in out) / len(out),
               "completion_rate": sum(o["completed"] for o in out) / len(out), "seconds": round(time.time() - t0)}
        self.eval_log.append(rec)
        if self.run_dir:
            os.makedirs(os.path.join(self.run_dir, "evals"), exist_ok=True)
            with gzip.open(os.path.join(self.run_dir, "evals", f"{self.n_eval:03d}.json.gz"), "wt") as f:
                json.dump({"summary": rec, "results": out}, f)
        return out

    def _partial(self):
        if not self.run_dir:
            return None
        os.makedirs(os.path.join(self.run_dir, "evals"), exist_ok=True)
        return os.path.join(self.run_dir, "evals", f"{self.n_eval + 1:03d}_partial.jsonl")

    # -- teacher-forced forward pass
    def format_prompt(self, problem):
        """the exact evaluation prompt (chat template, thinking on) for a problem; append generated text to it."""
        return self.target.prompt(problem + "\n" + CFG["target_eval"]["instruction"]) if self.target else \
            self.tok.apply_chat_template([{"role": "user", "content": problem + "\n" + CFG["target_eval"]["instruction"]}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)

    def forward(self, texts, layers=None, last_tokens=None):
        """teacher-forced pass of the CURRENT (edited) model over raw texts (no chat template is added).
        Returns {'tokens': [list of ids], 'logprobs': [tensor [T-1] log p(token t+1 | <= t)],
                 'last_logits': tensor [n, vocab]} and, for each requested layer l:
                 'resid'[l] (residual stream after layer l, [T, 1024]), 'mlp_in'[l] (input of mlp.down_proj, [T, 3584]),
                 'mlp_out'[l] (output of the MLP, [T, 1024]). Texts longer than the per-text limit keep their LAST tokens.
        last_tokens: keep only the last N positions of each returned activation."""
        mt, mpt, mtot = self.fwd_limits
        texts = list(texts)
        if len(texts) > mt:
            raise ValueError(f"at most {mt} texts per forward call")
        enc = [self.tok(t, add_special_tokens=False)["input_ids"][-mpt:] for t in texts]
        if sum(len(e) for e in enc) > mtot:
            raise ValueError(f"at most {mtot} tokens per forward call")
        layers = list(layers or [])
        out = {"tokens": enc, "logprobs": [], "last_logits": [], "resid": {l: [] for l in layers},
               "mlp_in": {l: [] for l in layers}, "mlp_out": {l: [] for l in layers}}
        caught = {}
        hooks = []
        for l in layers:
            L = self.hf.model.layers[l]
            hooks.append(L.register_forward_hook(lambda m, i, o, l=l: caught.__setitem__(("resid", l), o[0] if isinstance(o, tuple) else o)))
            hooks.append(L.mlp.down_proj.register_forward_hook(lambda m, i, o, l=l: caught.update({("mlp_in", l): i[0], ("mlp_out", l): o})))
        try:
            with torch.no_grad():
                for ids in enc:
                    caught.clear()
                    x = torch.tensor([ids], device=self.hf.device)
                    h = self.hf.model(x).last_hidden_state[0]
                    lps = []
                    for a in range(0, len(ids) - 1, 512):          # chunked: full logits of 4k tokens would be 4 GB
                        lg = self.hf.lm_head(h[a:min(a + 512, len(ids) - 1)]).float()
                        tgt = x[0, a + 1:a + 1 + lg.shape[0], None]
                        lps.append((lg.gather(1, tgt).squeeze(1) - torch.logsumexp(lg, -1)).cpu())
                    out["logprobs"].append(torch.cat(lps) if lps else torch.zeros(0))
                    out["last_logits"].append(self.hf.lm_head(h[-1:]).float()[0].cpu())
                    for (kind, l), v in caught.items():
                        v = v.reshape(-1, v.shape[-1]).float()
                        if last_tokens:
                            v = v[-last_tokens:]
                        out[kind][l].append(v.cpu())
        finally:
            for h in hooks:
                h.remove()
        out["last_logits"] = torch.stack(out["last_logits"]) if out["last_logits"] else None
        return out

    def namespace(self, level):
        ns = {"forward": self.forward, "eval": self.eval, "get_tensor": self.get_tensor, "apply_edit": self.apply_edit,
              "revert": self.revert, "list_tensors": self.list_tensors, "edit_summary": self.edit_summary,
              "format_prompt": self.format_prompt, "budget": self.budget.summary,
              "practice": {k: {kk: vv for kk, vv in v.items()} for k, v in self.practice.items()},
              "practice_ids": {"gsm8k": [k for k in self.practice if k.startswith("gsm8k_train_")],
                               "math": [k for k in self.practice if k.startswith("math_train_")],
                               "aime": [k for k in self.practice if k.startswith("aime_")]}}
        if level == "L1":
            ns.update(diff_of_means=diff_of_means, project_out=project_out)
        return ns


# ------------------------------------------------------------------ sandbox
BANNED_MODULES = ("torch.optim", "torch.autograd", "torch.func", "torch.nn.utils.stateless", "torch.utils",
                  "torch.hub", "torch.distributed", "torch.multiprocessing", "numpy.lib.format", "numpy.ctypeslib")
BANNED_ATTRS = {"backward", "optim", "autograd", "func", "enable_grad", "set_grad_enabled", "grad", "requires_grad_",
                "inference_mode", "load", "save", "hub", "from_file", "fromfile", "tofile", "loadtxt", "savetxt",
                "genfromtxt", "memmap", "ctypes", "load_state_dict", "state_dict", "zero_grad", "retain_grad", "register_hook",
                "vjp", "jvp", "jacrev", "jacfwd", "hessian", "system", "popen", "_C", "ops", "library",
                "jit", "utils", "distributed", "multiprocessing"}
# (torch.nn.functional and re.compile are allowed: no gradients or file access; torch.autograd.functional stays blocked
#  through "autograd". type() is allowed: the dunder ban already closes type()-based escapes.)
BANNED_NAMES = {"open", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr",
                "input", "breakpoint", "exit", "quit", "help", "memoryview", "__builtins__", "object", "super"}
SAFE_BUILTINS = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "format", "frozenset", "hash",
                 "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next", "pow", "print",
                 "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "divmod",
                 "chr", "ord", "hex", "bin", "callable", "ValueError", "KeyError", "IndexError", "TypeError",
                 "Exception", "RuntimeError", "ZeroDivisionError", "StopIteration", "AssertionError", "True", "False",
                 "None", "id", "sorted", "dir", "type", "hasattr", "AttributeError", "NameError", "NotImplementedError",
                 "OverflowError", "ArithmeticError", "LookupError", "complex"]


def _raise_grad(*a, **k):
    raise SandboxViolation("gradients are not allowed in this sandbox (backward / autograd / optimizers)")


class _BlockedOptim:
    def __init__(self, *a, **k):
        _raise_grad()


def _set_grad_enabled_guard(orig):
    def f(mode):
        if mode:
            _raise_grad()
        return orig(mode)
    return f


@contextlib.contextmanager
def no_grad_guard():
    """torch.no_grad() plus backward / autograd.grad / autograd.backward / every optimizer / enable_grad patched to
    raise (set_grad_enabled(True) raises too). The patches are undone before no_grad exits."""
    import torch.optim as optim
    saved = [(torch.Tensor, "backward", torch.Tensor.backward, _raise_grad),
             (torch.autograd, "backward", torch.autograd.backward, _raise_grad),
             (torch.autograd, "grad", torch.autograd.grad, _raise_grad),
             (optim.Optimizer, "__init__", optim.Optimizer.__init__, _raise_grad),
             (torch, "enable_grad", torch.enable_grad, _raise_grad),
             (torch, "set_grad_enabled", torch.set_grad_enabled, _set_grad_enabled_guard(torch.set_grad_enabled))]
    with torch.no_grad():
        try:
            for obj, name, _, new in saved:
                setattr(obj, name, new)
            yield
        finally:
            for obj, name, orig, _ in saved:
                setattr(obj, name, orig)


def check_code(code, allowed_imports):
    """static check of the editor's code; raises SandboxViolation (or SyntaxError) before anything runs."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            else:
                mods = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            for m in mods:
                if m.split(".")[0] not in allowed_imports:
                    raise SandboxViolation(f"import of {m!r} is not allowed (allowed: {', '.join(allowed_imports)})")
                if any(m == b or m.startswith(b + ".") for b in BANNED_MODULES) or \
                        any(p in BANNED_ATTRS for p in m.split(".")[1:]):
                    raise SandboxViolation(f"import of {m!r} is not allowed (no gradients / file or system access)")
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                raise SandboxViolation(f"attribute .{node.attr} is not allowed (no gradients / file or system access)")
            if node.attr.startswith("__"):
                raise SandboxViolation(f"dunder attribute .{node.attr} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or node.id.startswith("__"):
                raise SandboxViolation(f"name {node.id!r} is not allowed")
        elif isinstance(node, ast.keyword) and node.arg == "requires_grad":
            raise SandboxViolation("requires_grad is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            pass
    return tree


class Sandbox:
    def __init__(self, namespace, budget, allowed_imports=None, output_chars=6000, timeout_s=7200):
        self.allowed = list(allowed_imports or CFG["sandbox"]["allowed_imports"])
        self.budget = budget
        self.output_chars = output_chars
        self.timeout_s = timeout_s
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            full = [name] + [f"{name}.{f}" for f in (fromlist or ())]
            for m in full:
                if name.split(".")[0] not in self.allowed or any(m == b or m.startswith(b + ".") for b in BANNED_MODULES):
                    raise SandboxViolation(f"import of {m!r} is not allowed")
            return real_import(name, globals, locals, fromlist, level)

        bi = {k: getattr(builtins, k) for k in SAFE_BUILTINS if hasattr(builtins, k)}
        bi["__import__"] = guarded_import
        import numpy, math
        self.ns = {"__builtins__": bi, "torch": torch, "np": numpy, "numpy": numpy, "math": math}
        self.ns.update(namespace)

    def run(self, code):
        """returns (status, output). status: ok | error | rejected | budget."""
        try:
            tree = check_code(code, self.allowed)
        except SyntaxError as e:
            return "error", f"SyntaxError: {e}"
        except SandboxViolation as e:
            return "rejected", f"REJECTED before execution: {e}"
        try:
            self.budget.take("python_execs")
        except BudgetExceeded as e:
            return "budget", str(e)
        buf = io.StringIO()
        status = "ok"
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)

        def on_alarm(*a):
            raise TimeoutError(f"execution exceeded {self.timeout_s}s")
        old = signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(self.timeout_s)
        try:
            with contextlib.redirect_stdout(buf), no_grad_guard():
                exec(compile(tree, "<editor>", "exec"), self.ns)
                if last_expr is not None:
                    v = builtins.eval(compile(last_expr, "<editor>", "eval"), self.ns)
                    if v is not None:
                        print(repr(v))
        except Exception:
            status = "error"
            tb = traceback.format_exc().splitlines()
            keep = [l for l in tb if "<editor>" in l or not l.startswith("  File")]
            buf.write("\n" + "\n".join(keep[-12:]))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        out = buf.getvalue()
        if len(out) > self.output_chars:
            h = self.output_chars // 2
            out = out[:h] + f"\n... [{len(out) - 2 * h} characters cut] ...\n" + out[-h:]
        return status, out


# ------------------------------------------------------------------ assembling a live session
def hf_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(E.P("target"))
    m = AutoModelForCausalLM.from_pretrained(E.P("target"), dtype=torch.bfloat16).cuda().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok


def hf_name(name):
    return name.replace("model.language_model.", "model.")


class Session:
    """target engines + edit state + tools for one run (editor run, scripted baseline or dry run)."""

    def __init__(self, run_dir, level="L0", seed=0, target=None, gpu_util=None, with_hf=True):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.target = target or E.Target(gpu_memory_utilization=gpu_util)
        self.hf, self.tok = hf_model() if with_hf else (None, self.target.tok)
        params = dict(self.hf.named_parameters()) if self.hf else {}

        def push(tensors):
            self.target.load_tensors(tensors)
            for k, v in tensors.items():
                if params:
                    params[hf_name(k)].data.copy_(v.to(params[hf_name(k)].dtype))
        sb = CFG["sandbox"]
        self.state = EditState(load_base(), log_dir=run_dir, frozen=sb["frozen_tensors"],
                               refuse_rescale=sb["refuse_whole_tensor_rescale"], on_change=push)
        b = CFG["budget"]
        self.budget = Budget(b["eval_calls"], b["eval_max_ids"], b["python_execs"])
        self.tools = Tools(self.state, self.budget, self.target, self.hf, self.tok, run_dir, seed=seed)
        self.sandbox = Sandbox(self.tools.namespace(level), self.budget, output_chars=CFG["editor"]["output_chars"])
        if self.hf is not None:                    # compile / autotune the fla kernels once, outside the sandbox
            for n in (16, 300, 4096):
                self.tools.forward([" ".join(["warm"] * n)], layers=[0])

    def reset_to_base(self):
        """make sure both engines hold base weights (e.g. when a Target is reused between runs)."""
        self.state.revert("all")

    def finish(self, extra=None):
        man = self.state.save_checkpoint(os.path.join(self.run_dir, "final"), extra=extra)
        return man
