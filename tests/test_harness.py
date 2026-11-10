"""Harness guards: gradients rejected (AST and at runtime), budget enforced, held-out ids rejected, diff-log rebuild
reproduces identical weights. CPU only, on small synthetic tensors (plus one test on the real base checkpoint)."""
import os, sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness as H  # noqa: E402
import evaluate as E  # noqa: E402


def make_state(tmp_path, on_change=None):
    g = torch.Generator().manual_seed(0)
    base = {"model.language_model.layers.0.mlp.down_proj.weight": torch.randn(16, 40, generator=g).to(torch.bfloat16),
            "model.language_model.layers.1.mlp.down_proj.weight": torch.randn(16, 40, generator=g).to(torch.bfloat16),
            "model.language_model.layers.0.linear_attn.norm.weight": torch.randn(8, generator=g),        # float32
            "model.language_model.embed_tokens.weight": torch.randn(10, 16, generator=g).to(torch.bfloat16)}
    return H.EditState(base, log_dir=str(tmp_path / "run"), frozen=["model.language_model.embed_tokens.weight"],
                       on_change=on_change)


class FakeTarget:
    def __init__(self):
        self.loaded = []

    def load_tensors(self, t):
        self.loaded.append(dict(t))

    def prompt(self, text):
        return text

    def generate(self, rows, n, seed, kind="math"):
        return [[{"finish": "stop", "n_tokens": 5, "text": "\\boxed{1}", "pred": "1", "correct": r["answer"] == "1"}]
                for r in rows]


PRACTICE = {f"gsm8k_train_{i}": {"id": f"gsm8k_train_{i}", "problem": "p", "answer": str(i % 2)} for i in range(600)}


def make_sandbox(tmp_path, level="L1", eval_calls=3, execs=5):
    st = make_state(tmp_path)
    b = H.Budget(eval_calls=eval_calls, eval_max_ids=512, python_execs=execs)
    tools = H.Tools(st, b, target=FakeTarget(), practice=PRACTICE, run_dir=str(tmp_path / "run"))
    return H.Sandbox(tools.namespace(level), b), tools, st, b


# ---------------------------------------------------------------- gradients
GRAD_CODES = [
    "x = torch.ones(3)\nx.backward()",
    "import torch.optim\n",
    "from torch import optim",
    "from torch.optim import SGD",
    "opt = torch.optim.SGD([torch.zeros(1)], lr=0.1)",
    "torch.autograd.grad(torch.ones(1), torch.ones(1))",
    "import torch.autograd as ag",
    "with torch.enable_grad():\n    pass",
    "torch.set_grad_enabled(True)",
    "x = torch.zeros(2, requires_grad=True)",
    "x = torch.zeros(2)\nx.requires_grad_()",
    "y = torch.zeros(2).grad",
    "f = torch.func.grad(lambda x: x.sum())",
    "getattr(torch.ones(1), 'back' + 'ward')()",
    "torch.ones(1).__class__",
    "open('assets/data/math500/test.jsonl')",
    "import os",
    "torch.load('x.pt')",
    "np.load('x.npy')",
    "__import__('os')",
]


@pytest.mark.parametrize("code", GRAD_CODES)
def test_ast_rejects(tmp_path, code):
    sb, _, _, b = make_sandbox(tmp_path)
    status, out = sb.run(code)
    assert status == "rejected", (code, status, out)
    assert "REJECTED" in out
    assert b.used["python_execs"] == 0          # rejected code never ran


def test_runtime_guard_blocks_backward_and_optim():
    """even if code got past the AST check, backward / optimizers / autograd.grad raise inside the guard."""
    w = torch.ones(3, requires_grad=True)
    with H.no_grad_guard():
        with pytest.raises(H.SandboxViolation):
            torch.Tensor.backward(w.sum())
        with pytest.raises(H.SandboxViolation):
            torch.optim.SGD([w], lr=0.1)
        with pytest.raises(H.SandboxViolation):
            torch.autograd.grad(w.sum(), w)
        with pytest.raises(H.SandboxViolation):
            torch.enable_grad()
        assert not torch.is_grad_enabled()
        y = (w * 2).sum()
        assert not y.requires_grad                   # no graph is recorded under the guard
    # restored afterwards (the harness itself is untouched)
    (w * 2).sum().backward()
    assert w.grad is not None


def test_sandbox_runs_ordinary_code(tmp_path):
    sb, _, st, b = make_sandbox(tmp_path)
    status, out = sb.run("W = get_tensor('model.language_model.layers.0.mlp.down_proj.weight')\n"
                         "v = diff_of_means(torch.randn(10, 16), torch.randn(10, 16))\n"
                         "W2 = project_out(W, v, [0, 1, 2])\n"
                         "r = apply_edit('model.language_model.layers.0.mlp.down_proj.weight', W2)\n"
                         "print(r['entries_changed_vs_base'] > 0)\n"
                         "torch.is_grad_enabled()")
    assert status == "ok", out
    assert "True\nFalse" in out
    assert b.used["python_execs"] == 1
    assert st.changed() > 0


def test_l0_has_no_helpers(tmp_path):
    sb, _, _, _ = make_sandbox(tmp_path, level="L0")
    status, out = sb.run("diff_of_means")
    assert status == "error" and "NameError" in out


def test_frozen_tensor_refused(tmp_path):
    sb, _, _, _ = make_sandbox(tmp_path)
    status, out = sb.run("n='model.language_model.embed_tokens.weight'\napply_edit(n, get_tensor(n) + 1)")
    assert status == "error" and "frozen" in out


def test_whole_tensor_rescale_refused(tmp_path):
    st = make_state(tmp_path)
    n = "model.language_model.layers.0.mlp.down_proj.weight"
    with pytest.raises(H.SandboxViolation):
        st.apply(n, st.get(n).float() * 2.0)


# ---------------------------------------------------------------- budget
def test_python_exec_budget(tmp_path):
    sb, _, _, b = make_sandbox(tmp_path, execs=3)
    for _ in range(3):
        assert sb.run("1 + 1")[0] == "ok"
    status, out = sb.run("1 + 1")
    assert status == "budget" and "exhausted" in out


def test_eval_budget_and_size(tmp_path):
    sb, tools, _, b = make_sandbox(tmp_path, eval_calls=2)
    ids = list(PRACTICE)[:10]
    tools.eval(ids)
    tools.eval(ids, original=True)
    with pytest.raises(H.BudgetExceeded):
        tools.eval(ids)
    assert b.used["eval_calls"] == 2
    sb2, tools2, _, b2 = make_sandbox(tmp_path / "b", eval_calls=5)
    with pytest.raises(ValueError):
        tools2.eval(list(PRACTICE)[:513])           # more than 512 ids per call
    assert b2.used["eval_calls"] == 0


def test_eval_budget_via_sandbox(tmp_path):
    sb, _, _, _ = make_sandbox(tmp_path, eval_calls=1)
    assert sb.run("r = eval(practice_ids['gsm8k'][:4])\nlen(r)")[0] == "ok"
    status, out = sb.run("eval(practice_ids['gsm8k'][:4])")
    assert status == "error" and "BudgetExceeded" in out


# ---------------------------------------------------------------- held-out ids
@pytest.mark.parametrize("bad", ["math500/test/precalculus/807.json", "gsm8k_test_0", "ifeval_1000", "anything"])
def test_heldout_ids_rejected(tmp_path, bad):
    _, tools, _, b = make_sandbox(tmp_path)
    with pytest.raises(ValueError, match="practice ids only"):
        tools.eval([bad])
    assert b.used["eval_calls"] == 0


def test_real_practice_excludes_heldout():
    if not os.path.exists(os.path.join(E.P("practice"), "math_train.jsonl")):
        pytest.skip("run `evaluate.py prepare` first")
    prac = E.load_practice()
    held = {r["id"] for s in ("math500", "gsm8k_test", "ifeval") for r in E.load_heldout(s)}
    assert not (set(prac) & held)
    held_problems = {r["problem"] for s in ("math500", "gsm8k_test") for r in E.load_heldout(s)}
    assert not any(r["problem"] in held_problems for r in prac.values())


# ---------------------------------------------------------------- diff log rebuild
def test_rebuild_identical(tmp_path):
    pushed = []
    st = make_state(tmp_path, on_change=lambda t: pushed.append(t))
    a = "model.language_model.layers.0.mlp.down_proj.weight"
    b = "model.language_model.layers.1.mlp.down_proj.weight"
    c = "model.language_model.layers.0.linear_attn.norm.weight"
    g = torch.Generator().manual_seed(1)
    st.apply(a, H.project_out(st.get(a), torch.randn(16, generator=g), [0, 5, 7]))
    st.apply(b, st.get(b).float() + 0.01 * torch.randn(16, 40, generator=g))
    st.apply(a, H.project_out(st.get(a), torch.randn(16, generator=g), list(range(20))))
    st.apply(c, st.get(c) + 0.5)
    st.revert(b)
    st.apply(b, st.get(b).float().clone().index_fill_(1, torch.tensor([3]), 0.0))
    rebuilt = H.EditState.rebuild(st.base, st.log_dir)
    assert set(rebuilt) == set(st.current)
    for k in st.current:
        assert torch.equal(H.bits(rebuilt[k]), H.bits(st.current[k])), k
    assert rebuilt[c].dtype == torch.float32
    # checkpoint written from the state matches too
    man = st.save_checkpoint(str(tmp_path / "ckpt"))
    from safetensors.torch import load_file
    ck = load_file(str(tmp_path / "ckpt" / "edited.safetensors"))
    for k in ck:
        assert torch.equal(H.bits(ck[k]), H.bits(rebuilt[k]))
    assert man["n_changed"] == st.changed()
    assert len(pushed) == 6


def test_revert_all_restores_base(tmp_path):
    st = make_state(tmp_path)
    a = "model.language_model.layers.0.mlp.down_proj.weight"
    st.apply(a, st.get(a).float() + 1)
    st.revert("all")
    assert st.changed() == 0 and not st.current
    assert H.EditState.rebuild(st.base, st.log_dir) == {}


def test_rebuild_on_real_base(tmp_path):
    if not os.path.exists(os.path.join(E.P("target"), "model.safetensors")):
        pytest.skip("run `evaluate.py prepare` first")
    base = H.load_base()
    st = H.EditState(base, log_dir=str(tmp_path / "real"))
    n = "model.language_model.layers.15.mlp.down_proj.weight"
    v = torch.randn(1024, generator=torch.Generator().manual_seed(0))
    st.apply(n, H.project_out(st.get(n), v, list(range(64))))
    r = H.EditState.rebuild(base, st.log_dir)
    assert torch.equal(H.bits(r[n]), H.bits(st.current[n]))
    assert st.changed() <= 64 * 1024


def test_project_out_math():
    W = torch.randn(8, 5)
    v = torch.randn(8)
    W2 = H.project_out(W, v, [1, 3])
    u = v / v.norm()
    assert torch.allclose(u @ W2[:, [1, 3]], torch.zeros(2), atol=1e-5)
    assert torch.equal(W2[:, [0, 2, 4]], W[:, [0, 2, 4]])
