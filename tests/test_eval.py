"""Grading correctness: answer extraction (as MIT's mathqa.py), math-verify equivalence, IFEval checker, summary."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate as E  # noqa: E402


def test_extract_last_box_after_think():
    t = "<think>\nmaybe \\boxed{3}\n</think>\n\nSo the answer is \\boxed{\\frac{1}{2}}."
    assert E.extract(t) == "\\frac{1}{2}"


def test_extract_nested_and_unclosed():
    assert E.extract("x \\boxed{\\sqrt{2}+\\frac{a}{b}} y") == "\\sqrt{2}+\\frac{a}{b}"
    assert E.extract("\\boxed{1} then \\boxed{2") is None          # last box unclosed (cut) -> no answer
    assert E.extract("no box here") is None
    assert E.extract("\\boxed 5$") == "5"


def test_extract_thinking_cut():
    # a response cut inside thinking (no </think>): the whole text is read, as in MIT's grader
    assert E.extract("Wait, \\boxed{7} ... Wait, let me check") == "7"


def test_is_correct_equivalences():
    assert E.is_correct("0.5", "\\frac{1}{2}")
    assert E.is_correct("\\left( 3, \\frac{\\pi}{2} \\right)", "\\left( 3, \\frac{\\pi}{2} \\right)")
    assert E.is_correct("18", "18")
    assert not E.is_correct("17", "18")
    assert not E.is_correct(None, "18")


def test_ifeval_checker():
    row = {"prompt": "Write in all lowercase.", "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]}
    assert E.grade_ifeval(row, "<think>x</think>\n\nthis is a short answer written entirely in lowercase english letters.")
    assert not E.grade_ifeval(row, "<think>x</think>\n\nThis Is A Short Answer Written In Title Case English Letters.")
    assert not E.grade_ifeval(row, "this is a short answer written entirely in lowercase english letters.")        # no </think> -> empty answer


def test_summary_completion_metrics():
    per = [{"id": "a", "samples": [{"correct": True, "finish": "stop", "n_tokens": 10},
                                   {"correct": False, "finish": "length", "n_tokens": 100}]},
           {"id": "b", "samples": [{"correct": False, "finish": "stop", "n_tokens": 20},
                                   {"correct": True, "finish": "length", "n_tokens": 100}]}]
    s = E.summarize(per)
    assert s["acc"] == 0.5 and s["cut_share"] == 0.5 and s["completion_rate"] == 0.5
    assert s["completed_only_acc"] == 0.5 and s["n_completed"] == 2


def test_req_seed_stable():
    assert E.req_seed(0, "x", 1) == E.req_seed(0, "x", 1) != E.req_seed(0, "x", 2)
