# ============================================================
# ground_truth.py - SED Core Edit-Repair and Search Components
# Source: Computer_Code/SED-main
#
# Contains the debugger edit-operation algorithms and iterative
# search strategies central to Synthesize, Execute and Debug.
# No training loops, datasets, evaluation runners, notebooks, or CLI code.
# ============================================================

from abc import ABC, abstractmethod
from collections import defaultdict

import Levenshtein


# --- [Original file: program_synthesis/tools/edit.py] ---

def compute_edit_ops(source_seq, target_seq, stoi):
    """
    [TODO] Convert a source token sequence into an edit script for the target token sequence.

    Input:
        source_seq: (source_len,) - reference program tokens.
        target_seq: (target_len,) - desired repaired program tokens.
        stoi: callable - maps each token to a stable integer id encodable as one character.

    Output: iterator of triples `(source_index, op, value)`.
        source_index: int in `[0, source_len]`, indicating the current source cursor.
        op: one of `"keep"`, `"delete"`, `"insert"`, or `"replace"`.
        value: target token for insert/replace, otherwise None.

"""
    pass


def compute_edit_ops_no_stoi(source_seq, target_seq):
    """
    [TODO] Convert numeric source tokens into a target edit script without a vocabulary mapper.

    Input:
        source_seq: (source_len,) - integer token ids for the reference program.
        target_seq: (target_len,) - integer token ids for the repaired program.

    Output: iterator of triples `(source_index, op, value)`.
        source_index: int in `[0, source_len]`.
        op: one of `"keep"`, `"delete"`, `"insert"`, or `"replace"`.
        value: target integer token for insert/replace, otherwise None.

"""
    pass


def apply_edit_ops(source_seq, ops):
    """
    [TODO] Apply a debugger edit script to a source program sequence.

    Input:
        source_seq: (source_len,) - original token sequence.
        ops: (num_ops,) - triples `(source_index, op, value)` emitted by an edit-script function.

    Output: iterator over repaired tokens with length determined by the edit operations.

"""
    pass


# --- [Original file: program_synthesis/tools/iterative_search.py] ---

class Strategy(ABC):
    @abstractmethod
    def decide(self, candidates, evaluate):
        pass

    @staticmethod
    def get(descriptor):
        if ":" not in descriptor:
            descriptor += ":"
        start, *rest = descriptor.split(":")
        kwargs = eval("dict({})".format(":".join(*rest)))
        return {
            'greedy': lambda: GreedyStrategy,
            'best_first': lambda: BestFirstSearch,
            'diverse': lambda: DiversitySearch
        }[start](**kwargs)


def valid(considered_program, result):
    if not considered_program:
        return False
    if result['syntax-error'] > 0:
        return False
    return True


class GreedyStrategy(Strategy):
    def __init__(self, item):
        self.seen = set()
        del item  # no need

    def decide(self, candidates, evaluate):
        """
        [TODO] Choose the next candidate with the greedy SED repair-search policy.

        Input:
            candidates: (num_candidates,) - candidate token sequences for one program.
            evaluate: callable - maps a candidate to a result dict with syntax-error, correct, and total fields.

        Output: pair `(decision, candidate)`.
            decision: one of `"accept"`, `"expand"`, or `"give_up"`.
            candidate: tuple/list token sequence selected by the policy.

"""
        pass


class BestFirstSearch(Strategy):
    def __init__(self, item):
        self.seen = set()
        self.by_number_correct = defaultdict(list)

    def decide(self, candidates, evaluate):
        """
        [TODO] Select a repair candidate using best-first search over passed-test counts.

        Input:
            candidates: (num_candidates,) - beam candidates for the current repair iteration.
            evaluate: callable - maps a candidate to a result dict with syntax-error, correct, and total fields.

        Output: pair `(decision, candidate)`.
            decision: `"accept"` for a fully correct candidate, `"expand"` for the best partial candidate,
                or `"give_up"` if no valid candidate exists.
            candidate: tuple token sequence selected from the frontier or fallback input.

"""
        pass


class DiversitySearch(Strategy):
    """
    Add a new alternative to BestFirstStrategy that takes into
    account semantic diversity, that is it breaks ties by
    picking programs that pass different sets of test cases
    from ones expanded in the past. see res['individual']

    """
    def __init__(self, item):
        self.seen = set()
        self.seen_patterns = defaultdict(int)
        self.by_number_correct = defaultdict(list)

    def decide(self, candidates, evaluate):
        """
        [TODO] Select a repair candidate using best-first score plus semantic diversity.

        Input:
            candidates: (num_candidates,) - beam candidates for one repair iteration.
            evaluate: callable - maps a candidate to a result dict with syntax-error, correct, total,
                and individual test-pass pattern fields.

        Output: pair `(decision, candidate)`.
            decision: `"accept"`, `"expand"`, or `"give_up"`.
            candidate: tuple token sequence selected by score and diversity.

"""
        pass

    def diverse_decision(self, items):
        """
        [TODO] Break ties by choosing the candidate with the rarest semantic pass/fail pattern.

        Input:
            items: (num_items,) - pairs `(candidate, individual_pattern)` from one score bucket.

        Output: candidate - token sequence selected for expansion or acceptance.

"""
        pass


# ============================================================
# __main__: Automated test suite for 7 ablated functions
# ============================================================

if __name__ == "__main__":
    passed = 0
    failed = 0

    def check(test_name, condition, detail=""):
        global passed, failed
        if bool(condition):
            passed += 1
            print(f"  [{test_name}] PASS")
        else:
            failed += 1
            suffix = f" - {detail}" if detail else ""
            print(f"  [{test_name}] FAIL{suffix}")

    def skip_checks(count, reason):
        global failed
        failed += count
        print(f"  [skipped {count} check(s)] FAIL - {reason}")

    def make_eval(results):
        def evaluate(candidate):
            return results[tuple(candidate)]
        return evaluate

    print("=" * 70)
    print("SED: automated benchmark for edit repair and iterative search")
    print("Automated Test Suite - 7 ablated functions, 38 checks")
    print("=" * 70)
    print()

    # ==========================================================
    # Test 1/7: compute_edit_ops
    # ==========================================================
    print("-" * 60)
    print("[Test 1/7] compute_edit_ops - token edit script with vocabulary mapping")
    try:
        source = ("move", "turnLeft", "paint")
        target = ("move", "paint", "turnRight")
        vocab = {tok: idx + 11 for idx, tok in enumerate(sorted(set(source + target)))}
        ops = list(compute_edit_ops(source, target, vocab.__getitem__))
        reconstructed = tuple(apply_edit_ops(source, ops))
        check("compute_edit_ops output not None", ops is not None)
        check("compute_edit_ops reconstructs target", reconstructed == target, f"got {reconstructed}")
        check("compute_edit_ops source positions monotonic", all(ops[i][0] <= ops[i + 1][0] for i in range(len(ops) - 1)))
        check("compute_edit_ops emits keep", any(op == 'keep' for _, op, _ in ops))
        check("compute_edit_ops emits non-keep repair", any(op in {'insert', 'replace', 'delete'} for _, op, _ in ops))
    except Exception as exc:
        skip_checks(5, f"compute_edit_ops raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/7: compute_edit_ops_no_stoi
    # ==========================================================
    print("-" * 60)
    print("[Test 2/7] compute_edit_ops_no_stoi - numeric-token edit script")
    try:
        source = (40, 41, 42, 43)
        target = (40, 99, 42)
        ops = list(compute_edit_ops_no_stoi(source, target))
        reconstructed = tuple(apply_edit_ops(source, ops))
        check("compute_edit_ops_no_stoi output not None", ops is not None)
        check("compute_edit_ops_no_stoi reconstructs target", reconstructed == target, f"got {reconstructed}")
        check("compute_edit_ops_no_stoi keeps first token", ops[0] == (0, 'keep', None), f"got {ops[0] if ops else None}")
        check("compute_edit_ops_no_stoi has replacement", any(op == 'replace' for _, op, _ in ops))
        check("compute_edit_ops_no_stoi consumes all source", ops[-1][0] <= len(source), f"last op {ops[-1] if ops else None}")
    except Exception as exc:
        skip_checks(5, f"compute_edit_ops_no_stoi raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/7: apply_edit_ops
    # ==========================================================
    print("-" * 60)
    print("[Test 3/7] apply_edit_ops - keep/delete/insert/replace semantics")
    try:
        source = ("a", "b", "c")
        ops = [
            (0, "keep", None),
            (1, "delete", None),
            (2, "insert", "x"),
            (2, "replace", "z"),
        ]
        output = tuple(apply_edit_ops(source, ops))
        check("apply_edit_ops output not None", output is not None)
        check("apply_edit_ops exact output", output == ("a", "x", "z"), f"got {output}")
        check("apply_edit_ops output length", len(output) == 3, f"got {len(output)}")
        check("apply_edit_ops source unchanged", source == ("a", "b", "c"))
        bad_failed = False
        try:
            list(apply_edit_ops(source, [(1, "keep", None)]))
        except AssertionError:
            bad_failed = True
        check("apply_edit_ops validates source cursor", bad_failed)
    except Exception as exc:
        skip_checks(5, f"apply_edit_ops raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4/7: GreedyStrategy.decide
    # ==========================================================
    print("-" * 60)
    print("[Test 4/7] GreedyStrategy.decide - accept perfect or expand best unseen")
    try:
        strategy = GreedyStrategy(None)
        results = {
            (): {"syntax-error": 0, "correct": 0, "total": 5},
            ("bad",): {"syntax-error": 1, "correct": 0, "total": 5},
            ("partial",): {"syntax-error": 0, "correct": 3, "total": 5},
            ("good",): {"syntax-error": 0, "correct": 5, "total": 5},
        }
        decision1 = strategy.decide([(), ("bad",), ("partial",), ("good",)], make_eval(results))
        decision2 = strategy.decide([("good",), ("partial",)], make_eval(results))
        check("greedy decision output not None", decision1 is not None and decision2 is not None)
        check("greedy accepts perfect", decision1 == ("accept", ("good",)), f"got {decision1}")
        check("greedy skips seen accepted candidate", decision2 == ("expand", ("partial",)), f"got {decision2}")
        check("greedy records accepted candidate", ("good",) in strategy.seen)
        check("greedy records expanded candidate", ("partial",) in strategy.seen)
    except Exception as exc:
        skip_checks(5, f"GreedyStrategy.decide raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5/7: BestFirstSearch.decide
    # ==========================================================
    print("-" * 60)
    print("[Test 5/7] BestFirstSearch.decide - priority by number of passed tests")
    try:
        strategy = BestFirstSearch(None)
        results = {
            ("two",): {"syntax-error": 0, "correct": 2, "total": 5},
            ("four-a",): {"syntax-error": 0, "correct": 4, "total": 5},
            ("four-b",): {"syntax-error": 0, "correct": 4, "total": 5},
            ("perfect",): {"syntax-error": 0, "correct": 5, "total": 5},
        }
        decision1 = strategy.decide([("two",), ("four-a",), ("four-b",)], make_eval(results))
        decision2 = strategy.decide([("perfect",)], make_eval(results))
        check("best-first decision output not None", decision1 is not None and decision2 is not None)
        check("best-first expands highest partial", decision1 == ("expand", ("four-a",)), f"got {decision1}")
        check("best-first accepts perfect", decision2 == ("accept", ("perfect",)), f"got {decision2}")
        check("best-first records seen", ("two",) in strategy.seen and ("perfect",) in strategy.seen)
        check("best-first retains queued tie", strategy.by_number_correct[4] == [("four-b",)], f"got {strategy.by_number_correct[4]}")
    except Exception as exc:
        skip_checks(5, f"BestFirstSearch.decide raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 6/7: DiversitySearch.diverse_decision
    # ==========================================================
    print("-" * 60)
    print("[Test 6/7] DiversitySearch.diverse_decision - prefer rarer semantic pattern")
    try:
        strategy = DiversitySearch(None)
        strategy.seen_patterns[(True, False, False, False, False)] = 3
        strategy.seen_patterns[(False, True, False, False, False)] = 0
        items = [
            (("common",), [True, False, False, False, False]),
            (("rare",), [False, True, False, False, False]),
        ]
        decision = strategy.diverse_decision(items)
        check("diverse_decision output not None", decision is not None)
        check("diverse_decision chooses rare pattern", decision == ("rare",), f"got {decision}")
        check("diverse_decision removes chosen item", items == [(("common",), [True, False, False, False, False])], f"got {items}")
        check("diverse_decision preserves pattern counts", strategy.seen_patterns[(True, False, False, False, False)] == 3)
    except Exception as exc:
        skip_checks(4, f"DiversitySearch.diverse_decision raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 7/7: DiversitySearch.decide
    # ==========================================================
    print("-" * 60)
    print("[Test 7/7] DiversitySearch.decide - best-first with semantic diversity")
    try:
        strategy = DiversitySearch(None)
        strategy.seen_patterns[(1, 0, 0, 0, 0)] = 4
        results = {
            ("common",): {"syntax-error": 0, "correct": 4, "total": 5, "individual": [1, 0, 0, 0, 0]},
            ("rare",): {"syntax-error": 0, "correct": 4, "total": 5, "individual": [0, 1, 0, 0, 0]},
            ("perfect",): {"syntax-error": 0, "correct": 5, "total": 5, "individual": [1, 1, 1, 1, 1]},
        }
        decision1 = strategy.decide([("common",), ("rare",)], make_eval(results))
        decision2 = strategy.decide([("perfect",)], make_eval(results))
        check("diversity decision output not None", decision1 is not None and decision2 is not None)
        check("diversity expands rarer tie", decision1 == ("expand", ("rare",)), f"got {decision1}")
        check("diversity accepts perfect", decision2 == ("accept", ("perfect",)), f"got {decision2}")
        check("diversity records seen candidates", ("common",) in strategy.seen and ("rare",) in strategy.seen)
        check("diversity records semantic patterns", strategy.seen_patterns[(0, 1, 0, 0, 0)] == 1)
        check("diversity queues remaining same-score candidate", strategy.by_number_correct[4] == [(("common",), [1, 0, 0, 0, 0])], f"got {strategy.by_number_correct[4]}")
    except Exception as exc:
        skip_checks(6, f"DiversitySearch.decide raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Final Score
    # ==========================================================
    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The SED edit-repair core code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
