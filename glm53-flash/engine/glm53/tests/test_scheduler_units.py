"""`glm53.scheduler` unit tests: prefix matching, the host snapshot store, and the DFlash block-size policy.

These cover the pure logic that decides *what* gets recomputed (prefix matching, snapshot reuse,
speculative block-size selection) — the parts that decide correctness, not speed.  The engine is faked;
no TPU and no real KV cache is involved.
"""
import numpy as np
import pytest

from glm53 import scheduler as S


# ----------------------------------------------------------------------------- match_prefix
def test_match_prefix_hits_when_the_live_ids_are_a_prefix():
    k, fed = S.match_prefix([1, 2, 3], 2, [1, 2, 9, 9])
    assert k == 2
    assert fed == [1, 2, 9, 9]


def test_match_prefix_misses_when_the_ids_differ():
    assert S.match_prefix([1, 2, 3], 2, [1, 5, 9]) == (0, None)


def test_match_prefix_misses_at_position_zero():
    assert S.match_prefix([1, 2, 3], 0, [1, 2, 3]) == (0, None)


def test_match_prefix_misses_when_the_prompt_is_shorter_than_the_context():
    assert S.match_prefix([1, 2, 3], 5, [1, 2]) == (0, None)


def test_match_prefix_hits_the_full_prompt():
    k, fed = S.match_prefix([1, 2, 3], 3, [1, 2, 3])
    assert k == 3
    assert fed == [1, 2, 3]


def test_match_prefix_compares_only_the_first_pos_ids():
    # the tail of the live context is irrelevant; only live_ids[:pos] must match
    k, fed = S.match_prefix([1, 2, 99, 98], 2, [1, 2, 7])
    assert k == 2
    assert fed == [1, 2, 7]


# ----------------------------------------------------------------------------- Request
def test_request_normalises_its_inputs():
    r = S.Request(prompt=[1.0, 2.7], max_new=5, temperature="0.5", top_p="0.9")
    assert r.prompt == [1, 2]
    assert r.max_new == 5
    assert r.temperature == 0.5
    assert r.top_p == 0.9
    assert r.out == [] and r.error is None and r.stop_reason is None
    assert r.reused == 0


def test_request_cancel_sets_the_flag():
    r = S.Request([1], 3)
    assert not r.cancelled
    r.cancel()
    assert r.cancelled


def test_request_decode_s_is_zero_before_the_first_token():
    r = S.Request([1], 3)
    r.t_submit = 100.0
    r.t_first = 110.0
    r.t_end = 112.0
    assert r.decode_s == pytest.approx(2.0)


# ----------------------------------------------------------------------------- _argmax_first
def test_argmax_first_picks_the_largest_logit():
    assert S._argmax_first(np.array([0.1, 5.0, 2.0]), 1.0, 1.0) == 1


# ----------------------------------------------------------------------------- SnapStore
class _FakeEngine:
    """Minimal stand-in for `ResidentLayerEngine`'s snapshot API."""

    def __init__(self):
        self.snapshots = []

    def snapshot_prefix(self, caches, pos, rows_bucket=1024):
        self.snapshots.append((pos, rows_bucket))
        return {"pos": pos, "bytes": 100}

    def snapshot_to_host(self, snap):
        return {"host": True, "bytes": snap["bytes"]}

    def snapshot_from_host(self, host):
        return {"host": False, "bytes": host["bytes"]}

    def restore_prefix(self, snap):
        return {"caches": snap["bytes"]}


@pytest.fixture
def store():
    st = S.SnapStore(_FakeEngine(), max_bytes=250, log=lambda *a, **k: None)
    return st


def test_park_then_lookup_finds_the_parked_context(store):
    store.park([1, 2, 3], None, 3, logits=np.zeros(4))
    k, fed, e = store.lookup([1, 2, 3, 4])
    assert k == 3
    assert fed == [1, 2, 3, 4]
    assert e is not None
    assert e["n"] == 3


def test_park_is_idempotent_for_the_same_ids(store):
    e1 = store.park([1, 2], None, 2, logits=None)
    e2 = store.park([1, 2], None, 2, logits=None)
    assert e1 is e2
    assert len(store.entries) == 1
    assert store.bytes == 100


def test_park_counts_entries_and_bytes(store):
    store.park([1], None, 1, logits=None)
    store.park([2], None, 1, logits=None)
    assert store.bytes == 200
    assert store.state["snap_entries"] == 2
    assert store.state["snap_parks"] == 2


def test_park_evicts_the_oldest_unpinned_entry_over_the_budget(store):
    store.park([1], None, 1, logits=None)      # 100 bytes
    store.park([2], None, 1, logits=None)      # 200 bytes total
    store.park([3], None, 1, logits=None)      # 300 -> evict the oldest
    assert len(store.entries) == 2
    assert store.bytes == 200
    assert (1,) not in store.entries


def test_a_pinned_entry_is_never_evicted(store):
    store.park([1], None, 1, logits=None, pinned=True)
    store.park([2], None, 1, logits=None)
    store.park([3], None, 1, logits=None)
    assert (1,) in store.entries
    assert store.state["snap_pins"] == 1


def test_park_supersedes_older_strict_prefixes_but_keeps_the_latest(store):
    # three turns of one conversation: the store keeps the two latest turn boundaries,
    # the oldest strict prefix is dropped
    store.park([1], None, 1, logits=None)
    store.park([1, 2], None, 2, logits=None)
    store.park([1, 2, 3], None, 3, logits=None)
    assert (1,) not in store.entries
    assert (1, 2) in store.entries
    assert (1, 2, 3) in store.entries


def test_park_keeps_a_pinned_older_prefix(store):
    store.park([1, 2], None, 2, logits=None, pinned=True)
    store.park([1, 2, 3], None, 3, logits=None)
    assert (1, 2) in store.entries


def test_lookup_misses_when_no_entry_is_a_prefix(store):
    store.park([1, 2, 3], None, 3, logits=np.zeros(4))
    k, fed, e = store.lookup([9, 9, 9])
    assert (k, fed, e) == (0, None, None)


def test_lookup_skips_an_entry_that_only_covers_the_whole_prompt_without_logits(store):
    # a full-length match with no logits to continue from is not a usable prefix
    store.park([1, 2], None, 2, logits=None)
    k, fed, e = store.lookup([1, 2])
    assert (k, fed, e) == (0, None, None)


def test_lookup_prefers_the_longest_covering_entry(store):
    store.park([1, 2], None, 2, logits=np.zeros(4))
    store.park([1, 2, 3], None, 3, logits=np.zeros(4))
    k, fed, e = store.lookup([1, 2, 3, 4])
    assert k == 3
    assert e["n"] == 3


def test_restore_returns_caches_and_counts_a_hit(store):
    e = store.park([1, 2], None, 2, logits=None)
    caches = store.restore(e)
    assert caches == {"caches": 100}
    assert store.state["snap_hits"] == 1


def test_drop_releases_the_bytes(store):
    store.park([1], None, 1, logits=None)
    store._drop((1,))
    assert store.bytes == 0
    assert len(store.entries) == 0


# ----------------------------------------------------------------------------- DFlashPolicy
class _Stream:
    def __init__(self):
        self.tau = None


def _policy(**kw):
    cost = kw.pop("cost", {4: 20.0, 8: 30.0})
    return S.DFlashPolicy(cost, **kw)


def test_policy_prior_defaults_to_a_growing_curve():
    p = _policy()
    assert set(p.prior) == {4, 8}
    assert p.prior[8] > p.prior[4]


def test_policy_prior_can_be_overridden():
    p = S.DFlashPolicy({4: 20.0}, prior={4: 3.0})
    assert p.prior == {4: 3.0}


def test_rows_T_is_the_smallest_block():
    assert _policy(cost={8: 30.0, 4: 20.0, 6: 25.0}).rows_T == 4


def test_start_initialises_the_stream_state():
    p = _policy()
    s = _Stream()
    p.start(s)
    assert s.tau["ema"] == p.prior
    assert s.tau["last"] == {4: 0, 8: 0}
    assert s.tau["n"] == 0


def test_pick_prefers_the_cheapest_ms_per_token():
    p = _policy()
    s = _Stream()
    p.start(s)
    # ema 1.0 for both -> 20/1 = 20 ms/token for T=4 vs 30/1 for T=8
    assert p.pick(s) == 4
    # make T=8 far more productive: 8 tokens per step vs 1
    s.tau["ema"][4] = 1.0
    s.tau["ema"][8] = 8.0
    assert p.pick(s) == 8


def test_pick_probes_an_untried_block_after_the_probe_interval():
    p = _policy(probe=2)
    s = _Stream()
    p.start(s)
    p.pick(s)                       # first call: n=0, last=0 -> 0-0 < 2 -> no probe
    s.tau["n"] = 2
    assert p.pick(s) == 4          # T=4 is the first key with n-last >= probe


def test_update_moves_the_ema_toward_the_observed_token_count():
    p = S.DFlashPolicy({4: 20.0, 8: 30.0}, prior={4: 1.0, 8: 1.0}, alpha=0.5)
    s = _Stream()
    p.start(s)
    p.update(s, 4, 4)
    # for T=4: ema = 1.0 + 0.5*(min(4,4) - 1.0) = 2.5
    assert s.tau["ema"][4] == pytest.approx(2.5)
    # the block of 4 was fully accepted, so T=8 stays unknown -> its EMA is not touched
    assert s.tau["ema"][8] == pytest.approx(1.0)
    assert s.tau["n"] == 1


def test_update_uses_the_smaller_count_for_a_run_cut_before_the_block():
    p = S.DFlashPolicy({4: 20.0, 8: 30.0}, prior={4: 1.0, 8: 1.0}, alpha=0.5)
    s = _Stream()
    p.start(s)
    p.update(s, 8, 4)                     # the run of 8 ended after 4 tokens
    # a cut run tells every larger block what it would have earned: min(n_tokens, t)
    assert s.tau["ema"][4] == pytest.approx(2.5)
    assert s.tau["ema"][8] == pytest.approx(2.5)


def test_update_leaves_the_larger_block_unknown_after_a_full_acceptance():
    p = S.DFlashPolicy({4: 20.0, 8: 30.0}, prior={4: 1.0, 8: 1.0}, alpha=0.5)
    s = _Stream()
    p.start(s)
    p.update(s, 8, 8)
    # T=8 fully accepted: T=4 earned min(8,4) = 4 tokens, T=8 earned 8
    assert s.tau["ema"][4] == pytest.approx(2.5)
    assert s.tau["ema"][8] == pytest.approx(4.5)


def test_update_marks_the_tried_block():
    p = S.DFlashPolicy({4: 20.0, 8: 30.0}, prior={4: 1.0, 8: 1.0})
    s = _Stream()
    p.start(s)
    p.update(s, 4, 4)
    assert s.tau["last"][4] == 1
    assert s.tau["last"][8] == 0


def test_pick_rows_is_false_without_a_cost_entry():
    p = _policy(rows={2: (10.0, 20.0)})
    s = _Stream()
    p.start(s)
    assert p.pick_rows([s]) is False


def test_pick_rows_is_false_when_speculation_is_not_faster():
    # DFlash: 2 streams in 20 ms -> sum(ema)/20; plain: 2 tokens in 10 ms -> 2/10
    p = _policy(rows={2: (20.0, 10.0)})
    s1, s2 = _Stream(), _Stream()
    p.start(s1)
    p.start(s2)
    s1.tau["ema"][4] = 1.0
    s2.tau["ema"][4] = 1.0
    # 2/20 = 0.1 tokens/ms vs 2/10 = 0.2 tokens/ms -> plain wins
    assert p.pick_rows([s1, s2]) is False


def test_pick_rows_is_true_when_speculation_wins():
    p = _policy(rows={2: (10.0, 100.0)})
    s1, s2 = _Stream(), _Stream()
    p.start(s1)
    p.start(s2)
    s1.tau["ema"][4] = 4.0
    s2.tau["ema"][4] = 4.0
    # 8/10 = 0.8 tokens/ms vs 2/100 = 0.02 tokens/ms -> DFlash wins
    assert p.pick_rows([s1, s2]) is True


def test_pick_rows_honours_the_margin():
    p = _policy(rows={1: (10.0, 12.0)})
    s = _Stream()
    p.start(s)
    s.tau["ema"][4] = 1.0
    # 1/10 = 0.1 vs 1/12 = 0.0833 -> wins without a margin, loses with margin 1.5
    assert p.pick_rows([s]) is True
    assert p.pick_rows([s], margin=1.5) is False
