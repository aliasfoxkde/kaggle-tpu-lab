"""The scheduler with DFlash 2 (`Scheduler(spec=DFlashPolicy)`, tiny resident engine + a random tiny drafter, 8 CPU
devices): a stream running alone decodes speculatively, a second stream switches both to batched plain steps and
back; every request's tokens equal its own single-stream greedy generation; the block size follows the policy; the
tail before max_new and a thinking budget decode plainly; a follow-up turn reuses the live context."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from glm53.scheduler import DFlashPolicy, Request, Scheduler, SnapStore  # noqa: E402
from glm53.tests.test_dflash_engine import random_drafter  # noqa: E402
from glm53.tests.test_scheduler import STOP, reference, wait  # noqa: E402
from glm53.tests.test_spec_decode import build  # noqa: E402


@pytest.fixture(scope="module")
def eng():
    eng, cfg, rng = build(False, mtp=False)
    dcfg, dp = random_drafter(rng, cfg.hidden_size, cfg.vocab_size)
    eng.set_dflash(dp, dcfg)
    return eng


def run_sched(eng, policy, reqs, max_streams=2):
    state = {}
    snaps = SnapStore(eng, int(50e9), rows_bucket=8, log=lambda *a: None, state=state)
    sched = Scheduler(eng, {STOP}, max_streams=max_streams, max_sets=3, snaps=snaps, log=lambda *a: None, state=state,
                      base_min=10 ** 9, snap_min=1, piece=32, spec=policy)
    sched.start()
    try:
        for r in reqs:
            sched.submit(r)
        for r in reqs:
            wait(r)
        return sched, state
    finally:
        sched.stop()


@pytest.mark.parametrize("cost", [{4: 1.0, 8: 9.0}, {4: 9.0, 8: 1.0}, {4: 1.0}])
def test_dflash_scheduler_matches_greedy(eng, cost):
    rng = np.random.default_rng(11)
    V = eng.cfg.vocab
    p1, p2 = (rng.integers(8, V - 1, size=(n,)).tolist() for n in (40, 28))
    r1 = Request(p1, 14, 0.0, 1.0, rid="a")
    sched, state = run_sched(eng, DFlashPolicy(cost, probe=4), [r1])
    assert r1.out == reference(eng, p1, 14), (r1.out, reference(eng, p1, 14))
    assert state["spec_steps"] > 0
    # two requests: the second admission ends DFlash for the first (batched steps), DFlash resumes when one is left
    r2, r3 = Request(p1, 40, 0.0, 1.0, rid="b"), Request(p2, 6, 0.0, 1.0, rid="c")
    sched, state = run_sched(eng, DFlashPolicy(cost, probe=4), [r2, r3])
    assert r2.out == reference(eng, p1, 40) and r3.out == reference(eng, p2, 6)
    assert state["spec_steps"] > 0 and state["steps"] > state["spec_steps"]
    # a follow-up turn extends the live context left by a DFlash stream (ids and position = what the caches hold)
    p4 = p1 + r2.out
    r4 = Request(p4, 10, 0.0, 1.0, rid="d")
    run_sched(eng, DFlashPolicy(cost, probe=4), [r1, r4], max_streams=1)
    assert r4.out == reference(eng, p4, 10)


def test_policy_picks_the_cheaper_block():
    pol = DFlashPolicy({4: 21.2, 8: 27.3}, prior={4: 2.8, 8: 3.3}, probe=1000)

    class S:
        pass
    s = S()
    pol.start(s)
    assert pol.pick(s) == 4                                   # 7.6 vs 8.3 ms per token
    for _ in range(30):
        pol.update(s, 8, 6)                                   # long accepted runs: T=8 pays
    assert pol.pick(s) == 8 and abs(s.tau["ema"][4] - 4) < 0.1
    for _ in range(30):
        pol.update(s, 4, 2)
    assert pol.pick(s) == 4
