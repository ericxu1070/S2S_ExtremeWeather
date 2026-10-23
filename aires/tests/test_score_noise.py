"""The score-noise fit recovers a known atmosphere, and ctune's default replay is unchanged."""
import numpy as np
import pandas as pd

from aires import ctune
from aires import score_noise as SN


def _members(v, s, g, n=40000, m_members=6, seed=0):
    """Synthetic Gate 3: m ~ N(0, v), outcome = m + N(0, 1 - v), member = m + e + N(0, g)."""
    rng = np.random.default_rng(seed)
    m = rng.normal(scale=np.sqrt(v), size=n)
    real = m + rng.normal(scale=np.sqrt(1 - v), size=n)
    e = rng.normal(scale=np.sqrt(s), size=n)
    rows = [dict(walker=w, lead_days=9.0, member=j, box=m[w] + e[w] + x, realized_box=real[w])
            for j in range(m_members)
            for w, x in enumerate(rng.normal(scale=np.sqrt(g), size=n))]
    return pd.DataFrame(rows)


def test_fit_recovers_predictability_when_there_is_no_model_error():
    f = SN.fit(SN.subset_rho(_members(v=0.6, s=0.0, g=0.5)))[9.0]
    sc = SN.scenario({9.0: f}, frac=0.0)[9.0]
    assert abs(sc["v"] - 0.6) < 0.03
    assert abs(sc["g"] - 0.5) < 0.05
    assert f["max_resid"] < 0.01


def test_fit_with_model_error_is_bracketed_by_frac():
    # with s > 0 the frac=0 frame underestimates v; the true v lies inside [1/a, 1]
    f = SN.fit(SN.subset_rho(_members(v=0.6, s=0.2, g=0.5)))[9.0]
    lo, hi = SN.scenario({9.0: f}, 0.0)[9.0]["v"], SN.scenario({9.0: f}, 1.0)[9.0]["v"]
    assert lo < 0.6 < hi


def test_default_replay_is_bit_identical_to_explicit_one_minus_rho():
    rho = np.array([0.0, 0.6, 0.8, 0.95, 0.99])
    a = ctune.replay(rho, ctune.SCHEDULES["aires.md"], 32, np.random.default_rng(5))[1]
    b = ctune.replay(rho, ctune.SCHEDULES["aires.md"], 32, np.random.default_rng(5),
                     noise_var=1.0 - rho)[1]
    assert np.array_equal(a, b)
