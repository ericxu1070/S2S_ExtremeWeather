#!/usr/bin/env python
"""The RES core: Diffusion-Monte-Carlo importance splitting, model-agnostic.

This is the algorithm itself and nothing else. It knows how to score a population, tilt
it, clone and kill, keep the books that make the result an unbiased probability, and say
how badly the population has degenerated. It does not know what a walker is - GenCast,
an Ornstein-Uhlenbeck process and a lookup table all drive it through the same two
callables - which is what lets the whole thing be tested with zero GPU time.

The scheme
----------
At each resampling time ``t_k`` every walker gets a score ``theta``, standardized across
the population, and a splitting function

    V_k(i) = C_k * z(theta_k^i)

Its incremental weight is the CHANGE in that function along the leg it just travelled,

    W_k^i = exp( V_k(i) - V_{k-1}(parent of i) )        R_k = mean_i W_k^i

and the population is resampled with expected multiplicity ``W_k^i / R_k``, which sums to
``N`` in expectation and, thanks to pivotal sampling, to ``N`` exactly.

Because ``V`` is a function of state alone, the weights telescope along a lineage: the
product of ``W`` down any surviving branch is ``exp(V_K - V_0)``. So after the last
resampling the population is distributed under the tilted law ``exp(V_K) dP / E[exp(V_K)]``
and any expectation under the ORIGINAL law is recovered by undoing the tilt:

    E_P[phi]  =  Z * (1/N) * sum_i phi(X^i) * exp(-V_K(lineage of i)),   Z = prod_k R_k

That is the estimator :meth:`DMCResult.expectation` implements, and with
``phi = 1{A_L > a}`` it is the exceedance curve the experiment reports. Setting every
``C_k = 0`` makes ``V`` identically zero, so ``W = 1``, ``R_k = 1``, ``Z = 1`` and every
weight is 1: the algorithm degenerates *exactly* into direct sampling. That is the
strongest end-to-end invariant available and ``tests/test_dmc.py`` pins it.

Unbiasedness, honestly stated
-----------------------------
The estimator above is unbiased for a **deterministic** ``V``. Standardizing the score
across the live population - which the paper does, and which is what makes a single ``C_k``
schedule work across events whose scores have different units and spreads - makes ``V``
depend on the other walkers, so the scheme becomes self-interacting and the guarantee
weakens from unbiased to consistent. :func:`run` takes ``standardize=False`` (or explicit
``ref_mean``/``ref_sd``) so the strict regime can be tested, and
``test_estimator_is_unbiased_on_an_ornstein_uhlenbeck_walker`` runs there.

Effective sample size
---------------------
``ESS = (sum w)^2 / sum w^2`` on the resampling weights. It is the honest measure of how
much of the population is still doing work: ESS = N means uniform weights, ESS = 1 means
one walker has taken over. ``aires.md``'s tuning target is ESS > N/4 after the final step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["standardize", "pivotal_sample", "multiplicities", "ess",
           "Resampling", "DMCResult", "run"]


# --------------------------------------------------------------------------- #
# Score standardization
# --------------------------------------------------------------------------- #
def standardize(theta, ref_mean: float | None = None, ref_sd: float | None = None,
                ddof: int = 1) -> np.ndarray:
    """``(theta - mean) / sd`` across the population, or against a fixed reference.

    A degenerate population (every walker identical) has no spread to standardize by; it
    returns zeros rather than NaN, so a collapsed step tilts nothing instead of poisoning
    every downstream weight with NaN.
    """
    t = np.asarray(theta, dtype="float64")
    mu = float(np.mean(t)) if ref_mean is None else float(ref_mean)
    sd = float(np.std(t, ddof=ddof)) if ref_sd is None else float(ref_sd)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(t)
    return (t - mu) / sd


# --------------------------------------------------------------------------- #
# Pivotal sampling (Deville & Tille 1998)
# --------------------------------------------------------------------------- #
def pivotal_sample(p, rng: np.random.Generator) -> np.ndarray:
    """Draw a 0/1 vector with inclusion probabilities ``p`` and an EXACT total.

    ``sum(p)`` must be a whole number ``m``; exactly ``m`` entries come back 1, and entry
    ``i`` is 1 with probability exactly ``p_i``. Independent Bernoulli draws would give the
    right marginals and a fluctuating total - a population that grows or shrinks every
    step - and a multinomial draw fixes the total but inflates the variance. The pivotal
    method gives both.

    It works by repeatedly taking two undecided units and moving probability mass between
    them until at least one is settled at 0 or 1, which preserves each marginal exactly
    and conserves the sum at every move.
    """
    p = np.asarray(p, dtype="float64").copy()
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    if np.any(p < -1e-9) or np.any(p > 1 + 1e-9):
        raise ValueError(f"inclusion probabilities must lie in [0, 1]; got "
                         f"[{p.min()}, {p.max()}]")
    p = np.clip(p, 0.0, 1.0)
    total = p.sum()
    if abs(total - round(total)) > 1e-6:
        raise ValueError(f"sum of inclusion probabilities must be a whole number, "
                         f"got {total!r}")

    out = np.empty(n, dtype="float64")
    out[:] = np.nan
    # Units that arrive already at 0 or 1 are decided and must not enter the pivot loop:
    # two of them would give s = 2 and a 0/0 in the case-2 branch below.
    eps = 1e-12
    decided = (p <= eps) | (p >= 1.0 - eps)
    out[decided] = np.round(p[decided])
    undecided = np.flatnonzero(~decided)

    # `a` is the current pivot: the one unit still undecided from earlier rounds.
    a = None
    for j in undecided:
        if a is None:
            a = j
            continue
        pa, pb = p[a], p[j]
        s = pa + pb
        if s < 1.0:
            # One of the two must end at 0; the mass moves entirely onto the other.
            # The unit that KEEPS the mass is chosen with probability proportional to its
            # OWN share, which is what preserves its marginal: E[p_a'] = (pa/s)*s = pa.
            # Using pb/s here instead is subtly wrong in a way that is easy to miss - the
            # totals stay exact and the multiset of marginals is unchanged, but they come
            # back attached to the wrong units, so the population is tilted toward the
            # wrong walkers. It cost a debugging round; test_pivotal_marginals_are_not_
            # merely_permuted pins the mapping, not just the values.
            if rng.random() < (pa / s if s > 0 else 0.0):
                p[a], p[j] = s, 0.0
                out[j] = 0.0
                # `a` stays the pivot.
            else:
                p[a], p[j] = 0.0, s
                out[a] = 0.0
                a = j
        else:
            # One of the two must end at 1.
            if rng.random() < (1.0 - pb) / (2.0 - s):
                p[a], p[j] = 1.0, s - 1.0
                out[a] = 1.0
                a = j
            else:
                p[a], p[j] = s - 1.0, 1.0
                out[j] = 1.0
                # `a` stays the pivot.
    if a is not None and np.isnan(out[a]):
        # One unit can be left over as the pivot; since the total is a whole number and
        # every other unit is settled, its remaining mass is 0 or 1 by construction.
        out[a] = float(round(p[a]))
    return out.astype(bool)


def multiplicities(expected, rng: np.random.Generator) -> np.ndarray:
    """Integer offspring counts with ``E[N_i] = expected_i`` and ``sum N_i`` exact.

    ``expected`` is a walker's mean number of children and is routinely greater than 1 -
    that is what cloning IS - so this is not a selection problem but an allocation one:
    every walker gets ``floor(expected_i)`` children outright and the leftover fractions
    are allocated by :func:`pivotal_sample`, which is exactly the sub-problem it solves.
    """
    e = np.asarray(expected, dtype="float64")
    if np.any(e < -1e-9):
        raise ValueError("expected multiplicities must be non-negative")
    e = np.clip(e, 0.0, None)
    n = int(round(e.sum()))
    if abs(e.sum() - n) > 1e-6:
        raise ValueError(f"expected multiplicities must sum to a whole number, got {e.sum()}")
    base = np.floor(e).astype("int64")
    frac = e - base
    # Rounding can leave the fractional total a hair off a whole number; correct the
    # largest fraction rather than let pivotal_sample reject the vector.
    want = n - int(base.sum())
    drift = frac.sum() - want
    if abs(drift) > 1e-9:
        frac[int(np.argmax(frac))] -= drift
        frac = np.clip(frac, 0.0, 1.0)
    return base + pivotal_sample(frac, rng).astype("int64")


def parents_from_multiplicities(counts) -> np.ndarray:
    """``[0,2,0,1] -> [1,1,3]``: the parent index of each child, in walker order."""
    return np.repeat(np.arange(len(counts)), np.asarray(counts, dtype="int64"))


def ess(weights) -> float:
    """Kish effective sample size, ``(sum w)^2 / sum w^2``."""
    w = np.asarray(weights, dtype="float64")
    d = float(np.sum(w ** 2))
    return float(np.sum(w) ** 2 / d) if d > 0 else 0.0


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Resampling:
    """Everything that happened at one resampling time."""
    step: int
    C: float
    theta: np.ndarray          # raw score per walker, before resampling
    z: np.ndarray              # standardized score
    V: np.ndarray              # splitting function C_k * z
    log_W: np.ndarray          # V_k(i) - V_{k-1}(parent of i)
    R: float                   # mean_i exp(log_W_i) - the step's normalization
    counts: np.ndarray         # offspring per walker
    parents: np.ndarray        # parent index of each new walker
    ess: float                 # effective sample size of the resampling weights

    @property
    def max_multiplicity(self) -> int:
        return int(self.counts.max()) if self.counts.size else 0

    @property
    def killed(self) -> int:
        return int((self.counts == 0).sum())

    def to_dict(self) -> dict:
        return dict(step=self.step, C=self.C, R=self.R, ess=self.ess,
                    theta=np.asarray(self.theta).tolist(),
                    z=np.asarray(self.z).tolist(), V=np.asarray(self.V).tolist(),
                    log_W=np.asarray(self.log_W).tolist(),
                    counts=np.asarray(self.counts).astype(int).tolist(),
                    parents=np.asarray(self.parents).astype(int).tolist())

    @classmethod
    def from_dict(cls, d: dict) -> "Resampling":
        return cls(step=int(d["step"]), C=float(d["C"]), R=float(d["R"]),
                   ess=float(d["ess"]),
                   theta=np.asarray(d["theta"], dtype="float64"),
                   z=np.asarray(d["z"], dtype="float64"),
                   V=np.asarray(d["V"], dtype="float64"),
                   log_W=np.asarray(d["log_W"], dtype="float64"),
                   counts=np.asarray(d["counts"], dtype="int64"),
                   parents=np.asarray(d["parents"], dtype="int64"))


@dataclass
class DMCResult:
    n_walkers: int
    C: tuple
    steps: list[Resampling] = field(default_factory=list)
    final_V: np.ndarray | None = None     # V_K carried by each surviving walker
    log_Z: float = 0.0
    genealogy: np.ndarray | None = None   # (K, N) parent index per step
    final_parents: np.ndarray | None = None   # survivors of the last resampling
    states: object = None                 # population at the horizon, aligned with weights

    # -- the estimator ----------------------------------------------------- #
    @property
    def weights(self) -> np.ndarray:
        """``exp(-V_K)`` per surviving walker: the tilt the algorithm has to undo."""
        if self.final_V is None:
            return np.ones(self.n_walkers)
        return np.exp(-np.asarray(self.final_V, dtype="float64"))

    def expectation(self, phi) -> float:
        """``E_P[phi]`` from the tilted population. See the module docstring."""
        f = np.asarray(phi, dtype="float64")
        return float(np.exp(self.log_Z) * np.mean(f * self.weights))

    def exceedance(self, values, thresholds) -> np.ndarray:
        """``P(values > a)`` for each ``a``, under the ORIGINAL (untilted) law."""
        v = np.asarray(values, dtype="float64")
        return np.array([self.expectation(v > float(a)) for a in np.atleast_1d(thresholds)])

    def normalization_check(self) -> float:
        """``Z * mean(exp(-V_K))``, which estimates ``E_P[1]`` and must come out near 1.

        A free self-consistency test: it uses the same machinery as every probability the
        experiment reports, so a bookkeeping error in ``Z`` or in the final weights shows
        up here without needing a known answer to compare against.

        It is noisy, though - a product of K estimated normalizations times a mean
        dominated by its smallest term. On the reference OU problem at N = 256 a single
        run has a standard deviation of ~0.5 and ranges over roughly 0.6 to 3, so read it
        across runs, not from one.

        **"Near 1" is the MEAN, not the typical reading.** At the pilot's size (N = 64,
        K = 5, the aires.md schedule) the distribution over 2000 replays is strongly
        right-skewed: mean 0.99 but median 0.73, sd 1.17, with 36% of runs below 0.6 and
        5% below 0.28. The production run read 0.568, which is its 33rd percentile.
        Judging a single run against 1.0 will therefore condemn a healthy one about a
        third of the time; compare it against the replayed distribution for the schedule
        actually used, and prefer the check that has teeth - whether the exceedance curve
        agrees with direct sampling in the range where BOTH have members.
        """
        return self.expectation(np.ones(self.n_walkers))

    # -- diagnostics -------------------------------------------------------- #
    @property
    def ess_by_step(self) -> np.ndarray:
        return np.array([s.ess for s in self.steps])

    @property
    def max_multiplicity_by_step(self) -> np.ndarray:
        return np.array([s.max_multiplicity for s in self.steps])

    def ancestry(self) -> np.ndarray:
        """Index of the step-0 walker each final walker descends from."""
        if self.genealogy is None or not len(self.genealogy):
            return np.arange(self.n_walkers)
        line = np.arange(self.n_walkers)
        for parents in reversed(self.genealogy):
            line = np.asarray(parents)[line]
        return line

    @property
    def n_founders(self) -> int:
        """How many of the original walkers still have descendants."""
        return int(np.unique(self.ancestry()).size)

    # -- serialization ------------------------------------------------------ #
    #
    # The production driver runs the loop across two conda envs and a Slurm walltime, so
    # the bookkeeping has to survive as a file: `--stage res` produces it and `--stage
    # compare` (and every figure) consumes it, with no GPU in between. `states` is NOT
    # serialized - in production it is a handle to a directory tree, and the population's
    # actual values are re-derived from the walker diagnostics on disk.
    def to_dict(self) -> dict:
        return dict(
            n_walkers=int(self.n_walkers), C=list(self.C), log_Z=float(self.log_Z),
            steps=[s.to_dict() for s in self.steps],
            final_V=None if self.final_V is None else np.asarray(self.final_V).tolist(),
            genealogy=None if self.genealogy is None
            else np.asarray(self.genealogy).astype(int).tolist(),
            final_parents=None if self.final_parents is None
            else np.asarray(self.final_parents).astype(int).tolist(),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "DMCResult":
        return cls(
            n_walkers=int(d["n_walkers"]), C=tuple(d["C"]), log_Z=float(d["log_Z"]),
            steps=[Resampling.from_dict(s) for s in d["steps"]],
            final_V=None if d.get("final_V") is None
            else np.asarray(d["final_V"], dtype="float64"),
            genealogy=None if d.get("genealogy") is None
            else np.asarray(d["genealogy"], dtype="int64"),
            final_parents=None if d.get("final_parents") is None
            else np.asarray(d["final_parents"], dtype="int64"),
        )

    def summary(self) -> str:
        rows = [f"  {'step':>4} {'C_k':>5} {'R_k':>8} {'ESS':>7} {'ESS/N':>6} "
                f"{'killed':>7} {'max mult':>9}"]
        for s in self.steps:
            rows.append(f"  {s.step:>4} {s.C:>5.2f} {s.R:>8.4f} {s.ess:>7.2f} "
                        f"{s.ess / self.n_walkers:>6.2f} {s.killed:>7d} "
                        f"{s.max_multiplicity:>9d}")
        rows.append(f"  log Z = {self.log_Z:+.4f}   normalization check = "
                    f"{self.normalization_check():.4f} (should be ~1)")
        rows.append(f"  founders surviving: {self.n_founders}/{self.n_walkers}")
        return "\n".join(rows)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def run(n_walkers: int, C, propagate, score, *, rng: np.random.Generator | None = None,
        standardize_scores: bool = True, ref: tuple | None = None,
        init=None, verbose: bool = False) -> DMCResult:
    """Run the DMC loop for ``len(C)`` resampling steps.

    Parameters
    ----------
    n_walkers
        Population size, held exactly constant by :func:`multiplicities`.
    C
        The splitting schedule, one coefficient per resampling time. ``C_k = 0`` makes
        step ``k`` free-running.
    propagate
        ``propagate(step, parents, states) -> states``: select the given parents, then
        advance them one leg. ``step`` is 1-based; ``parents`` is ``None`` at step 1, where
        ``init`` is used. Whatever it returns is opaque to this module and handed straight
        back to ``score``.

        **It is called once more than there are resampling times**, with
        ``step = len(C) + 1``, to carry the survivors of the last resampling from ``t_K``
        to the verification horizon ``t_f``. That final call is not optional bookkeeping:
        the population that comes out of a resampling is ``states[parents]``, not
        ``states``, so without it ``result.states`` would be the pre-resampling population
        while ``result.weights`` describes the post-resampling one, and every probability
        derived from the pair would be wrong. (An experiment whose last resampling sits
        exactly at ``t_f`` passes a ``propagate`` that simply gathers and returns.)
    score
        ``score(step, states) -> array of length n_walkers``. The observable's forecast
        value per walker - the FCN3 ensemble mean in production, the current index value
        for the persistence baseline.
    standardize_scores, ref
        ``ref=(mean, sd)`` fixes the standardization instead of taking it from the live
        population, which is what makes the estimator strictly unbiased rather than merely
        consistent. ``standardize_scores=False`` uses the raw score as ``z``.
    """
    rng = np.random.default_rng() if rng is None else rng
    C = tuple(float(c) for c in C)
    res = DMCResult(n_walkers=n_walkers, C=C)

    states = init
    parents = None
    V_prev = np.zeros(n_walkers)          # V_0 == 0
    genealogy = []

    for k, Ck in enumerate(C, start=1):
        states = propagate(k, parents, states)
        theta = np.asarray(score(k, states), dtype="float64")
        if theta.shape != (n_walkers,):
            raise ValueError(f"score() must return {n_walkers} values, got {theta.shape}")
        if not np.all(np.isfinite(theta)):
            raise ValueError(f"step {k}: score returned non-finite values")

        if standardize_scores:
            z = standardize(theta, *(ref if ref is not None else (None, None)))
        else:
            z = theta.copy()
        V = Ck * z

        log_W = V - V_prev
        W = np.exp(log_W - log_W.max())               # stabilised; the shift cancels in R
        R = float(np.mean(W)) * float(np.exp(log_W.max()))
        expected = np.exp(log_W) / R if R > 0 else np.ones(n_walkers)
        # Guard the exact-sum contract against float drift before pivotal sampling.
        expected *= n_walkers / expected.sum()

        counts = multiplicities(expected, rng)
        new_parents = parents_from_multiplicities(counts)
        step = Resampling(step=k, C=Ck, theta=theta, z=z, V=V, log_W=log_W, R=R,
                          counts=counts, parents=new_parents, ess=ess(expected))
        res.steps.append(step)
        genealogy.append(new_parents)
        res.log_Z += float(np.log(R))
        if verbose:
            print(f"  [dmc] step {k}: C={Ck:.2f} R={R:.4f} ESS={step.ess:.1f}/{n_walkers} "
                  f"killed={step.killed} max_mult={step.max_multiplicity}", flush=True)

        V_prev = V[new_parents]           # the survivors carry their parent's V forward
        parents = new_parents

    # The last resampling produced children that have not moved yet. Carry them to the
    # horizon so `states`, `final_V` and `weights` all describe the SAME population.
    res.final_V = V_prev
    res.genealogy = np.array(genealogy) if genealogy else None
    res.final_parents = parents
    res.states = propagate(len(C) + 1, parents, states)
    return res
