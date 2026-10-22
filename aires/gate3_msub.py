"""Gate 3 score skill at M = 1..6, by subsampling the cached M=6 score cubes. CPU only.

Is M=6 FCN3 scoring overpaying? Scoring is ~87% of an FCN3-walker case's H100-h and its
cost is linear in M, so this asks how much rank skill each smaller M keeps. Every member
subset of each size is scored. Score members use common random numbers across walkers,
so a production M=k run is one fixed subset shared by all walkers - that is what each
combination here is, and why subsets are never drawn per walker.

    PYTHONPATH=. python -m aires.gate3_msub [--event PNW_HeatDome_2021]

Reads the Gate 3 JSON and score cubes, writes `<event>_gate3_msub.csv` beside them. M=6 is
the single full set and is asserted to reproduce the Gate 3 JSON's rho_s.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np
import pandas as pd
import xarray as xr

from aires import aconfig as A
from aires import gate3 as G
from aires import score as S


def members(ev, res: dict) -> pd.DataFrame:
    """Per-member box index for every (walker, lead) cube, checked against Gate 3's theta."""
    from aires import aindex as AI
    rows = []
    for i, w in enumerate(res["walkers"]):
        for l in res["leads"]:
            t = S.Task(ev.name, w, float(l))
            with xr.open_dataset(t.cube_path) as cube:
                v = np.asarray(AI.indices(cube, ev.name, ev.peak, ev.metric,
                                          where=f"score cube {t}")["box"].values, dtype=float)
            assert v.size == res["m_members"], (str(t), v.size)
            assert abs(v.mean() - res["theta"]["box"][str(l)][i]) < 1e-6, str(t)
            rows += [dict(walker=w, lead_days=float(l), member=m, box=x,
                          realized_box=res["realized"]["box"][i]) for m, x in enumerate(v)]
    return pd.DataFrame(rows)


def by_m(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for l, g in df.groupby("lead_days"):
        piv = g.pivot(index="walker", columns="member", values="box")
        real = g.groupby("walker")["realized_box"].first().loc[piv.index].to_numpy()
        for m in range(1, piv.shape[1] + 1):
            rhos = np.array([G._spearman(piv[list(c)].mean(axis=1), real)[0]
                             for c in combinations(piv.columns, m)])
            out.append(dict(lead_days=l, M=m, n_subsets=rhos.size, rho_mean=rhos.mean(),
                            rho_min=rhos.min(), rho_max=rhos.max(),
                            frac_pass=(rhos >= A.GATE3["spearman_min"]).mean()))
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--event", default="PNW_HeatDome_2021")
    a = ap.parse_args(argv)
    ev = G._event(a.event)
    res = json.loads(G.result_path(ev).read_text())
    t = by_m(members(ev, res))
    full = t[t.M == res["m_members"]].set_index("lead_days")["rho_mean"]
    for l, s in res["stats"]["box"].items():
        assert abs(full[float(l)] - s["spearman"]) < 1e-9, (l, full[float(l)], s["spearman"])
    out = A.gate3_dir(ev.name) / f"{ev.name}_gate3_msub.csv"
    t.to_csv(out, index=False)
    print(t.round(3).to_string(index=False))
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
