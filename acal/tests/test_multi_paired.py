"""Light tests for acal.multi_paired (pure helpers, synthetic data)."""
import numpy as np
import pandas as pd

from acal import multi_paired as MP


def test_p_txt_formats():
    st = dict(n=5, win=3, tie=1, loss=1, wilcoxon_p=0.0004)
    assert MP._p_txt(st) == "3-1-1  p<0.001"
    st["wilcoxon_p"] = 0.1234
    assert MP._p_txt(st).endswith("p=0.123")
    assert MP._p_txt(None) == "n/a"


def test_score_row_counts():
    sc = pd.DataFrame(dict(family=["heat", "heat", "cold", "cold"],
                           p=[0.0, 0.2, 0.0, 0.4], lift=[0.5, 2.0, 0.0, 3.0]))
    r = MP.score_row(sc, "p", "lift")
    assert (r["n_zero_heat"], r["n_zero_cold"], r["n_zero"]) == (1, 1, 2)
    assert (r["n_lift_gt1_heat"], r["n_lift_gt1_cold"]) == (1, 1)
    assert np.isclose(r["p_med_all"], 0.1) and np.isclose(r["lift_med_cold"], 1.5)
