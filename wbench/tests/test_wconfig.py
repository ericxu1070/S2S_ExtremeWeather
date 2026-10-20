"""The benchmark's constants against their sources, so a reference number cannot drift."""
from __future__ import annotations

from aires import aconfig as A
from aires import run_aires as R
from fcn3 import fevents as F
from wbench import wconfig as W


def test_reference_rates_match_their_sources():
    assert W.REF_A3MEGA["gencast_0p25_prod_s_per_step"] == R.SECONDS_PER_GENCAST_STEP
    assert W.REF_A3MEGA["fcn3_s_per_member_step"] == R.SECONDS_PER_FCN3_STEP


def test_one_segment_is_the_production_resampling_interval():
    assert W.SEG_DAYS == 3.0
    assert W.CONFIGS["gencast_0p25"].seg_steps == A.RES_SEG_STEPS == 6
    assert W.CONFIGS["fcn3_native"].seg_steps == 12
    assert W.CONFIGS["fcn3_e2s"].step_h == F.STEP_H == 6


def test_gencast_configs_go_through_the_walker_registry():
    # The checkpoint is chosen by aconfig.walker_model_kwargs, never by hand: job 1155
    # rolled the Mini model on 0.25 deg fields because a name was built by hand.
    assert A.walker_model_kwargs(W.GENCAST_RES["gencast_0p25"])["params_file"] == \
        "GenCast 0p25deg <2019.npz"
    assert A.walker_model_kwargs(W.GENCAST_RES["gencast_1p0"])["params_file"] == \
        "GenCast 1p0deg <2019.npz"


def test_case_schedule_is_the_production_one():
    assert W.CASE_HORIZON_DAYS == 21
    assert W.CASE_LEGS == len(A.RES_LEAD_DAYS) + 1
    assert len(W.CASE_C) == len(W.CASE_LEAD_DAYS)


def test_every_config_is_ordered_and_labelled():
    assert set(W.ORDER) == set(W.CONFIGS)
    assert W.BASELINE in W.CONFIGS
