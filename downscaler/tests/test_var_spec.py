"""Variable-spec parsing and the ERA5 per-channel transforms.

The precip transform is the one that matters: ERA5 tp is a 12 h accumulation in METRES,
while the HRRR log_tp target is a 6 h accumulation of ln(mm + 1e-5). If the units or the log
are wrong the driver channel silently lands on a scale ~1000x off from its target, which
would not crash anything -- it would just quietly fail to train.
"""

import numpy as np
import pytest

from data.era5_hrrr_dataset import ERA5_TRANSFORMS, _parse_var_spec

LOG_FLOOR = np.log(1e-5)   # -11.5129: the value a fully dry pixel maps to, in BOTH datasets


def test_plain_name():
    assert _parse_var_spec("t2m") == ("t2m", None, None)


def test_name_with_level():
    assert _parse_var_spec("u:850") == ("u", 850.0, None)


def test_dict_with_level():
    assert _parse_var_spec({"name": "u", "level": 850}) == ("u", 850.0, None)


def test_dict_with_transform():
    name, level, transform = _parse_var_spec(
        {"name": "total_precipitation_12hr", "transform": "log_precip_m_to_mm"}
    )
    assert (name, level, transform) == ("total_precipitation_12hr", None, "log_precip_m_to_mm")


def test_unknown_transform_is_rejected_loudly():
    with pytest.raises(ValueError, match="unknown transform"):
        _parse_var_spec({"name": "tp", "transform": "definitely_not_a_transform"})


def test_precip_transform_matches_the_hrrr_target_convention():
    """ERA5 metres -> the same ln(mm + 1e-5) the HRRR log_tp target uses."""
    f = ERA5_TRANSFORMS["log_precip_m_to_mm"]

    # A dry pixel must land exactly on the target's dry floor, or "no rain" means two
    # different numbers on the input and output sides.
    assert f(np.array([0.0]))[0] == pytest.approx(LOG_FLOOR)

    # 10 mm of rain = 0.010 m -> ln(10 + 1e-5)
    assert f(np.array([0.010]))[0] == pytest.approx(np.log(10.0 + 1e-5), rel=1e-6)

    # 100 mm -> ln(100 + 1e-5)
    assert f(np.array([0.100]))[0] == pytest.approx(np.log(100.0 + 1e-5), rel=1e-6)


def test_precip_transform_is_monotonic_and_finite():
    f = ERA5_TRANSFORMS["log_precip_m_to_mm"]
    x = np.linspace(0.0, 0.2, 500)
    y = f(x)
    assert np.all(np.isfinite(y))
    assert np.all(np.diff(y) > 0)


def test_precip_transform_clamps_negatives():
    """ERA5 accumulations can carry tiny negative values from the archive's packing.
    log() of a negative is nan, which would poison the whole channel."""
    f = ERA5_TRANSFORMS["log_precip_m_to_mm"]
    y = f(np.array([-1e-9, -0.5, 0.0]))
    assert np.all(np.isfinite(y))
    assert np.allclose(y, LOG_FLOOR)


def test_mm_variant_agrees_with_the_metre_variant():
    f_m = ERA5_TRANSFORMS["log_precip_m_to_mm"]
    f_mm = ERA5_TRANSFORMS["log_precip_mm"]
    assert f_m(np.array([0.025]))[0] == pytest.approx(f_mm(np.array([25.0]))[0])
