# xres events — dates & verification windows

The 12 out-of-sample CONUS extreme events used in the cross-resolution experiment
(`xres/xconfig.py::XRES_EVENTS`). Each event is anchored to a **peak date**; the
**verification week** is the 7 days ending on the peak, and each lead's **init date** is
peak − 14 d (week 2), − 21 d (week 3), − 28 d (week 4). All events are post-2019, i.e.
out-of-sample for the `<2019` GenCast checkpoints.

| # | Event | Family | Headline metric | Peak | Verification week | Init wk-2 | Init wk-3 | Init wk-4 |
|--:|---|---|---|---|---|---|---|---|
| 1 | PNW_HeatDome_2021 | Heat | t2m anomaly (K) | 2021-06-28 | 2021-06-22 → 06-28 | 2021-06-14 | 2021-06-07 | 2021-05-31 |
| 2 | Southwest_HeatWave_2020 | Heat | t2m anomaly (K) | 2020-08-16 | 2020-08-10 → 08-16 | 2020-08-02 | 2020-07-26 | 2020-07-19 |
| 3 | California_HeatWave_2022 | Heat | t2m anomaly (K) | 2022-09-06 | 2022-08-31 → 09-06 | 2022-08-23 | 2022-08-16 | 2022-08-09 |
| 4 | WinterStorm_Uri_2021 | Cold | t2m anomaly (K) | 2021-02-15 | 2021-02-09 → 02-15 | 2021-02-01 | 2021-01-25 | 2021-01-18 |
| 5 | PolarVortex_2019 | Cold | t2m anomaly (K) | 2019-01-30 | 2019-01-24 → 01-30 | 2019-01-16 | 2019-01-09 | 2019-01-02 |
| 6 | WinterStorm_Elliott_2022 | Cold | t2m anomaly (K) | 2022-12-23 | 2022-12-17 → 12-23 | 2022-12-09 | 2022-12-02 | 2022-11-25 |
| 7 | HurricaneIda_2021 | Hurricane | u850 speed (m/s) | 2021-08-29 | 2021-08-23 → 08-29 | 2021-08-15 | 2021-08-08 | 2021-08-01 |
| 8 | HurricaneIan_2022 | Hurricane | u850 speed (m/s) | 2022-09-28 | 2022-09-22 → 09-28 | 2022-09-14 | 2022-09-07 | 2022-08-31 |
| 9 | HurricaneIdalia_2023 | Hurricane | u850 speed (m/s) | 2023-08-30 | 2023-08-24 → 08-30 | 2023-08-16 | 2023-08-09 | 2023-08-02 |
| 10 | California_AR_Floods_2023 | Extreme rain | total precip (mm) | 2023-01-10 | 2023-01-04 → 01-10 | 2022-12-27 | 2022-12-20 | 2022-12-13 |
| 11 | Kentucky_Floods_2022 | Extreme rain | total precip (mm) | 2022-07-28 | 2022-07-22 → 07-28 | 2022-07-14 | 2022-07-07 | 2022-06-30 |
| 12 | Vermont_Floods_2023 | Extreme rain | total precip (mm) | 2023-07-10 | 2023-07-04 → 07-10 | 2023-06-26 | 2023-06-19 | 2023-06-12 |

Notes:

- **Cold events** (4–6) are in `xconfig.COLD_EVENTS`: their extreme tail is the LOW side of
  the t2m-anomaly distribution (the "extreme member" selection flips sign there).
- Hurricane and rain events also cache a free **secondary metric** (weekly-max |U850| speed
  / max 12 h precip) alongside the headline.
- The four events used in the `figures/xres/maps3var/` map grids are **1, 4, 7, 10** — one
  per family.
