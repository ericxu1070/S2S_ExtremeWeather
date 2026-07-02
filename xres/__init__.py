"""Cross-resolution GenCast S2S extreme-weather experiment (0.25deg vs 1.0deg).

A self-contained follow-up to the ``gencast_s2s`` week2/3/4 pipeline. It runs the SAME
out-of-sample extreme events at TWO resolutions and compares them:

    * 0.25deg  -> ``GenCast 0p25deg <2019.npz``  (full model, mesh 6, native 721x1440)
    * 1.0deg   -> ``GenCast 1p0deg  <2019.npz``  (full model, mesh 5, 181x360)

Both are full <2019 checkpoints differing only in resolution, so the comparison isolates
the effect of grid resolution. Three event families, each with its own headline metric:

    heat / cold        -> weekly-mean 2 m-temperature ANOMALY (K)        [t2m_anom]
    OOD hurricanes     -> weekly-mean 850 hPa wind SPEED (m/s)           [u850_speed]
    OOD extreme rain   -> weekly-TOTAL precipitation (mm)                [tp_total]

Unlike the original pipeline (which kept only the verification-week T2m field), this
experiment CACHES THE ENTIRE GenCast output (all target variables, every rollout step,
every member) cropped to the CONUS box, so any metric can be re-derived offline without
re-running the model. Only the 2-week horizon is run here; weeks 3/4 are held.

Stages (``run_xres.py``): prep (CPU) -> infer (one GPU node per resolution) -> compare
(CPU; reads both resolutions to build the 6-panel maps and the 3 combined PDFs).
"""

__all__ = ["xconfig", "xmetrics", "xdata", "xhrrr", "xinference",
           "xplotting", "xcombined"]
