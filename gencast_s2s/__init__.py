"""GenCast S2S extreme-weather ensemble pipeline.

Converted from the Colab notebook ``gencast_extreme_events_ensemble.ipynb`` into a
batch pipeline that runs on an 8xH100 node under Slurm. Three experiments share the
same code, differing only in forecast horizon (and therefore initialization date):

    * week2  -> 14-day lead (init = peak - 14d)   [the original notebook experiment]
    * week3  -> 21-day lead (init = peak - 21d)
    * week4  -> 28-day lead (init = peak - 28d)

In every experiment the *verification target is the same calendar week* (the 7 days
ending on the event peak); only the lead time to reach it changes. This isolates how
week-N predictability of the extreme degrades as the model is initialized earlier.
"""

__all__ = ["config", "data", "download", "model", "inference", "plotting"]
