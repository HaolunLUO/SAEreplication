#!/usr/bin/env python3
"""Gate before the wide LOSO run. Exit 0 = pass, 1 = fail, 2 = inputs incomplete.

Pass requires, on the 45 is_lang & sig_glove electrodes:
  - shuffled features (partial mode), every v3-grid lag, for sae, residual,
    surprisal and the SAE gain: |mean Fisher-z| < 0.01. The fraction of lags
    with |mean| > 2 channel SEM is written out and is not part of the pass;
  - timing-partialled GloVe-300 curve peaks at +200 to +300 ms.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import timing_control as tc  # noqa: E402

KEY = ["subject", "channel"]
MAX_ABS = 0.01
MAX_FRAC_2SEM = 0.10
GLOVE_PEAK = (200.0, 300.0)


def main():
    both = tc.electrode_table()
    both = both[both.both][KEY]
    out = {"n_both": int(len(both))}
    ok = True

    g = pd.read_csv(tc.TABLES / "tc_glove_both_lag_scores.csv").merge(both, on=KEY)
    gp = g[g["mode"] == "partial"].groupby("lag_ms")["fisher_z_mean"].mean()
    peak = float(gp.idxmax())
    out["glove_partial_peak_lag_ms"] = peak
    out["glove_partial_peak_z"] = float(gp.max())
    out["glove_partial_z_-250"] = float(gp.get(-250.0, np.nan))
    out["glove_partial_ratio_-250_over_peak"] = float(gp.get(-250.0, np.nan) / gp.max())
    out["glove_pass"] = bool(GLOVE_PEAK[0] <= peak <= GLOVE_PEAK[1])
    ok &= out["glove_pass"]

    p = tc.TABLES / "tc_both_lag_scores.csv"
    d = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=KEY + ["condition", "mode"])
    s = d[(d.condition == "v2_shuffle") & (d["mode"] == "partial")].merge(both, on=KEY)
    s = s[s.lag_ms.isin(tc.MAIN_LAGS)]
    n_sh = s[KEY].drop_duplicates().shape[0]
    out["n_shuffle_electrodes"] = int(n_sh)
    if n_sh < len(both):
        out["status"] = "incomplete"
        print(json.dumps(out, indent=1))
        return 2
    wide = s.pivot_table(index=KEY + ["lag_ms"], columns="feature_space", values="fisher_z_mean").reset_index()
    wide["sae_gain"] = wide["sae"] - wide["surprisal"]
    for col in ("sae", "residual", "surprisal", "sae_gain"):
        c = wide.groupby("lag_ms")[col].agg(["mean", "sem"])
        mx = float(c["mean"].abs().max())
        frac = float((c["mean"].abs() > 2 * c["sem"]).mean())
        # The 10% fraction is diagnostic, because 51 correlated lags have about
        # a 5% null rate and a flat curve already means max |mean Fisher-z| < 0.01.
        passed = mx < MAX_ABS
        out[f"shuffle_{col}_max_abs_mean_z"] = mx
        out[f"shuffle_{col}_frac_lags_gt_2sem"] = frac
        out[f"shuffle_{col}_pass"] = bool(passed)
        ok &= passed
    out["status"] = "pass" if ok else "fail"
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
