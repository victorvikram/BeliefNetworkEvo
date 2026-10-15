"""
Spill 06b: Pooling the structure-guided sorting estimates across the two panel eras
(exploratory).

spill_06 found arrangement-based sorting in the 2006-14 panels; its pre-specified
replication in the 2016-20 panel missed the verdict rule, with a raw-correlation estimate
almost identical to the original but half the sample. This combines the two relative-fit
coefficients by inverse-variance (fixed-effect) weighting and tests their difference.
Standard errors are recovered from each coefficient and its two-sided HC1 p-value. Not
pre-specified.

Usage: PYTHONIOENCODING=utf-8 python scripts/spill_06b_pool_panels.py
       (run spill_06 with and without --panel 2020 first)
Outputs: stdout
"""
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import norm

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def coef_se(results, est):
    b = results["coefs"][est]["rel_fit"]
    p = results["pvalues"][est]["rel_fit"]
    return b, abs(b) / norm.isf(p / 2)


def main():
    eras = {}
    for name, tag in (("2006-14", ""), ("2016-20", "_2020")):
        with open(CACHE_DIR / f"spill_06_results{tag}.pkl", "rb") as fh:
            eras[name] = pickle.load(fh)

    print("=" * 70)
    print("SPILL 06b: POOLED STRUCTURE-GUIDED SORTING ESTIMATE (exploratory)")
    print("=" * 70)
    for est in ("partial", "raw"):
        (b1, s1), (b2, s2) = (coef_se(eras[e], est) for e in eras)
        w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
        b = (w1 * b1 + w2 * b2) / (w1 + w2)
        se = (w1 + w2) ** -0.5
        z_diff = (b1 - b2) / np.hypot(s1, s2)
        print(f"\n[{est}]")
        print(f"  2006-14: OR {np.exp(b1):.3f} (log-OR {b1:+.3f}, SE {s1:.3f})")
        print(f"  2016-20: OR {np.exp(b2):.3f} (log-OR {b2:+.3f}, SE {s2:.3f})")
        print(f"  pooled:  OR {np.exp(b):.3f} [{np.exp(b - 1.96 * se):.3f}, {np.exp(b + 1.96 * se):.3f}], "
              f"p={2 * norm.sf(abs(b / se)):.5f}")
        print(f"  difference between eras: z={z_diff:+.2f}, p={2 * norm.sf(abs(z_diff)):.3f}")


if __name__ == "__main__":
    main()
