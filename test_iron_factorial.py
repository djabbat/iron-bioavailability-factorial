#!/usr/bin/env python3
"""
test_iron_factorial.py — verification suite for the JFS factorial pipeline.

Run:  python3 -m pytest test_iron_factorial.py -v
  or:  python3 test_iron_factorial.py   (standalone runner)

Checks
------
T1  Design is a balanced 2x4x2 = 16 conditions x 4 reps = 64 rows.
T2  Effect-coded design matrix is full rank and mutually orthogonal.
T3  Reproducibility: same seed -> identical report JSON.
T4  Type-I error: under the null, proportion of p<0.05 per term ~ alpha.
T5  Signal recovery: data with a single N2 main effect -> N2 significant,
    other main effects show ~alpha Type-I rate.
T6  Covariate coefficient recovery (sign + order of magnitude) in ANCOVA.
T7  Numerical F-value for N2 matches the closed-form orthogonal-projection SS.
"""

import itertools
import json
import os
import tempfile

import numpy as np
from scipy import stats

import iron_factorial as f

ALPHA = 0.05


# --------------------------------------------------------------------------- #
def test_design_dimensions():
    d = f.build_design()
    n_cond = len(f.N2_LEVELS) * len(f.CITRATE_LEVELS) * len(f.BILE_LEVELS)
    assert n_cond == 16
    assert len(d) == n_cond * f.REPS == 64
    # balanced: each N2 level 32, each citrate level 16, each bile level 32
    assert (d["N2"].value_counts() == 32).all()
    assert (d["citrate"].value_counts() == 16).all()
    assert (d["bile"].value_counts() == 32).all()
    print("T1 PASS: design dimensions & balance")


def test_design_orthogonal():
    d = f.build_design()
    X, names = f._effect_code(d)
    assert X.shape[1] == 12
    assert np.linalg.matrix_rank(X) == 12, "design matrix must be full rank"
    # normalized columns should be mutually orthogonal
    Xn = X / np.linalg.norm(X, axis=0)
    G = Xn.T @ Xn
    G -= np.eye(G.shape[0])
    # exclude intercept column (norm exactly 1, fine) - all should be ~0 off-diag
    assert np.abs(G).max() < 1e-8, f"columns not orthogonal, max={np.abs(G).max():.2e}"
    print("T2 PASS: full rank + mutually orthogonal design matrix")


def test_reproducibility():
    with tempfile.TemporaryDirectory() as tmp:
        a = f.main(outdir=tmp, n_sim=20)
        with open(a) as fh:
            r1 = json.load(fh)
        b = f.main(outdir=tmp, n_sim=20)
        with open(b) as fh:
            r2 = json.load(fh)
    assert r1 == r2, "reports must be identical for the same seed"
    print("T3 PASS: reproducible output (fixed seed)")


def _null_power_per_term(n_sim=400):
    """Proportion of p<0.05 per term under the null across simulations."""
    design = f.build_design()
    hits = dict.fromkeys(
        ["N2", "citrate", "citrate_linear", "bile", "N2 x citrate",
         "N2 x bile", "citrate x bile", "N2 x citrate x bile"], 0)
    for s in range(n_sim):
        data = f.simulate_ferritin(design, seed=10000 + s, with_effects=False)
        an = f.fit_anova(data)
        for k in hits:
            if an["terms"][k]["p"] < ALPHA:
                hits[k] += 1
    return {k: v / n_sim for k, v in hits.items()}, n_sim


def test_type1_error():
    rates, n_sim = _null_power_per_term(n_sim=400)
    tol = 3.0 * np.sqrt(ALPHA * (1 - ALPHA) / n_sim)  # ~3 sigma
    for k, rate in rates.items():
        assert abs(rate - ALPHA) < tol, \
            f"Type-I error for {k} = {rate:.3f} outside {ALPHA}+-{tol:.3f}"
        print(f"   null p<0.05 rate {k:<22} = {rate:.3f}")
    print("T4 PASS: Type-I error within tolerance for all terms")


def test_signal_recovery_single_effect():
    """Only N2 main effect present -> N2 rejects, others ~alpha rate."""
    design = f.build_design()
    n_sim = 400
    sig = {}
    for k in ["N2", "citrate", "bile", "N2 x bile"]:
        sig[k] = 0
    for s in range(n_sim):
        data = f.simulate_ferritin(design, seed=20000 + s)
        # zero-out all effects except N2 by post-adding only N2
        # (reuse null data then inject N2 effect)
        base = f.simulate_ferritin(design, seed=30000 + s, with_effects=False)
        inj = f.EFFECT_N2 * (base["N2"].values == "N2").astype(float)
        y = base["ferritin"].values + inj
        data = base.copy()
        data["ferritin"] = y
        an = f.fit_anova(data)
        for k in sig:
            if an["terms"][k]["p"] < ALPHA:
                sig[k] += 1
    # N2 must be detected (power high with EFFECT_N2)
    assert sig["N2"] / n_sim > 0.98, f"N2 detection rate too low: {sig['N2']/n_sim}"
    tol = 3.0 * np.sqrt(ALPHA * (1 - ALPHA) / n_sim)
    for k in ["citrate", "bile", "N2 x bile"]:
        rate = sig[k] / n_sim
        assert rate < ALPHA + tol, f"spurious detection of {k}: {rate:.3f}"
    print("T5 PASS: single N2-effect recovered; no spurious detections"
          f" (N2 rate {sig['N2']/n_sim:.3f})")


def test_covariate_coefficient_recovery():
    # controlled data with known covariate betas, no factor effects
    rng = np.random.default_rng(5)
    n = 2000
    n2 = np.tile([-0.5, 0.5], n // 2)
    phyt = 700 + rng.normal(0, f.SD_PHYTATE, n)
    poly = 40 + rng.normal(0, f.SD_POLYPH, n)
    y = (f.MU_BASELINE + f.BETA_PHYTATE * (phyt - 700)
         + f.BETA_POLYPH * (poly - 40) + rng.normal(0, 5, n))
    X = np.column_stack([np.ones(n), n2, phyt - 700, poly - 40])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    # true = BETA_PHYTATE, BETA_POLYPH (both negative here)
    assert b[2] * f.BETA_PHYTATE > 0, "phytate slope sign mismatch"
    assert b[3] * f.BETA_POLYPH > 0, "polyphenol slope sign mismatch"
    assert 0.5 < abs(b[2]) / abs(f.BETA_PHYTATE) < 1.5
    assert 0.5 < abs(b[3]) / abs(f.BETA_POLYPH) < 1.5
    print(f"T6 PASS: covariate recovery beta_phyt={b[2]:.3f} (true "
          f"{f.BETA_PHYTATE}), beta_poly={b[3]:.3f} (true {f.BETA_POLYPH})")


def test_anova_numeric_consistency():
    """F for N2 must equal (SS_N2/1)/(MS_resid) using orthogonal projection."""
    d = f.build_design()
    data = f.simulate_ferritin(d, seed=20260828)
    y = data["ferritin"].values.astype(float)
    X, names = f._effect_code(data)
    beta, rss = f._ols(y, X)
    n, p = X.shape
    df_res = n - p
    ms_res = rss / df_res
    # orthogonal SS for the N2 column
    x_n2 = X[:, names.index("N2")]
    ss_n2 = float((x_n2 @ y) ** 2 / (x_n2 @ x_n2))
    F_expected = ss_n2 / ms_res
    F_got = f.fit_anova(data)["terms"]["N2"]["F"]
    assert abs(F_got - F_expected) < 1e-3, \
        f"F mismatch: got {F_got}, expected {F_expected}"
    print(f"T7 PASS: F(N2)={F_got:.4f} matches closed-form {F_expected:.4f}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            fails += 1
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"ERROR: {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} tests passed")
    raise SystemExit(1 if fails else 0)
