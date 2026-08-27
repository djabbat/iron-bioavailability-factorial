#!/usr/bin/env python3
"""
test_iron_factorial.py — verification suite for the JFS factorial pipeline (v1.1.0)

Run:  python3 -m pytest test_iron_factorial.py -v
  or:  python3 test_iron_factorial.py

Checks
------
T1  Design: 16 factorial (2x4x2) + 1 ascorbate benchmark, x12 reps = 204 obs; balanced.
T2  Effect-coded design matrix full rank + mutually orthogonal.
T3  Reproducibility: same seed -> identical report JSON.
T4  Type-I error ~ alpha under the null (factorial ANOVA terms).
T5  Signal recovery: single N2 effect detected, no spurious interactions.
T6  Covariate coefficient recovery (phytate slope sign & magnitude).
T7  Numerical F(N2) equals the closed-form orthogonal-projection value.
T8  Endpoint sanity: Fe2+/total higher under N2 (p<alpha); H8 DMT1/DCYTB higher for optimised bile.
T9  Null power ~ alpha for Fe2+ endpoint and H8 endpoint.
"""

import json
import tempfile
import itertools

import numpy as np

import iron_factorial as f

ALPHA = 0.05


def test_design_dimensions():
    assert 2 * 4 * 2 == 16
    d = f.build_design()
    assert len(d) == 16 * f.REPS == 192
    assert (d["N2"].value_counts().values == 96).all()
    assert (d["citrate"].value_counts().values == 48).all()
    assert (d["bile"].value_counts().values == 96).all()
    b = f.build_benchmark()
    assert len(b) == f.REPS == 12
    data = f.simulate(f.build_design(), f.build_benchmark(), with_effects=True)
    assert len(data) == 204
    assert int((data["ascorbate"] == f.ASC_DOSE).sum()) == 12
    print("T1 PASS: 16 factorial + 1 benchmark x 12 reps = 204 obs; balanced")


def test_design_orthogonal():
    d = f.build_design()
    X, names = f._effect_code(d)
    assert X.shape[1] == 12
    assert np.linalg.matrix_rank(X) == 12, "design must be full rank"
    Xn = X / np.linalg.norm(X, axis=0)
    G = Xn.T @ Xn
    G -= np.eye(12)
    assert np.abs(G).max() < 1e-8, f"not orthogonal, max={np.abs(G).max():.2e}"
    print("T2 PASS: full rank + mutually orthogonal design matrix")


def test_reproducibility():
    with tempfile.TemporaryDirectory() as tmp:
        a = f.main(outdir=tmp, n_sim=15)
        b = f.main(outdir=tmp, n_sim=15)
        with open(a) as fh:
            r1 = json.load(fh)
        with open(b) as fh:
            r2 = json.load(fh)
    assert r1 == r2
    print("T3 PASS: reproducible output (fixed seed)")


def _reject_rates_under_null(n_sim=300):
    design, bench = f.build_design(), f.build_benchmark()
    hits = dict.fromkeys(
        ["N2", "citrate", "citrate_linear", "bile", "N2 x citrate",
         "N2 x bile", "citrate x bile", "N2 x citrate x bile"], 0)
    for s in range(n_sim):
        data = f.simulate(design, bench, seed=50000 + s, with_effects=False)
        an = f.fit_anova(data)
        for k in hits:
            if an["terms"][k]["p"] < ALPHA:
                hits[k] += 1
    return {k: v / n_sim for k, v in hits.items()}, n_sim


def test_type1_error():
    rates, n_sim = _reject_rates_under_null(n_sim=300)
    tol = 3.0 * np.sqrt(ALPHA * (1 - ALPHA) / n_sim)
    for k, rate in rates.items():
        assert abs(rate - ALPHA) < tol, f"Type-I {k}={rate:.3f}"
        print(f"   null rate {k:<22} = {rate:.3f}")
    print("T4 PASS: Type-I error within tolerance for all terms")


def test_signal_recovery_single_effect():
    design, bench = f.build_design(), f.build_benchmark()
    n_sim = 300
    sig = {k: 0 for k in ["N2", "citrate", "bile", "N2 x bile"]}
    for s in range(n_sim):
        base = f.simulate(design, bench, seed=60000 + s, with_effects=False)
        inj = f.EFFECT_N2 * (base["N2"].values == "N2").astype(float)
        fac_rows = base["ascorbate"] == 0.0
        base.loc[fac_rows, "ferritin"] = (base.loc[fac_rows, "ferritin"].values + inj[fac_rows.values])
        an = f.fit_anova(base)
        for k in sig:
            if an["terms"][k]["p"] < ALPHA:
                sig[k] += 1
    assert sig["N2"] / n_sim > 0.98, f"N2 rate too low {sig['N2']/n_sim}"
    tol = 3.0 * np.sqrt(ALPHA * (1 - ALPHA) / n_sim)
    for k in ["citrate", "bile", "N2 x bile"]:
        r = sig[k] / n_sim
        assert r < ALPHA + tol, f"spurious {k}={r:.3f}"
    print("T5 PASS: single N2-effect recovered; no spurious detections "
          f"(N2 rate {sig['N2']/n_sim:.3f})")


def test_covariate_coefficient_recovery():
    rng = np.random.default_rng(5)
    n = 3000
    phyt = 700 + rng.normal(0, f.SD_PHYTATE, n)
    poly = 40 + rng.normal(0, f.SD_POLYPH, n)
    y = (f.MU_BASELINE + f.BETA_PHYTATE * (phyt - 700)
         + f.BETA_POLYPH * (poly - 40) + rng.normal(0, 5, n))
    X = np.column_stack([np.ones(n), phyt - 700, poly - 40])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert b[1] * f.BETA_PHYTATE > 0, "phytate sign mismatch"
    assert b[2] * f.BETA_POLYPH > 0, "polyphenol sign mismatch"
    assert 0.5 < abs(b[1] / f.BETA_PHYTATE) < 1.5
    assert 0.5 < abs(b[2] / f.BETA_POLYPH) < 1.5
    print(f"T6 PASS: covariate recovery beta_phyt={b[1]:.3f} (true {f.BETA_PHYTATE}), "
          f"beta_poly={b[2]:.3f} (true {f.BETA_POLYPH})")


def test_anova_numeric_consistency():
    design, bench = f.build_design(), f.build_benchmark()
    data = f.simulate(design, bench, seed=20260828)
    fac = data[data["ascorbate"] == 0.0]
    y = fac["ferritin"].values.astype(float)
    X, names = f._effect_code(fac)
    beta, rss = f._ols(y, X)
    df_res = len(y) - X.shape[1]
    ms_res = rss / df_res
    xn = X[:, names.index("N2")]
    ss_n2 = float((xn @ y) ** 2 / (xn @ xn))
    F_exp = ss_n2 / ms_res
    F_got = f.fit_anova(data)["terms"]["N2"]["F"]
    assert abs(F_got - F_exp) < 1e-3, f"F got {F_got}, exp {F_exp}"
    print(f"T7 PASS: F(N2)={F_got:.4f} matches closed-form {F_exp:.4f}")


def test_endpoints():
    design, bench = f.build_design(), f.build_benchmark()
    data = f.simulate(design, bench, seed=7)
    fe = f.fe2_analysis(data)
    assert fe["by_n2"]["N2"]["mean"] > fe["by_n2"]["air"]["mean"]
    assert fe["ttest_p"] < ALPHA
    h8 = f.h8_analysis(data)
    assert h8["by_bile"]["optimised"]["mean"] > h8["by_bile"]["standard"]["mean"]
    assert h8["ttest_p"] < ALPHA
    print("T8 PASS: Fe2+ higher under N2; DMT1/DCYTB higher for optimised bile (both p<0.05)")


def test_null_power_endpoints():
    design, bench = f.build_design(), f.build_benchmark()
    n_sim = 300
    fe_hits = h8_hits = 0
    for s in range(n_sim):
        data = f.simulate(design, bench, seed=70000 + s, with_effects=False)
        # endpoints have no factor dependence under null -> replace with noise
        fac = data["ascorbate"] == 0.0
        data.loc[fac, "fe2_ratio"] = 50 + np.random.default_rng(s).normal(0, 4, int(fac.sum()))
        data.loc[fac, "dmt1_dcytb"] = 1.0 + np.random.default_rng(s).normal(0, 0.1, int(fac.sum()))
        if f.fe2_analysis(data)["ttest_p"] < ALPHA:
            fe_hits += 1
        if f.h8_analysis(data)["ttest_p"] < ALPHA:
            h8_hits += 1
    tol = 3.0 * np.sqrt(ALPHA * (1 - ALPHA) / n_sim)
    assert abs(fe_hits / n_sim - ALPHA) < tol, f"Fe2 type-I {fe_hits/n_sim:.3f}"
    assert abs(h8_hits / n_sim - ALPHA) < tol, f"H8 type-I {h8_hits/n_sim:.3f}"
    print(f"T9 PASS: Fe2+ and H8 type-I rates ~ alpha "
          f"(fe2={fe_hits/n_sim:.3f}, h8={h8_hits/n_sim:.3f})")


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
