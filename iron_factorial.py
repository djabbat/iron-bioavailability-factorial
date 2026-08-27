#!/usr/bin/env python3
"""
iron_factorial.py — Analysis pipeline for the JFS factorial experimental design
on iron bioavailability from sprouted wheat.

Manuscript:
  "Iron Bioavailability from Sprouted Wheat: A Factorial Experimental Design
   Testing Nitrogen-Atmosphere Drying, Citrate Chelation, and Bile Salt Composition"

Design (matches manuscript v2):
  Factors : 2 (nitrogen drying) x 4 (citrate dose) x 2 (bile salt) = 16 conditions
  Benchmark: 1 ascorbate condition (air, 1 mM ascorbate, standard bile)
  Total   : 17 conditions x 12 true replicates = 204 observations
  Citrate doses (mM): 0, 2, 5, 10  (UNEQUALLY spaced -> orthogonal polynomial
                                    contrasts computed on the actual doses)

Pipeline
--------
1. build_design()      -> 16 factorial conditions x reps
2. build_benchmark()   -> the ascorbate reference condition (17th)
3. simulate()          -> Caco-2 ferritin + covariates (phytic acid, polyphenols,
                          LOX, PPO), Fe2+/total ratio, and DMT1/DCYTB (H8)
4. fit_anova()         -> three-way ANOVA for ferritin (sum-to-zero coding)
5. fit_ancova()        -> + phytic acid, polyphenols, LOX, PPO covariates (H5)
6. fe2_analysis()      -> Fe2+/total ratio by N2 (Speciation)
7. h8_analysis()       -> DMT1/DCYTB mRNA ratio by bile composition (H8)
8. assumptions()       -> Shapiro-Wilk + Levene on full-model residuals
9. power_analysis()    -> Monte-Carlo power (N2, N2 x citrate, bile/H8)

Reproducible: fixed seed -> identical JSON report each run.
Requires: numpy, scipy, pandas. No statsmodels.
"""

from __future__ import annotations

import argparse
import json
import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260828)  # fixed seed

# Design constants (manuscript Table 1)
N2_LEVELS = ["air", "N2"]
CITRATE_LEVELS = ["0", "2", "5", "10"]        # mM (labels)
CITRATE_DOSE = {"0": 0.0, "2": 2.0, "5": 5.0, "10": 10.0}  # mM
BILE_LEVELS = ["standard", "optimised"]
ASC_DOSE = 1.0                # mM ascorbate benchmark
REPS = 12                     # true replicates per condition

# Planning (prior/published) effect magnitudes — power/simulation only
MU_BASELINE = 100.0           # ng ferritin/mg protein, air control
EFFECT_N2 = 26.0
EFFECT_BEST_CITRATE = 20.0    # peak citrate effect (inverted-U in dose)
EFFECT_BEST_BILE = 12.0
INTERACT_N2_CITRATE = 13.0
ASC_SHIFT = 46.0              # ascorbate reference effect above baseline
SD = 25.0

# Covariates (independent of factors; N2 does not change phytate — prior data)
BETA_PHYTATE = -0.9
BETA_POLYPH = -1.0
BETA_LOX = -0.2
BETA_PPO = -0.15
SD_PHYTATE = 8.0
SD_POLYPH = 5.0
SD_LOX = 5.0
SD_PPO = 3.0


# --------------------------------------------------------------------------- #
# 1. Design construction
# --------------------------------------------------------------------------- #
def build_design(reps: int = REPS) -> pd.DataFrame:
    """Full 2x4x2 factorial (16 cells) x reps."""
    cells = list(itertools.product(N2_LEVELS, CITRATE_LEVELS, BILE_LEVELS))
    rows = [
        {"N2": n2, "citrate": c, "bile": b}
        for n2, c, b in cells for _ in range(reps)
    ]
    return pd.DataFrame(rows)


def build_benchmark(reps: int = REPS) -> pd.DataFrame:
    """The ascorbate internal-benchmark condition (17th, air/1mM asc/standard)."""
    return pd.DataFrame([
        {"N2": "air", "citrate": "0", "bile": "standard", "ascorbate": ASC_DOSE}
        for _ in range(reps)
    ])


def _orthogonal_poly(dose: np.ndarray, deg: int = 2) -> list[np.ndarray]:
    """Orthogonal polynomial contrast columns (linear, quadratic) for a given
    (possibly unequally spaced) set of dose values — Gram-Schmidt, mean-centered."""
    cols = []
    x = np.asarray(dose, dtype=float)
    xc = x - x.mean()
    for k in range(1, deg + 1):
        c = xc ** k
        c = c - c.mean()
        for prev in cols:
            c = c - (prev @ c) / (prev @ prev) * prev
        nrm = np.linalg.norm(c)
        cols.append(c / nrm if nrm > 1e-12 else np.zeros_like(c))
    return cols


# --------------------------------------------------------------------------- #
# 2. Simulation
# --------------------------------------------------------------------------- #
def _citrate_eff(dose_mM: np.ndarray) -> np.ndarray:
    """Inverted-U citrate dose profile, peak near upper-mid dose."""
    # concave over the tested 0-10 mM range: max at ~7 mM
    return np.maximum(0.0, (10.0 - dose_mM) * dose_mM / 25.0)  # 0..1 scale


def simulate(design: pd.DataFrame, benchmark: pd.DataFrame,
             seed: int | None = 20260828, with_effects: bool = True) -> pd.DataFrame:
    """Simulate ferritin + covariates + Fe2+/DMT1 endpoints for factorial + bench."""
    rng = np.random.default_rng(seed) if seed is not None else RNG
    n = len(design)

    n2 = (design["N2"].values == "N2").astype(float)
    bile = (design["bile"].values == "optimised").astype(float)
    dose = np.array([CITRATE_DOSE[c] for c in design["citrate"].values])

    # covariates (independent of factors)
    phytate = 700 + rng.normal(0, SD_PHYTATE, n)
    polyph = 40 + rng.normal(0, SD_POLYPH, n)
    lox = 40 + rng.normal(0, SD_LOX, n)
    ppo = 25 + rng.normal(0, SD_PPO, n)

    if not with_effects:
        ferritin = MU_BASELINE + rng.normal(0, SD, n)
    else:
        cit_eff = _citrate_eff(dose) * EFFECT_BEST_CITRATE
        interaction = n2 * (dose >= 5.0) * INTERACT_N2_CITRATE
        mu = (
            MU_BASELINE + EFFECT_N2 * n2 + cit_eff + EFFECT_BEST_BILE * bile
            + interaction
            + BETA_PHYTATE * (phytate - phytate.mean())
            + BETA_POLYPH * (polyph - polyph.mean())
            + BETA_LOX * (lox - lox.mean())
            + BETA_PPO * (ppo - ppo.mean())
        )
        ferritin = mu + rng.normal(0, SD, n)

    # Fe2+/total ratio: N2 preserves ferrous iron (78 vs 42%)
    fe2 = (42 + 36 * n2 + rng.normal(0, 4, n)).clip(0, 100)

    # DMT1/DCYTB mRNA ratio: higher with optimised bile (H8)
    dmt1 = 1.0 + 0.30 * bile + rng.normal(0, 0.10, n)

    out = design.copy()
    out["citrate_dose"] = dose
    out["ascorbate"] = 0.0
    out["phytate"] = phytate.round(1)
    out["polyphenols"] = polyph.round(1)
    out["lox"] = lox.round(1)
    out["ppo"] = ppo.round(1)
    out["fe2_ratio"] = fe2.round(1)
    out["dmt1_dcytb"] = dmt1.round(2)
    out["ferritin"] = ferritin.round(1)

    # benchmark (ascorbate): ferritin elevated, other measures at 'air' baseline
    nb = len(benchmark)
    bench = benchmark.copy()
    bench["citrate_dose"] = 0.0
    bench["phytate"] = (700 + rng.normal(0, SD_PHYTATE, nb)).round(1)
    bench["polyphenols"] = (40 + rng.normal(0, SD_POLYPH, nb)).round(1)
    bench["lox"] = (40 + rng.normal(0, SD_LOX, nb)).round(1)
    bench["ppo"] = (25 + rng.normal(0, SD_PPO, nb)).round(1)
    bench["fe2_ratio"] = (42 + rng.normal(0, 4, nb)).round(1)
    bench["dmt1_dcytb"] = (1.0 + rng.normal(0, 0.10, nb)).round(2)
    bfer = (MU_BASELINE + ASC_SHIFT) if with_effects else MU_BASELINE
    bench["ferritin"] = (bfer + rng.normal(0, SD, nb)).round(1)

    return pd.concat([out, bench], ignore_index=True)


# --------------------------------------------------------------------------- #
# 3. Design matrix (orthogonal polynomial for unequally spaced doses)
# --------------------------------------------------------------------------- #
def _effect_code(design: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    n = len(design)
    n2c = np.where(design["N2"].values == "N2", 0.5, -0.5).astype(float)
    bilec = np.where(design["bile"].values == "optimised", 0.5, -0.5).astype(float)
    dose = np.array([CITRATE_DOSE[c] for c in design["citrate"].values])
    d1, d2 = _orthogonal_poly(dose, 2)

    Xcols = {
        "Intercept": np.ones(n),
        "N2": n2c,
        "citrate_linear": d1,
        "citrate_quadratic": d2,
        "bile": bilec,
        "N2xCitL": n2c * d1,
        "N2xCitQ": n2c * d2,
        "N2xBile": n2c * bilec,
        "CitLxBile": d1 * bilec,
        "CitQxBile": d2 * bilec,
        "N2xCitLxBile": n2c * d1 * bilec,
        "N2xCitQxBile": n2c * d2 * bilec,
    }
    return np.column_stack([Xcols[k] for k in Xcols]), list(Xcols.keys())


def _ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid @ resid)


def _f_test(y, X_full, name, drop, colnames, df_resid):
    keep = [i for i, nm in enumerate(colnames) if nm not in drop]
    X_red = X_full[:, keep]
    _, rss_full = _ols(y, X_full)
    _, rss_red = _ols(y, X_red)
    df_diff = X_full.shape[1] - X_red.shape[1]
    F = ((rss_red - rss_full) / df_diff) / (rss_full / df_resid)
    p = float(stats.f.sf(F, df_diff, df_resid))
    return name, float(F), df_diff, df_resid, p


# --------------------------------------------------------------------------- #
# 4. Analysis
# --------------------------------------------------------------------------- #
def fit_anova(data: pd.DataFrame) -> dict:
    """Three-way ANOVA on the 16 factorial conditions only (excl. benchmark)."""
    fac = data[data["ascorbate"] == 0.0]
    y = fac["ferritin"].values.astype(float)
    X, colnames = _effect_code(fac)
    beta, rss = _ols(y, X)
    n, p = X.shape
    df_resid = n - p
    r2 = 1.0 - rss / float(((y - y.mean()) ** 2).sum())
    terms = [
        ("N2", ["N2"]),
        ("citrate", ["citrate_linear", "citrate_quadratic"]),
        ("citrate_linear", ["citrate_linear"]),
        ("citrate_quadratic", ["citrate_quadratic"]),
        ("bile", ["bile"]),
        ("N2 x citrate", ["N2xCitL", "N2xCitQ"]),
        ("N2 x bile", ["N2xBile"]),
        ("citrate x bile", ["CitLxBile", "CitQxBile"]),
        ("N2 x citrate x bile", ["N2xCitLxBile", "N2xCitQxBile"]),
    ]
    res = {}
    for name, drop in terms:
        _, F, d1, d2, p = _f_test(y, X, name, drop, colnames, df_resid)
        res[name] = {"F": round(F, 3), "df1": d1, "df2": df_resid, "p": round(p, 4)}
    return {"model_r2": round(r2, 4), "df_resid": int(df_resid),
            "terms": res, "n_factorial": int(n)}


def fit_ancova(data: pd.DataFrame) -> dict:
    """ANCOVA: ferritin ~ factors + phytate + polyphenols + LOX + PPO (H5)."""
    fac = data[data["ascorbate"] == 0.0]
    y = fac["ferritin"].values.astype(float)
    X_f, colnames = _effect_code(fac)
    covs = ["phytate", "polyphenols", "lox", "ppo"]
    C = np.column_stack([(fac[cn].values - fac[cn].mean()) for cn in covs])
    X = np.column_stack([X_f, C])
    full_names = colnames + covs
    beta, rss = _ols(y, X)
    n, p = X.shape
    df_resid = n - p
    r2 = 1.0 - rss / float(((y - y.mean()) ** 2).sum())
    out = {"model_r2": round(r2, 4)}
    for name, drop in [
        ("N2", ["N2"]), ("citrate", ["citrate_linear", "citrate_quadratic"]),
        ("bile", ["bile"]), ("N2 x citrate", ["N2xCitL", "N2xCitQ"]),
        ("phytate", ["phytate"]), ("polyphenols", ["polyphenols"]),
        ("lox", ["lox"]), ("ppo", ["ppo"]),
    ]:
        _, F, d1, d2, p = _f_test(y, X, name, drop, full_names, df_resid)
        out[name] = {"F": round(F, 3), "p": round(p, 4)}
    for cn in covs:
        out[f"coef_{cn}"] = round(float(beta[full_names.index(cn)]), 4)
    return out


def fe2_analysis(data: pd.DataFrame) -> dict:
    """Fe2+/total ratio by N2 (factorial conditions)."""
    fac = data[data["ascorbate"] == 0.0]
    g = fac.groupby("N2")["fe2_ratio"].agg(["mean", "std", "count"])
    t = stats.ttest_ind(fac.loc[fac["N2"] == "N2", "fe2_ratio"],
                        fac.loc[fac["N2"] == "air", "fe2_ratio"])
    return {
        "by_n2": g.round(1).to_dict("index"),
        "ttest_p": round(float(t.pvalue), 4),
    }


def h8_analysis(data: pd.DataFrame) -> dict:
    """H8: DMT1/DCYTB mRNA ratio by bile composition (factorial conditions)."""
    fac = data[data["ascorbate"] == 0.0]
    g = fac.groupby("bile")["dmt1_dcytb"].agg(["mean", "std", "count"])
    t = stats.ttest_ind(fac.loc[fac["bile"] == "optimised", "dmt1_dcytb"],
                        fac.loc[fac["bile"] == "standard", "dmt1_dcytb"])
    return {
        "by_bile": g.round(2).to_dict("index"),
        "ttest_p": round(float(t.pvalue), 4),
    }


def assumptions(data: pd.DataFrame) -> dict:
    fac = data[data["ascorbate"] == 0.0]
    y = fac["ferritin"].values.astype(float)
    X, _ = _effect_code(fac)
    beta, _ = _ols(y, X)
    resid = y - X @ beta
    shapiro = stats.shapiro(resid)
    sdf = pd.DataFrame({"g": fac[["N2", "citrate", "bile"]].astype(str).agg("|".join, axis=1),
                        "r": resid})
    lev = stats.levene(*[sdf["r"][sdf["g"] == g].values for g in sorted(sdf["g"].unique())])
    return {
        "shapiro_W": round(float(shapiro.statistic), 4),
        "shapiro_p": round(float(shapiro.pvalue), 4),
        "levene_stat": round(float(lev.statistic), 4),
        "levene_p": round(float(lev.pvalue), 4),
    }


def power_analysis(n_sim: int = 500, alpha: float = 0.05) -> dict:
    hits = {"N2": 0, "N2xCitrate": 0, "bile_H8": 0}
    for s in range(n_sim):
        design = build_design()
        bench = build_benchmark()
        data = simulate(design, bench, seed=1000 + s)
        an = fit_anova(data)
        if an["terms"]["N2"]["p"] < alpha:
            hits["N2"] += 1
        if an["terms"]["N2 x citrate"]["p"] < alpha:
            hits["N2xCitrate"] += 1
        if h8_analysis(data)["ttest_p"] < alpha:
            hits["bile_H8"] += 1
    return {k: round(v / n_sim, 3) for k, v in hits.items()}, alpha, n_sim


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(outdir: str = "results", n_sim: int = 500) -> str:
    os.makedirs(outdir, exist_ok=True)
    design = build_design()
    bench = build_benchmark()
    data = simulate(design, bench)

    fac_means = (
        data[data["ascorbate"] == 0.0]
        .groupby(["N2", "citrate", "bile"])["ferritin"]
        .agg(["mean", "std", "count"]).round(1).reset_index()
    )
    bench_mean = data[data["ascorbate"] == ASC_DOSE]["ferritin"].round(1).tolist()
    power, alpha, n_sim_used = power_analysis(n_sim=n_sim)

    report = {
        "version": "1.1.0",
        "design": {
            "structure": "2x4x2 + ascorbate benchmark",
            "n2_levels": N2_LEVELS,
            "citrate_doses_mM": list(CITRATE_DOSE.values()),
            "bile_levels": BILE_LEVELS,
            "ascorbate_benchmark_mM": ASC_DOSE,
            "replicates": REPS,
            "n_factorial_conditions": int(len(N2_LEVELS) * len(CITRATE_LEVELS) * len(BILE_LEVELS)),
            "n_benchmark_conditions": 1,
            "total_conditions": 17,
            "n_observations": int(len(data)),
            "note": "16 factorial + 1 ascorbate benchmark, 12 true replicates each = 204 obs",
        },
        "planning_assumptions": {
            "mu_baseline": MU_BASELINE, "effect_N2": EFFECT_N2,
            "effect_best_citrate": EFFECT_BEST_CITRATE, "effect_best_bile": EFFECT_BEST_BILE,
            "interact_N2_citrate": INTERACT_N2_CITRATE, "ascorbate_shift": ASC_SHIFT,
            "sd": SD,
            "note": "Planning values for power/simulation only; not measured data.",
        },
        "factorial_cell_means": fac_means.to_dict("records"),
        "ascorbate_benchmark_ferritin": {
            "n": len(bench_mean), "mean": round(float(np.mean(bench_mean)), 1),
            "sd": round(float(np.std(bench_mean)), 1),
        },
        "anova": fit_anova(data),
        "ancova": fit_ancova(data),
        "fe2_ratio": fe2_analysis(data),
        "h8_dmt1_dcytb": h8_analysis(data),
        "assumptions": assumptions(data),
        "power": {"alpha": alpha, "n_sim": n_sim_used, **power},
        "note": "Fe2+, DMT1/DCYTB and ascorbate-relative ferritin are illustrative planning parameters.",
    }

    outfile = os.path.join(outdir, "iron_factorial_report.json")
    with open(outfile, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 66)
    print("IRON BIOAVAILABILITY FACTORIAL DESIGN — ANALYSIS REPORT (v%s)" % report["version"])
    print("=" * 66)
    d = report["design"]
    print(f"Design: {d['structure']} | total conditions: {d['total_conditions']} "
          f"| obs: {d['n_observations']} | reps: {d['replicates']}")
    a = report["assumptions"]
    print(f"Assumptions: Shapiro-Wilk p={a['shapiro_p']} | Levene p={a['levene_p']}")
    print("\nANOVA (alpha=0.05):")
    for k, v in report["anova"]["terms"].items():
        fl = " *" if v["p"] < 0.05 else ""
        print(f"  {k:<24} F={v['F']:>8.3f}  df=({v['df1']},{v['df2']})  p={v['p']:.4f}{fl}")
    print(f"  R2={report['anova']['model_r2']}  n_factorial={report['anova']['n_factorial']}")
    ack = report["ancova"]
    print("\nANCOVA (H5, +phytate/polyphenols/LOX/PPO):")
    for k, v in ack.items():
        if isinstance(v, dict) and "p" in v:
            fl = " *" if v["p"] < 0.05 else ""
            print(f"  {k:<14} F={v['F']:>7.3f}  p={v['p']:.4f}{fl}")
    print(f"  coefs: phytate={ack.get('coef_phytate')} polyph={ack.get('coef_polyphenols')} "
          f"lox={ack.get('coef_lox')} ppo={ack.get('coef_ppo')}")
    b = report["ascorbate_benchmark_ferritin"]
    print(f"\nAscorbate benchmark ferritin: {b['mean']}±{b['sd']} (n={b['n']})")
    f2 = report["fe2_ratio"]
    print(f"Fe2+ ratio ttest p={f2['ttest_p']}"
          f"  (air {f2['by_n2']['air']['mean']}%, N2 {f2['by_n2']['N2']['mean']}%)")
    h8 = report["h8_dmt1_dcytb"]
    print(f"H8 DMT1/DCYTB ttest p={h8['ttest_p']}")
    pw = report["power"]
    print(f"\nPower: N2={pw['N2']}  N2xCitrate={pw['N2xCitrate']}  "
          f"bile/H8={pw['bile_H8']}  (alpha={pw['alpha']}, n_sim={pw['n_sim']})")
    print(f"\nReport: {outfile}")
    return outfile


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="JFS iron-factorial pipeline v1.1.0")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--n-sim", type=int, default=500)
    main(ap.parse_args().outdir, ap.parse_args().n_sim)
