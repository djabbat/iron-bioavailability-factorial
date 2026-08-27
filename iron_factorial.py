#!/usr/bin/env python3
"""
iron_factorial.py — Analysis pipeline for the JFS factorial experimental design
on iron bioavailability from sprouted wheat.

Manuscript:
  "Iron Bioavailability from Sprouted Wheat: A Factorial Experimental Design
   Testing Nitrogen-Atmosphere Drying, Citrate Chelation, and Bile Salt Composition"

Design: 2 (nitrogen drying) x 4 (citrate dose) x 2 (bile salt) = 16 conditions, n=4.

Pipeline
--------
1. build_design()          -> full 2x4x2 factorial design matrix (16 cells x n reps)
2. power_analysis()        -> Monte-Carlo power for the primary effect (alpha=0.05)
3. simulate_ferritin()     -> simulated Caco-2 ferritin data with main effects,
                              N2 x citrate interaction, and covariates
                              (phytate, polyphenols)
4. fit_anova()             -> three-way ANOVA (sum-to-zero effect coding), F-tests
5. fit_ancova()            -> add phytate & polyphenol covariates (H5 test)
6. assumptions()           -> Shapiro-Wilk (normality) + Levene (homoscedasticity)
7. reproduce from a fixed seed -> identical JSON report every run

Only numpy/scipy/pandas are required (no statsmodels).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

# Global reproducibility
RNG = np.random.default_rng(20260828)  # fixed seed -> reproducible output

# Design constants (match manuscript Section 2.1)
N2_LEVELS = ["air", "N2"]                 # Factor A
CITRATE_LEVELS = ["0", "low", "mid", "high"]  # Factor B (dose), ordered
BILE_LEVELS = ["porcine", "optimised"]    # Factor C
REPS = 4                                # true replicates per condition

# Level-string -> index 0..3 for orthogonal polynomial contrasts
_CITRATE_INDEX = {lvl: i for i, lvl in enumerate(CITRATE_LEVELS)}

# Assumed (prior / published) effect magnitudes, used ONLY for power + simulation.
# These are planning values, not measured data.
MU_BASELINE = 100.0          # ng ferritin/mg protein, air control
EFFECT_N2 = 26.0             # N2 main effect (ng/mg)
EFFECT_BEST_CITRATE = 20.0   # peak citrate effect at 'high' dose (inverted-U)
EFFECT_BEST_BILE = 12.0      # bile composition main effect
INTERACT_N2_CITRATE = 13.0   # N2 x citrate positive interaction
SD = 25.0                    # between-run SD (CV ~25%)

# Covariate generation (phytate, polyphenols) — independent of the processing
# factors, matching the prior finding that N2 does not alter phytic acid.
BETA_PHYTATE = -0.9          # ng ferritin per mg IP6/g (identifiable, secondary to factors)
BETA_POLYPH = -1.0           # ng ferritin per mg GAE/g
SD_PHYTATE = 8.0
SD_POLYPH = 5.0


# --------------------------------------------------------------------------- #
# 1. Design matrix
# --------------------------------------------------------------------------- #
def build_design(reps: int = REPS) -> pd.DataFrame:
    """Return the full orthogonal 2x4x2 factorial design (16 cells x reps)."""
    cells = [
        (n2, c, b)
        for n2 in N2_LEVELS
        for c in CITRATE_LEVELS
        for b in BILE_LEVELS
    ]
    rows = []
    for n2, c, b in cells:
        for _ in range(reps):
            rows.append({"N2": n2, "citrate": c, "bile": b})
    return pd.DataFrame(rows)


def _citrate_dose(citrate: str) -> float:
    """Map ordered citrate level to a numeric dose for dose-response contrast."""
    return {"0": 0.0, "low": 1.0, "mid": 2.0, "high": 3.0}[citrate]


# --------------------------------------------------------------------------- #
# 2. Simulation of ferritin data (planning model)
# --------------------------------------------------------------------------- #
def _citrate_profile(dose: float | np.ndarray) -> float | np.ndarray:
    """Citrate dose-response (inverted-U) normalised to peak at 'high'."""
    # unimodal: rises through 'low'/'mid', peaks at 'high', returns partway at end
    # -> use a concave bump
    return np.maximum(0.0, np.asarray(dose, dtype=float) * (4.0 - np.asarray(dose, dtype=float)) * 0.375)


def simulate_ferritin(design: pd.DataFrame,
                      seed: int | None = 20260828,
                      with_effects: bool = True) -> pd.DataFrame:
    """Simulate Caco-2 ferritin + covariates under the planning model.

    If ``with_effects=False`` all factor and covariate effects are set to zero
    (pure null model), used for Type-I-error (power under H0) checks.
    """
    rng = np.random.default_rng(seed) if seed is not None else RNG
    n = len(design)
    n2_num = (design["N2"].values == "N2").astype(float)
    bile_num = (design["bile"].values == "optimised").astype(float)
    dose = np.array([_citrate_dose(c) for c in design["citrate"].values])

    # Covariates: independent of the factors (consistent with the prior finding
    # that N2 does not change phytic acid). Measured per experimental run.
    phytate = 700 + rng.normal(0, SD_PHYTATE, n)   # mg IP6/g
    polyph = 40 + rng.normal(0, SD_POLYPH, n)      # mg GAE/g

    if not with_effects:
        mu = np.full(n, MU_BASELINE)
    else:
        citrate_eff = _citrate_profile(dose) * EFFECT_BEST_CITRATE
        interaction = n2_num * (dose >= 2.0) * INTERACT_N2_CITRATE
        mu = (
            MU_BASELINE
            + EFFECT_N2 * n2_num
            + citrate_eff
            + EFFECT_BEST_BILE * bile_num
            + interaction
            + BETA_PHYTATE * (phytate - phytate.mean())
            + BETA_POLYPH * (polyph - polyph.mean())
        )
    ferritin = mu + rng.normal(0, SD, n)

    out = design.copy()
    out["citrate_dose"] = dose
    out["phytate"] = phytate.round(1)
    out["polyphenols"] = polyph.round(1)
    out["ferritin"] = ferritin.round(1)
    return out


# --------------------------------------------------------------------------- #
# 3. OLS / F-test helpers (three-way factorial, sum-to-zero effect coding)
# --------------------------------------------------------------------------- #
def _effect_code(design: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build a full-rank sum-to-zero design matrix for N2*citrate*bile.

    Weighted-mean (effect) coding with explicit orthogonal polynomial contrasts
    for the 4 equally-spaced citrate dose levels:
        linear    = (-3, -1, 1, 3)
        quadratic = ( 1, -1, -1, 1)
    together with 2-level codes of +/-0.5 for N2 and bile. Because the design
    is balanced, all resulting columns are mutually orthogonal.
    """
    n = len(design)
    n2c = np.where(design["N2"].values == "N2", 0.5, -0.5).astype(float)     # A
    bilec = np.where(design["bile"].values == "optimised", 0.5, -0.5).astype(float)  # C

    # Orthogonal polynomial contrasts for citrate (index 0..3 = 0/low/mid/high)
    lin = np.array([-3.0, -1.0, 1.0, 3.0])
    quad = np.array([1.0, -1.0, -1.0, 1.0])
    idx = np.array([_CITRATE_INDEX[c] for c in design["citrate"].values])
    d1 = lin[idx]
    d2 = quad[idx]

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


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares coefficients + residual sum of squares."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid @ resid)


def _f_test(y, X_full, name, drop: list[str], colnames, df_resid_full):
    """Type-III-style F test by full-vs-reduced comparison for given terms.

    The intercept is retained in BOTH models (nested comparison); only the
    term columns in `drop` are removed from the reduced model.
    """
    keep = [i for i, nm in enumerate(colnames) if nm not in drop]
    X_red = X_full[:, keep]
    beta, rss_full = _ols(y, X_full)
    _, rss_red = _ols(y, X_red)
    df_diff = X_full.shape[1] - X_red.shape[1]
    F = ((rss_red - rss_full) / df_diff) / (rss_full / df_resid_full)
    p = float(stats.f.sf(F, df_diff, df_resid_full))
    return name, float(F), df_diff, df_resid_full, p


def fit_anova(data: pd.DataFrame) -> dict:
    """Three-way factorial ANOVA; returns terms + F-statistics + p-values."""
    y = data["ferritin"].values.astype(float)
    X, colnames = _effect_code(data)
    beta, rss = _ols(y, X)
    n, p = X.shape
    df_resid = n - p
    ms_resid = rss / df_resid
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
    results = {}
    for name, drop in terms:
        _, F, d1, d2, p = _f_test(y, X, name, drop, colnames, df_resid)
        results[name] = {"F": round(F, 3), "df1": d1, "df2": df_resid, "p": round(p, 4)}
    return {"model_r2": round(r2, 4), "df_resid": int(df_resid),
            "ms_residual": round(ms_resid, 3), "terms": results}


def fit_ancova(data: pd.DataFrame) -> dict:
    """ANCOVA: ferritin ~ factors + phytate + polyphenols (H5: composition is covariate)."""
    y = data["ferritin"].values.astype(float)
    X_f, colnames = _effect_code(data)
    cov = np.column_stack([
        (data["phytate"].values - data["phytate"].mean()),
        (data["polyphenols"].values - data["polyphenols"].mean()),
    ])
    X = np.column_stack([X_f, cov])
    full_names = colnames + ["phytate", "polyphenols"]
    beta, rss = _ols(y, X)
    n, p = X.shape
    df_resid = n - p
    r2 = 1.0 - rss / float(((y - y.mean()) ** 2).sum())

    # covariate significance
    out = {"model_r2": round(r2, 4), "df_resid": int(df_resid)}
    for term, drop in [
        ("N2", ["N2"]),
        ("citrate", ["citrate_linear", "citrate_quadratic"]),
        ("bile", ["bile"]),
        ("N2 x citrate", ["N2xCitL", "N2xCitQ"]),
        ("phytate", ["phytate"]),
        ("polyphenols", ["polyphenols"]),
    ]:
        _, F, d1, d2, p = _f_test(y, X, term, drop, full_names, df_resid)
        out[term] = {"F": round(F, 3), "df1": d1, "df2": df_resid, "p": round(p, 4)}
    # covariate coefficients
    out["coef_phytate"] = round(float(beta[full_names.index("phytate")]), 4)
    out["coef_polyphenols"] = round(float(beta[full_names.index("polyphenols")]), 4)
    return out


def assumptions(data: pd.DataFrame) -> dict:
    """Normality (Shapiro-Wilk) + homoscedasticity (Levene) on the full model."""
    y = data["ferritin"].values.astype(float)
    X, _ = _effect_code(data)
    beta, _ = _ols(y, X)
    resid = y - X @ beta
    shapiro = stats.shapiro(resid)
    # Levene on residuals grouped by cell (16 groups)
    groups = [
        f"{r.N2}|{r.citrate}|{r.bile}"
        for r in data[["N2", "citrate", "bile"]].itertuples(index=False)
    ]
    sdf = pd.DataFrame({"g": groups, "r": resid})
    cell_means = sdf.groupby("g")["r"].mean()
    # centred residuals within cell
    lev = stats.levene(
        *[sdf["r"][sdf["g"] == g].values for g in sorted(sdf["g"].unique())]
    )
    return {
        "shapiro_W": round(float(shapiro.statistic), 4),
        "shapiro_p": round(float(shapiro.pvalue), 4),
        "levene_stat": round(float(lev.statistic), 4),
        "levene_p": round(float(lev.pvalue), 4),
    }


def power_analysis(n_sim: int = 500, alpha: float = 0.05) -> dict:
    """Monte-Carlo power for the N2 main effect and N2 x citrate interaction."""
    hits_n2 = 0
    hits_inter = 0
    for i in range(n_sim):
        design = build_design()
        data = simulate_ferritin(design, seed=1000 + i)
        an = fit_anova(data)
        if an["terms"]["N2"]["p"] < alpha:
            hits_n2 += 1
        if an["terms"]["N2 x citrate"]["p"] < alpha:
            hits_inter += 1
    return {
        "n_sim": n_sim,
        "alpha": alpha,
        "power_N2_main": round(hits_n2 / n_sim, 3),
        "power_N2xCitrate": round(hits_inter / n_sim, 3),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(outdir: str = "results", n_sim: int = 500) -> str:
    os.makedirs(outdir, exist_ok=True)

    design = build_design()
    data = simulate_ferritin(design)

    # Group means per cell (for anticipated Figure 2)
    cell_means = (
        data.groupby(["N2", "citrate", "bile"])["ferritin"]
        .agg(["mean", "std", "count"])
        .round(1)
    )
    cell_rows = [
        {
            "N2": r.N2, "citrate": r.citrate, "bile": r.bile,
            "mean": r.mean, "std": r.std, "n": int(r.count),
        }
        for r in cell_means.reset_index().itertuples(index=False)
    ]

    report = {
        "design": {
            "structure": "2x4x2",
            "n2_levels": N2_LEVELS,
            "citrate_levels": CITRATE_LEVELS,
            "bile_levels": BILE_LEVELS,
            "replicates": REPS,
            "total_conditions": int(
                len(N2_LEVELS) * len(CITRATE_LEVELS) * len(BILE_LEVELS)
            ),
            "n_observations": int(len(data)),
        },
        "planning_assumptions": {
            "mu_baseline_ng_mg": MU_BASELINE,
            "effect_N2": EFFECT_N2,
            "effect_best_citrate": EFFECT_BEST_CITRATE,
            "effect_best_bile": EFFECT_BEST_BILE,
            "interact_N2_citrate": INTERACT_N2_CITRATE,
            "sd": SD,
            "note": "Planning values for power/simulation only; not measured data.",
        },
        "cell_means": cell_rows,
        "anova": fit_anova(data),
        "ancova": fit_ancova(data),
        "assumptions": assumptions(data),
        "power": power_analysis(n_sim=n_sim),
        "seed_note": "Simulation is illustrative (planned effect model). "
                     "Assumptions tests report the full-model residuals.",
    }

    outfile = os.path.join(outdir, "iron_factorial_report.json")
    with open(outfile, "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable summary
    print("=" * 64)
    print("IRON BIOAVAILABILITY FACTORIAL DESIGN — ANALYSIS REPORT")
    print("=" * 64)
    print(f"Design: 2x4x2  |  conditions: "
          f"{report['design']['total_conditions']}  |  observations: {report['design']['n_observations']}  |  reps: {REPS}")
    print(f"\nAssumptions (full-model residuals):")
    a = report["assumptions"]
    print(f"  Shapiro-Wilk  W={a['shapiro_W']}  p={a['shapiro_p']}")
    print(f"  Levene        stat={a['levene_stat']}  p={a['levene_p']}")
    print(f"\nANOVA (alpha=0.05):")
    for k, v in report["anova"]["terms"].items():
        flag = " *" if v["p"] < 0.05 else ""
        print(f"  {k:<24} F={v['F']:>7.3f}  df=({v['df1']},{v['df2']})  p={v['p']:.4f}{flag}")
    print(f"  model R2 = {report['anova']['model_r2']}")
    print(f"\nANCOVA with covariates (H5):")
    for k, v in report["ancova"].items():
        if isinstance(v, dict) and "p" in v:
            flag = " *" if v["p"] < 0.05 else ""
            print(f"  {k:<14} F={v['F']:>7.3f}  p={v['p']:.4f}{flag}")
    print(f"  coef_phytate={report['ancova'].get('coef_phytate')}  "
          f"coef_polyphenols={report['ancova'].get('coef_polyphenols')}")
    print(f"\nPower (Monte-Carlo, n_sim={report['power']['n_sim']}, alpha=0.05):")
    print(f"  N2 main           : {report['power']['power_N2_main']}")
    print(f"  N2 x citrate      : {report['power']['power_N2xCitrate']}")
    print(f"\nReport written to: {outfile}")
    return outfile


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="JFS iron-factorial analysis pipeline")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--n-sim", type=int, default=500,
                    help="Monte-Carlo simulations for power")
    args = ap.parse_args()
    main(args.outdir, args.n_sim)
