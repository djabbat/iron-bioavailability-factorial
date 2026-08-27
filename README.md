# Iron Bioavailability from Sprouted Wheat — Factorial Analysis Code

Open-source analysis pipeline for the factorial experimental design (2×4×2) on
iron bioavailability from sprouted wheat, submitted to the *Journal of Food
Science* (ScholarOne submission). This repository contains the reproducible
statistical analysis plan, power analysis and simulation code — it is a
**planning/analysis tool**, not experimental data.

## Paper
Tqemaladze, J. *Iron Bioavailability from Sprouted Wheat: A Factorial
Experimental Design Testing Nitrogen-Atmosphere Drying, Citrate Chelation, and
Bile Salt Composition.* Journal of Food Science (submitted).

## What this code does
- Builds the full orthogonal **2×4×2 factorial design** (nitrogen drying ×
  citrate dose × bile salt) = 16 conditions × 4 true replicates = 64 observations.
- Computes **Monte-Carlo statistical power** for the primary (N₂) effect and the
  N₂×citrate interaction at α = 0.05.
- Simulates Caco-2 ferritin data under a pre-registered planning model (with
  phytic-acid and polyphenol covariates, per the journal's compositional-analysis
  requirement).
- Fits a **three-way factorial ANOVA** (sum-to-zero effect coding) with
  normality (Shapiro–Wilk) and homoscedasticity (Levene) checks, and an
  **ANCOVA** adding phytic acid and polyphenols as covariates (H5 test).
- Writes a reproducible JSON report (`results/iron_factorial_report.json`).

## Requirements
Python ≥ 3.9 with `numpy`, `scipy`, `pandas`.

## Usage
```bash
# Full analysis + report (Monte-Carlo power, n_sim simulations)
python3 iron_factorial.py --outdir results --n-sim 400

# Verify statistical correctness (7 checks)
python3 test_iron_factorial.py
```

## Verification suite (`test_iron_factorial.py`)
The pipeline is verified (all tests pass) for:
1. Balanced design dimensions (16 conditions × 4 reps = 64).
2. Full-rank, mutually orthogonal effect-coded design matrix.
3. Reproducibility under a fixed seed.
4. **Type-I error ≈ α** for every term under the null model.
5. Correct detection of a single planted effect with no spurious interactions.
6. Recovery of covariate coefficients in ANCOVA.
7. Numerical consistency of the F-test against the closed-form orthogonal
   projection.

## Statistical model (summary)
- Factors: N₂ (2), citrate dose (4, orthogonal linear/quadratic contrasts), bile
  salt (2).
- Primary endpoint: Caco-2 ferritin (ng/mg protein); covariates: phytic acid
  (mg IP₆/g), polyphenols (mg GAE/g).
- Tests at α = 0.05 in R-equivalent fashion; assumptions validated by
  Shapiro–Wilk and Levene.

## License
MIT — see [LICENSE](LICENSE).

## Data availability
This repository contains **planning code and simulation**, not biological data.
Experimental data will be deposited to OSF/Zenodo on completion of the study.

## Authors
Jaba Tqemaladze, MD — Georgia Longevity Alliance.
ORCID: 0000-0001-8651-7243
