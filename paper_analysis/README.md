# Analysis code for the labnofer PhysioNet Challenge 2026 methodological study

Code and aggregate results behind the manuscript on leave-one-site-out (LOSO)
validation versus hidden-site scores in the George B. Moody PhysioNet Challenge 2026.

## What is here
- `canonical_analysis.py` – LOSO predictions for the five candidate models, paired bootstrap, per-site and pooled comparison. Writes `canonical_results.json`.
- `omega2_robustness.py`, `site_probe.py` – site-versus-outcome variance decomposition (eta^2 / omega^2) and the multivariate site probe.
- `reproduce_results.py` – demographics-only floor and the OSF-embedding configuration grid.
- `val_vs_test_across_teams.py` – validation-versus-test comparison across teams, from the official results table in `final_results/`.
- `make_fig*.py` – figures (PNG copies included).
- `extract_*.py`, `fetch_*.py`, `caisr_features.py`, `team_code_spectral.py` – feature extraction used to build the (unreleased) feature cache.
- `fast_metric.py` – vectorised age-conditioned AUROC, checked identical to the official `compute_auroc_age`.
- `*.json` – aggregate outputs only (no per-record values).

## What is deliberately not here
Challenge data are distributed under the BDSP Credentialed Health Data License. Per-record
features, embeddings, out-of-fold predictions and the EDF files are therefore not redistributed.
To rerun, obtain the data under your own credentialed access, build the feature caches with the
`extract_*` scripts, and set `PHYSIO_ROOT` to the directory holding them.

## Notes
The submitted model (commit 443af5c) is on `main` of the team repository; this branch only adds the analysis.
