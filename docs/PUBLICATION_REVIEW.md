# Publication review — 2026-10-13

## Prepared changes

- Safely extracted the supplied ZIP after checking paths and symlink entries.
- Preserved LICENSE and NOTICE.md exactly, including reference author attribution.
- Matched the supplied README metrics to the supplied metrics.json and retained
  original reports under docs/evidence/supplied-*.
- Corrected the final walk-forward fold boundary, which included the first test
  week, and added a regression test.
- Clarified that evaluation uses rolling one-week observed-history features,
  rather than fixed-origin recursive multi-week evaluation. Hyperparameters are
  fixed; validation is diagnostic, not an implemented model selection step.
- Recorded fresh tests, CLI training, HTTP smoke checks, and installed versions.
- Included NOTICE.md in the Docker image and excluded generated artifacts,
  environment files, virtualenv, caches, and raw CSVs from the publication ZIP.

## Limits

The supplied attribution describes an independent implementation and a historical
MIT declaration by the reference repository. This review preserves that statement;
it does not independently establish upstream license status or originality.
Scores are from synthetic data. Real-data validation, recursive multi-week
backtesting, dashboard validation, Docker, MLflow, and hosted CI need separate
verification. Dependencies have minimum bounds rather than a portable lockfile;
current-environment.txt records this Linux test environment only.

## Publication handoff

No GitHub repository was created, no code pushed, and no external settings changed.
After reviewing the package, provide explicit authorization, the target repository
URL or owner/name, desired visibility if creating a repository, and target branch.
For an existing repository, its current contents and branch must be inspected
before choosing a non-destructive publication path. Do not overwrite existing
history or files without review.

The package retains the original repository layout, GitHub Actions workflow,
README, source, tests, Docker/Compose, dashboard, and attribution.
