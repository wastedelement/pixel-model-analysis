# Verification completed while organizing this release

- Python syntax was checked for every released source file.
- CPU tests check the LR observation/decay budgets, coverage update caps,
  reported seed/rate choices, and counts of retained numeric observations.
- The existing Section 4.3 dependency-light tests pass; its tensor-transfer
  test requires PyTorch and is skipped if it is not installed.
- The retained sampling, dataset, training, and evaluation functions were
  compared with the local experiment snapshots. Changes to launch/configuration
  code are limited to paths, unused-model support, explicit dependency errors,
  runtime metadata, output guards, and the documented portable entry points.
- The matched BERT control checks the packaged data-loader file hash. The
  generated holdout manifest separately records hashes of actual data subsets.
- Published-result exports are numeric-only selections from retained records,
  not copies of private run configurations, source paths, or hardware logs.
- Files are scanned for project-specific personal identifiers, server paths,
  node names, email addresses, SSH/private-key material, and credential patterns.

Not verified here: clean-environment installation, native rendering/font
equivalence, new GPU training, and exact reproduction of every historical
checkpoint. Some source weights and original robustness programs are not
available in this release. See COVERAGE.md rather than inferring full paper
reproducibility from the presence of a training script.
