# Release verification

Checks completed during packaging:

- All Python files parse successfully.
- The seven dependency-light unit tests pass. The optional tensor-transfer
  test is skipped because PyTorch is not installed in the packaging environment.
- The MNLI command-line help and recorded-results summary execute successfully.
- AST comparisons confirm that the SST-2 training/evaluation functions are
  unchanged from the retained experiment script. The MNLI dataset/evaluation
  and training loop are unchanged, apart from restricting the task argument to
  MNLI. The original stratified sampler and MNLI data-loading functions match.
- All 30 MNLI numeric records were checked against their original ten-epoch
  logs and file hashes. Success labels, first-fit epochs, training curves,
  matched-development curves, update counts, and data exposure agree.
- The numeric supplement retains six SST-2 condition scores and all 30 MNLI
  repetitions. It contains no raw paths or unfiltered diagnostic metadata.
- File contents and filenames were scanned for known project-specific author
  identifiers, absolute personal/cluster paths, cluster node names, email
  addresses, SSH/private-key material, and common credential patterns.

The submitted ZIP uses a fixed file allowlist, relative member names, neutral
timestamps, and no filesystem owner IDs, archive comments, extended attributes,
or hidden/system files. Archive integrity and its extracted contents are checked
after creation.

Not performed during packaging: clean-environment dependency installation,
native renderer validation, model downloads, or new GPU training. These checks
therefore establish source-level consistency and anonymization, not a new
end-to-end numerical replication. Exact checkpoint-intervention reproduction
also requires the original source-task weights, which are not bundled.
