# Data and model inputs

Run scripts from the repository root. Default input directories are:

```text
data/
  sst2/
    train-00000-of-00001.parquet
    validation-00000-of-00001.parquet
    test-00000-of-00001.parquet
  mnli/
    multinli_1.0_train.jsonl
    multinli_1.0_dev_matched.jsonl
    multinli_1.0_dev_mismatched.jsonl
  cola/
    in_domain_train.tsv
    in_domain_dev.tsv
    out_of_domain_dev.tsv
  xling/
    xnli/<language>/train-*.parquet
    xnli/<language>/validation-*.parquet
    xnli/<language>/test-*.parquet
    paws-x/<language>/train.parquet
    paws-x/<language>/validation.parquet
    paws-x/<language>/test.parquet
    sib-200/data/<language-code>/{train,dev,test}.tsv
    taxi-1500-labeled/<language>/{train,dev,test}.tsv
```

All datasets are separate downloads, not files distributed here. Preserve the
original file order because the historical samplers use it.

Environment variables:
- `SST2_DATA_DIR`, `MNLI_DATA_DIR`, `COLA_DATA_DIR`: English dataset roots.
- `XLING_DATA_DIR`: crosslingual dataset root.
- `TAXI1500_LABELED_DIR`: processed Taxi1500 TSV root, if separate.
- `ROBUSTNESS_HF_CACHE`: local Hugging Face cache.
- `EXPERIMENT_RUNS_DIR`, `TRANSFER_RUNS_DIR`: training output roots.
- `MNLI_HOLDOUT_MANIFEST`: rebuilt matched-LR holdout manifest.

English SST-2 parquet rows use `sentence` and `label`. MultiNLI 1.0 JSONL rows
use `sentence1`, `sentence2`, and `gold_label`; the label IDs are entailment=0,
neutral=1, contradiction=2. For XNLI parquet files, numeric label IDs must use
that same ordering. PAWS-X uses `sentence1`, `sentence2`, and binary `label`.
SIB-200 uses `text` and `category`. Processed Taxi1500 TSV files require
`id`, `text`, and `label`, with topic labels listed in `xling_data_utils.py`.

Model locations can be public model identifiers or local directories. Non-base
PIXEL variants default to neutral `checkpoints/` locations; provide compatible
weights and renderers separately. Transfer scripts accept a BERT model-path
override or PIXEL `--init-checkpoint`, and PIXEL input length can be specified
with `--max-seq-length`.

For SST-2 mapped test evaluation, download the authors' Stanford Sentiment
Treebank `trainDevTestTrees_PTB.zip` separately, and pass it to
`finetuning/prepare_test_labels.py --archive ... --output ...`.
No generated or predicted test labels are used.
