# Section 4.3 experiment software

This anonymous supplement contains the code for the two pretraining–adaptation
experiments in Section 4.3. It is a partial software release, not the complete
paper repository. No datasets, model weights, source-task training pipelines,
server submission scripts, or private run metadata are included.

## Contents

- `sst2.py`: Random, Frozen, and Full FT on SST-2 for BERT-base and PIXEL-base.
- `mnli.py`: matched MNLI runs from original, PAWS-X-adapted, or
  Taxi1500-adapted PIXEL encoder weights.
- `_pixel_training.py`, `_diagnostics.py`: retained training loop and passive
  diagnostics used by the MNLI wrapper.
- `data_utils.py`, `metrics.py`, `protocol.py`: data loading, sampling, metrics,
  and the fixed intervention protocol.
- `recorded_results.json`, `summarize_results.py`: anonymized numeric results
  for checking the reported comparisons without training or downloading data.
- `tests/`: dependency-light protocol checks and an optional tensor-transfer test.

## Environment and external dependencies

Training requires a Linux CUDA environment and one GPU per process. The original
runs used RTX 4090/4090D GPUs; the same hardware is not enforced by this release.
The retained environment used Python 3.10, PyTorch 2.2.2 with CUDA 12.1,
Transformers 4.17.0, Datasets 2.14.6, and NumPy 1.23.5.

Install PyTorch for the available CUDA runtime and the Python dependencies in
`requirements.txt`. The requirements file is not a complete lockfile: most
versions not individually recorded for the original runs are left unpinned. Installation
and end-to-end GPU training have not been rerun in the packaging environment.
Torchvision is paired with PyTorch using the upstream compatibility versions;
some other packages have compatibility bounds for the older dependency APIs.
These bounds do not assert the exact versions used by the original runs.

Install the upstream [PIXEL implementation](https://github.com/xplip/pixel) and
its native rendering prerequisites, including Pango, Cairo, GObject introspection,
and the checkpoint's fonts. The recorded checkout was based on revision
`1b0da29`. This package uses only the plain `PangoCairoTextRenderer`; it does not
include the local bigram-rendering extensions used elsewhere in the paper.
Make the upstream `src` directory importable, for example:

```sh
git clone https://github.com/xplip/pixel.git vendor/pixel
git -C vendor/pixel checkout 1b0da29
export PYTHONPATH="$PWD/vendor/pixel/src:${PYTHONPATH:-}"
```

Check that `from pixel import PIXELForSequenceClassification,
PangoCairoTextRenderer` succeeds. Do not install an unrelated package named
`pixel` from a package index. Obtain the original model and renderer assets from
`Team-PIXEL/pixel-base`, and BERT assets from `bert-base-uncased`, or supply local
copies through the command-line arguments. The original PIXEL renderer uses
16-pixel patches, RGB rendering, font size 8, DPI 120, and the Go Noto Current
font assets specified by the checkpoint. Preserve those renderer assets and
font settings when comparing conditions.

## Data

Download the datasets separately under their original terms. The default layout is:

```text
data/
  sst2/
    train-00000-of-00001.parquet
    validation-00000-of-00001.parquet
  mnli/
    multinli_1.0_train.jsonl
    multinli_1.0_dev_matched.jsonl
    multinli_1.0_dev_mismatched.jsonl
```

SST-2 parquet files must contain `sentence` and `label` columns in the original
GLUE row order. MultiNLI uses the original MultiNLI 1.0 JSONL distribution,
with `sentence1`, `sentence2`, and `gold_label` fields. The MNLI label mapping is
entailment=0, neutral=1, contradiction=2; do not substitute another mapping.
Rows without a consensus label are skipped. Sampling uses the original
stratified sampler with seed 42 for training and seed 43 for development,
independently of the model-training seed. Preprocessing strips leading/trailing
whitespace. Dataset root paths can also be supplied by `SST2_DATA_DIR`,
`MNLI_DATA_DIR`, or `--data-dir`.

The SST-2 comparison uses a fixed 24,000-example subset and the full 872-example
validation set. The MNLI experiment uses 24,000 training examples and fixed
2,000-example subsets from each official development split. `protocol.py`
records the MNLI subset hashes, and `mnli.py` verifies them before training.
Reordering the input files can change the sampled subsets. To check data only:

```sh
python mnli.py --seed 100 --initialization original --repeat 1 --check-data-only
```

The historical target subset has 29 normalized sentence overlaps with the
complete official development data, as reported in the paper's appendix.
The paired intervention retains that historical partition unchanged.

## Experiment 1: SST-2 training conditions

The released experiment is the Random/Frozen/Full FT comparison in the paper,
not the separate unified-readout follow-up explored during development.

- **Random**: randomly initialize the encoder and task head, then train both.
- **Frozen**: freeze the pretrained encoder, mean-pool its output, and train
  only a linear classifier.
- **Full FT**: initialize from pretraining and jointly train the encoder and
  classification head.

Run all six conditions with seed 42:

```sh
for model in bert pixel-base; do
  python sst2.py --model "$model" --mode full_finetune --initialization random
  python sst2.py --model "$model" --mode frozen_probe --initialization pretrained
  python sst2.py --model "$model" --mode full_finetune --initialization pretrained
done
```

Use `--model-path checkpoints/pixel-base` (or a BERT checkpoint directory) to
override the public model identifier. `--runs-root` controls output placement.
Existing run directories are not overwritten.

| Setting | BERT Random / Full FT | PIXEL Random / Full FT | Frozen readout, both models |
|---|---:|---:|---:|
| Learning rate | 2e-5 | 3e-5 | 1e-3 |
| Minibatch size | 32 | 32 | 256 feature vectors |
| Gradient accumulation | 1 | 4 | 1 |
| Maximum epochs | 10 | 30 | 50 |
| Patience | 3 | 8 | 8 |
| Warmup | 10% of scheduled updates | 100 updates | None |
| Weight decay | 0.01 | 0.01 | 0.01 |
| Input length | 128 tokens | 64 patches | Same encoder-specific lengths |

Frozen features are extracted once in batches of 64. Random and Full FT use
AdamW, linear learning-rate decay, and gradient clipping at 1.0. The linear
probe uses AdamW without a scheduler. Checkpoints/readouts are selected by
validation accuracy. Training accuracy in the joint-training logs is measured
online during training, including dropout; it is not a separate evaluation pass.
The original model-specific schedules and classification readouts are retained.
Frozen and Full FT do not share a single identical readout architecture.

## Experiment 2: MNLI initialization intervention

Supply the two previously adapted PIXEL classification checkpoints separately.
Each must contain a `pytorch_model.bin` tensor state dictionary with `vit.*`
encoder keys. The PAWS-X and Taxi1500 weights used in the paper are **not** in
this archive; without them this is an implementation release, not a self-contained
exact numerical reproduction of the checkpoint intervention. Using newly trained
source checkpoints tests the same intervention but need not reproduce its numbers.

The recorded PAWS-X source was trained on 49,401 English paraphrase examples;
the selected checkpoint was at epoch 3 (development accuracy 0.893). The
Taxi1500 source used 860 English training examples and 106 development examples,
with six topic labels; its selected checkpoint was at epoch 55 (macro-F1 about
0.541). Source tasks are different from MNLI inference. The retained overlap
audit found no normalized sentence overlap between either source training set
and the target subsets. Source training code and raw texts are not redistributed.

Run one original condition, or an adapted condition, as follows:

```sh
python mnli.py --seed 100 --initialization original --repeat 1
python mnli.py --seed 100 --initialization pawsx --repeat 1 \
  --source-checkpoint checkpoints/pawsx/pytorch_model.bin
python mnli.py --seed 100 --initialization taxi1500 --repeat 1 \
  --source-checkpoint checkpoints/taxi1500/pytorch_model.bin
```

Run each condition with seeds 100 and 5678, repeats 1 through 5, in fresh Python
processes: 30 runs in total. A repeat resets the **same** seed; it is not an
additional independent seed. CUDA deterministic algorithms were not enforced
in the original repetitions. Results can vary between reruns and environments.
These two seeds were selected after previously observed failures; their success
rates are case-study results, not an estimate over randomly sampled seeds.

| Setting | Seed 100 | Seed 5678 |
|---|---:|---:|
| Learning rate | 3e-5 | 3e-5 |
| Minibatch / accumulation | 32 / 4 | 32 / 4 |
| Warmup updates | 100 | 0 |
| Observed epochs | 10 | 10 |
| Learning-rate schedule horizon | 80 epochs | 80 epochs |
| Observed optimizer updates | 1,880 | 1,880 |
| Scheduled optimizer updates | 15,040 | 15,040 |
| Maximum input length | 256 patches | 256 patches |

AdamW uses weight decay 0.01, dropout 0.1, clipping at 1.0, and linear decay.
Early stopping is disabled. **Do not shorten the decay horizon to ten epochs**
or replace it with a 1,880-update schedule: that changes the experiment.
For each seed, the data, fresh task head, optimizer, target position embeddings,
and RNG initialization are held fixed across the three conditions. Only encoder
tensors from the source checkpoint are transferred, excluding position
embeddings. Source classifiers, poolers, and optimizer states are not transferred.
This preserves the common 256-patch positions even when the Taxi1500 checkpoint
was trained at a different input length. Runtime checks verify preservation of
the target head, positions, RNG states, data hashes, and observation budget.
Passive diagnostics retain gradient samples and per-epoch data-order hashes
without recording hostnames, scheduler accounts, or job identifiers.

A run is counted as successful when its epoch-10 online training loss is below
1.0 and training accuracy exceeds 50%. The numeric release retains all successes
and failures. It does not apply a development-accuracy threshold.

## Inspect results and run local checks

```sh
python summarize_results.py
python -m unittest discover -s tests -v
```

The summary utility needs only the Python standard library. To summarize new
MNLI outputs, use `python summarize_results.py --runs-root outputs/mnli`.
The SST-2 numbers are the retained, two-decimal figure/table values; the MNLI
arrays were extracted from the recorded epoch logs, including failed runs.
The tests do not replace full GPU execution; the tensor-transfer test skips
when PyTorch is unavailable.
See `VERIFICATION.md` for the checks completed while preparing this release.

## Anonymization and scope of changes

Personal and institutional paths have been replaced with relative paths or
arguments. The archive excludes author/contact information, original cluster
node names, Slurm scripts/accounts/job IDs, SSH settings, repository history,
credentials, raw checkpoint configs, and unfiltered historical logs. Public
upstream project and dataset names remain as necessary attribution.

The training/evaluation functions and original sampling algorithm were retained.
Packaging changes remove unused model variants and CoLA helpers, make paths
configurable, prevent SST-2 output overwrites, and consolidate the historical
MNLI launch wrappers into one portable entry point. The new wrapper loads only
tensor weights and retains the original encoder-transfer and passive-diagnostic
logic. No additional training experiments were run while preparing this archive.

See `THIRD_PARTY_NOTICES.md` for dependency attribution. Dependencies, fonts,
datasets, and pretrained weights remain subject to their own terms.
