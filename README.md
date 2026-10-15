# Pixel Model Analysis

Selected experiment code for **Beyond Clean Accuracy: The Practical Challenges
and Benefits of Pixel-Based Language Models**.

This repository follows the experiments retained in the paper. It is a partial
release, not an archive of every exploratory run. Code for abandoned CKA studies,
unified-readout follow-ups, initialization searches, and teaser design drafts is
not included.

## What is included?

| Paper content | Location | Release scope |
|---|---|---|
| §4.1: learning rate and training duration | `finetuning/` | PIXEL/BERT training, holdout preparation/evaluation, recorded six-rate results |
| §4.2: supervision budget and SST-2 generalization | `finetuning/` | fixed-update training, checkpoint audit, test-label preparation and evaluation |
| §4.3: pretraining–adaptation gap | `pretraining/` | Random/Frozen/Full FT and repeated MNLI initialization intervention |
| §5: typographical robustness | `robustness/` | primary figure analysis and retained numeric evaluations; original noise-generation/inference pipeline not included yet |
| §6: zero-shot crosslingual transfer | `transfer/` | English training and target-language evaluation, with dataset loaders |
| Necessary experiment settings and qualifications | `docs/` | paper-to-code map, data layout, known reproduction limits |

## Start here

Use Python 3.10 and install a CUDA-compatible PyTorch environment. See
`requirements.txt`, `docs/DATA.md`, and the upstream PIXEL installation
instructions. The recorded environment used PyTorch 2.2.2 with CUDA 12.1,
Transformers 4.17.0, Datasets 2.14.6, and NumPy 1.23.5.
The dependency list is not a fully verified installation lockfile.

Training uses one GPU per process. No cluster-specific account or node is
required. Supply model/data locations using command-line arguments or the
documented environment variables. Do not copy a complete server environment
or home directory into this repository.

To inspect commands without starting training:

```sh
python run_experiment.py --experiment mnli-lr --model pixel-base --seed 42 --lr 3e-5
python run_experiment.py --experiment sample-coverage --model bert --task sst2 --seed 42 --train-size 8000
```

The launcher is dry-run by default. Add `--execute` only when ready to allocate
training resources.

For the matched MNLI LR screen, first prepare its fixed holdout:

```sh
python finetuning/prepare_mnli_holdout.py
```

For the two §4.3 experiments, see [pretraining/README.md](pretraining/README.md).
Their numeric results can be summarized without a GPU:

```sh
python pretraining/summarize_results.py
```

For zero-shot transfer, run from the repository root:

```sh
python transfer/train_xling_subword.py --task xnli --model bert
python transfer/train_xling_pixel.py --task pawsx --model pixel-base
```

The transfer programs train on English only and evaluate the selected checkpoint
on target-language test sets. These commands use retained defaults; historical
published runs can involve additional documented overrides. See the coverage
notes before interpreting a fresh run as an exact replication.

## Data, checkpoints, and scope

Datasets, fonts, pretrained/adapted weights, and perturbed sentence files are not
redistributed. Obtain them separately under their original terms.
Exact reproduction of the MNLI initialization intervention requires the original
PAWS-X- and Taxi1500-adapted weights, which are not bundled.

The ordinary PIXEL-base renderer works with the upstream PIXEL implementation.
The local bigram-rendering extensions used for other PIXEL variants are not
included. Such variants require a compatible renderer; the code raises an
explicit error when it is unavailable. Do not substitute a different renderer
and report the resulting scores as the original experiment.

Numeric records retain failed runs. Training accuracy logged during optimization
is an online epoch average; checkpoint audits evaluate a fixed model in
evaluation mode. MNLI held-out development examples are not the official
unlabelled benchmark test set.

See [docs/COVERAGE.md](docs/COVERAGE.md) for exactly what this release does and
does not reproduce. No new GPU training was performed while organizing it.

## Attribution and publication

The arXiv identifier and citation will be added after the preprint is available.
A project-wide code license has not yet been selected; dependencies and external
assets retain their own licenses. See `THIRD_PARTY_NOTICES.md`.

An AI assistant helped organize, sanitize, and document this code release and
develop portable launch/evaluation utilities. The source-level checks are listed
in `docs/VERIFICATION.md`; they do not replace GPU replication.
