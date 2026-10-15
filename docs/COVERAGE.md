# Paper-to-code map and reproduction limits

## Selection principle

Include the training, evaluation, and analysis supporting the retained paper.
Do not release the complete exploratory workspace. The manuscript used for this
inventory is the last locally synchronized version, with sections on optimization,
supervision budget, pretraining–adaptation gap, typographical robustness, and
zero-shot transfer. The manuscript itself is not changed by this release.

## §4.1: optimization and training duration

- `finetuning/train_pixel_sweep.py`: retained PIXEL training core.
- `finetuning/train_bert_lr.py`: the matched BERT LR control.
- `finetuning/train_subword_sweep.py`: BERT longer-budget/general training.
- `finetuning/prepare_mnli_holdout.py`: portable extraction of the recorded
  selection/holdout sampling algorithm.
- `finetuning/evaluate_mnli_holdout.py`: portable evaluation entry point
  added for this release, using the retained model/evaluation functions.
- `results/training.json`: 60 original PIXEL and 60 BERT LR observations,
  including failures. Six rates: 1e-6, 5e-6, 1e-5, 3e-5, 4e-5, 5e-5.
- The LR screen observes 10 epochs while retaining an 80-epoch decay horizon.
  Both models use minibatch 32, accumulation 4, 100 warmup updates, MSL 256,
  and no early stopping.
- The longer-budget comparison uses model-specific recipes, not the controlled
  LR-screen settings: PIXEL LR 1e-5, at most 80 epochs, warmup 400, patience 12;
  BERT LR 2e-5, at most 10 epochs, 10% warmup, patience 3.
- Warmup follow-up and supplementary CoLA results can use the retained training
  programs, but their full historical run orchestration is not released.
  The launcher intentionally does not offer a new parameter search.

## §4.2: supervision budget and generalization

- Both training programs support exact update caps. The coverage study uses
  15,000 updates, warmup 400, effective batch 128, and no early stopping.
- SST-2: 2k, 8k, 24k distinct examples; BERT LR 2e-5, PIXEL LR 3e-5.
- MNLI: 8k, 24k; both models LR 2e-5.
- The paired seed set is 42, 1234, 5678. The SST-2 follow-up adds only the
  seven reported seeds at 24k examples.
- `audit_sst2_checkpoint.py` re-evaluates training/development accuracy.
- `prepare_test_labels.py` reconstructs binary SST-2 test labels from the
  authors' tree-root labels. It retains four reviewed spelling variants and
  checks agreement with all 872 known validation labels; no labels are predicted.
- `evaluate_test_checkpoint.py` evaluates the development-selected checkpoint,
  reporting both full and overlap-excluded test accuracy.
- The numeric coverage records retain all original paired seeds. The paper's
  outcome-conditioned PIXEL SST-2 24k plot uses only its two high-generalization
  runs; do not silently delete the excluded observation from the released data.

## §4.3: pretraining–adaptation gap

The self-contained `pretraining/` directory retains the previously checked
Section 4.3 software release and its detailed settings. It contains the original
Random/Frozen/Full FT comparison, not the abandoned unified frozen readout.

The MNLI intervention retains seed 100 with warmup 100 and seed 5678 with
warmup 0, five repetitions per initialization, 10 observed epochs, and the
80-epoch LR horizon. Only non-position encoder tensors are transferred.
Source-task heads, poolers, and optimizer states are discarded.

The original adapted weights and source-task training pipelines are not in
this release. A newly trained source checkpoint is a new experimental input,
not an exact replacement for the original one.

## §5: typographical robustness

`robustness/plot_metrics.py` is the retained primary figure-analysis program.
`results/robustness/` contains its numeric inputs for BERT-base and PIXEL-base
on SST-2 and MNLI, including both MNLI development splits. The figure uses the
matched split, as in the retained figure program.

These records come from the retained figure inputs. They are not a reconstruction
of every model/checkpoint variant in the appendix's absolute-score table.
The original MulTypo perturbation-generation and English inference programs
could not be retrieved during this release and are not replaced with guessed
implementations. Exact noisy-text reproduction therefore remains unavailable.

## §6 and supporting transfer results

The retained `transfer/` programs support XNLI, PAWS-X, SIB-200, and Taxi1500.
ELECTRA support is removed because ELECTRA is not a reported paper model.
The data loader includes Spanish XNLI and SIB-200 evaluation, historically run
separately using the same sampling rule; other evaluation languages follow
the retained task-specific lists.

XNLI uses 20k English training examples, not the 24k used for the optimization
screen. PAWS-X uses the full 49,401-example English training split.
Selection uses English development data. Target-language scores are never used
for checkpoint selection.

Historical checkpoint selection, continuation runs, model-specific rendering,
and input lengths must be matched to reproduce a particular published table
cell. The generic defaults alone do not certify reproduction of every model.
The original bigram renderer modifications are not bundled.

## Deliberately excluded

- CKA, parameter drift, onset-geometry, and gradient-direction explorations
  not retained in the paper.
- Disjoint-MNLI/XNLI rescue attempts and failed-seed discovery launchers.
- The abandoned unified-readout experiment and additional scratch replications
  beyond the retained Random comparison.
- Crosslingual alignment/hypothesis experiments not retained in the manuscript.
- Slurm scripts, node workarounds, SSH configuration, private protocols, drafts,
  Overleaf archives, slide decks, and teaser-design iterations.
