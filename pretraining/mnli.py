"""Run one matched MNLI initialization comparison; one process per run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import data_utils
from protocol import (
    SEEDS, INITIALIZATIONS, TRAIN_N, BATCH_SIZE, GRAD_ACCUM, SCREEN_EPOCHS,
    SCHEDULE_EPOCHS, MAX_SEQ_LENGTH, LEARNING_RATE, EXPECTED_TARGET,
    EXPECTED_DEVS, digest, warmup_steps, updates_per_epoch, training_fit,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def state_hash(state):
    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().contiguous().cpu()
        hasher.update(name.encode())
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def transfer_encoder(model, source, torch):
    """Transfer encoder tensors, retaining the original target positions."""
    target = model.vit.state_dict()
    if set(source) != set(target):
        raise ValueError("Source and target encoder keys differ")
    position_key = "embeddings.position_embeddings"
    position = target[position_key].detach().clone()
    for key in target:
        if key == position_key:
            continue
        if source[key].shape != target[key].shape:
            raise ValueError("Source and target tensor shapes differ: " + key)
        target[key] = source[key]
    model.vit.load_state_dict(target, strict=True)
    if not torch.equal(position, model.vit.state_dict()[position_key]):
        raise RuntimeError("Target position embeddings changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--initialization", required=True, choices=INITIALIZATIONS)
    parser.add_argument("--repeat", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--original-checkpoint", default="Team-PIXEL/pixel-base")
    parser.add_argument("--source-checkpoint", type=Path,
                        help="Adapted PIXEL pytorch_model.bin, or its directory")
    parser.add_argument("--data-dir", help="Directory containing the original MultiNLI JSONL files")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/mnli"))
    parser.add_argument("--check-data-only", action="store_true",
                        help="Verify the exact target subsets without importing training dependencies")
    args = parser.parse_args()
    if args.data_dir:
        data_utils.MNLI_DIR = args.data_dir
    if args.initialization == "original" and args.source_checkpoint is not None:
        parser.error("The original condition must not receive a source checkpoint")
    if args.initialization != "original" and args.source_checkpoint is None and not args.check_data_only:
        parser.error("An adapted condition requires --source-checkpoint")

    rows = data_utils.get_train("mnli")
    evals = data_utils.get_evals("mnli")
    if len(rows) != TRAIN_N or digest(rows) != EXPECTED_TARGET:
        raise ValueError("Training subset differs from the recorded experiment; check data version and row order")
    if set(evals) != set(EXPECTED_DEVS):
        raise ValueError("Development split names differ")
    for key, expected in EXPECTED_DEVS.items():
        if len(evals[key]) != 2000 or digest(evals[key]) != expected:
            raise ValueError("Development subset differs from the recorded experiment: " + key)
    if args.check_data_only:
        print(json.dumps({"train_n": len(rows), "train_sha256": digest(rows),
                          "dev_sha256": {key: digest(value) for key, value in evals.items()}}, indent=2))
        return

    import _pixel_training as module
    from _diagnostics import attach_diagnostics
    torch = module.torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script does not train on CPU")
    # Repeated runs intentionally reset the same seed. The repeat is only an output label.
    suffix = f"_{args.initialization}_s{args.seed}_repeat{args.repeat}"
    out = args.runs_root / ("mnli_pixel-base" + suffix)
    if out.exists():
        raise FileExistsError("Refusing to overwrite or resume an existing run")
    source = None
    source_sha256 = None
    if args.source_checkpoint is not None:
        path = args.source_checkpoint
        if path.is_dir():
            path = path / "pytorch_model.bin"
        # Only load tensor state dictionaries, never executable pickled objects.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        source = {key[len("vit."):]: value for key, value in checkpoint.items() if key.startswith("vit.")}
        if not source or "embeddings.position_embeddings" not in source:
            raise ValueError("Expected a PIXEL classification checkpoint containing vit.* tensors")
        source_sha256 = file_hash(path)

    module.MODEL_CFG["pixel-base"]["path"] = args.original_checkpoint
    module.RUNS_ROOT = str(args.runs_root)
    module.get_train = lambda task: list(rows)
    module.get_evals = lambda task: {key: list(value) for key, value in evals.items()}
    state = attach_diagnostics(module, out)
    original_resize = module.resize_model_embeddings

    def resize(model, length):
        result = original_resize(model, length)
        if length != MAX_SEQ_LENGTH:
            raise ValueError("Target position capacity changed")
        position_key = "embeddings.position_embeddings"
        position = model.vit.state_dict()[position_key].detach().clone()
        heads = lambda: {key: value for key, value in model.state_dict().items()
                         if key.startswith(("classifier.", "pooler."))}
        head_hash = state_hash(heads())
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
        if source is not None:
            transfer_encoder(model, source, torch)
        if head_hash != state_hash(heads()):
            raise RuntimeError("MNLI head initialization changed during encoder transfer")
        if not torch.equal(position, model.vit.state_dict()[position_key]):
            raise RuntimeError("Target positions changed during encoder transfer")
        if not torch.equal(cpu_rng, torch.get_rng_state()):
            raise RuntimeError("CPU RNG changed during encoder transfer")
        after_cuda = torch.cuda.get_rng_state_all()
        if len(cuda_rng) != len(after_cuda) or not all(torch.equal(a, b) for a, b in zip(cuda_rng, after_cuda)):
            raise RuntimeError("CUDA RNG changed during encoder transfer")
        state.update({
            "seed": args.seed, "repeat": args.repeat, "initialization_name": args.initialization,
            "initial_task_head_sha256": head_hash,
            "encoder_sha256": state_hash(model.vit.state_dict()),
            "position_sha256": state_hash({position_key: position}),
            "source_checkpoint_sha256": source_sha256,
            "rng_after_initialization": {
                "cpu": state_hash({"cpu": cpu_rng}),
                "cuda": state_hash({str(i): value for i, value in enumerate(cuda_rng)}),
            },
        })
        write_json(out / "initialization.json", state)
        return result

    module.resize_model_embeddings = resize
    # Crucial: observe ten epochs but retain the original 80-epoch decay horizon.
    argv = ["mnli.py", "--task", "mnli", "--model", "pixel-base", "--run-suffix", suffix,
            "--seed", str(args.seed), "--lr", str(LEARNING_RATE),
            "--max-epochs", str(SCHEDULE_EPOCHS), "--screen-epochs", str(SCREEN_EPOCHS),
            "--warmup-steps", str(warmup_steps(args.seed)), "--batch-size", str(BATCH_SIZE),
            "--grad-accum", str(GRAD_ACCUM), "--train-size", str(TRAIN_N),
            "--max-seq-length", str(MAX_SEQ_LENGTH), "--disable-early-stopping"]
    started = time.monotonic()
    old_argv = sys.argv
    sys.argv = argv
    try:
        module.main()
        records = [json.loads(line) for line in (out / "training_log.jsonl").read_text().splitlines()]
        if len(records) != SCREEN_EPOCHS or records[-1]["optimizer_updates"] != updates_per_epoch() * SCREEN_EPOCHS:
            raise RuntimeError("Unexpected observation budget")
        if records[-1]["examples_seen"] != TRAIN_N * SCREEN_EPOCHS or state["train_subset_sha256"] != EXPECTED_TARGET:
            raise RuntimeError("Data exposure or training subset changed")
        fits = [record["epoch"] for record in records if training_fit(record)]
        write_json(out / "completion.json", {
            "seed": args.seed, "repeat": args.repeat, "initialization": args.initialization,
            "epochs": SCREEN_EPOCHS, "optimizer_updates": records[-1]["optimizer_updates"],
            "fit_epoch10": training_fit(records[-1]), "first_fit_epoch": min(fits) if fits else None,
            "fit_epochs": fits, "train_loss_epoch10": records[-1]["train_loss"],
            "train_acc_epoch10": records[-1]["train_acc"], "dev_epoch10": records[-1]["eval"],
            "train_subset_sha256": digest(rows), "source_checkpoint_sha256": source_sha256,
            "elapsed_seconds": time.monotonic() - started,
        })
    finally:
        sys.argv = old_argv
        if out.exists():
            write_json(out / "diagnostic_metadata.json", state)


if __name__ == "__main__":
    main()
