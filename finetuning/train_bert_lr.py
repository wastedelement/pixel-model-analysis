"""BERT control for the final PIXEL MNLI early-LR screen (2026-09-29).

Derived from unchanged train_subword_sweep.py (SHA 920c3391...). Observe ten
full epochs while retaining the 80-epoch LR horizon. Check fixed dataset
hashes against the existing PIXEL holdout manifest; evaluate the saved
best-selection checkpoint on that same held-out dev complement after training.
Never select, stop, or tune on holdout; do not overwrite previous outputs.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from data_utils import TASK_DATA_PLAN, get_evals, get_train, stratified_subsample
from metrics import matthews_corrcoef


MODEL_PATHS = {
    "bert": "bert-base-uncased",
}
RUNS_ROOT = os.environ.get("EXPERIMENT_RUNS_DIR", "outputs/finetuning")
HOLDOUT_MANIFEST = Path(os.environ.get("MNLI_HOLDOUT_MANIFEST", "outputs/mnli_holdout/manifest.json"))
EXPECTED_DATA_SHA = "4fd501addb8cfb9c1f5a0439dd2473c11180ec3a42c579e704ad8294f02b2223"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def screen_budget(n_batches, grad_accum, max_epochs, screen_epochs):
    if not 1 <= screen_epochs <= max_epochs or grad_accum <= 0 or n_batches <= 0:
        raise ValueError("Invalid screen/schedule budget")
    per_epoch = math.ceil(n_batches / grad_accum)
    return per_epoch, per_epoch * max_epochs, screen_epochs


class TextDataset(Dataset):
    def __init__(self, items, tokenizer, max_length, text_keys):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.key1, self.key2 = text_keys

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        texts = (item[self.key1],) if self.key2 is None else (item[self.key1], item[self.key2])
        encoded = self.tokenizer(
            *texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(item["label"])),
        }
        if "token_type_ids" in encoded:
            result["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
        return result


def evaluate(model, loader, device):
    model.eval()
    losses, predictions, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            losses.append(float(output.loss.item()))
            predictions.append(output.logits.argmax(dim=-1).cpu())
            labels.append(batch["labels"].cpu())
    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()
    accuracy = float((predictions == labels).mean())
    mcc = float(matthews_corrcoef(labels, predictions)) if len(set(labels)) > 1 else 0.0
    return {"loss": float(np.mean(losses)), "acc": accuracy, "mcc": mcc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_DATA_PLAN))
    parser.add_argument("--model", default="bert", choices=list(MODEL_PATHS))
    parser.add_argument("--run-suffix", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--screen-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Optional exact optimizer-update budget; overrides the scheduler horizon derived from epochs.",
    )
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=None,
        help="When max-updates is set, evaluate only after this many optimizer updates.",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Collect the full requested update trajectory even after validation plateaus.",
    )
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    if args.model_path:
        MODEL_PATHS[args.model] = args.model_path
    if args.task != "mnli" or args.max_updates is not None or args.eval_every_updates is not None:
        raise ValueError("This control uses MNLI with a full-epoch observation cutoff only")
    if not args.disable_early_stopping:
        raise ValueError("Early stopping must be disabled for this controlled screen")
    if not 1 <= args.screen_epochs <= args.max_epochs:
        raise ValueError("Invalid screen-epochs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    if file_sha(Path(__file__).with_name("data_utils.py")) != EXPECTED_DATA_SHA:
        raise RuntimeError("data_utils.py changed; refusing unmatched comparison")
    manifest = json.loads(HOLDOUT_MANIFEST.read_text())

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    plan = TASK_DATA_PLAN[args.task]
    model_path = MODEL_PATHS[args.model]
    run_name = f"{args.task}_{args.model}{args.run_suffix}"
    output_dir = os.path.join(RUNS_ROOT, run_name)
    if Path(output_dir).exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    os.makedirs(output_dir, exist_ok=False)

    config = {
        "task": args.task,
        "model": args.model,
        "seed": args.seed,
        "model_path": model_path,
        "train_n": args.train_size if args.train_size is not None else plan["train_n"],
        "max_seq_length": args.max_seq_length,
        "hparams": {
            "lr": args.lr,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": args.warmup_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
        },
        "screen_epochs": args.screen_epochs,
        "hardware": {"gpu": torch.cuda.get_device_name(0),
                     "torch": torch.__version__, "cuda": torch.version.cuda,
                     "python": sys.version,
                     "default_dtype": str(torch.get_default_dtype()),
                     "cuda_determinism_enforced": False},
        "source_sha256": file_sha(__file__),
        "data_utils_sha256": EXPECTED_DATA_SHA,
        "holdout_manifest_sha256": file_sha(HOLDOUT_MANIFEST),
        "optimizer": {"name": "AdamW", "weight_decay": 0.01, "betas": [0.9, 0.999],
                      "eps": 1e-8, "clip_norm": 1.0},
        "max_updates": args.max_updates,
        "eval_every_updates": args.eval_every_updates,
        "disable_early_stopping": args.disable_early_stopping,
        "run_suffix": args.run_suffix,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    print(f"[INFO] {config}", flush=True)

    train_items = get_train(args.task)
    if args.train_size is not None:
        if args.train_size > len(train_items):
            raise ValueError(
                f"train-size={args.train_size} exceeds the fixed base subset of {len(train_items)}"
            )
        train_items = stratified_subsample(train_items, args.train_size, "label", seed=42)
    config["train_n"] = len(train_items)
    assert digest(train_items) == manifest["train_subset_sha256"], "Training subset mismatch"
    config["train_subset_sha256"] = digest(train_items)
    with open(os.path.join(output_dir, "run_config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    eval_items = get_evals(args.task)
    primary_split = next(iter(eval_items))
    assert primary_split == "dev_matched"
    config["selection_sha256"] = {}
    for name, items in eval_items.items():
        assert len(items) == 2000
        assert digest(items) == manifest["audit"][name]["selection_sha256"], name
        config["selection_sha256"][name] = digest(items)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=plan["num_labels"], output_loading_info=True,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    )
    assert model.config.hidden_dropout_prob == 0.1
    assert model.config.attention_probs_dropout_prob == 0.1
    config["loading_info"] = loading
    config["dropout"] = 0.1
    config["num_workers"] = 0
    config["initial_new_parameter_hashes"] = {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in model.state_dict().items() if name in set(loading["missing_keys"])
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the multi-seed training experiment")
    device = torch.device("cuda")
    model.to(device)

    train_loader = DataLoader(
        TextDataset(train_items, tokenizer, args.max_seq_length, plan["text_keys"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    eval_loaders = {
        name: DataLoader(
            TextDataset(items, tokenizer, args.max_seq_length, plan["text_keys"]),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
        )
        for name, items in eval_items.items()
    }

    updates_per_epoch, total_steps, epoch_limit = screen_budget(
        len(train_loader), args.grad_accum, args.max_epochs, args.screen_epochs)
    config["updates_per_epoch"] = updates_per_epoch
    config["scheduler_horizon_updates"] = total_steps
    config["observation_updates"] = updates_per_epoch * epoch_limit
    config["effective_batch"] = args.batch_size * args.grad_accum
    with open(os.path.join(output_dir, "run_config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    print(f"[BUDGET] observed_epochs={epoch_limit} schedule_epochs={args.max_epochs} "
          f"updates_per_epoch={updates_per_epoch} schedule_updates={total_steps}", flush=True)
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else int(total_steps * args.warmup_ratio)
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    metric_name = "mcc" if args.task == "cola" else "acc"
    best_metric, best_epoch, no_improvement = -1.0, -1, 0

    log_path = os.path.join(output_dir, "training_log.jsonl")
    with open(log_path, "w") as log_handle:
        global_updates = 0
        examples_seen = 0
        eval_every = args.eval_every_updates or updates_per_epoch
        next_eval_update = eval_every

        for epoch in range(1, epoch_limit + 1):
            model.train()
            started = time.time()
            train_loss, train_correct, train_total = 0.0, 0, 0
            optimizer.zero_grad()
            n_batches = len(train_loader)
            for batch_index, batch in enumerate(train_loader):
                batch = {key: value.to(device) for key, value in batch.items()}
                examples_seen += batch["labels"].size(0)
                output = model(**batch)
                window_start = (batch_index // args.grad_accum) * args.grad_accum
                window_size = min(args.grad_accum, n_batches - window_start)
                (output.loss / window_size).backward()
                if ((batch_index + 1) % args.grad_accum == 0
                        or (batch_index + 1) == n_batches):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_updates += 1
                train_loss += float(output.loss.item()) * batch["labels"].size(0)
                train_correct += int((output.logits.argmax(-1) == batch["labels"]).sum().item())
                train_total += batch["labels"].size(0)
                if args.max_updates is not None and global_updates >= args.max_updates:
                    break

            reached_update_cap = (
                args.max_updates is not None and global_updates >= args.max_updates
            )
            should_evaluate = (
                args.max_updates is None
                or global_updates >= next_eval_update
                or reached_update_cap
            )
            if not should_evaluate:
                continue

            while next_eval_update <= global_updates:
                next_eval_update += eval_every

            eval_metrics = {
                name: evaluate(model, loader, device) for name, loader in eval_loaders.items()
            }
            primary = eval_metrics[primary_split][metric_name]
            record = {
                "epoch": epoch,
                "optimizer_updates": global_updates,
                "examples_seen": examples_seen,
                "train_loss": train_loss / train_total,
                "train_acc": train_correct / train_total,
                "epoch_time_sec": round(time.time() - started, 1),
                "eval": eval_metrics,
                "primary_metric": metric_name,
                "primary_value": primary,
            }
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(
                f"[E{epoch}/{epoch_limit} U{global_updates}/{total_steps}] primary({metric_name})={primary:.4f} "
                f"train_acc={record['train_acc']:.4f}",
                flush=True,
            )

            if primary > best_metric:
                best_metric, best_epoch, no_improvement = primary, epoch, 0
                checkpoint_dir = os.path.join(output_dir, "ckpt")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                with open(os.path.join(checkpoint_dir, "best_info.json"), "w") as handle:
                    json.dump(
                        {
                            "best_epoch": epoch,
                            "best_optimizer_update": global_updates,
                            "examples_seen": examples_seen,
                            "metric": metric_name,
                            "value": primary,
                            "seed": args.seed,
                            "hparams": config["hparams"],
                        },
                        handle,
                        indent=2,
                    )
            else:
                no_improvement += 1
                if no_improvement >= args.patience and not args.disable_early_stopping:
                    print(
                        f"[EARLY STOP] best {metric_name}={best_metric:.4f} @ E{best_epoch}",
                        flush=True,
                    )
                    break

            if reached_update_cap:
                print(f"[UPDATE CAP] reached {global_updates} optimizer updates", flush=True)
                break

    print(f"[TRAIN DONE] best {metric_name}={best_metric:.4f} @ E{best_epoch}", flush=True)
    rows = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
    assert len(rows) == args.screen_epochs
    assert rows[-1]["optimizer_updates"] == updates_per_epoch * args.screen_epochs
    assert rows[-1]["examples_seen"] == len(train_items) * args.screen_epochs
    fit = lambda row: row["train_loss"] < 1.0 and row["train_acc"] > 0.5
    first_fit = next((row["epoch"] for row in rows if fit(row)), None)
    fit_summary = {"endpoint_fit": fit(rows[-1]), "first_fit_epoch": first_fit,
                   "crossed_then_endpoint_failed": first_fit is not None and not fit(rows[-1]),
                   "epoch10": rows[-1], "best_epoch": best_epoch, "best_selection_dev": best_metric}
    Path(output_dir, "fit_summary.json").write_text(json.dumps(fit_summary, indent=2) + "\n")

    # Training and selection are over. Only the existing best checkpoint is evaluated.
    del optimizer, scheduler, model
    torch.cuda.empty_cache()
    selected_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(output_dir, "ckpt")).to(device)
    holdout_metrics = {}
    for split, items in manifest["holdout"].items():
        assert len(items) == 2000
        assert digest(items) == manifest["audit"][split]["holdout_sha256"]
        loader = DataLoader(TextDataset(items, tokenizer, args.max_seq_length, plan["text_keys"]),
                            batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
        holdout_metrics[split] = evaluate(selected_model, loader, device)
    holdout = {"purpose": "same held-out development complement as PIXEL, not official test",
               "selection_rule": "best original matched-selection-dev checkpoint",
               "best_epoch": best_epoch, "seed": args.seed, "lr": args.lr,
               "manifest_sha256": file_sha(HOLDOUT_MANIFEST),
               "holdout_sha256": {k: manifest["audit"][k]["holdout_sha256"] for k in holdout_metrics},
               "eval": holdout_metrics}
    Path(output_dir, "holdout_result.json").write_text(json.dumps(holdout, indent=2) + "\n")
    print("[HOLDOUT]", json.dumps(holdout_metrics), flush=True)
    print(f"[DONE] seed={args.seed} lr={args.lr} fit={fit_summary['endpoint_fit']}", flush=True)


if __name__ == "__main__":
    main()
