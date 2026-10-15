"""Fine-tune a subword classifier with an explicit, recorded random seed.

This is the multi-seed counterpart of train_subword.py.  The training subset
and evaluation subset remain fixed by data_utils; only initialization,
dropout, and data-loader order vary with --seed.
"""

import argparse
import json
import math
import os
import random
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
    parser.add_argument("--max-epochs", type=int, default=10)
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    plan = TASK_DATA_PLAN[args.task]
    model_path = MODEL_PATHS[args.model]
    run_name = f"{args.task}_{args.model}{args.run_suffix}"
    output_dir = os.path.join(RUNS_ROOT, run_name)
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
    with open(os.path.join(output_dir, "run_config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    eval_items = get_evals(args.task)
    primary_split = next(iter(eval_items))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=plan["num_labels"]
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the multi-seed training experiment")
    device = torch.device("cuda")
    model.to(device)

    train_loader = DataLoader(
        TextDataset(train_items, tokenizer, args.max_seq_length, plan["text_keys"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    eval_loaders = {
        name: DataLoader(
            TextDataset(items, tokenizer, args.max_seq_length, plan["text_keys"]),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=2,
        )
        for name, items in eval_items.items()
    }

    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = (
        args.max_updates
        if args.max_updates is not None
        else updates_per_epoch * args.max_epochs
    )
    if total_steps <= 0:
        raise ValueError("The optimizer-update budget must be positive")
    if args.eval_every_updates is not None and args.eval_every_updates <= 0:
        raise ValueError("--eval-every-updates must be positive")
    epoch_limit = args.max_epochs
    if args.max_updates is not None:
        epoch_limit = max(epoch_limit, math.ceil(args.max_updates / updates_per_epoch))
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

    print(f"[DONE] best {metric_name}={best_metric:.4f} @ E{best_epoch}", flush=True)


if __name__ == "__main__":
    main()
