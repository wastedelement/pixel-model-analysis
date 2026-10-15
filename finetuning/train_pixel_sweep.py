"""
Fine-tune a PIXEL variant (pixel-base / pixel-m4 / pixel-bigrams / pixel-tiny-continuous)
on SST-2 / CoLA / MNLI. Records per-epoch train_loss, train_acc, val_loss, val_acc, val_mcc.

Renders text on the fly with PangoCairoTextRenderer.
"""
import os
import sys
import json
import argparse
import time
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from metrics import matthews_corrcoef
from PIL import Image

from pixel import (
    PangoCairoTextRenderer,
    PIXELForSequenceClassification,
    PoolingMode,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    resize_model_embeddings,
)
try:
    from pixel.data.rendering import PangoCairoBigramsTextRenderer
except ImportError:
    PangoCairoBigramsTextRenderer = None
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_train, get_evals, stratified_subsample, TASK_DATA_PLAN

# ---------------------------------------------------------------------------
# Per-variant configuration.
#   - pixel-base         : plain PangoCairo renderer, official recipe (Rust et al. 2023)
#   - pixel-m4           : bigrams renderer, m4 recipe (Kesen et al. 2025, lr=1e-5)
#   - pixel-bigrams      : bigrams renderer (Borenstein 2024, same family as m4)
#   - pixel-tiny-continuous: bigrams renderer (same Borenstein line)
# Reference: https://github.com/ilkerkesen/pixel-m4/.github/FINETUNING.md
# ---------------------------------------------------------------------------
MODEL_CFG = {
    "pixel-base": {
        "path":     "Team-PIXEL/pixel-base",
        "renderer": PangoCairoTextRenderer,
        "renderer_id": "PangoCairoTextRenderer",
        "lr":       3e-5,
    },
    "pixel-m4": {
        "path":     "checkpoints/pixel-m4",
        "renderer": PangoCairoBigramsTextRenderer,
        "renderer_id": "PangoCairoBigramsTextRenderer",
        "lr":       1e-5,
    },
    "pixel-bigrams": {
        "path":     "checkpoints/pixel-bigrams",
        "renderer": PangoCairoBigramsTextRenderer,
        "renderer_id": "PangoCairoBigramsTextRenderer",
        "lr":       1e-5,
    },
    "pixel-tiny-continuous": {
        # Borenstein 2024's "continuous" renderer is not available in our fork,
        # so we substitute plain PangoCairo (same as pixel-base). See docs/
        # REPORT_methodology.md for discussion.
        "path":     "checkpoints/pixel-tiny-continuous",
        "renderer": PangoCairoTextRenderer,
        "renderer_id": "PangoCairoTextRenderer",
        "lr":       3e-5,
    },
}

# Shared hyper-parameters per task. lr comes from MODEL_CFG (variant-specific).
# max_epochs is a hard cap; early_stopping_patience drives actual termination.
# Follows PIXEL/m4 official recipe (FINETUNING.md): eff_bs 64-128, warmup 100,
# dropout 0.1, load_best_model_at_end=True, patience-based early stop.
HP = {
    "sst2": {"max_epochs": 30, "bs": 32, "grad_accum": 4, "warmup_steps": 100,
             "dropout": 0.1, "patience": 8},
    "cola": {"max_epochs": 30, "bs": 32, "grad_accum": 4, "warmup_steps": 100,
             "dropout": 0.1, "patience": 8},
    # MNLI cap raised to 50 because PIXEL models on MNLI commonly stay on the
    # ln(num_classes) plateau for 10+ epochs before breaking out. patience=8
    # is the real stopper.
    "mnli": {"max_epochs": 50, "bs": 32, "grad_accum": 4, "warmup_steps": 100,
             "dropout": 0.1, "patience": 8},
}

# Historical task-specific defaults; controlled experiment recipes override these.
LR_OVERRIDES = {
    ("pixel-base",            "mnli"): 1e-5,
    # tiny-cont stuck at ln(3) with 3e-5 and 1e-5; last try with 5e-6.
    ("pixel-tiny-continuous", "mnli"): 5e-6,
}

MODEL_PATHS = {k: v["path"] for k, v in MODEL_CFG.items()}

# Per-task max_seq_length follows the PIXEL paper's GLUE fine-tuning settings.
# These are also perfect squares, so resize_model_embeddings stays in the safe
# interpolate path and avoids the off-by-one float bug for pixel-m4.
MAX_SEQ_LEN = {
    "sst2": 64,    # 8*8, PIXEL paper GLUE setting
    "cola": 64,    # 8*8, PIXEL paper GLUE setting
    "mnli": 256,   # 16*16, premise+hypothesis can be longer
}

RUNS_ROOT = os.environ.get("EXPERIMENT_RUNS_DIR", "outputs/finetuning")
SEED = 42


class PixelDataset(Dataset):
    """Renders text on-the-fly via PangoCairoTextRenderer."""

    def __init__(self, items, processor, transforms, max_seq_length, text_keys):
        self.items = items
        self.proc = processor
        self.tfms = transforms
        self.msl = max_seq_length
        self.k1, self.k2 = text_keys

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if self.k2 is None:
            enc = self.proc(text=glue_strip_spaces(it[self.k1]))
        else:
            enc = self.proc(text=(glue_strip_spaces(it[self.k1]),
                                  glue_strip_spaces(it[self.k2])))
        pv = self.tfms(Image.fromarray(enc.pixel_values))
        am = get_attention_mask(enc.num_text_patches, seq_length=self.msl)
        return {
            "pixel_values":   pv,
            "attention_mask": am,
            "labels":         torch.tensor(int(it["label"])),
        }


def evaluate(model, loader, device):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(**b)
            losses.append(out.loss.item())
            preds.append(out.logits.argmax(-1).cpu())
            labels.append(b["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = float((preds == labels).mean())
    mcc = float(matthews_corrcoef(labels, preds)) if len(set(labels)) > 1 else 0.0
    return {"loss": float(np.mean(losses)), "acc": acc, "mcc": mcc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sst2", "cola", "mnli"])
    ap.add_argument("--model", required=True, choices=list(MODEL_PATHS.keys()))
    ap.add_argument("--run-suffix", default="")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Optional exact optimizer-update budget; overrides the scheduler horizon derived from epochs.",
    )
    ap.add_argument(
        "--eval-every-updates",
        type=int,
        default=None,
        help="When max-updates is set, evaluate only after this many optimizer updates.",
    )
    ap.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Collect the full requested update trajectory even after validation plateaus.",
    )
    ap.add_argument(
        "--screen-epochs",
        type=int,
        default=None,
        help="Stop after this many epochs while retaining the max-epochs LR schedule.",
    )
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()
    if args.model_path:
        MODEL_CFG[args.model]["path"] = args.model_path

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task, model_key = args.task, args.model
    plan = TASK_DATA_PLAN[task]
    hp = dict(HP[task])  # copy
    vcfg = MODEL_CFG[model_key]
    hp["lr"] = LR_OVERRIDES.get((model_key, task), vcfg["lr"])
    if args.lr is not None:
        hp["lr"] = args.lr
    if args.max_epochs is not None:
        hp["max_epochs"] = args.max_epochs
    if args.screen_epochs is not None and not 1 <= args.screen_epochs <= hp["max_epochs"]:
        raise ValueError("screen-epochs must be between 1 and max-epochs")
    if args.patience is not None:
        hp["patience"] = args.patience
    if args.warmup_steps is not None:
        hp["warmup_steps"] = args.warmup_steps
    if args.batch_size is not None:
        hp["bs"] = args.batch_size
    if args.grad_accum is not None:
        hp["grad_accum"] = args.grad_accum
    RendererCls = vcfg["renderer"]
    if RendererCls is None:
        raise ImportError("This checkpoint requires the bigram renderer; it is not bundled in this release")
    if vcfg["renderer_id"] == "PangoCairoBigramsTextRenderer":
        assert "bigrams" in RendererCls.__module__, RendererCls.__module__
    else:
        assert "bigrams" not in RendererCls.__module__, RendererCls.__module__
    run_name = f"{task}_{model_key}{args.run_suffix}"
    out_dir = os.path.join(RUNS_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=False)

    print(f"[INFO] === {task} / {model_key} ===")
    print(f"[INFO] out_dir={out_dir}  hparams={hp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = vcfg["path"]
    renderer_fqcn = f"{RendererCls.__module__}.{RendererCls.__name__}"
    print(
        f"[INFO] renderer = {vcfg['renderer_id']} ({renderer_fqcn})  lr = {hp['lr']}"
    )

    # ---- data ----
    train_items = get_train(task)
    if args.train_size is not None:
        if args.train_size > len(train_items):
            raise ValueError(
                f"train-size={args.train_size} exceeds the fixed base subset of {len(train_items)}"
            )
        train_items = stratified_subsample(train_items, args.train_size, "label", seed=42)
    eval_dict   = get_evals(task)
    primary_eval = list(eval_dict.keys())[0]
    print(f"[INFO] train={len(train_items)}, evals=" +
          ", ".join(f"{k}={len(v)}" for k, v in eval_dict.items()))

    # ---- renderer + transforms ----
    processor = RendererCls.from_pretrained(model_path, rgb=True)
    max_seq_length = args.max_seq_length if args.max_seq_length is not None else MAX_SEQ_LEN[task]
    processor.max_seq_length = max_seq_length        # follow run_glue.py pattern
    transforms = get_transforms(
        do_resize=True,
        size=(processor.pixels_per_patch, processor.pixels_per_patch * max_seq_length),
    )
    print(f"[INFO] renderer max_seq_length forced to {max_seq_length} "
          f"(pixels_per_patch={processor.pixels_per_patch})")

    # ---- model ----
    # Apply dropout from HP (official uses 0.1, default config has 0.0)
    model = PIXELForSequenceClassification.from_pretrained(
        model_path,
        num_labels=plan["num_labels"],
        pooling_mode=PoolingMode.MEAN,
        add_layer_norm=True,
        hidden_dropout_prob=hp["dropout"],
        attention_probs_dropout_prob=hp["dropout"],
    ).to(device)
    resize_model_embeddings(model, max_seq_length)
    # Defensive: hard-set image_size everywhere ViTPatchEmbeddings might check it.
    correct_size = [processor.pixels_per_patch,
                    processor.pixels_per_patch * max_seq_length]
    model.config.image_size = correct_size
    model.image_size = correct_size
    pe = model.vit.embeddings.patch_embeddings
    pe.image_size = correct_size
    if hasattr(pe, "num_patches"):
        pe.num_patches = max_seq_length
    print(f"[INFO] forced image_size = {correct_size}, num_patches={max_seq_length}")
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({"task": task, "model": model_key, "seed": args.seed,
                   "train_n": len(train_items),
                   "hparams": hp, "max_seq_length": max_seq_length,
                   "max_updates": args.max_updates,
                   "screen_epochs": args.screen_epochs,
                   "eval_every_updates": args.eval_every_updates,
                   "disable_early_stopping": args.disable_early_stopping,
                   "renderer": vcfg["renderer_id"],
                   "renderer_class_fqcn": renderer_fqcn,
                   "model_path": model_path,
                   "run_suffix": args.run_suffix}, f, indent=2)

    # ---- loaders ----
    train_ds = PixelDataset(train_items, processor, transforms, max_seq_length, plan["text_keys"])
    train_loader = DataLoader(train_ds, batch_size=hp["bs"], shuffle=True, num_workers=0)
    eval_loaders = {
        name: DataLoader(
            PixelDataset(its, processor, transforms, max_seq_length, plan["text_keys"]),
            batch_size=hp["bs"] * 2, shuffle=False, num_workers=0)
        for name, its in eval_dict.items()
    }

    # Official recipe: single optimizer group, fixed 100 warmup steps.
    grad_accum = hp["grad_accum"]
    updates_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_updates = (
        args.max_updates
        if args.max_updates is not None
        else updates_per_epoch * hp["max_epochs"]
    )
    if total_updates <= 0:
        raise ValueError("The optimizer-update budget must be positive")
    if args.eval_every_updates is not None and args.eval_every_updates <= 0:
        raise ValueError("--eval-every-updates must be positive")
    epoch_limit = hp["max_epochs"]
    if args.max_updates is not None:
        epoch_limit = max(epoch_limit, math.ceil(args.max_updates / updates_per_epoch))
    if args.screen_epochs is not None:
        epoch_limit = min(epoch_limit, args.screen_epochs)
    warmup_steps = hp["warmup_steps"]
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[INFO] trainable param tensors={n_params}, "
          f"eff_bs={hp['bs']*grad_accum}, "
          f"updates/epoch={updates_per_epoch}, total_updates={total_updates}")
    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=hp["lr"], weight_decay=0.01,
    )
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_updates)

    log_path = os.path.join(out_dir, "training_log.jsonl")
    log_f = open(log_path, "w")
    best_metric = -1.0; best_epoch = -1
    metric_key = "mcc" if task == "cola" else "acc"
    patience = hp["patience"]
    no_improve = 0

    global_updates = 0
    examples_seen = 0
    eval_every = args.eval_every_updates or updates_per_epoch
    next_eval_update = eval_every

    for epoch in range(1, epoch_limit + 1):
        model.train()
        t0 = time.time()
        sum_loss, n_correct, n_total = 0.0, 0, 0
        optim.zero_grad()
        n_batches = len(train_loader)
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            examples_seen += batch["labels"].size(0)
            out = model(**batch)
            # Use a smaller denominator for the final partial accumulation
            # window, so no examples are silently discarded or underweighted.
            window_start = (i // grad_accum) * grad_accum
            window_size = min(grad_accum, n_batches - window_start)
            loss = out.loss / window_size
            loss.backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad()
                global_updates += 1
            sum_loss += out.loss.item() * batch["labels"].size(0)
            preds = out.logits.argmax(-1)
            n_correct += (preds == batch["labels"]).sum().item()
            n_total += batch["labels"].size(0)
            if args.max_updates is not None and global_updates >= args.max_updates:
                break

        train_loss = sum_loss / n_total
        train_acc = n_correct / n_total
        ep_time = time.time() - t0

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

        eval_metrics = {}
        for name, loader in eval_loaders.items():
            r = evaluate(model, loader, device)
            eval_metrics[name] = {"loss": r["loss"], "acc": r["acc"], "mcc": r["mcc"]}

        primary = eval_metrics[primary_eval][metric_key]

        rec = {
            "epoch": epoch,
            "optimizer_updates": global_updates,
            "examples_seen": examples_seen,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "epoch_time_sec": round(ep_time, 1),
            "eval": eval_metrics,
            "primary_metric": metric_key,
            "primary_value": primary,
        }
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()

        ev_str = " ".join(f"{n}_acc={m['acc']:.4f}" for n, m in eval_metrics.items())
        print(f"[E{epoch}/{epoch_limit} U{global_updates}/{total_updates}] {ep_time:5.1f}s "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  {ev_str}  primary({metric_key})={primary:.4f}",
              flush=True)

        if primary > best_metric:
            best_metric = primary; best_epoch = epoch
            no_improve = 0
            ckpt_dir = os.path.join(out_dir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            with open(os.path.join(ckpt_dir, "best_info.json"), "w") as f:
                json.dump({"best_epoch": epoch, "metric": metric_key,
                           "best_optimizer_update": global_updates,
                           "examples_seen": examples_seen,
                           "value": primary, "max_seq_length": max_seq_length,
                           "num_labels": plan["num_labels"], "seed": args.seed,
                           "hparams": hp, "renderer": vcfg["renderer_id"],
                           "renderer_class_fqcn": renderer_fqcn}, f, indent=2)
            print(f"           --> saved new best ({metric_key}={primary:.4f})", flush=True)
        else:
            no_improve += 1
            print(f"           -- no improvement for {no_improve}/{patience} epoch(s)", flush=True)
            if no_improve >= patience and not args.disable_early_stopping:
                print(f"[EARLY STOP] no val improvement in {patience} epochs. Best {metric_key}={best_metric:.4f} @ E{best_epoch}.",
                      flush=True)
                break

        if reached_update_cap:
            print(f"[UPDATE CAP] reached {global_updates} optimizer updates", flush=True)
            break

    log_f.close()
    print(f"[DONE] best epoch={best_epoch}  best {metric_key}={best_metric:.4f}")


if __name__ == "__main__":
    main()
