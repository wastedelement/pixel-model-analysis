import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import macro_f1, matthews_corrcoef
from xling_data_utils import TASK_DATA_PLAN, get_final_evals, get_train, get_train_evals, LANGS_8

MODEL_CFG = {
    "pixel-base": {
        "path": "Team-PIXEL/pixel-base",
        "renderer": PangoCairoTextRenderer,
        "lr": 3e-5,
    },
    "pixel-m4": {
        "path": "checkpoints/pixel-m4",
        "renderer": PangoCairoBigramsTextRenderer,
        "lr": 1e-5,
    },
    "pixel-bigrams": {
        "path": "checkpoints/pixel-bigrams",
        "renderer": PangoCairoBigramsTextRenderer,
        "lr": 1e-5,
    },
    "pixel-tiny-continuous": {
        "path": "checkpoints/pixel-tiny-continuous",
        "renderer": PangoCairoTextRenderer,
        "lr": 3e-5,
    },
}

HP = {
    "xnli": {"max_epochs": 40, "bs": 32, "grad_accum": 4, "warmup_steps": 100, "dropout": 0.1, "patience": 8},
    "sib200": {"max_epochs": 150, "bs": 16, "grad_accum": 2, "warmup_steps": 100, "dropout": 0.1, "patience": 20},
    "taxi1500": {"max_epochs": 40, "bs": 16, "grad_accum": 2, "warmup_steps": 100, "dropout": 0.1, "patience": 10},
    "pawsx": {"max_epochs": 20, "bs": 32, "grad_accum": 4, "warmup_steps": 100, "dropout": 0.1, "patience": 8},
}

LR_OVERRIDES = {
    ("pixel-base", "xnli"): 1e-5,
    ("pixel-base", "sib200"): 1e-5,
    ("pixel-tiny-continuous", "xnli"): 1e-5,
    ("pixel-tiny-continuous", "sib200"): 1e-5,
    ("pixel-base", "pawsx"): 1e-5,
}

MAX_SEQ_LEN = {
    "xnli": 256,
    "sib200": 256,
    "taxi1500": 196,
    "pawsx": 256,
}

RUNS_ROOT = os.environ.get("TRANSFER_RUNS_DIR", "outputs/transfer")
SEED = 42


class PixelDataset(Dataset):
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
            enc = self.proc(text=glue_strip_spaces(str(it[self.k1])))
        else:
            enc = self.proc(text=(glue_strip_spaces(str(it[self.k1])),
                                  glue_strip_spaces(str(it[self.k2]))))
        return {
            "pixel_values": self.tfms(Image.fromarray(enc.pixel_values)),
            "attention_mask": get_attention_mask(enc.num_text_patches, seq_length=self.msl),
            "labels": torch.tensor(int(it["label"])),
        }


def force_pixel_size(model, pixels_per_patch, max_seq_length):
    resize_model_embeddings(model, max_seq_length)
    correct_size = [pixels_per_patch, pixels_per_patch * max_seq_length]
    model.config.image_size = correct_size
    model.image_size = correct_size
    pe = model.vit.embeddings.patch_embeddings
    pe.image_size = correct_size
    if hasattr(pe, "num_patches"):
        pe.num_patches = max_seq_length


def evaluate(model, loader, device):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(out.loss.item())
            preds.append(out.logits.argmax(-1).cpu())
            labels.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = float((preds == labels).mean())
    f1 = float(macro_f1(labels, preds))
    mcc = float(matthews_corrcoef(labels, preds)) if len(set(labels)) > 1 else 0.0
    return {"loss": float(np.mean(losses)), "acc": acc, "f1": f1, "mcc": mcc, "n": int(len(labels))}


def make_loaders(eval_dict, processor, transforms, max_seq_length, text_keys, batch_size):
    return {
        name: DataLoader(
            PixelDataset(items, processor, transforms, max_seq_length, text_keys),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        for name, items in eval_dict.items()
    }


def run_final_eval(out_dir, task, model_key, device, text_keys, fallback_fonts_dir=None, results_name="results_xling.json"):
    ckpt_dir = os.path.join(out_dir, "ckpt")
    info = json.load(open(os.path.join(ckpt_dir, "best_info.json")))
    max_seq_length = int(info["max_seq_length"])
    if fallback_fonts_dir is None:
        fallback_fonts_dir = info.get("fallback_fonts_dir")
    renderer_cls = MODEL_CFG[model_key]["renderer"]
    if renderer_cls is None:
        raise ImportError("The bigram renderer required by this checkpoint is not bundled")
    processor = renderer_cls.from_pretrained(ckpt_dir, rgb=True, fallback_fonts_dir=fallback_fonts_dir)
    processor.max_seq_length = max_seq_length
    transforms = get_transforms(
        do_resize=True,
        size=(processor.pixels_per_patch, processor.pixels_per_patch * max_seq_length),
    )
    model = PIXELForSequenceClassification.from_pretrained(
        ckpt_dir,
        num_labels=TASK_DATA_PLAN[task]["num_labels"],
        pooling_mode=PoolingMode.MEAN,
        add_layer_norm=True,
    ).to(device)
    force_pixel_size(model, processor.pixels_per_patch, max_seq_length)
    eval_dict = get_final_evals(task)
    loaders = make_loaders(eval_dict, processor, transforms, max_seq_length, text_keys, batch_size=16)
    results = {}
    for name, loader in loaders.items():
        metrics = evaluate(model, loader, device)
        results[name] = metrics
        print(f"[FINAL] {name}: acc={metrics['acc']:.4f} f1={metrics['f1']:.4f} mcc={metrics['mcc']:.4f} n={metrics['n']}", flush=True)
    with open(os.path.join(out_dir, results_name), "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_DATA_PLAN.keys()))
    ap.add_argument("--model", required=True, choices=list(MODEL_CFG.keys()))
    ap.add_argument("--run-suffix", default="")
    ap.add_argument("--results-name", default="results_xling.json")
    ap.add_argument("--fallback-fonts-dir", default=None)
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--max-seq-length", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    task = args.task
    model_key = args.model
    plan = TASK_DATA_PLAN[task]
    hp = dict(HP[task])
    vcfg = MODEL_CFG[model_key]
    hp["lr"] = LR_OVERRIDES.get((model_key, task), vcfg["lr"])
    if args.max_epochs is not None:
        hp["max_epochs"] = args.max_epochs
    if args.patience is not None:
        hp["patience"] = args.patience
    if args.learning_rate is not None:
        hp["lr"] = args.learning_rate
    if args.warmup_steps is not None:
        hp["warmup_steps"] = args.warmup_steps
    if vcfg["renderer"] is None:
        raise ImportError("The bigram renderer required by this checkpoint is not bundled")
    if args.max_seq_length is not None:
        MAX_SEQ_LEN[task] = args.max_seq_length
    init_path = args.init_checkpoint or vcfg["path"]
    out_dir = os.path.join(RUNS_ROOT, f"{task}_{model_key}{args.run_suffix}")
    if not args.eval_only and os.path.exists(out_dir):
        raise FileExistsError("Refusing to overwrite an existing training run")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] === {task} / {model_key} ===")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] hparams={hp}")
    print(f"[INFO] renderer class = {vcfg['renderer'].__name__}")
    print(f"[INFO] initialization checkpoint = {init_path}")
    print(f"[INFO] languages={list(LANGS_8.keys())}")
    print(f"[INFO] fallback_fonts_dir={args.fallback_fonts_dir}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this experiment. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
    device = torch.device("cuda")
    print(f"[INFO] device={device} gpu={torch.cuda.get_device_name(0)}", flush=True)
    text_keys = plan["text_keys"]
    metric_key = plan["metric"]
    primary_eval = plan["primary_eval"]

    if args.eval_only:
        run_final_eval(out_dir, task, model_key, device, text_keys, args.fallback_fonts_dir, args.results_name)
        return

    train_items = get_train(task)
    train_eval_dict = get_train_evals(task)
    print(f"[INFO] train={len(train_items)}, train_evals=" +
          ", ".join(f"{k}={len(v)}" for k, v in train_eval_dict.items()))

    processor = vcfg["renderer"].from_pretrained(init_path, rgb=True, fallback_fonts_dir=args.fallback_fonts_dir)
    max_seq_length = MAX_SEQ_LEN[task]
    processor.max_seq_length = max_seq_length
    transforms = get_transforms(
        do_resize=True,
        size=(processor.pixels_per_patch, processor.pixels_per_patch * max_seq_length),
    )
    print(f"[INFO] renderer max_seq_length forced to {max_seq_length} "
          f"(pixels_per_patch={processor.pixels_per_patch})")

    model = PIXELForSequenceClassification.from_pretrained(
        init_path,
        num_labels=plan["num_labels"],
        pooling_mode=PoolingMode.MEAN,
        add_layer_norm=True,
        hidden_dropout_prob=hp["dropout"],
        attention_probs_dropout_prob=hp["dropout"],
    ).to(device)
    force_pixel_size(model, processor.pixels_per_patch, max_seq_length)

    train_loader = DataLoader(
        PixelDataset(train_items, processor, transforms, max_seq_length, text_keys),
        batch_size=hp["bs"], shuffle=True, num_workers=0)
    train_eval_loaders = make_loaders(
        train_eval_dict, processor, transforms, max_seq_length, text_keys, batch_size=hp["bs"] * 2)

    grad_accum = hp["grad_accum"]
    updates_per_epoch = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_updates = updates_per_epoch * hp["max_epochs"]
    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=hp["lr"], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, hp["warmup_steps"], total_updates)

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({"task": task, "model": model_key, "hparams": hp, "plan": plan,
                   "languages": LANGS_8, "model_path": vcfg["path"],
                   "initialization_checkpoint": init_path,
                   "renderer": vcfg["renderer"].__name__, "max_seq_length": max_seq_length,
                   "fallback_fonts_dir": args.fallback_fonts_dir}, f, indent=2)

    log_f = open(os.path.join(out_dir, "training_log.jsonl"), "w")
    best_metric, best_epoch, no_improve = -1.0, -1, 0

    for epoch in range(1, hp["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        sum_loss, n_correct, n_total = 0.0, 0, 0
        optim.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad()
            sum_loss += out.loss.item() * batch["labels"].size(0)
            preds = out.logits.argmax(-1)
            n_correct += (preds == batch["labels"]).sum().item()
            n_total += batch["labels"].size(0)

        train_loss = sum_loss / n_total
        train_acc = n_correct / n_total
        ep_time = time.time() - t0
        eval_metrics = {name: evaluate(model, loader, device) for name, loader in train_eval_loaders.items()}
        primary = eval_metrics[primary_eval][metric_key]
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "epoch_time_sec": round(ep_time, 1),
            "eval": eval_metrics,
            "primary_metric": metric_key,
            "primary_value": primary,
        }
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        ev_str = " ".join(f"{n}_acc={m['acc']:.4f}/{n}_f1={m['f1']:.4f}" for n, m in eval_metrics.items())
        print(f"[E{epoch}/{hp['max_epochs']}] {ep_time:5.1f}s train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} {ev_str} primary({metric_key})={primary:.4f}", flush=True)

        if primary > best_metric:
            best_metric, best_epoch, no_improve = primary, epoch, 0
            ckpt_dir = os.path.join(out_dir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            with open(os.path.join(ckpt_dir, "best_info.json"), "w") as f:
                json.dump({"best_epoch": epoch, "metric": metric_key, "value": primary,
                           "max_seq_length": max_seq_length, "num_labels": plan["num_labels"],
                           "label_names": plan["label_names"], "renderer": vcfg["renderer"].__name__,
                           "fallback_fonts_dir": args.fallback_fonts_dir}, f, indent=2)
            print(f"           --> saved new best ({metric_key}={primary:.4f})", flush=True)
        else:
            no_improve += 1
            print(f"           -- no improvement for {no_improve}/{hp['patience']} epoch(s)", flush=True)
            if no_improve >= hp["patience"]:
                print(f"[EARLY STOP] best {metric_key}={best_metric:.4f} @ E{best_epoch}", flush=True)
                break

    log_f.close()
    run_final_eval(out_dir, task, model_key, device, text_keys, args.fallback_fonts_dir, args.results_name)
    print(f"[DONE] best epoch={best_epoch} best {metric_key}={best_metric:.4f}")


if __name__ == "__main__":
    main()
