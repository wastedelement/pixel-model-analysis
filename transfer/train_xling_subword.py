import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import macro_f1, matthews_corrcoef
from xling_data_utils import TASK_DATA_PLAN, get_final_evals, get_train, get_train_evals, LANGS_8

MODEL_PATHS = {
    "bert": "bert-base-uncased",
    "mbert": "bert-base-multilingual-cased",
    "roberta": "roberta-base",
    "xlmr": "xlm-roberta-base",
}

HP = {
    "xnli": {"max_epochs": 10, "max_len": 256, "lr": 2e-5, "bs": 32, "warmup": 0.1, "patience": 3},
    "sib200": {"max_epochs": 50, "max_len": 128, "lr": 2e-5, "bs": 16, "warmup": 0.1, "patience": 10},
    "taxi1500": {"max_epochs": 20, "max_len": 128, "lr": 2e-5, "bs": 32, "warmup": 0.1, "patience": 5},
    "pawsx": {"max_epochs": 10, "max_len": 128, "lr": 2e-5, "bs": 32, "warmup": 0.1, "patience": 3},
}

RUNS_ROOT = os.environ.get("TRANSFER_RUNS_DIR", "outputs/transfer")
SEED = 42


class TextDataset(Dataset):
    def __init__(self, items, tokenizer, max_len, text_keys):
        self.items = items
        self.tok = tokenizer
        self.max_len = max_len
        self.k1, self.k2 = text_keys

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if self.k2 is None:
            enc = self.tok(str(it[self.k1]), truncation=True, padding="max_length",
                           max_length=self.max_len, return_tensors="pt")
        else:
            enc = self.tok(str(it[self.k1]), str(it[self.k2]), truncation=True,
                           padding="max_length", max_length=self.max_len,
                           return_tensors="pt")
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(it["label"])),
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return out


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


def make_loaders(eval_dict, tokenizer, hp, text_keys, bs_mult=2):
    return {
        name: DataLoader(
            TextDataset(items, tokenizer, hp["max_len"], text_keys),
            batch_size=hp["bs"] * bs_mult,
            shuffle=False,
            num_workers=2,
        )
        for name, items in eval_dict.items()
    }


def run_final_eval(out_dir, task, tokenizer, model, device, hp, text_keys, results_name="results_xling.json"):
    eval_dict = get_final_evals(task)
    loaders = make_loaders(eval_dict, tokenizer, hp, text_keys, bs_mult=2)
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
    ap.add_argument("--model", required=True, choices=list(MODEL_PATHS.keys()))
    ap.add_argument("--run-suffix", default="")
    ap.add_argument("--results-name", default="results_xling.json")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()
    if args.model_path:
        MODEL_PATHS[args.model] = args.model_path

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    task = args.task
    model_key = args.model
    plan = TASK_DATA_PLAN[task]
    hp = HP[task]
    out_dir = os.path.join(RUNS_ROOT, f"{task}_{model_key}{args.run_suffix}")
    if not args.eval_only and os.path.exists(out_dir):
        raise FileExistsError("Refusing to overwrite an existing training run")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] === {task} / {model_key} ===")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] hparams={hp}")
    print(f"[INFO] languages={list(LANGS_8.keys())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODEL_PATHS[model_key]
    text_keys = plan["text_keys"]
    metric_key = plan["metric"]
    primary_eval = plan["primary_eval"]

    if args.eval_only:
        ckpt_dir = os.path.join(out_dir, "ckpt")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
        run_final_eval(out_dir, task, tokenizer, model, device, hp, text_keys, args.results_name)
        return

    train_items = get_train(task)
    train_eval_dict = get_train_evals(task)
    print(f"[INFO] train={len(train_items)}, train_evals=" +
          ", ".join(f"{k}={len(v)}" for k, v in train_eval_dict.items()))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=plan["num_labels"]).to(device)

    train_loader = DataLoader(
        TextDataset(train_items, tokenizer, hp["max_len"], text_keys),
        batch_size=hp["bs"], shuffle=True, num_workers=2)
    train_eval_loaders = make_loaders(train_eval_dict, tokenizer, hp, text_keys, bs_mult=2)

    total_steps = len(train_loader) * hp["max_epochs"]
    warmup_steps = int(total_steps * hp["warmup"])
    optim = AdamW(model.parameters(), lr=hp["lr"], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({"task": task, "model": model_key, "hparams": hp, "plan": plan,
                   "languages": LANGS_8, "model_path": model_path}, f, indent=2)

    log_f = open(os.path.join(out_dir, "training_log.jsonl"), "w")
    best_metric, best_epoch, no_improve = -1.0, -1, 0

    for epoch in range(1, hp["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        sum_loss, n_correct, n_total = 0.0, 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
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
            tokenizer.save_pretrained(ckpt_dir)
            with open(os.path.join(ckpt_dir, "best_info.json"), "w") as f:
                json.dump({"best_epoch": epoch, "metric": metric_key, "value": primary,
                           "num_labels": plan["num_labels"], "label_names": plan["label_names"]}, f, indent=2)
            print(f"           --> saved new best ({metric_key}={primary:.4f})", flush=True)
        else:
            no_improve += 1
            print(f"           -- no improvement for {no_improve}/{hp['patience']} epoch(s)", flush=True)
            if no_improve >= hp["patience"]:
                print(f"[EARLY STOP] best {metric_key}={best_metric:.4f} @ E{best_epoch}", flush=True)
                break

    log_f.close()
    ckpt_dir = os.path.join(out_dir, "ckpt")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    run_final_eval(out_dir, task, tokenizer, model, device, hp, text_keys, args.results_name)
    print(f"[DONE] best epoch={best_epoch} best {metric_key}={best_metric:.4f}")


if __name__ == "__main__":
    main()
