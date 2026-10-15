"""Compare pretrained and random initialization on SST-2.

The frozen-probe mode freezes the pretrained backbone and trains only a
linear classifier. The full-finetune mode supports both pretrained and
randomly initialized models.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from pixel import (
    PIXELConfig,
    PangoCairoTextRenderer,
    PIXELForSequenceClassification,
    PoolingMode,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    resize_model_embeddings,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import TASK_DATA_PLAN, get_evals, get_train, load_sst2, stratified_subsample


MODEL_CFG = {
    "bert": {
        "family": "subword",
        "path": "bert-base-uncased",
        "renderer": None,
    },
    "pixel-base": {
        "family": "pixel",
        "path": "Team-PIXEL/pixel-base",
        "renderer": PangoCairoTextRenderer,
    },
}

FULL_HP = {
    "bert": {"max_epochs": 10, "batch_size": 32, "grad_accum": 1,
              "warmup_ratio": 0.1, "lr": 2e-5, "patience": 3},
    "pixel-base": {"max_epochs": 30, "batch_size": 32, "grad_accum": 4,
                    "warmup_steps": 100, "lr": 3e-5, "patience": 8},
}

PROBE_HP = {"max_epochs": 50, "batch_size": 256, "lr": 1e-3, "patience": 8}
MAX_SEQ_LENGTH = 64
RUNS_ROOT = "outputs/sst2"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, items, tokenizer, max_length):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        encoded = self.tokenizer(
            item["text"], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        output = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(item["label"])),
        }
        if "token_type_ids" in encoded:
            output["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
        return output


class PixelDataset(Dataset):
    def __init__(self, items, processor, transforms):
        self.items = items
        self.processor = processor
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        encoded = self.processor(text=glue_strip_spaces(item["text"]))
        return {
            "pixel_values": self.transforms(Image.fromarray(encoded.pixel_values)),
            "attention_mask": get_attention_mask(
                encoded.num_text_patches, seq_length=MAX_SEQ_LENGTH,
            ),
            "labels": torch.tensor(int(item["label"])),
        }


def force_pixel_size(model, pixels_per_patch):
    resize_model_embeddings(model, MAX_SEQ_LENGTH)
    image_size = [pixels_per_patch, pixels_per_patch * MAX_SEQ_LENGTH]
    model.config.image_size = image_size
    model.image_size = image_size
    patch_embeddings = model.vit.embeddings.patch_embeddings
    patch_embeddings.image_size = image_size
    if hasattr(patch_embeddings, "num_patches"):
        patch_embeddings.num_patches = MAX_SEQ_LENGTH


def load_items(train_size):
    items = load_sst2("train")
    return stratified_subsample(items, train_size, "label") if train_size else items


def load_pixel_processor(model_key):
    config = MODEL_CFG[model_key]
    processor = config["renderer"].from_pretrained(config["path"], rgb=True)
    processor.max_seq_length = MAX_SEQ_LENGTH
    transforms = get_transforms(
        do_resize=True,
        size=(processor.pixels_per_patch, processor.pixels_per_patch * MAX_SEQ_LENGTH),
    )
    return processor, transforms


def build_backbone(model_key, initialization, dropout=0.1):
    config = MODEL_CFG[model_key]
    if config["family"] == "subword":
        if initialization == "pretrained":
            backbone = AutoModel.from_pretrained(config["path"])
        else:
            model_config = AutoConfig.from_pretrained(config["path"])
            backbone = AutoModel.from_config(model_config)
        return backbone

    if initialization == "pretrained":
        classifier = PIXELForSequenceClassification.from_pretrained(
            config["path"], num_labels=2, pooling_mode=PoolingMode.MEAN,
            add_layer_norm=True, hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
    else:
        model_config = PIXELConfig.from_pretrained(config["path"])
        model_config.num_labels = 2
        model_config.hidden_dropout_prob = dropout
        model_config.attention_probs_dropout_prob = dropout
        classifier = PIXELForSequenceClassification(
            model_config, pooling_mode=PoolingMode.MEAN, add_layer_norm=True,
        )
    force_pixel_size(classifier, 16)
    return classifier


def build_full_model(model_key, initialization):
    config = MODEL_CFG[model_key]
    if config["family"] == "subword":
        if initialization == "pretrained":
            return AutoModelForSequenceClassification.from_pretrained(
                config["path"], num_labels=2,
            )
        model_config = AutoConfig.from_pretrained(config["path"])
        model_config.num_labels = 2
        return AutoModelForSequenceClassification.from_config(model_config)
    return build_backbone(model_key, initialization, dropout=0.1)


def make_loader(model_key, items, processor=None, transforms=None, batch_size=32, shuffle=False):
    if MODEL_CFG[model_key]["family"] == "subword":
        dataset = TextDataset(items, processor, max_length=128)
    else:
        dataset = PixelDataset(items, processor, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def model_inputs(batch):
    return {key: value for key, value in batch.items() if key != "labels"}


def extract_features(model_key, backbone, loader, device):
    backbone.eval()
    feature_batches, label_batches = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in model_inputs(batch).items()}
            labels = batch["labels"]
            outputs = backbone(**inputs, return_dict=True)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].float()
            if hidden.shape[1] == mask.shape[1] + 1:
                hidden = hidden[:, 1:, :]
            if mask.shape[1] != hidden.shape[1]:
                mask = mask[:, :hidden.shape[1]]
            denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            features = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator
            feature_batches.append(features.cpu())
            label_batches.append(labels)
    return torch.cat(feature_batches), torch.cat(label_batches)


def accuracy(predictions, labels):
    return float((predictions == labels).float().mean().item())


def run_probe(model_key, train_items, eval_items, seed, out_dir, device):
    processor = None
    transforms = None
    if MODEL_CFG[model_key]["family"] == "subword":
        processor = AutoTokenizer.from_pretrained(MODEL_CFG[model_key]["path"])
    else:
        processor, transforms = load_pixel_processor(model_key)

    backbone = build_backbone(model_key, "pretrained")
    backbone = backbone.vit if MODEL_CFG[model_key]["family"] == "pixel" else backbone
    backbone.to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    train_loader = make_loader(model_key, train_items, processor, transforms,
                               batch_size=64, shuffle=False)
    eval_loader = make_loader(model_key, eval_items, processor, transforms,
                              batch_size=64, shuffle=False)
    train_features, train_labels = extract_features(model_key, backbone, train_loader, device)
    eval_features, eval_labels = extract_features(model_key, backbone, eval_loader, device)

    probe = torch.nn.Linear(train_features.shape[1], 2).to(device)
    optimizer = AdamW(probe.parameters(), lr=PROBE_HP["lr"], weight_decay=0.01)
    train_data = TensorDataset(train_features, train_labels)
    train_feature_loader = DataLoader(train_data, batch_size=PROBE_HP["batch_size"], shuffle=True)
    best_accuracy, best_epoch, no_improve = -1.0, -1, 0
    history = []
    for epoch in range(1, PROBE_HP["max_epochs"] + 1):
        probe.train()
        for features, labels in train_feature_loader:
            logits = probe(features.to(device))
            loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            predictions = probe(eval_features.to(device)).argmax(-1).cpu()
        value = accuracy(predictions, eval_labels)
        history.append({"epoch": epoch, "eval_accuracy": value})
        if value > best_accuracy:
            best_accuracy, best_epoch, no_improve = value, epoch, 0
        else:
            no_improve += 1
            if no_improve >= PROBE_HP["patience"]:
                break

    result = {
        "model": model_key,
        "mode": "frozen_probe",
        "initialization": "pretrained",
        "seed": seed,
        "train_n": len(train_items),
        "eval_n": len(eval_items),
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "history": history,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as handle:
        json.dump(result, handle, indent=2)


def evaluate_full(model, loader, device):
    model.eval()
    predictions, labels = [], []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            losses.append(float(output.loss.item()))
            predictions.append(output.logits.argmax(-1).cpu())
            labels.append(batch["labels"].cpu())
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    return {"loss": float(np.mean(losses)), "accuracy": accuracy(predictions, labels)}


def run_full(model_key, initialization, train_items, eval_items, seed, out_dir, device):
    processor = None
    transforms = None
    if MODEL_CFG[model_key]["family"] == "subword":
        processor = AutoTokenizer.from_pretrained(MODEL_CFG[model_key]["path"])
    else:
        processor, transforms = load_pixel_processor(model_key)

    model = build_full_model(model_key, initialization).to(device)
    train_loader = make_loader(model_key, train_items, processor, transforms,
                               batch_size=FULL_HP[model_key]["batch_size"], shuffle=True)
    eval_loader = make_loader(model_key, eval_items, processor, transforms,
                              batch_size=FULL_HP[model_key]["batch_size"] * 2, shuffle=False)
    hp = FULL_HP[model_key]
    updates_per_epoch = max(1, (len(train_loader) + hp["grad_accum"] - 1) // hp["grad_accum"])
    total_updates = updates_per_epoch * hp["max_epochs"]
    warmup_steps = int(total_updates * hp["warmup_ratio"]) if "warmup_ratio" in hp else hp["warmup_steps"]
    optimizer = AdamW(model.parameters(), lr=hp["lr"], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    best_accuracy, best_epoch, no_improve = -1.0, -1, 0
    history = []
    for epoch in range(1, hp["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / hp["grad_accum"]
            loss.backward()
            if (step + 1) % hp["grad_accum"] == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            train_loss += float(output.loss.item()) * batch["labels"].size(0)
            train_correct += int((output.logits.argmax(-1) == batch["labels"]).sum().item())
            train_total += batch["labels"].size(0)
        metrics = evaluate_full(model, eval_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "eval": metrics,
        }
        history.append(record)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, best_epoch, no_improve = metrics["accuracy"], epoch, 0
            checkpoint_dir = os.path.join(out_dir, "ckpt")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            if processor is not None:
                processor.save_pretrained(checkpoint_dir)
        else:
            no_improve += 1
            if no_improve >= hp["patience"]:
                break

    result = {
        "model": model_key,
        "mode": "full_finetune",
        "initialization": initialization,
        "seed": seed,
        "train_n": len(train_items),
        "eval_n": len(eval_items),
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "history": history,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as handle:
        json.dump(result, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CFG))
    parser.add_argument("--mode", required=True, choices=["frozen_probe", "full_finetune"])
    parser.add_argument("--initialization", required=True, choices=["pretrained", "random"])
    parser.add_argument("--train-size", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-root", default=RUNS_ROOT)
    parser.add_argument("--model-path", help="Public model identifier or local checkpoint directory")
    parser.add_argument("--data-dir", help="Directory containing the SST-2 parquet files")
    args = parser.parse_args()
    if args.mode == "frozen_probe" and args.initialization != "pretrained":
        parser.error("frozen_probe requires --initialization pretrained")

    if args.model_path:
        MODEL_CFG[args.model]["path"] = args.model_path
    if args.data_dir:
        import data_utils
        data_utils.SST2_DIR = args.data_dir
    set_seed(args.seed)
    train_items = load_items(args.train_size)
    eval_items = get_evals("sst2")["validation"]
    run_name = f"{args.model}_{args.mode}_{args.initialization}_n{args.train_size}_seed{args.seed}"
    out_dir = os.path.join(args.runs_root, run_name)
    os.makedirs(out_dir, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this training script. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
    device = torch.device("cuda")
    print(f"[INFO] device={device} gpu={torch.cuda.get_device_name(0)}", flush=True)
    config = {
        "model": args.model,
        "mode": args.mode,
        "initialization": args.initialization,
        "seed": args.seed,
        "train_size": args.train_size,
        "max_seq_length": MAX_SEQ_LENGTH,
        "model_path": MODEL_CFG[args.model]["path"],
        "renderer": MODEL_CFG[args.model]["renderer"].__name__ if MODEL_CFG[args.model]["renderer"] else None,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    print(f"[INFO] {config}", flush=True)
    start = time.time()
    if args.mode == "frozen_probe":
        run_probe(args.model, train_items, eval_items, args.seed, out_dir, device)
    else:
        run_full(args.model, args.initialization, train_items, eval_items, args.seed, out_dir, device)
    print(f"[DONE] elapsed_sec={time.time() - start:.1f} output={out_dir}", flush=True)


if __name__ == "__main__":
    main()
